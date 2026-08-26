//! Homography -> 6DoF pose for planar targets.
//!
//! Conventions:
//! - **Object frame**: origin at the marker center, X right (+marker x),
//!   Y up (-marker y), Z out of the marker face toward the viewer.
//!   Units: `phys_width` per marker width (1.0 default).
//! - **Camera frame**: OpenCV — X right, Y down, Z forward. A visible target
//!   has t_z > 0. (Renderer adapters convert to their own conventions.)
//!
//! Estimation: Zhang-style decomposition of K^-1 * H_obj (columns ~ r1, r2, t)
//! with SVD orthonormalization, then Levenberg-Marquardt refinement of the
//! reprojection error over a 3x3 grid of plane points. When the previous
//! frame's pose is available it seeds the refinement, which keeps the
//! solution on the same branch of the near-frontal planar-pose ambiguity.
//!
//! The focal length is not knowable a priori on the web: `FocalEstimator`
//! accumulates the two Zhang self-calibration constraints each tilted view
//! of the plane provides and serves a median-filtered estimate, starting
//! from a typical-phone-FOV default.

use nalgebra::{Matrix3, Matrix6, Rotation3, UnitQuaternion, Vector3, Vector6};

#[derive(Clone, Copy, Debug)]
pub struct Intrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

impl Intrinsics {
    /// `focal_ratio` = f / image_width (square pixels, centered principal point).
    pub fn from_focal_ratio(focal_ratio: f64, width: f64, height: f64) -> Self {
        let f = focal_ratio * width;
        Self { fx: f, fy: f, cx: width / 2.0, cy: height / 2.0 }
    }
}

/// Typical phone main camera: ~65 deg horizontal FOV -> f ~= 0.785 * width.
pub const DEFAULT_FOCAL_RATIO: f64 = 0.785;

#[derive(Clone, Copy, Debug)]
pub struct Pose {
    pub rotation: UnitQuaternion<f64>,
    pub translation: Vector3<f64>,
}

/// Maps object-plane homogeneous coords (X, Y, 1) to marker pixel coords.
pub fn marker_from_object(marker_w: f64, marker_h: f64, phys_width: f64) -> Matrix3<f64> {
    let s = phys_width / marker_w;
    Matrix3::new(
        1.0 / s, 0.0, marker_w / 2.0,
        0.0, -1.0 / s, marker_h / 2.0,
        0.0, 0.0, 1.0,
    )
}

/// Object coords of a marker-pixel point (Z = 0 plane).
pub fn object_point(mx: f64, my: f64, marker_w: f64, marker_h: f64, phys_width: f64) -> Vector3<f64> {
    let s = phys_width / marker_w;
    Vector3::new((mx - marker_w / 2.0) * s, (marker_h / 2.0 - my) * s, 0.0)
}

pub fn project_point(pose: &Pose, k: &Intrinsics, p_obj: &Vector3<f64>) -> Option<(f64, f64)> {
    let pc = pose.rotation * p_obj + pose.translation;
    if pc.z < 1e-9 {
        return None;
    }
    Some((k.fx * pc.x / pc.z + k.cx, k.fy * pc.y / pc.z + k.cy))
}

/// Zhang-style closed-form decomposition (no refinement).
pub fn pose_from_homography(h_obj: &Matrix3<f64>, k: &Intrinsics) -> Option<Pose> {
    let kinv = Matrix3::new(
        1.0 / k.fx, 0.0, -k.cx / k.fx,
        0.0, 1.0 / k.fy, -k.cy / k.fy,
        0.0, 0.0, 1.0,
    );
    let b = kinv * h_obj;
    let b1 = b.column(0).into_owned();
    let b2 = b.column(1).into_owned();
    let b3 = b.column(2).into_owned();
    let n1 = b1.norm();
    let n2 = b2.norm();
    if n1 < 1e-12 || n2 < 1e-12 {
        return None;
    }
    let mut lambda = 2.0 / (n1 + n2);
    if b3[2] * lambda < 0.0 {
        lambda = -lambda; // target must sit in front of the camera
    }
    let r1 = b1 * lambda;
    let r2 = b2 * lambda;
    let r3 = r1.cross(&r2);
    let r_approx = Matrix3::from_columns(&[r1, r2, r3]);
    let svd = r_approx.svd(true, true);
    let (u, v_t) = (svd.u?, svd.v_t?);
    let mut r = u * v_t;
    if r.determinant() < 0.0 {
        let mut u2 = u;
        for i in 0..3 {
            u2[(i, 2)] = -u2[(i, 2)];
        }
        r = u2 * v_t;
    }
    let t = b3 * lambda;
    if !t.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some(Pose {
        rotation: UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r)),
        translation: t,
    })
}

fn reprojection_error_sq(pose: &Pose, k: &Intrinsics, obj: &[Vector3<f64>], img: &[(f64, f64)]) -> f64 {
    let mut e = 0.0;
    for (p, &(u, v)) in obj.iter().zip(img) {
        match project_point(pose, k, p) {
            Some((pu, pv)) => e += (pu - u).powi(2) + (pv - v).powi(2),
            None => return f64::INFINITY,
        }
    }
    e
}

fn apply_delta(pose: &Pose, d: &Vector6<f64>) -> Pose {
    let dr = UnitQuaternion::from_scaled_axis(Vector3::new(d[0], d[1], d[2]));
    Pose {
        rotation: dr * pose.rotation,
        translation: pose.translation + Vector3::new(d[3], d[4], d[5]),
    }
}

/// Levenberg-Marquardt refinement with a numeric Jacobian (few points, cheap).
pub fn refine_pose(
    pose: &Pose,
    k: &Intrinsics,
    obj: &[Vector3<f64>],
    img: &[(f64, f64)],
    iters: usize,
) -> Pose {
    let n = obj.len();
    let mut cur = *pose;
    let mut cur_err = reprojection_error_sq(&cur, k, obj, img);
    let mut lambda = 1e-3;
    let eps = 1e-6;
    for _ in 0..iters {
        // residuals + numeric Jacobian (central differences)
        let mut jtj = Matrix6::<f64>::zeros();
        let mut jtr = Vector6::<f64>::zeros();
        let mut ok = true;
        for i in 0..n {
            let (pu, pv) = match project_point(&cur, k, &obj[i]) {
                Some(p) => p,
                None => {
                    ok = false;
                    break;
                }
            };
            let ru = pu - img[i].0;
            let rv = pv - img[i].1;
            let mut ju = Vector6::<f64>::zeros();
            let mut jv = Vector6::<f64>::zeros();
            for p in 0..6 {
                let mut dp = Vector6::<f64>::zeros();
                dp[p] = eps;
                let plus = apply_delta(&cur, &dp);
                dp[p] = -eps;
                let minus = apply_delta(&cur, &dp);
                let (a, b) = match (project_point(&plus, k, &obj[i]), project_point(&minus, k, &obj[i])) {
                    (Some(a), Some(b)) => (a, b),
                    _ => {
                        ok = false;
                        break;
                    }
                };
                ju[p] = (a.0 - b.0) / (2.0 * eps);
                jv[p] = (a.1 - b.1) / (2.0 * eps);
            }
            if !ok {
                break;
            }
            jtj += ju * ju.transpose() + jv * jv.transpose();
            jtr += ju * ru + jv * rv;
        }
        if !ok {
            break;
        }
        let damped = jtj + Matrix6::identity() * lambda * jtj.trace().max(1e-12) / 6.0;
        let Some(step) = damped.lu().solve(&(-jtr)) else { break };
        let candidate = apply_delta(&cur, &step);
        let err = reprojection_error_sq(&candidate, k, obj, img);
        if err < cur_err {
            cur = candidate;
            cur_err = err;
            lambda = (lambda / 3.0).max(1e-9);
            if step.norm() < 1e-10 {
                break;
            }
        } else {
            lambda = (lambda * 5.0).min(1e3);
        }
    }
    cur
}

/// Full estimation from a marker-px homography. `prev` (if consistent with
/// the measurement) seeds refinement to stay on the same ambiguity branch.
pub fn estimate_pose(
    h_marker: &Matrix3<f64>,
    marker_w: f64,
    marker_h: f64,
    phys_width: f64,
    k: &Intrinsics,
    prev: Option<&Pose>,
) -> Option<Pose> {
    // 3x3 grid of plane points: observations from the homography, object
    // coords from the marker geometry.
    let mut obj = Vec::with_capacity(9);
    let mut img = Vec::with_capacity(9);
    for gy in 0..3 {
        for gx in 0..3 {
            let mx = marker_w * gx as f64 / 2.0;
            let my = marker_h * gy as f64 / 2.0;
            let p = h_marker * Vector3::new(mx, my, 1.0);
            if p.z.abs() < 1e-12 {
                return None;
            }
            img.push((p.x / p.z, p.y / p.z));
            obj.push(object_point(mx, my, marker_w, marker_h, phys_width));
        }
    }
    let h_obj = h_marker * marker_from_object(marker_w, marker_h, phys_width);
    let decomposed = pose_from_homography(&h_obj, k)?;
    let init = match prev {
        Some(p) if reprojection_error_sq(p, k, &obj, &img) < reprojection_error_sq(&decomposed, k, &obj, &img).max(400.0) => *p,
        _ => decomposed,
    };
    let refined = refine_pose(&init, k, &obj, &img, 8);
    if !refined.translation.iter().all(|v| v.is_finite()) || refined.translation.z <= 0.0 {
        return None;
    }
    Some(refined)
}

/// Online focal estimation from the plane's self-calibration constraints
/// (each sufficiently tilted view yields up to two estimates of f).
pub struct FocalEstimator {
    /// Valid f/width ratio samples (bounded ring).
    samples: Vec<f64>,
    pos: usize,
    observed: usize,
    /// Currently served estimate; updated at milestones with hysteresis so a
    /// per-frame drifting median cannot make rendered content "breathe".
    served: Option<f64>,
    next_recompute: usize,
    pub default_ratio: f64,
}

impl FocalEstimator {
    pub fn new(default_ratio: f64) -> Self {
        Self {
            samples: Vec::new(),
            pos: 0,
            observed: 0,
            served: None,
            next_recompute: 12,
            default_ratio,
        }
    }

    fn median(&self) -> f64 {
        let mut s = self.samples.clone();
        s.sort_by(f64::total_cmp);
        s[s.len() / 2]
    }

    /// Feed one object->image homography (any uniform-scale object frame).
    pub fn observe(&mut self, h_obj: &Matrix3<f64>, width: f64, height: f64) {
        let (cx, cy) = (width / 2.0, height / 2.0);
        let col = |i: usize| (h_obj[(0, i)], h_obj[(1, i)], h_obj[(2, i)]);
        let (a0, b0, w0) = col(0);
        let (a1, b1, w1) = col(1);
        let u0 = a0 - cx * w0;
        let v0 = b0 - cy * w0;
        let u1 = a1 - cx * w1;
        let v1 = b1 - cy * w1;
        let mut pushed = 0usize;
        let mut push = |samples: &mut Vec<f64>, pos: &mut usize, f2: f64| {
            if f2.is_finite() && f2 > 0.0 {
                let ratio = f2.sqrt() / width;
                if (0.4..=2.5).contains(&ratio) {
                    if samples.len() < 90 {
                        samples.push(ratio);
                    } else {
                        samples[*pos] = ratio;
                        *pos = (*pos + 1) % 90;
                    }
                    return 1;
                }
            }
            0
        };
        let d_a = w0 * w1;
        if d_a.abs() > 1e-12 {
            pushed += push(&mut self.samples, &mut self.pos, -(u0 * u1 + v0 * v1) / d_a);
        }
        let d_b = w1 * w1 - w0 * w0;
        if d_b.abs() > 1e-12 {
            pushed += push(&mut self.samples, &mut self.pos, (u0 * u0 + v0 * v0 - u1 * u1 - v1 * v1) / d_b);
        }
        self.observed += pushed;
        // Milestone recompute (12, 24, 48, ... observations) with 2%
        // hysteresis: the served value moves rarely and only meaningfully.
        if self.observed >= self.next_recompute && !self.samples.is_empty() {
            self.next_recompute = self.observed * 2;
            let m = self.median();
            match self.served {
                Some(cur) if ((m - cur) / cur).abs() <= 0.02 => {}
                _ => self.served = Some(m),
            }
        }
    }

    /// Currently served estimate; the default until enough evidence.
    pub fn estimate(&self) -> f64 {
        self.served.unwrap_or(self.default_ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_pose() -> Pose {
        Pose {
            rotation: UnitQuaternion::from_scaled_axis(Vector3::new(0.35, -0.25, 0.15)),
            translation: Vector3::new(0.12, -0.06, 2.4),
        }
    }

    /// Build the exact homography a pose induces: H_obj columns = K[r1 r2 t].
    fn homography_for(pose: &Pose, k: &Intrinsics, marker_w: f64, marker_h: f64, phys_width: f64) -> Matrix3<f64> {
        let km = Matrix3::new(k.fx, 0.0, k.cx, 0.0, k.fy, k.cy, 0.0, 0.0, 1.0);
        let r = pose.rotation.to_rotation_matrix();
        let rt = Matrix3::from_columns(&[
            r.matrix().column(0).into_owned(),
            r.matrix().column(1).into_owned(),
            pose.translation,
        ]);
        let h_obj = km * rt;
        // marker-px homography = H_obj * (object -> marker)^-1
        h_obj * marker_from_object(marker_w, marker_h, phys_width).try_inverse().unwrap()
    }

    #[test]
    fn recovers_exact_pose_from_homography() {
        let k = Intrinsics { fx: 500.0, fy: 500.0, cx: 320.0, cy: 240.0 };
        let gt = sample_pose();
        let h = homography_for(&gt, &k, 320.0, 320.0, 1.0);
        let est = estimate_pose(&h, 320.0, 320.0, 1.0, &k, None).unwrap();
        let ang = est.rotation.angle_to(&gt.rotation).to_degrees();
        let dt = (est.translation - gt.translation).norm();
        assert!(ang < 0.1, "rotation error {ang:.3} deg");
        assert!(dt < 0.005, "translation error {dt:.4}");
    }

    #[test]
    fn previous_pose_seeds_refinement() {
        let k = Intrinsics { fx: 500.0, fy: 500.0, cx: 320.0, cy: 240.0 };
        let gt = sample_pose();
        let h = homography_for(&gt, &k, 320.0, 320.0, 1.0);
        // Slightly stale previous pose (as between consecutive frames).
        let prev = Pose {
            rotation: UnitQuaternion::from_scaled_axis(Vector3::new(0.34, -0.24, 0.15)),
            translation: gt.translation + Vector3::new(0.01, 0.005, -0.02),
        };
        let est = estimate_pose(&h, 320.0, 320.0, 1.0, &k, Some(&prev)).unwrap();
        assert!(est.rotation.angle_to(&gt.rotation).to_degrees() < 0.1);
    }

    #[test]
    fn physical_width_scales_translation() {
        let k = Intrinsics { fx: 500.0, fy: 500.0, cx: 320.0, cy: 240.0 };
        let gt = sample_pose();
        let h = homography_for(&gt, &k, 320.0, 320.0, 1.0);
        // Same homography, marker declared 0.2 "meters" wide: geometry is
        // similar with translation scaled by 0.2.
        let est = estimate_pose(&h, 320.0, 320.0, 0.2, &k, None).unwrap();
        let dt = (est.translation - gt.translation * 0.2).norm();
        assert!(dt < 0.01, "scaled translation error {dt:.4}");
    }

    #[test]
    fn focal_estimator_converges() {
        let f_gt = 520.0;
        let k = Intrinsics { fx: f_gt, fy: f_gt, cx: 320.0, cy: 240.0 };
        let mut est = FocalEstimator::new(DEFAULT_FOCAL_RATIO);
        for i in 0..40 {
            let ph = i as f64 * 0.37;
            let pose = Pose {
                rotation: UnitQuaternion::from_scaled_axis(Vector3::new(
                    0.5 * ph.sin(),
                    0.4 * (ph * 1.3).cos(),
                    0.2 * (ph * 0.7).sin(),
                )),
                translation: Vector3::new(0.2 * ph.cos(), 0.1 * ph.sin(), 2.0 + 0.3 * ph.sin()),
            };
            let h = homography_for(&pose, &k, 320.0, 320.0, 1.0);
            let h_obj = h * marker_from_object(320.0, 320.0, 1.0);
            est.observe(&h_obj, 640.0, 480.0);
        }
        let ratio = est.estimate();
        let err = (ratio - f_gt / 640.0).abs() / (f_gt / 640.0);
        assert!(err < 0.05, "focal ratio {ratio:.3} vs gt {:.3}", f_gt / 640.0);
    }
}
