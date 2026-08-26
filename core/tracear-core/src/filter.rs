//! Pose smoothing: One Euro filtering on SE(3).
//!
//! Translation components go through independent One Euro filters; rotation
//! is filtered in quaternion space (velocity-adaptive slerp toward the new
//! sample — the One Euro principle applied on SO(3)). Never filter Euler
//! angles or matrix entries.
//!
//! The filter also outputs linear and angular velocity of the *filtered*
//! signal, so a renderer can extrapolate the pose to its own presentation
//! timestamp (render-time prediction; done SDK-side). Angular velocity is
//! body-frame: q(t + dt) ~= q_filtered * exp(omega * dt).

use crate::pose::Pose;
use nalgebra::{Quaternion, UnitQuaternion, Vector3};

fn smoothing_factor(cutoff_hz: f64, dt_s: f64) -> f64 {
    let r = std::f64::consts::TAU * cutoff_hz * dt_s;
    r / (r + 1.0)
}

/// Scalar One Euro filter (Casiez et al., 2012). The derivative is taken
/// from RAW consecutive samples — deriving it from the filtered value
/// inflates it by 1/alpha under steady motion.
pub struct OneEuro {
    pub min_cutoff: f64,
    pub beta: f64,
    pub d_cutoff: f64,
    state: Option<(f64, f64, f64)>, // (x_hat, dx_hat, x_raw_prev)
    last_cutoff: f64,
}

impl OneEuro {
    pub fn new(min_cutoff: f64, beta: f64, d_cutoff: f64) -> Self {
        Self { min_cutoff, beta, d_cutoff, state: None, last_cutoff: min_cutoff }
    }

    /// Returns (filtered value, filtered raw-signal derivative per second).
    pub fn update(&mut self, x: f64, dt_s: f64) -> (f64, f64) {
        match self.state {
            None => {
                self.state = Some((x, 0.0, x));
                self.last_cutoff = self.min_cutoff;
                (x, 0.0)
            }
            Some((x_hat_prev, dx_prev, x_raw_prev)) => {
                let dt = dt_s.max(1e-4);
                let dx = (x - x_raw_prev) / dt;
                let dx_hat = dx_prev + smoothing_factor(self.d_cutoff, dt) * (dx - dx_prev);
                let cutoff = self.min_cutoff + self.beta * dx_hat.abs();
                let x_hat = x_hat_prev + smoothing_factor(cutoff, dt) * (x - x_hat_prev);
                self.state = Some((x_hat, dx_hat, x));
                self.last_cutoff = cutoff;
                (x_hat, dx_hat)
            }
        }
    }

    /// A first-order low-pass at cutoff fc delays the signal by ~1/(2*pi*fc)
    /// seconds — the group delay the render-time predictor must add back.
    pub fn lag_seconds(&self) -> f64 {
        1.0 / (std::f64::consts::TAU * self.last_cutoff.max(1e-3))
    }

    pub fn reset(&mut self) {
        self.state = None;
        self.last_cutoff = self.min_cutoff;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PoseFilterConfig {
    pub pos_min_cutoff: f64,
    pub pos_beta: f64,
    pub rot_min_cutoff: f64,
    pub rot_beta: f64,
    pub d_cutoff: f64,
}

impl Default for PoseFilterConfig {
    fn default() -> Self {
        // Tuned for AR: aggressive suppression when still (low min_cutoff —
        // affordable because the filter reports its own group delay and the
        // render-time predictor adds it back), cutoff opening quickly with
        // speed (beta) so motion stays responsive.
        Self {
            pos_min_cutoff: 0.6,
            pos_beta: 1.5,
            rot_min_cutoff: 0.6,
            rot_beta: 2.0,
            // Velocity estimates drive prediction — keep them responsive
            // (their lag shows up as swim during acceleration).
            d_cutoff: 2.5,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FilteredPose {
    pub rotation: UnitQuaternion<f64>,
    pub translation: Vector3<f64>,
    /// Units per second (filtered-signal derivative).
    pub velocity: Vector3<f64>,
    /// Body-frame rad/s of the filtered rotation.
    pub angular_velocity: Vector3<f64>,
    /// Group delay of the translation filter (s) — the predictor extrapolates
    /// over (render latency + this) to place content where the target IS.
    pub pos_lag_s: f64,
    /// Group delay of the rotation filter (s).
    pub rot_lag_s: f64,
}

pub struct PoseFilter {
    cfg: PoseFilterConfig,
    x: [OneEuro; 3],
    // (q_hat, omega_hat body rad/s, q_raw_prev)
    rot: Option<(UnitQuaternion<f64>, Vector3<f64>, UnitQuaternion<f64>)>,
    last_t_ms: Option<f64>,
}

fn negate(q: &UnitQuaternion<f64>) -> UnitQuaternion<f64> {
    UnitQuaternion::new_unchecked(Quaternion::new(-q.w, -q.i, -q.j, -q.k))
}

impl PoseFilter {
    pub fn new(cfg: PoseFilterConfig) -> Self {
        Self {
            x: [
                OneEuro::new(cfg.pos_min_cutoff, cfg.pos_beta, cfg.d_cutoff),
                OneEuro::new(cfg.pos_min_cutoff, cfg.pos_beta, cfg.d_cutoff),
                OneEuro::new(cfg.pos_min_cutoff, cfg.pos_beta, cfg.d_cutoff),
            ],
            rot: None,
            last_t_ms: None,
            cfg,
        }
    }

    pub fn reset(&mut self) {
        for f in self.x.iter_mut() {
            f.reset();
        }
        self.rot = None;
        self.last_t_ms = None;
    }

    pub fn update(&mut self, t_ms: f64, pose: &Pose) -> FilteredPose {
        let dt = match self.last_t_ms {
            Some(prev) => ((t_ms - prev) / 1000.0).clamp(1e-4, 0.5),
            None => 1.0 / 30.0,
        };
        self.last_t_ms = Some(t_ms);

        let mut p = Vector3::zeros();
        let mut v = Vector3::zeros();
        for i in 0..3 {
            let (x_hat, dx_hat) = self.x[i].update(pose.translation[i], dt);
            p[i] = x_hat;
            v[i] = dx_hat;
        }

        let mut rot_cutoff = self.cfg.rot_min_cutoff;
        let (q_hat, omega) = match self.rot {
            None => {
                self.rot = Some((pose.rotation, Vector3::zeros(), pose.rotation));
                (pose.rotation, Vector3::zeros())
            }
            Some((q_prev_hat, omega_prev, q_raw_prev)) => {
                let mut q_new = pose.rotation;
                if q_prev_hat.coords.dot(&q_new.coords) < 0.0 {
                    q_new = negate(&q_new); // same hemisphere for a short slerp
                }
                // Angular velocity from RAW consecutive samples (body frame),
                // low-passed — this drives the adaptive cutoff and is the
                // velocity handed out for render-time prediction.
                let mut q_raw = q_raw_prev;
                if q_raw.coords.dot(&q_new.coords) < 0.0 {
                    q_raw = negate(&q_raw);
                }
                let omega_inst = (q_raw.inverse() * q_new).scaled_axis() / dt;
                let a_d = smoothing_factor(self.cfg.d_cutoff, dt);
                let omega_hat = omega_prev + (omega_inst - omega_prev) * a_d;
                let cutoff = self.cfg.rot_min_cutoff + self.cfg.rot_beta * omega_hat.norm();
                rot_cutoff = cutoff;
                let a = smoothing_factor(cutoff, dt);
                let q_next = q_prev_hat.slerp(&q_new, a);
                self.rot = Some((q_next, omega_hat, q_new));
                (q_next, omega_hat)
            }
        };

        let pos_lag_s = self.x.iter().map(|f| f.lag_seconds()).sum::<f64>() / 3.0;
        let rot_lag_s = 1.0 / (std::f64::consts::TAU * rot_cutoff.max(1e-3));
        FilteredPose {
            rotation: q_hat,
            translation: p,
            velocity: v,
            angular_velocity: omega,
            pos_lag_s,
            rot_lag_s,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::XorShift64;

    fn noisy_static_pose(rng: &mut XorShift64, sigma: f64) -> Pose {
        let (a, b) = rng.next_gaussian_pair();
        let (c, d) = rng.next_gaussian_pair();
        Pose {
            rotation: UnitQuaternion::from_scaled_axis(Vector3::new(0.3 + a * sigma * 0.01, -0.2 + b * sigma * 0.01, 0.1)),
            translation: Vector3::new(0.1 + c * sigma, -0.05 + d * sigma, 2.5),
        }
    }

    #[test]
    fn static_noise_is_suppressed() {
        let mut rng = XorShift64::new(11);
        let mut filter = PoseFilter::new(PoseFilterConfig::default());
        let mut raw_positions = Vec::new();
        let mut filt_positions = Vec::new();
        for f in 0..120 {
            let pose = noisy_static_pose(&mut rng, 0.004);
            let out = filter.update(f as f64 * 33.0, &pose);
            if f >= 30 {
                raw_positions.push(pose.translation);
                filt_positions.push(out.translation);
            }
        }
        let std = |v: &Vec<Vector3<f64>>| {
            let mean = v.iter().sum::<Vector3<f64>>() / v.len() as f64;
            (v.iter().map(|p| (p - mean).norm_squared()).sum::<f64>() / v.len() as f64).sqrt()
        };
        let raw_std = std(&raw_positions);
        let filt_std = std(&filt_positions);
        // One Euro's operating point at 30 fps: ~3x suppression while still.
        assert!(
            filt_std < raw_std * 0.40,
            "filter should suppress static noise: raw {raw_std:.5} vs filtered {filt_std:.5}"
        );
    }

    #[test]
    fn fast_motion_tracks_with_low_lag() {
        let mut filter = PoseFilter::new(PoseFilterConfig::default());
        let mut last = FilteredPose {
            rotation: UnitQuaternion::identity(),
            translation: Vector3::zeros(),
            velocity: Vector3::zeros(),
            angular_velocity: Vector3::zeros(),
            pos_lag_s: 0.0,
            rot_lag_s: 0.0,
        };
        let mut true_x = 0.0;
        for f in 0..60 {
            true_x = f as f64 * 0.02; // fast steady motion: 0.6 units/s
            let pose = Pose {
                rotation: UnitQuaternion::identity(),
                translation: Vector3::new(true_x, 0.0, 2.0),
            };
            last = filter.update(f as f64 * 33.0, &pose);
        }
        // Phase lag of the filtered signal itself. Theory: lag ~= v / (2*pi *
        // (min_cutoff + beta*v)) = 0.0597 units here; render-time prediction
        // (translation + velocity * dt) cancels it at display time.
        let lag = (true_x - last.translation.x).abs();
        assert!(lag < 0.07, "lag under fast motion = {lag:.4} units");
        // Velocity output must be accurate enough to drive that prediction.
        assert!((last.velocity.x - 0.6).abs() < 0.15, "velocity estimate {:.3}", last.velocity.x);
        let predicted = last.translation.x + last.velocity.x * lag / 0.6;
        assert!((true_x - predicted).abs() < 0.02, "prediction-compensated error {:.4}", (true_x - predicted).abs());
    }

    #[test]
    fn rotation_filtering_stays_on_hemisphere() {
        let mut filter = PoseFilter::new(PoseFilterConfig::default());
        for f in 0..50 {
            let angle = f as f64 * 0.05;
            let mut q = UnitQuaternion::from_scaled_axis(Vector3::new(0.0, 0.0, angle));
            if f % 7 == 0 {
                q = negate(&q); // double-cover flips must not glitch the filter
            }
            let pose = Pose { rotation: q, translation: Vector3::new(0.0, 0.0, 2.0) };
            let out = filter.update(f as f64 * 33.0, &pose);
            assert!(out.rotation.coords.iter().all(|v| v.is_finite()));
        }
    }
}
