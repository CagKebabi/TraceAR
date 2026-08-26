//! Session: the full runtime — detect/track pipeline + focal estimation +
//! pose estimation + SE(3) filtering, per marker. This is what the WASM
//! Engine wraps and what native benches drive.

use crate::filter::{FilteredPose, PoseFilter, PoseFilterConfig};
use crate::image::GrayImage;
use crate::marker::CompiledMarker;
use crate::pipeline::{MarkerStatus, Pipeline, PipelineResult};
use crate::pose::{
    estimate_pose, marker_from_object, FocalEstimator, Intrinsics, Pose, DEFAULT_FOCAL_RATIO,
};

pub struct SessionConfig {
    pub default_focal_ratio: f64,
    pub filter: PoseFilterConfig,
    /// Consecutive not-found frames before the pose filter state resets.
    pub reset_after_misses: u32,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            default_focal_ratio: DEFAULT_FOCAL_RATIO,
            filter: PoseFilterConfig::default(),
            reset_after_misses: 10,
        }
    }
}

struct MarkerPoseState {
    phys_width: f64,
    last_pose: Option<Pose>,
    filter: PoseFilter,
    misses: u32,
}

pub struct SessionResult {
    pub tracking: PipelineResult,
    /// Filtered pose (with velocities) when the target was found and pose
    /// estimation succeeded.
    pub pose: Option<FilteredPose>,
}

pub struct Session {
    pub pipeline: Pipeline,
    pub config: SessionConfig,
    focal: FocalEstimator,
    states: Vec<MarkerPoseState>,
}

impl Session {
    pub fn new(config: SessionConfig) -> Self {
        let focal = FocalEstimator::new(config.default_focal_ratio);
        Self { pipeline: Pipeline::new(), focal, states: Vec::new(), config }
    }

    pub fn add_marker(&mut self, marker: CompiledMarker, phys_width: f64) -> usize {
        let idx = self.pipeline.add_marker(marker);
        self.states.push(MarkerPoseState {
            phys_width: if phys_width > 0.0 { phys_width } else { 1.0 },
            last_pose: None,
            filter: PoseFilter::new(self.config.filter),
            misses: 0,
        });
        idx
    }

    pub fn marker(&self, index: usize) -> Option<&CompiledMarker> {
        self.pipeline.marker(index)
    }

    pub fn marker_count(&self) -> usize {
        self.pipeline.marker_count()
    }

    /// Current focal estimate as f / frame_width.
    pub fn focal_ratio(&self) -> f64 {
        self.focal.estimate()
    }

    pub fn reset(&mut self) {
        self.pipeline.reset();
        for s in self.states.iter_mut() {
            s.last_pose = None;
            s.filter.reset();
            s.misses = 0;
        }
    }

    pub fn process(&mut self, frame: &GrayImage, t_ms: f64) -> Vec<SessionResult> {
        let results = self.pipeline.process(frame, t_ms);
        let k = Intrinsics::from_focal_ratio(self.focal.estimate(), frame.w as f64, frame.h as f64);
        let mut out = Vec::with_capacity(results.len());
        for (i, r) in results.into_iter().enumerate() {
            let state = &mut self.states[i];
            let mut pose_out = None;
            match (&r.homography, r.status) {
                (Some(h), status) => {
                    state.misses = 0;
                    let marker = self.pipeline.marker(i).unwrap();
                    let (mw, mh) = (marker.width as f64, marker.height as f64);
                    // Tracked homographies are sub-pixel — the good data for
                    // online focal self-calibration.
                    if status == MarkerStatus::Tracked {
                        let h_obj = h * marker_from_object(mw, mh, state.phys_width);
                        self.focal.observe(&h_obj, frame.w as f64, frame.h as f64);
                    }
                    if let Some(p) =
                        estimate_pose(h, mw, mh, state.phys_width, &k, state.last_pose.as_ref())
                    {
                        state.last_pose = Some(p);
                        pose_out = Some(state.filter.update(t_ms, &p));
                    }
                }
                (None, _) => {
                    state.misses += 1;
                    if state.misses >= self.config.reset_after_misses {
                        state.last_pose = None;
                        state.filter.reset();
                    }
                }
            }
            out.push(SessionResult { tracking: r, pose: pose_out });
        }
        out
    }

    /// Stateless one-shot detection (detectImage API): raw pose, no filter.
    pub fn detect_only(&self, frame: &GrayImage) -> Vec<SessionResult> {
        let results = self.pipeline.detect_only(frame);
        let k = Intrinsics::from_focal_ratio(self.focal.estimate(), frame.w as f64, frame.h as f64);
        results
            .into_iter()
            .enumerate()
            .map(|(i, r)| {
                let pose = r.homography.as_ref().and_then(|h| {
                    let marker = self.pipeline.marker(i)?;
                    let p = estimate_pose(
                        h,
                        marker.width as f64,
                        marker.height as f64,
                        self.states[i].phys_width,
                        &k,
                        None,
                    )?;
                    Some(FilteredPose {
                        rotation: p.rotation,
                        translation: p.translation,
                        velocity: nalgebra::Vector3::zeros(),
                        angular_velocity: nalgebra::Vector3::zeros(),
                    })
                });
                SessionResult { tracking: r, pose }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::warp_onto_aa;
    use crate::marker::{compile_marker, CompileConfig};
    use crate::pose::{object_point, project_point};
    use crate::synthetic;

    #[test]
    fn static_scene_produces_stable_filtered_pose() {
        let marker_img = synthetic::textured_image(320, 320, 7);
        let mut session = Session::new(SessionConfig::default());
        session.add_marker(compile_marker(&marker_img, &CompileConfig::default()), 1.0);
        let h_gt = synthetic::trajectory_homography(320.0, 320.0, 640.0, 480.0, 0.4);
        let bg = synthetic::textured_image(640, 480, 42);

        let corners_obj: Vec<_> = [(0.0, 0.0), (320.0, 0.0), (320.0, 320.0), (0.0, 320.0)]
            .iter()
            .map(|&(x, y)| object_point(x, y, 320.0, 320.0, 1.0))
            .collect();
        let mut projected: Vec<[(f64, f64); 4]> = Vec::new();
        for f in 0..40u64 {
            let mut frame = warp_onto_aa(&marker_img, &h_gt.try_inverse().unwrap(), &bg);
            synthetic::add_gaussian_noise(&mut frame, 2.0, 1000 + f);
            let res = &session.process(&frame, f as f64 * 33.0)[0];
            if f >= 5 {
                let pose = res.pose.as_ref().expect("pose should be available");
                let k = Intrinsics::from_focal_ratio(session.focal_ratio(), 640.0, 480.0);
                let p = crate::pose::Pose { rotation: pose.rotation, translation: pose.translation };
                let mut c = [(0.0, 0.0); 4];
                for (j, obj) in corners_obj.iter().enumerate() {
                    c[j] = project_point(&p, &k, obj).expect("in front of camera");
                }
                projected.push(c);
            }
        }
        // Filtered reprojected-corner jitter (M3 target < 0.15 px).
        let n = projected.len() as f64;
        let mut jitter = 0.0;
        for c in 0..4 {
            let mx = projected.iter().map(|f| f[c].0).sum::<f64>() / n;
            let my = projected.iter().map(|f| f[c].1).sum::<f64>() / n;
            let var = projected.iter().map(|f| (f[c].0 - mx).powi(2) + (f[c].1 - my).powi(2)).sum::<f64>() / n;
            jitter += var.sqrt();
        }
        jitter /= 4.0;
        assert!(jitter < 0.15, "filtered reprojected jitter = {jitter:.4} px");
    }

    #[test]
    fn pose_is_geometrically_plausible() {
        let marker_img = synthetic::textured_image(320, 320, 7);
        let mut session = Session::new(SessionConfig::default());
        session.add_marker(compile_marker(&marker_img, &CompileConfig::default()), 1.0);
        let h_gt = synthetic::trajectory_homography(320.0, 320.0, 640.0, 480.0, 0.4);
        let bg = synthetic::textured_image(640, 480, 21);
        let mut frame = warp_onto_aa(&marker_img, &h_gt.try_inverse().unwrap(), &bg);
        synthetic::add_gaussian_noise(&mut frame, 2.0, 5);
        let res = &session.process(&frame, 0.0)[0];
        let pose = res.pose.as_ref().expect("pose");
        // In front of the camera at a plausible planar-target distance
        // (marker width 1 unit, ~0.5 screen coverage -> a few units away).
        assert!(pose.translation.z > 0.5 && pose.translation.z < 10.0, "z = {}", pose.translation.z);
        // Reprojection of the marker center must land near the H-projected center.
        let k = Intrinsics::from_focal_ratio(session.focal_ratio(), 640.0, 480.0);
        let p = crate::pose::Pose { rotation: pose.rotation, translation: pose.translation };
        let center = project_point(&p, &k, &nalgebra::Vector3::zeros()).unwrap();
        let hc = crate::homography::project(&h_gt, 160.0, 160.0);
        let err = ((center.0 - hc.0).powi(2) + (center.1 - hc.1).powi(2)).sqrt();
        assert!(err < 3.0, "center reprojection error = {err:.2} px");
    }
}
