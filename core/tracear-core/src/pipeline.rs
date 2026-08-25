//! The detect <-> track state machine — the runtime entry point.
//!
//! Per marker: while lost, run detection; once acquired, run the cheap
//! sub-pixel tracker every frame and fall back to detection (same frame) the
//! moment track quality collapses.

use crate::detector::{detect_marker, DetectorConfig};
use crate::image::GrayImage;
use crate::marker::CompiledMarker;
use crate::tracker::{track_frame, TrackMode, TrackState, TrackerConfig};
use nalgebra::Matrix3;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MarkerStatus {
    NotFound,
    /// Acquired by full detection this frame.
    Detected,
    /// Followed by the sub-pixel tracker this frame.
    Tracked,
}

pub struct PipelineResult {
    pub status: MarkerStatus,
    pub homography: Option<Matrix3<f64>>,
    /// Detection: RANSAC inliers. Tracking: surviving patches.
    pub n_good: usize,
    /// Detection: total matches. Tracking: attempted patches.
    pub n_total: usize,
    /// 0..1 confidence.
    pub quality: f32,
}

#[derive(Default)]
pub struct Pipeline {
    pub detector_config: DetectorConfig,
    pub tracker_config: TrackerConfig,
    markers: Vec<CompiledMarker>,
    states: Vec<Option<TrackState>>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self {
            detector_config: DetectorConfig::default(),
            tracker_config: TrackerConfig::default(),
            markers: Vec::new(),
            states: Vec::new(),
        }
    }

    pub fn add_marker(&mut self, marker: CompiledMarker) -> usize {
        self.markers.push(marker);
        self.states.push(None);
        self.markers.len() - 1
    }

    pub fn marker(&self, index: usize) -> Option<&CompiledMarker> {
        self.markers.get(index)
    }

    pub fn marker_count(&self) -> usize {
        self.markers.len()
    }

    /// Drop all tracking state (markers stay).
    pub fn reset(&mut self) {
        for s in self.states.iter_mut() {
            *s = None;
        }
    }

    /// Stateless one-shot detection on a still image (detectImage API).
    pub fn detect_only(&self, frame: &GrayImage) -> Vec<PipelineResult> {
        self.markers
            .iter()
            .map(|m| match detect_marker(m, frame, &self.detector_config) {
                Some(d) => PipelineResult {
                    status: MarkerStatus::Detected,
                    homography: Some(d.homography),
                    n_good: d.inliers,
                    n_total: d.matches,
                    quality: (d.inliers as f32 / 40.0).min(1.0),
                },
                None => PipelineResult {
                    status: MarkerStatus::NotFound,
                    homography: None,
                    n_good: 0,
                    n_total: 0,
                    quality: 0.0,
                },
            })
            .collect()
    }

    /// Process one camera frame (stateful). `t` is the frame's capture
    /// timestamp in ms (any monotonic clock): frames do not arrive uniformly
    /// (a detection frame takes ~5x a tracking frame), and the tracker's
    /// motion prediction must be scaled by the real time gap.
    pub fn process(&mut self, frame: &GrayImage, t: f64) -> Vec<PipelineResult> {
        // The tracker wants a lightly blurred frame; blur once, shared.
        let blurred = if self.states.iter().any(|s| s.is_some()) {
            Some(frame.box_blur(1))
        } else {
            None
        };
        let mut out = Vec::with_capacity(self.markers.len());
        for i in 0..self.markers.len() {
            let tracked = match (&self.states[i], &blurred) {
                (Some(state), Some(bf)) => {
                    let dt_prev = state.t_last - state.t_prev;
                    let pred_scale = if dt_prev > 1e-6 { (t - state.t_last) / dt_prev } else { 1.0 };
                    // Normal pass first; on failure one wide-presearch
                    // recovery pass — still ~10x cheaper than full detection.
                    track_frame(&self.markers[i], bf, state, pred_scale, TrackMode::Normal, &self.tracker_config)
                        .or_else(|| {
                            track_frame(
                                &self.markers[i],
                                bf,
                                state,
                                pred_scale,
                                TrackMode::Recovery,
                                &self.tracker_config,
                            )
                        })
                }
                _ => None,
            };
            if let Some(tr) = tracked {
                let state = self.states[i].as_mut().unwrap();
                state.h_prev = Some(state.h);
                state.t_prev = state.t_last;
                state.t_last = t;
                state.h = tr.h;
                state.last_pred_err = tr.mean_pred_err;
                state.frames_tracked += 1;
                let survival = tr.survived as f32 / tr.attempted.max(1) as f32;
                out.push(PipelineResult {
                    status: MarkerStatus::Tracked,
                    homography: Some(tr.h),
                    n_good: tr.survived,
                    n_total: tr.attempted,
                    quality: (survival * tr.mean_ncc.max(0.0)).min(1.0),
                });
                continue;
            }
            // Lost (or never had) the target: full detection on the raw frame.
            self.states[i] = None;
            match detect_marker(&self.markers[i], frame, &self.detector_config) {
                Some(d) => {
                    self.states[i] = Some(TrackState::new(d.homography, t));
                    out.push(PipelineResult {
                        status: MarkerStatus::Detected,
                        homography: Some(d.homography),
                        n_good: d.inliers,
                        n_total: d.matches,
                        quality: (d.inliers as f32 / 40.0).min(1.0),
                    });
                }
                None => out.push(PipelineResult {
                    status: MarkerStatus::NotFound,
                    homography: None,
                    n_good: 0,
                    n_total: 0,
                    quality: 0.0,
                }),
            }
        }
        out
    }
}
