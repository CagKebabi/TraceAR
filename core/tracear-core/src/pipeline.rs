//! The detect <-> track state machine — the runtime entry point.
//!
//! Per marker: while lost, run detection; once acquired, run the cheap
//! sub-pixel tracker every frame and fall back to detection (same frame) the
//! moment track quality collapses.
//!
//! With many markers, detection cost is kept flat by a per-frame budget:
//! frame features are extracted once and shared (see `FrameFeatures`), and at
//! most `DetectSchedule::max_per_frame` lost markers attempt detection per
//! frame. Recently-lost markers get priority every frame (fast re-acquire of
//! the target the user is actually pointing at); long-lost markers take
//! round-robin turns, so a 10-target session costs the same per frame as a
//! 2-target one — only time-to-first-acquire of an idle target grows.

use crate::detector::{detect_marker_in, extract_frame_features, DetectorConfig};
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

/// Per-frame detection budget for multi-marker sessions.
pub struct DetectSchedule {
    /// Max full detection attempts per frame across all lost markers.
    pub max_per_frame: usize,
    /// A marker lost fewer than this many frames ago keeps priority: it is
    /// attempted every frame (within budget) instead of waiting its turn.
    pub priority_frames: u32,
    /// Long-lost ("cold") markers are only scanned every N-th frame — the
    /// frame-feature extraction dominates detection cost, so amortizing the
    /// cold scan keeps idle multi-target sessions cheap. Whenever the number
    /// of lost markers fits the per-frame budget there is nothing to
    /// amortize and every frame scans (a 1–2 target session behaves exactly
    /// like pre-0.2).
    pub cold_scan_interval: u64,
}

impl Default for DetectSchedule {
    fn default() -> Self {
        Self { max_per_frame: 2, priority_frames: 30, cold_scan_interval: 3 }
    }
}

/// Sentinel for "never found / long lost" — always outside the priority window.
const LOST_AGE_COLD: u32 = u32::MAX;

#[derive(Default)]
pub struct Pipeline {
    pub detector_config: DetectorConfig,
    pub tracker_config: TrackerConfig,
    pub schedule: DetectSchedule,
    markers: Vec<CompiledMarker>,
    states: Vec<Option<TrackState>>,
    /// Frames since this marker was last tracked/detected (LOST_AGE_COLD = never).
    lost_age: Vec<u32>,
    /// Round-robin cursor over long-lost markers.
    detect_cursor: usize,
    /// Frames processed so far (drives the cold-scan cadence).
    frame_index: u64,
    /// Diagnostic: marker indices that attempted full detection last frame.
    pub last_detect_indices: Vec<usize>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self {
            detector_config: DetectorConfig::default(),
            tracker_config: TrackerConfig::default(),
            schedule: DetectSchedule::default(),
            markers: Vec::new(),
            states: Vec::new(),
            lost_age: Vec::new(),
            detect_cursor: 0,
            frame_index: 0,
            last_detect_indices: Vec::new(),
        }
    }

    pub fn add_marker(&mut self, marker: CompiledMarker) -> usize {
        self.markers.push(marker);
        self.states.push(None);
        self.lost_age.push(LOST_AGE_COLD);
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
        for a in self.lost_age.iter_mut() {
            *a = LOST_AGE_COLD;
        }
        self.detect_cursor = 0;
        self.frame_index = 0;
    }

    /// Stateless one-shot detection on a still image (detectImage API).
    /// Frame features are extracted once and shared across all markers.
    pub fn detect_only(&self, frame: &GrayImage) -> Vec<PipelineResult> {
        let feats = extract_frame_features(frame, &self.detector_config);
        self.markers
            .iter()
            .map(|m| match detect_marker_in(m, &feats, &self.detector_config) {
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
        let n = self.markers.len();
        // The tracker wants a lightly blurred frame plus its half-resolution
        // downsample (coarse stage); build once, shared across markers.
        let blurred = if self.states.iter().any(|s| s.is_some()) {
            let b = frame.box_blur(1);
            let half = b.downsample_half();
            Some((b, half))
        } else {
            None
        };

        // Pass 1: tracking for every marker that has state.
        let mut out: Vec<Option<PipelineResult>> = (0..n).map(|_| None).collect();
        for i in 0..n {
            let tracked = match (&self.states[i], &blurred) {
                (Some(state), Some((bf, half))) => {
                    let dt_prev = state.t_last - state.t_prev;
                    let pred_scale = if dt_prev > 1e-6 { (t - state.t_last) / dt_prev } else { 1.0 };
                    // Normal pass first; on failure one wide-coarse-search
                    // recovery pass — still ~7x cheaper than full detection.
                    track_frame(&self.markers[i], bf, half, state, pred_scale, TrackMode::Normal, &self.tracker_config)
                        .or_else(|| {
                            track_frame(
                                &self.markers[i],
                                bf,
                                half,
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
                self.lost_age[i] = 0;
                let survival = tr.survived as f32 / tr.attempted.max(1) as f32;
                out[i] = Some(PipelineResult {
                    status: MarkerStatus::Tracked,
                    homography: Some(tr.h),
                    n_good: tr.survived,
                    n_total: tr.attempted,
                    quality: (survival * tr.mean_ncc.max(0.0)).min(1.0),
                });
            } else {
                self.states[i] = None;
                self.lost_age[i] = self.lost_age[i].saturating_add(1);
            }
        }

        // Pass 2: pick which lost markers get a detection attempt this frame.
        // Priority: recently lost (same-frame recovery keeps a briefly-occluded
        // target from visibly dropping), then round-robin over the cold ones.
        // Cold markers only scan on cadence frames — unless the lost set fits
        // the budget outright, in which case there is nothing to amortize.
        let lost_count = out.iter().filter(|r| r.is_none()).count();
        let interval = self.schedule.cold_scan_interval.max(1);
        let cold_scan = lost_count <= self.schedule.max_per_frame || self.frame_index % interval == 0;
        self.frame_index += 1;
        let mut budget = self.schedule.max_per_frame;
        let mut attempts: Vec<usize> = Vec::new();
        for i in 0..n {
            if budget == 0 {
                break;
            }
            if out[i].is_none() && self.lost_age[i] <= self.schedule.priority_frames {
                attempts.push(i);
                budget -= 1;
            }
        }
        if n > 0 && cold_scan {
            let mut advanced = 0;
            for k in 0..n {
                if budget == 0 {
                    break;
                }
                let i = (self.detect_cursor + k) % n;
                if out[i].is_none() && self.lost_age[i] > self.schedule.priority_frames {
                    attempts.push(i);
                    budget -= 1;
                    advanced = k + 1;
                }
            }
            self.detect_cursor = (self.detect_cursor + advanced) % n;
        }

        // Pass 3: run the attempts against one shared feature extraction.
        self.last_detect_indices = attempts.clone();
        if !attempts.is_empty() {
            let feats = extract_frame_features(frame, &self.detector_config);
            for &i in &attempts {
                if let Some(d) = detect_marker_in(&self.markers[i], &feats, &self.detector_config) {
                    self.states[i] = Some(TrackState::new(d.homography, t));
                    self.lost_age[i] = 0;
                    out[i] = Some(PipelineResult {
                        status: MarkerStatus::Detected,
                        homography: Some(d.homography),
                        n_good: d.inliers,
                        n_total: d.matches,
                        quality: (d.inliers as f32 / 40.0).min(1.0),
                    });
                }
            }
        }

        out.into_iter()
            .map(|r| {
                r.unwrap_or(PipelineResult {
                    status: MarkerStatus::NotFound,
                    homography: None,
                    n_good: 0,
                    n_total: 0,
                    quality: 0.0,
                })
            })
            .collect()
    }
}
