//! Runtime detection: find a compiled marker in a camera frame and return the
//! marker->frame homography.
//!
//! Frame-side work (pyramid + FAST + BRIEF) does not depend on the marker, so
//! it is factored into [`FrameFeatures`]: extract once per frame, then match
//! any number of markers against it. With multi-target sessions this is the
//! difference between O(markers) and O(1) feature extractions per frame.

use crate::brief::Descriptor;
use crate::features::extract_features;
use crate::homography::{self, quad_sane};
use crate::image::{build_pyramid, GrayImage};
use crate::marker::CompiledMarker;
use crate::matcher;
use nalgebra::Matrix3;

pub struct DetectorConfig {
    pub fast_threshold: u8,
    pub max_features_per_level: usize,
    pub pyramid_min_side: usize,
    pub pyramid_max_levels: usize,
    pub match_max_dist: u32,
    pub match_ratio: f32,
    /// "Second best" for the ratio test must be at least this far away in
    /// marker space (see matcher docs).
    pub second_dist_px: f32,
    pub ransac_thresh_px: f64,
    pub ransac_iters: usize,
    pub min_inliers: usize,
    pub min_inlier_ratio: f32,
    pub seed: u64,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            fast_threshold: 20,
            max_features_per_level: 350,
            pyramid_min_side: 80,
            pyramid_max_levels: 4,
            match_max_dist: 64,
            match_ratio: 0.8,
            second_dist_px: 8.0,
            ransac_thresh_px: 3.0,
            ransac_iters: 500,
            min_inliers: 10,
            // False-positive rejection is carried by the absolute inlier count
            // and the quad sanity check; this ratio is only a loose bound. A
            // small marker in a feature-rich scene legitimately produces many
            // unrelated background matches, so keep this permissive.
            min_inlier_ratio: 0.12,
            seed: 42,
        }
    }
}

pub struct Detection {
    /// Maps marker level-0 px -> frame level-0 px.
    pub homography: Matrix3<f64>,
    pub inliers: usize,
    pub matches: usize,
}

/// Marker-independent per-frame detection data: keypoints and descriptors from
/// every pyramid level, positions already converted to frame level-0 px.
pub struct FrameFeatures {
    pub positions: Vec<(f64, f64)>,
    pub descriptors: Vec<Descriptor>,
}

pub fn extract_frame_features(frame: &GrayImage, cfg: &DetectorConfig) -> FrameFeatures {
    let pyr = build_pyramid(frame, cfg.pyramid_min_side, cfg.pyramid_max_levels);
    let mut positions: Vec<(f64, f64)> = Vec::new();
    let mut descriptors = Vec::new();
    for (li, level) in pyr.levels.iter().enumerate() {
        let s = (1usize << li) as f64;
        let (pos, desc) = extract_features(level, cfg.fast_threshold, cfg.max_features_per_level);
        for (i, &(x, y)) in pos.iter().enumerate() {
            positions.push((x as f64 * s, y as f64 * s));
            descriptors.push(desc[i]);
        }
    }
    FrameFeatures { positions, descriptors }
}

/// Match one marker against pre-extracted frame features.
pub fn detect_marker_in(
    marker: &CompiledMarker,
    feats: &FrameFeatures,
    cfg: &DetectorConfig,
) -> Option<Detection> {
    if feats.descriptors.len() < cfg.min_inliers {
        return None;
    }
    let matches = matcher::match_descriptors(
        &feats.descriptors,
        &marker.descriptors,
        &marker.positions,
        cfg.match_max_dist,
        cfg.match_ratio,
        cfg.second_dist_px,
    );
    if matches.len() < cfg.min_inliers {
        return None;
    }
    let src: Vec<(f64, f64)> = matches
        .iter()
        .map(|m| {
            let p = marker.positions[m.train as usize];
            (p.0 as f64, p.1 as f64)
        })
        .collect();
    let dst: Vec<(f64, f64)> = matches.iter().map(|m| feats.positions[m.query as usize]).collect();
    let res = homography::ransac(&src, &dst, cfg.ransac_thresh_px, cfg.ransac_iters, cfg.seed)?;
    let inliers = res.inliers.len();
    if inliers < cfg.min_inliers {
        return None;
    }
    if (inliers as f32) < cfg.min_inlier_ratio * matches.len() as f32 {
        return None;
    }
    if !quad_sane(&res.h, marker.width as f64, marker.height as f64) {
        return None;
    }
    Some(Detection { homography: res.h, inliers, matches: matches.len() })
}

/// One-shot convenience: extract features and match a single marker.
pub fn detect_marker(marker: &CompiledMarker, frame: &GrayImage, cfg: &DetectorConfig) -> Option<Detection> {
    detect_marker_in(marker, &extract_frame_features(frame, cfg), cfg)
}
