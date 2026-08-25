//! Runtime detection: find a compiled marker in a camera frame and return the
//! marker->frame homography.

use crate::features::extract_features;
use crate::homography;
use crate::image::{build_pyramid, GrayImage};
use crate::marker::CompiledMarker;
use crate::matcher;
use nalgebra::{Matrix3, Vector3};

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

/// Reject homographies that map the marker to a degenerate quad: reflected,
/// non-convex, tiny, or crossing the plane at infinity.
fn quad_sane(h: &Matrix3<f64>, mw: f64, mh: f64) -> bool {
    let corners = [(0.0, 0.0), (mw, 0.0), (mw, mh), (0.0, mh)];
    let mut proj = [(0.0f64, 0.0f64); 4];
    let mut w_sign = 0.0f64;
    for (i, &(x, y)) in corners.iter().enumerate() {
        let p = h * Vector3::new(x, y, 1.0);
        if !p.x.is_finite() || !p.y.is_finite() || p.z.abs() < 1e-9 {
            return false;
        }
        if i == 0 {
            w_sign = p.z.signum();
        } else if p.z.signum() != w_sign {
            return false; // crosses infinity — physically impossible view
        }
        proj[i] = (p.x / p.z, p.y / p.z);
    }
    // Convexity: all consecutive-edge cross products share a sign.
    let mut sign = 0.0f64;
    for i in 0..4 {
        let a = proj[i];
        let b = proj[(i + 1) % 4];
        let c = proj[(i + 2) % 4];
        let cross = (b.0 - a.0) * (c.1 - b.1) - (b.1 - a.1) * (c.0 - b.0);
        if cross.abs() < 1e-9 {
            return false;
        }
        if sign == 0.0 {
            sign = cross.signum();
        } else if cross.signum() != sign {
            return false;
        }
    }
    // Shoelace area
    let mut area2 = 0.0;
    for i in 0..4 {
        let a = proj[i];
        let b = proj[(i + 1) % 4];
        area2 += a.0 * b.1 - b.0 * a.1;
    }
    area2.abs() / 2.0 > 400.0 // at least ~20x20 px on screen
}

pub fn detect_marker(marker: &CompiledMarker, frame: &GrayImage, cfg: &DetectorConfig) -> Option<Detection> {
    let pyr = build_pyramid(frame, cfg.pyramid_min_side, cfg.pyramid_max_levels);
    let mut fpos: Vec<(f64, f64)> = Vec::new();
    let mut fdesc = Vec::new();
    for (li, level) in pyr.levels.iter().enumerate() {
        let s = (1usize << li) as f64;
        let (pos, desc) = extract_features(level, cfg.fast_threshold, cfg.max_features_per_level);
        for (i, &(x, y)) in pos.iter().enumerate() {
            fpos.push((x as f64 * s, y as f64 * s));
            fdesc.push(desc[i]);
        }
    }
    if fdesc.len() < cfg.min_inliers {
        return None;
    }
    let matches = matcher::match_descriptors(
        &fdesc,
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
    let dst: Vec<(f64, f64)> = matches.iter().map(|m| fpos[m.query as usize]).collect();
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
