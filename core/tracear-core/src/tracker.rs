//! Frame-to-frame sub-pixel tracking — the jitter killer.
//!
//! Given the previous frame's homography, each precompiled marker patch is
//! re-warped into frame space and aligned to the live frame with
//! translation-only inverse-compositional Lucas-Kanade to sub-pixel
//! precision, validated with NCC, and the surviving correspondences update
//! the homography via Huber-IRLS. Corner detectors quantize to ~0.5 px; LK
//! converges to ~0.05 px, and pose noise scales directly with point noise —
//! this is why tracking, not re-detection, is what makes the pose stable.
//!
//! Works on a lightly blurred (box radius 1) level-0 frame: templates were
//! compiled from equally blurred marker levels, and the smoothing widens the
//! LK convergence basin.

use crate::homography::{dlt, dlt_weighted, project, projected_quad_area, quad_sane};
use crate::image::GrayImage;
use crate::marker::{CompiledMarker, TrackPatch, PATCH_CENTER, PATCH_SIZE};
use nalgebra::Matrix3;

pub struct TrackerConfig {
    /// Max patches aligned per frame.
    pub max_patches: usize,
    /// Half side of the LK window; window = (2h+1)^2 = 9x9 by default.
    pub half_window: i32,
    /// Coarse NCC pre-search radius (px) used on the first frame after
    /// detection, when no motion prediction exists yet.
    pub presearch_radius: i32,
    pub presearch_step: i32,
    pub lk_max_iters: usize,
    /// Convergence threshold on the LK update norm (px).
    pub lk_epsilon: f32,
    /// A patch may not travel farther than this from its predicted start.
    pub max_displacement: f32,
    /// Minimum normalized cross-correlation for a patch to survive.
    pub min_ncc: f32,
    pub min_patches: usize,
    /// Minimum survived/attempted ratio.
    pub min_survival: f32,
    pub irls_iters: usize,
    pub huber_px: f64,
    /// Frame-to-frame corner motion above this means the fit went wild.
    pub max_corner_jump_px: f64,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            max_patches: 40,
            half_window: 4,
            presearch_radius: 4,
            presearch_step: 2,
            lk_max_iters: 8,
            lk_epsilon: 0.02,
            max_displacement: 10.0,
            min_ncc: 0.65,
            min_patches: 12,
            min_survival: 0.45,
            irls_iters: 3,
            huber_px: 1.5,
            max_corner_jump_px: 80.0,
        }
    }
}

pub struct TrackState {
    /// Current marker -> frame homography.
    pub h: Matrix3<f64>,
    /// Previous frame's homography (for constant-velocity prediction).
    pub h_prev: Option<Matrix3<f64>>,
    pub frames_tracked: u64,
}

impl TrackState {
    pub fn new(h: Matrix3<f64>) -> Self {
        Self { h, h_prev: None, frames_tracked: 0 }
    }
}

pub struct TrackResult {
    pub h: Matrix3<f64>,
    pub attempted: usize,
    pub survived: usize,
    pub mean_ncc: f32,
}

/// Bilinear sample inside a stored PATCH_SIZE^2 template; None outside support.
#[inline]
fn sample_patch(template: &[u8], x: f32, y: f32) -> Option<f32> {
    let max = (PATCH_SIZE - 1) as f32;
    if !(0.0..=max).contains(&x) || !(0.0..=max).contains(&y) {
        return None;
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(PATCH_SIZE - 1);
    let y1 = (y0 + 1).min(PATCH_SIZE - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let a = template[y0 * PATCH_SIZE + x0] as f32;
    let b = template[y0 * PATCH_SIZE + x1] as f32;
    let c = template[y1 * PATCH_SIZE + x0] as f32;
    let d = template[y1 * PATCH_SIZE + x1] as f32;
    Some(a * (1.0 - fx) * (1.0 - fy) + b * fx * (1.0 - fy) + c * (1.0 - fx) * fy + d * fx * fy)
}

/// Zero-mean NCC between two equally sized sample vectors.
fn ncc(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let ma = a.iter().sum::<f32>() / n;
    let mb = b.iter().sum::<f32>() / n;
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for i in 0..a.len() {
        let da = a[i] - ma;
        let db = b[i] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    if va < 1e-6 || vb < 1e-6 {
        return 0.0;
    }
    cov / (va * vb).sqrt()
}

struct WarpedPatch {
    marker_pos: (f64, f64),
    /// Predicted position of the patch center in the frame.
    pred: (f32, f32),
    /// (2h+1)^2 template values in frame-space geometry.
    template: Vec<f32>,
    /// Template gradients (same grid).
    gx: Vec<f32>,
    gy: Vec<f32>,
}

/// Warp one stored patch into frame-space geometry around its current
/// projection. Returns None when the warp needs support outside the stored
/// template (extreme distortion) — such patches are geometrically unusable.
fn warp_template(
    patch: &TrackPatch,
    level_scale: f32,
    h: &Matrix3<f64>,
    h_inv: &Matrix3<f64>,
    pred: (f32, f32),
    hw: i32,
) -> Option<WarpedPatch> {
    let center = project(h, patch.x as f64, patch.y as f64);
    let side = (2 * hw + 3) as usize; // +1 ring for central-difference gradients
    let mut raw = vec![0.0f32; side * side];
    for (row, dy) in (-hw - 1..=hw + 1).enumerate() {
        for (col, dx) in (-hw - 1..=hw + 1).enumerate() {
            let fx = center.0 + dx as f64;
            let fy = center.1 + dy as f64;
            let m = project(h_inv, fx, fy);
            let lx = (m.0 as f32 - patch.x) * level_scale + PATCH_CENTER;
            let ly = (m.1 as f32 - patch.y) * level_scale + PATCH_CENTER;
            raw[row * side + col] = sample_patch(&patch.template, lx, ly)?;
        }
    }
    let win = (2 * hw + 1) as usize;
    let mut template = Vec::with_capacity(win * win);
    let mut gx = Vec::with_capacity(win * win);
    let mut gy = Vec::with_capacity(win * win);
    for r in 1..=win {
        for c in 1..=win {
            template.push(raw[r * side + c]);
            gx.push((raw[r * side + c + 1] - raw[r * side + c - 1]) * 0.5);
            gy.push((raw[(r + 1) * side + c] - raw[(r - 1) * side + c]) * 0.5);
        }
    }
    Some(WarpedPatch { marker_pos: (patch.x as f64, patch.y as f64), pred, template, gx, gy })
}

/// Sample the frame window centered at (px, py); None if any sample would
/// leave the frame.
fn sample_frame_window(frame: &GrayImage, px: f32, py: f32, hw: i32, out: &mut Vec<f32>) -> bool {
    let margin = (hw + 1) as f32;
    if px < margin || py < margin || px > frame.w as f32 - 1.0 - margin || py > frame.h as f32 - 1.0 - margin {
        return false;
    }
    out.clear();
    for dy in -hw..=hw {
        for dx in -hw..=hw {
            out.push(frame.bilinear(px + dx as f32, py + dy as f32));
        }
    }
    true
}

pub fn track_frame(
    marker: &CompiledMarker,
    frame: &GrayImage,
    state: &TrackState,
    cfg: &TrackerConfig,
) -> Option<TrackResult> {
    if marker.tracking_levels.is_empty() {
        return None;
    }
    let h = state.h;
    let h_inv = h.try_inverse()?;
    let (mw, mh) = (marker.width as f64, marker.height as f64);

    // Pick the tracking level whose resolution best matches the on-screen scale.
    let screen_scale = (projected_quad_area(&h, mw, mh) / (mw * mh)).sqrt();
    if !(screen_scale.is_finite() && screen_scale > 1e-3) {
        return None;
    }
    let level = marker
        .tracking_levels
        .iter()
        .min_by(|a, b| {
            let da = (screen_scale.ln() - (a.scale as f64).ln()).abs();
            let db = (screen_scale.ln() - (b.scale as f64).ln()).abs();
            da.total_cmp(&db)
        })
        .unwrap();

    // Predict each patch's frame position (constant velocity when possible).
    let margin = (cfg.half_window + cfg.presearch_radius + 3) as f32;
    let mut candidates: Vec<(usize, (f32, f32), f32)> = Vec::new(); // (patch idx, pred, score)
    for (i, p) in level.patches.iter().enumerate() {
        let now = project(&h, p.x as f64, p.y as f64);
        let pred = match &state.h_prev {
            Some(hp) => {
                let before = project(hp, p.x as f64, p.y as f64);
                (2.0 * now.0 - before.0, 2.0 * now.1 - before.1)
            }
            None => now,
        };
        let (px, py) = (pred.0 as f32, pred.1 as f32);
        if px < margin || py < margin || px > frame.w as f32 - 1.0 - margin || py > frame.h as f32 - 1.0 - margin {
            continue;
        }
        candidates.push((i, (px, py), p.score));
    }

    // Spread the active set over the frame: one best patch per 48px cell.
    let cell = 48usize;
    let cols = frame.w.div_ceil(cell);
    let rows = frame.h.div_ceil(cell);
    let mut best_per_cell: Vec<Option<usize>> = vec![None; cols * rows]; // index into candidates
    for (ci, &(_, (px, py), score)) in candidates.iter().enumerate() {
        let cx = (px as usize / cell).min(cols - 1);
        let cy = (py as usize / cell).min(rows - 1);
        let slot = &mut best_per_cell[cy * cols + cx];
        if slot.map_or(true, |prev| candidates[prev].2 < score) {
            *slot = Some(ci);
        }
    }
    let mut active: Vec<usize> = best_per_cell.into_iter().flatten().collect();
    active.sort_by(|&a, &b| candidates[b].2.total_cmp(&candidates[a].2));
    active.truncate(cfg.max_patches);

    // Align every active patch.
    let hw = cfg.half_window;
    let win_len = ((2 * hw + 1) * (2 * hw + 1)) as usize;
    let mut window = Vec::with_capacity(win_len);
    let mut attempted = 0usize;
    let mut corr_src: Vec<(f64, f64)> = Vec::new();
    let mut corr_dst: Vec<(f64, f64)> = Vec::new();
    let mut nccs: Vec<f32> = Vec::new();

    for &ci in &active {
        let (pi, pred, _) = candidates[ci];
        let Some(wp) = warp_template(&level.patches[pi], level.scale, &h, &h_inv, pred, hw) else {
            continue;
        };
        attempted += 1;

        // Inverse-compositional precomputation: 2x2 normal matrix of the
        // template gradients, inverted once.
        let (mut sxx, mut sxy, mut syy) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..win_len {
            sxx += wp.gx[i] * wp.gx[i];
            sxy += wp.gx[i] * wp.gy[i];
            syy += wp.gy[i] * wp.gy[i];
        }
        let det = sxx * syy - sxy * sxy;
        if det.abs() < 1e-3 {
            continue; // flat or edge-only patch
        }
        let (i00, i01, i11) = (syy / det, -sxy / det, sxx / det);

        // Coarse NCC pre-search on the first tracked frame (no prediction yet).
        let (mut px, mut py) = wp.pred;
        if state.frames_tracked == 0 {
            let mut best = f32::MIN;
            let (mut bx, mut by) = (px, py);
            let r = cfg.presearch_radius;
            let step = cfg.presearch_step.max(1);
            let mut oy = -r;
            while oy <= r {
                let mut ox = -r;
                while ox <= r {
                    let (tx, ty) = (wp.pred.0 + ox as f32, wp.pred.1 + oy as f32);
                    if sample_frame_window(frame, tx, ty, hw, &mut window) {
                        let s = ncc(&wp.template, &window);
                        if s > best {
                            best = s;
                            bx = tx;
                            by = ty;
                        }
                    }
                    ox += step;
                }
                oy += step;
            }
            px = bx;
            py = by;
        }

        // Translation-only inverse-compositional LK to convergence.
        let mut ok = false;
        for _ in 0..cfg.lk_max_iters {
            if !sample_frame_window(frame, px, py, hw, &mut window) {
                break;
            }
            let (mut bx, mut by) = (0.0f32, 0.0f32);
            for i in 0..win_len {
                let e = window[i] - wp.template[i];
                bx += wp.gx[i] * e;
                by += wp.gy[i] * e;
            }
            let dx = i00 * bx + i01 * by;
            let dy = i01 * bx + i11 * by;
            px -= dx;
            py -= dy;
            let dpx = px - wp.pred.0;
            let dpy = py - wp.pred.1;
            if dpx * dpx + dpy * dpy > cfg.max_displacement * cfg.max_displacement {
                break;
            }
            if dx * dx + dy * dy < cfg.lk_epsilon * cfg.lk_epsilon {
                ok = true;
                break;
            }
        }
        if !ok {
            continue;
        }
        if !sample_frame_window(frame, px, py, hw, &mut window) {
            continue;
        }
        let score = ncc(&wp.template, &window);
        if score < cfg.min_ncc {
            continue;
        }
        corr_src.push(wp.marker_pos);
        corr_dst.push((px as f64, py as f64));
        nccs.push(score);
    }

    let survived = corr_src.len();
    if survived < cfg.min_patches || (survived as f32) < cfg.min_survival * attempted.max(1) as f32 {
        return None;
    }

    // Huber-IRLS homography from the surviving correspondences.
    let mut h_new = dlt(&corr_src, &corr_dst)?;
    for _ in 0..cfg.irls_iters {
        let weights: Vec<f64> = corr_src
            .iter()
            .zip(&corr_dst)
            .map(|(&(sx, sy), &(dx, dy))| {
                let p = project(&h_new, sx, sy);
                let r = ((p.0 - dx).powi(2) + (p.1 - dy).powi(2)).sqrt();
                if r <= cfg.huber_px {
                    1.0
                } else {
                    cfg.huber_px / r
                }
            })
            .collect();
        h_new = dlt_weighted(&corr_src, &corr_dst, &weights)?;
    }

    // Sanity: bounded frame-to-frame motion, non-degenerate quad.
    if !quad_sane(&h_new, mw, mh) {
        return None;
    }
    for &(cx, cy) in &[(0.0, 0.0), (mw, 0.0), (mw, mh), (0.0, mh)] {
        let a = project(&h, cx, cy);
        let b = project(&h_new, cx, cy);
        if ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt() > cfg.max_corner_jump_px {
            return None;
        }
    }

    let mean_ncc = nccs.iter().sum::<f32>() / survived as f32;
    Some(TrackResult { h: h_new, attempted, survived, mean_ncc })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::warp_onto_aa;
    use crate::marker::{compile_marker, CompileConfig};
    use crate::synthetic;

    /// Render the marker into a frame under `h_gt`, lightly blurred like the
    /// pipeline's tracking input.
    fn render_tracking_frame(marker_img: &GrayImage, h_gt: &Matrix3<f64>, seed: u64) -> GrayImage {
        let bg = synthetic::textured_image(640, 480, seed);
        let mut f = warp_onto_aa(marker_img, &h_gt.try_inverse().unwrap(), &bg);
        synthetic::add_gaussian_noise(&mut f, 2.0, seed.wrapping_mul(31) + 7);
        f.box_blur(1)
    }

    fn corner_err(a: &Matrix3<f64>, b: &Matrix3<f64>, mw: f64, mh: f64) -> f64 {
        [(0.0, 0.0), (mw, 0.0), (mw, mh), (0.0, mh)]
            .iter()
            .map(|&(x, y)| {
                let pa = project(a, x, y);
                let pb = project(b, x, y);
                ((pa.0 - pb.0).powi(2) + (pa.1 - pb.1).powi(2)).sqrt()
            })
            .sum::<f64>()
            / 4.0
    }

    #[test]
    fn recovers_from_perturbed_homography() {
        let marker_img = synthetic::textured_image(320, 320, 7);
        let marker = compile_marker(&marker_img, &CompileConfig::default());
        let h_gt = synthetic::trajectory_homography(320.0, 320.0, 640.0, 480.0, 0.13);
        let frame = render_tracking_frame(&marker_img, &h_gt, 99);

        // Perturb the "previous" homography by ~2 px translation.
        let mut h0 = h_gt;
        h0[(0, 2)] += 2.1;
        h0[(1, 2)] -= 1.7;
        let state = TrackState::new(h0);
        let res = track_frame(&marker, &frame, &state, &TrackerConfig::default())
            .expect("tracking should succeed");
        let err = corner_err(&res.h, &h_gt, 320.0, 320.0);
        assert!(err < 0.35, "corner error after track = {err:.3} px (survived {}/{})", res.survived, res.attempted);
        assert!(res.survived >= 15, "only {} patches survived", res.survived);
    }

    #[test]
    fn fails_gracefully_when_marker_absent() {
        let marker_img = synthetic::textured_image(320, 320, 7);
        let marker = compile_marker(&marker_img, &CompileConfig::default());
        let h_gt = synthetic::trajectory_homography(320.0, 320.0, 640.0, 480.0, 0.13);
        // Frame WITHOUT the marker.
        let frame = synthetic::textured_image(640, 480, 555).box_blur(1);
        let state = TrackState::new(h_gt);
        assert!(track_frame(&marker, &frame, &state, &TrackerConfig::default()).is_none());
    }
}
