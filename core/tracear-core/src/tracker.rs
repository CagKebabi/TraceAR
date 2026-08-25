//! Frame-to-frame sub-pixel tracking — the jitter killer.
//!
//! Given the previous frame's homography, precompiled marker patches are
//! re-warped into frame space and aligned to the live frame with
//! translation-only inverse-compositional Lucas-Kanade to sub-pixel
//! precision, validated with NCC, and the surviving correspondences update
//! the homography via Huber-IRLS. Corner detectors quantize to ~0.5 px; LK
//! converges to ~0.05 px, and pose noise scales directly with point noise —
//! this is why tracking, not re-detection, is what makes the pose stable.
//!
//! Alignment is **coarse-to-fine**: when the velocity model erred last frame
//! (or on the hand-off frame, or in recovery), a small patch set is first
//! aligned on the half-resolution frame. At half resolution both the
//! displacement AND the motion blur shrink 2x, so a handheld pan that smears
//! the full-resolution image beyond recognition still registers coarsely;
//! the result seeds the full-resolution refinement. If refinement fails
//! (blur too strong for fine templates) the coarse pose itself keeps the
//! track alive rather than falling back to a ~10x more expensive detection.
//! On a still phone all of this is skipped and only the fine stage runs.
//!
//! Further real-camera measures:
//! - Motion prediction is scaled by the actual inter-frame time gap
//!   (`pred_scale`) — frames do not arrive uniformly.
//! - LK is zero-mean (auto-exposure brightness offsets).
//! - Loose NCC/survival thresholds are guarded by a median-reprojection-
//!   residual gate, so they cannot admit a geometrically bad fit.
//!
//! Works on a lightly blurred (box radius 1) level-0 frame plus its
//! half-resolution downsample: templates were compiled from equally blurred
//! marker levels, and the smoothing widens the LK convergence basin.

use crate::homography::{dlt, dlt_weighted, project, projected_quad_area, quad_sane};
use crate::image::GrayImage;
use crate::marker::{CompiledMarker, TrackPatch, PATCH_CENTER, PATCH_SIZE};
use nalgebra::Matrix3;

pub struct TrackerConfig {
    /// Max patches aligned in the fine stage.
    pub max_patches: usize,
    /// Half side of the LK window; window = (2h+1)^2 = 9x9 by default.
    pub half_window: i32,
    /// Presearch radius (px) for the fine stage on a hand-off frame where
    /// the coarse stage could not run/succeed.
    pub presearch_radius: i32,
    pub presearch_step: i32,
    /// Coarse stage (half-resolution) parameters. The coarse presearch
    /// radius is in half-res px — it covers twice that many full-res px.
    pub coarse_max_patches: usize,
    pub coarse_presearch_radius: i32,
    pub coarse_recovery_radius: i32,
    pub coarse_min_ncc: f32,
    pub coarse_min_patches: usize,
    pub coarse_min_survival: f32,
    /// The coarse stage activates when last frame's mean prediction error
    /// was at least this many px (always on hand-off & recovery).
    pub coarse_trigger_px: f32,
    pub lk_max_iters: usize,
    /// Convergence threshold on the LK update norm (px).
    pub lk_epsilon: f32,
    /// A patch may not travel farther than this from its search start.
    pub max_displacement: f32,
    /// Minimum normalized cross-correlation for a fine patch to survive.
    pub min_ncc: f32,
    pub min_patches: usize,
    /// Minimum survived/attempted ratio (fine stage).
    pub min_survival: f32,
    pub irls_iters: usize,
    pub huber_px: f64,
    /// Median reprojection residual of survivors must stay below this
    /// (in the aligned image's own px).
    pub max_median_residual_px: f64,
    /// Frame-to-frame corner motion above this means the fit went wild.
    pub max_corner_jump_px: f64,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            max_patches: 40,
            half_window: 4,
            presearch_radius: 8,
            presearch_step: 2,
            coarse_max_patches: 16,
            coarse_presearch_radius: 6,
            coarse_recovery_radius: 10,
            coarse_min_ncc: 0.45,
            coarse_min_patches: 6,
            coarse_min_survival: 0.25,
            coarse_trigger_px: 1.0,
            lk_max_iters: 10,
            lk_epsilon: 0.02,
            max_displacement: 14.0,
            min_ncc: 0.55,
            min_patches: 12,
            min_survival: 0.30,
            irls_iters: 3,
            huber_px: 1.5,
            max_median_residual_px: 1.5,
            max_corner_jump_px: 80.0,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TrackMode {
    Normal,
    /// Wider coarse search — a last attempt before full re-detection.
    Recovery,
}

pub struct TrackState {
    /// Current marker -> frame homography.
    pub h: Matrix3<f64>,
    /// Previous frame's homography (for constant-velocity prediction).
    pub h_prev: Option<Matrix3<f64>>,
    /// Capture timestamp (ms) of the frame `h` came from.
    pub t_last: f64,
    /// Capture timestamp (ms) of the frame `h_prev` came from.
    pub t_prev: f64,
    /// Mean |final - predicted| patch position error of the last tracked
    /// frame; drives coarse-stage activation.
    pub last_pred_err: f32,
    pub frames_tracked: u64,
}

impl TrackState {
    pub fn new(h: Matrix3<f64>, t: f64) -> Self {
        Self { h, h_prev: None, t_last: t, t_prev: t, last_pred_err: f32::MAX, frames_tracked: 0 }
    }
}

pub struct TrackResult {
    pub h: Matrix3<f64>,
    pub attempted: usize,
    pub survived: usize,
    pub mean_ncc: f32,
    /// Mean |final - predicted| over survivors (velocity-model quality),
    /// in level-0 px.
    pub mean_pred_err: f32,
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

fn median(v: &mut [f32]) -> f32 {
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

struct WarpedPatch {
    marker_pos: (f64, f64),
    /// (2h+1)^2 zero-mean template values in image-space geometry.
    template: Vec<f32>,
    /// Template gradients (same grid).
    gx: Vec<f32>,
    gy: Vec<f32>,
}

/// Warp one stored patch into image-space geometry around its current
/// projection. Returns None when the warp needs support outside the stored
/// template (extreme distortion) — such patches are geometrically unusable.
fn warp_template(
    patch: &TrackPatch,
    level_scale: f32,
    h: &Matrix3<f64>,
    h_inv: &Matrix3<f64>,
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
    // Zero-mean the template: with the frame window zero-meaned per iteration
    // too, LK becomes invariant to the brightness offsets a phone camera's
    // auto-exposure introduces between frames. (Gradients are unaffected.)
    let mean = template.iter().sum::<f32>() / template.len() as f32;
    for v in template.iter_mut() {
        *v -= mean;
    }
    Some(WarpedPatch { marker_pos: (patch.x as f64, patch.y as f64), template, gx, gy })
}

/// Sample the image window centered at (px, py); false if any sample would
/// leave the image.
fn sample_frame_window(img: &GrayImage, px: f32, py: f32, hw: i32, out: &mut Vec<f32>) -> bool {
    let margin = (hw + 1) as f32;
    if px < margin || py < margin || px > img.w as f32 - 1.0 - margin || py > img.h as f32 - 1.0 - margin {
        return false;
    }
    out.clear();
    for dy in -hw..=hw {
        for dx in -hw..=hw {
            out.push(img.bilinear(px + dx as f32, py + dy as f32));
        }
    }
    true
}

/// Best-NCC position on a grid around `center`. Returns (x, y, ncc).
fn presearch(
    img: &GrayImage,
    template: &[f32],
    center: (f32, f32),
    radius: i32,
    step: i32,
    hw: i32,
    window: &mut Vec<f32>,
) -> (f32, f32, f32) {
    let mut best = (center.0, center.1, f32::MIN);
    let step = step.max(1);
    let mut oy = -radius;
    while oy <= radius {
        let mut ox = -radius;
        while ox <= radius {
            let tx = center.0 + ox as f32;
            let ty = center.1 + oy as f32;
            if sample_frame_window(img, tx, ty, hw, window) {
                let s = ncc(template, window);
                if s > best.2 {
                    best = (tx, ty, s);
                }
            }
            ox += step;
        }
        oy += step;
    }
    best
}

struct AlignParams {
    max_patches: usize,
    /// Presearch (radius, step) around each start position; None = LK only.
    presearch: Option<(i32, i32)>,
    min_ncc: f32,
    min_patches: usize,
    min_survival: f32,
}

struct AlignOutcome {
    /// marker level-0 px -> this image's px.
    h: Matrix3<f64>,
    attempted: usize,
    survived: usize,
    mean_ncc: f32,
    /// Mean |final - predicted| in this image's px.
    mean_pred_err: f32,
}

/// One LK+IRLS alignment pass of the marker against `img`, starting from
/// `h_init` (marker level-0 px -> img px). `velocity` optionally supplies
/// (h_prev_in_img_px, alpha) for constant-velocity start positions.
#[allow(clippy::too_many_arguments)]
fn align_once(
    marker: &CompiledMarker,
    img: &GrayImage,
    h_init: &Matrix3<f64>,
    velocity: Option<(Matrix3<f64>, f64)>,
    cfg: &TrackerConfig,
    p: &AlignParams,
) -> Option<AlignOutcome> {
    let h_inv = h_init.try_inverse()?;
    let (mw, mh) = (marker.width as f64, marker.height as f64);
    let screen_scale = (projected_quad_area(h_init, mw, mh) / (mw * mh)).sqrt();
    if !(screen_scale.is_finite() && screen_scale > 1e-3) {
        return None;
    }
    // Marker tracking level whose resolution best matches this image.
    let level = marker
        .tracking_levels
        .iter()
        .min_by(|a, b| {
            let da = (screen_scale.ln() - (a.scale as f64).ln()).abs();
            let db = (screen_scale.ln() - (b.scale as f64).ln()).abs();
            da.total_cmp(&db)
        })?;

    let hw = cfg.half_window;
    let search_radius = p.presearch.map_or(0, |(r, _)| r);
    let margin = (hw + search_radius + 3) as f32;
    let mut candidates: Vec<(usize, (f32, f32), f32)> = Vec::new(); // (patch idx, start, score)
    for (i, patch) in level.patches.iter().enumerate() {
        let now = project(h_init, patch.x as f64, patch.y as f64);
        let pred = match &velocity {
            Some((hp, alpha)) => {
                let before = project(hp, patch.x as f64, patch.y as f64);
                (now.0 + (now.0 - before.0) * alpha, now.1 + (now.1 - before.1) * alpha)
            }
            None => now,
        };
        let (px, py) = (pred.0 as f32, pred.1 as f32);
        if px < margin || py < margin || px > img.w as f32 - 1.0 - margin || py > img.h as f32 - 1.0 - margin {
            continue;
        }
        candidates.push((i, (px, py), patch.score));
    }

    // Spread the active set over the image: one best patch per cell.
    let cell = (img.w / 13).clamp(16, 64);
    let cols = img.w.div_ceil(cell);
    let rows = img.h.div_ceil(cell);
    let mut best_per_cell: Vec<Option<usize>> = vec![None; cols * rows];
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
    active.truncate(p.max_patches);

    let win_len = ((2 * hw + 1) * (2 * hw + 1)) as usize;
    let mut window = Vec::with_capacity(win_len);
    let mut attempted = 0usize;
    let mut corr_src: Vec<(f64, f64)> = Vec::new();
    let mut corr_dst: Vec<(f64, f64)> = Vec::new();
    let mut nccs: Vec<f32> = Vec::new();
    let mut pred_errs: Vec<f32> = Vec::new();

    for &ci in &active {
        let (pi, pred, _) = candidates[ci];
        let Some(wp) = warp_template(&level.patches[pi], level.scale, h_init, &h_inv, hw) else {
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

        let (mut px, mut py) = pred;
        if let Some((radius, step)) = p.presearch {
            let (bx, by, _) = presearch(img, &wp.template, pred, radius, step, hw, &mut window);
            px = bx;
            py = by;
        }

        // Translation-only inverse-compositional LK to convergence.
        let start = (px, py);
        let mut ok = false;
        for _ in 0..cfg.lk_max_iters {
            if !sample_frame_window(img, px, py, hw, &mut window) {
                break;
            }
            let win_mean = window.iter().sum::<f32>() / win_len as f32;
            let (mut bx, mut by) = (0.0f32, 0.0f32);
            for i in 0..win_len {
                let e = (window[i] - win_mean) - wp.template[i];
                bx += wp.gx[i] * e;
                by += wp.gy[i] * e;
            }
            let dx = i00 * bx + i01 * by;
            let dy = i01 * bx + i11 * by;
            px -= dx;
            py -= dy;
            let dpx = px - start.0;
            let dpy = py - start.1;
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
        if !sample_frame_window(img, px, py, hw, &mut window) {
            continue;
        }
        let score = ncc(&wp.template, &window);
        if score < p.min_ncc {
            continue;
        }
        corr_src.push(wp.marker_pos);
        corr_dst.push((px as f64, py as f64));
        nccs.push(score);
        pred_errs.push(((px - pred.0).powi(2) + (py - pred.1).powi(2)).sqrt());
    }

    let survived = corr_src.len();
    if survived < p.min_patches || (survived as f32) < p.min_survival * attempted.max(1) as f32 {
        return None;
    }

    // Huber-IRLS homography from the surviving correspondences.
    let mut h_new = dlt(&corr_src, &corr_dst)?;
    for _ in 0..cfg.irls_iters {
        let weights: Vec<f64> = corr_src
            .iter()
            .zip(&corr_dst)
            .map(|(&(sx, sy), &(dx, dy))| {
                let pr = project(&h_new, sx, sy);
                let r = ((pr.0 - dx).powi(2) + (pr.1 - dy).powi(2)).sqrt();
                if r <= cfg.huber_px {
                    1.0
                } else {
                    cfg.huber_px / r
                }
            })
            .collect();
        h_new = dlt_weighted(&corr_src, &corr_dst, &weights)?;
    }

    // Survivors must agree with the fit.
    let mut residuals: Vec<f32> = corr_src
        .iter()
        .zip(&corr_dst)
        .map(|(&(sx, sy), &(dx, dy))| {
            let pr = project(&h_new, sx, sy);
            (((pr.0 - dx).powi(2) + (pr.1 - dy).powi(2)) as f32).sqrt()
        })
        .collect();
    if median(&mut residuals) as f64 > cfg.max_median_residual_px {
        return None;
    }

    Some(AlignOutcome {
        h: h_new,
        attempted,
        survived,
        mean_ncc: nccs.iter().sum::<f32>() / survived as f32,
        mean_pred_err: pred_errs.iter().sum::<f32>() / survived as f32,
    })
}

/// `pred_scale` rescales the constant-velocity prediction to the actual time
/// gap: (t_now - t_last) / (t_last - t_prev). Camera frames do NOT arrive
/// uniformly — a detection frame takes ~5x longer than a tracking frame, so
/// assuming equal spacing overshoots the prediction right after hand-off and
/// loses the target. Pass 1.0 for uniformly spaced input.
///
/// `frame` is the lightly blurred level-0 frame; `frame_half` its 2x
/// downsample (used by the coarse stage).
pub fn track_frame(
    marker: &CompiledMarker,
    frame: &GrayImage,
    frame_half: &GrayImage,
    state: &TrackState,
    pred_scale: f64,
    mode: TrackMode,
    cfg: &TrackerConfig,
) -> Option<TrackResult> {
    if marker.tracking_levels.is_empty() {
        return None;
    }
    let h = state.h;
    let (mw, mh) = (marker.width as f64, marker.height as f64);
    let alpha = pred_scale.clamp(0.0, 2.5);

    // Coarse stage: hand-off frames (no velocity yet), frames after a bad
    // prediction, and recovery attempts.
    let coarse_needed =
        state.frames_tracked == 0 || state.last_pred_err >= cfg.coarse_trigger_px || mode == TrackMode::Recovery;
    let down = Matrix3::new(0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0);
    let up = Matrix3::new(2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0);

    let mut fine_init = h;
    let mut fine_velocity = state.h_prev.map(|hp| (hp, alpha));
    let mut coarse_outcome: Option<AlignOutcome> = None;
    if coarse_needed {
        let radius = match mode {
            TrackMode::Recovery => cfg.coarse_recovery_radius,
            TrackMode::Normal => cfg.coarse_presearch_radius,
        };
        let params = AlignParams {
            max_patches: cfg.coarse_max_patches,
            presearch: Some((radius, cfg.presearch_step)),
            min_ncc: cfg.coarse_min_ncc,
            min_patches: cfg.coarse_min_patches,
            min_survival: cfg.coarse_min_survival,
        };
        let h1 = down * h;
        let vel1 = state.h_prev.map(|hp| (down * hp, alpha));
        match align_once(marker, frame_half, &h1, vel1, cfg, &params) {
            Some(out) => {
                fine_init = up * out.h;
                fine_velocity = None; // fine_init already describes THIS frame
                coarse_outcome = Some(out);
            }
            None => {
                if mode == TrackMode::Recovery {
                    return None;
                }
            }
        }
    }

    // Fine stage at full resolution.
    let fine_presearch = if state.frames_tracked == 0 && coarse_outcome.is_none() {
        Some((cfg.presearch_radius, cfg.presearch_step))
    } else {
        None
    };
    let fine_params = AlignParams {
        max_patches: cfg.max_patches,
        presearch: fine_presearch,
        min_ncc: cfg.min_ncc,
        min_patches: cfg.min_patches,
        min_survival: cfg.min_survival,
    };
    let fine = align_once(marker, frame, &fine_init, fine_velocity, cfg, &fine_params);

    let (h_new, attempted, survived, mean_ncc, mean_pred_err) = match (fine, &coarse_outcome) {
        (Some(f), coarse) => {
            // Keep the coarse stage armed while motion persists: the true
            // velocity-model error this frame is what the coarse pass saw.
            let pred_err = match coarse {
                Some(c) => (c.mean_pred_err * 2.0).max(f.mean_pred_err),
                None => f.mean_pred_err,
            };
            (f.h, f.attempted, f.survived, f.mean_ncc, pred_err)
        }
        (None, Some(c)) => {
            // Fine templates defeated by blur — the coarse pose still keeps
            // the track alive (slightly less precise for this frame).
            (fine_init, c.attempted, c.survived, c.mean_ncc, c.mean_pred_err * 2.0)
        }
        (None, None) => return None,
    };

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

    Some(TrackResult { h: h_new, attempted, survived, mean_ncc, mean_pred_err })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::warp_onto_aa;
    use crate::marker::{compile_marker, CompileConfig};
    use crate::synthetic;

    /// Render the marker into a frame under `h_gt`, lightly blurred like the
    /// pipeline's tracking input; returns (frame, frame_half).
    fn render_tracking_frames(marker_img: &GrayImage, h_gt: &Matrix3<f64>, seed: u64) -> (GrayImage, GrayImage) {
        let bg = synthetic::textured_image(640, 480, seed);
        let mut f = warp_onto_aa(marker_img, &h_gt.try_inverse().unwrap(), &bg);
        synthetic::add_gaussian_noise(&mut f, 2.0, seed.wrapping_mul(31) + 7);
        let f = f.box_blur(1);
        let half = f.downsample_half();
        (f, half)
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
        let (frame, half) = render_tracking_frames(&marker_img, &h_gt, 99);

        // Perturb the "previous" homography by ~2 px translation.
        let mut h0 = h_gt;
        h0[(0, 2)] += 2.1;
        h0[(1, 2)] -= 1.7;
        let state = TrackState::new(h0, 0.0);
        let res = track_frame(&marker, &frame, &half, &state, 1.0, TrackMode::Normal, &TrackerConfig::default())
            .expect("tracking should succeed");
        let err = corner_err(&res.h, &h_gt, 320.0, 320.0);
        assert!(err < 0.35, "corner error after track = {err:.3} px (survived {}/{})", res.survived, res.attempted);
        assert!(res.survived >= 15, "only {} patches survived", res.survived);
    }

    #[test]
    fn coarse_stage_reaches_large_offsets() {
        let marker_img = synthetic::textured_image(320, 320, 7);
        let marker = compile_marker(&marker_img, &CompileConfig::default());
        let h_gt = synthetic::trajectory_homography(320.0, 320.0, 640.0, 480.0, 0.13);
        let (frame, half) = render_tracking_frames(&marker_img, &h_gt, 99);

        // ~16 px off — far beyond the plain LK basin.
        let mut h0 = h_gt;
        h0[(0, 2)] += 12.0;
        h0[(1, 2)] -= 10.0;
        let mut state = TrackState::new(h0, 0.0);
        state.frames_tracked = 5; // not a hand-off frame
        state.last_pred_err = 0.0; // coarse stage disabled in Normal mode
        let cfg = TrackerConfig::default();
        assert!(
            track_frame(&marker, &frame, &half, &state, 1.0, TrackMode::Normal, &cfg).is_none(),
            "plain fine LK should not reach a 16 px offset"
        );
        let res = track_frame(&marker, &frame, &half, &state, 1.0, TrackMode::Recovery, &cfg)
            .expect("coarse recovery should re-lock");
        let err = corner_err(&res.h, &h_gt, 320.0, 320.0);
        assert!(err < 0.5, "corner error after recovery = {err:.3} px");
    }

    #[test]
    fn coarse_pose_carries_through_heavy_blur() {
        let marker_img = synthetic::textured_image(320, 320, 7);
        let marker = compile_marker(&marker_img, &CompileConfig::default());
        let h_gt = synthetic::trajectory_homography(320.0, 320.0, 640.0, 480.0, 0.13);
        let (frame, _) = render_tracking_frames(&marker_img, &h_gt, 99);
        // Heavy blur (~motion blur): fine 9x9 templates lose NCC, coarse
        // half-res alignment must still keep the track alive.
        let heavy = frame.box_blur(2);
        let half = heavy.downsample_half();

        let mut h0 = h_gt;
        h0[(0, 2)] += 4.0;
        h0[(1, 2)] += 3.0;
        let mut state = TrackState::new(h0, 0.0);
        state.frames_tracked = 3;
        state.last_pred_err = 5.0; // motion: coarse stage armed
        let res = track_frame(&marker, &heavy, &half, &state, 1.0, TrackMode::Normal, &TrackerConfig::default())
            .expect("track must survive heavy blur via the coarse pose");
        let err = corner_err(&res.h, &h_gt, 320.0, 320.0);
        assert!(err < 3.0, "corner error under heavy blur = {err:.2} px");
    }

    #[test]
    fn fails_gracefully_when_marker_absent() {
        let marker_img = synthetic::textured_image(320, 320, 7);
        let marker = compile_marker(&marker_img, &CompileConfig::default());
        let h_gt = synthetic::trajectory_homography(320.0, 320.0, 640.0, 480.0, 0.13);
        // Frame WITHOUT the marker.
        let frame = synthetic::textured_image(640, 480, 555).box_blur(1);
        let half = frame.downsample_half();
        let state = TrackState::new(h_gt, 0.0);
        let cfg = TrackerConfig::default();
        assert!(track_frame(&marker, &frame, &half, &state, 1.0, TrackMode::Normal, &cfg).is_none());
        assert!(track_frame(&marker, &frame, &half, &state, 1.0, TrackMode::Recovery, &cfg).is_none());
    }
}
