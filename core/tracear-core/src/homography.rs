//! Homography estimation: normalized DLT and RANSAC with adaptive iteration
//! count, degenerate-sample rejection, and inlier refit.

use crate::rng::XorShift64;
use nalgebra::{DMatrix, Matrix3, Vector3};

pub fn project(h: &Matrix3<f64>, x: f64, y: f64) -> (f64, f64) {
    let p = h * Vector3::new(x, y, 1.0);
    (p.x / p.z, p.y / p.z)
}

fn normalize_points(pts: &[(f64, f64)]) -> Option<(Matrix3<f64>, Vec<(f64, f64)>)> {
    let n = pts.len() as f64;
    let (mut cx, mut cy) = (0.0, 0.0);
    for &(x, y) in pts {
        cx += x;
        cy += y;
    }
    cx /= n;
    cy /= n;
    let mut mean_dist = 0.0;
    for &(x, y) in pts {
        mean_dist += ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
    }
    mean_dist /= n;
    if mean_dist < 1e-9 {
        return None;
    }
    let s = std::f64::consts::SQRT_2 / mean_dist;
    let t = Matrix3::new(s, 0.0, -s * cx, 0.0, s, -s * cy, 0.0, 0.0, 1.0);
    let normed = pts.iter().map(|&(x, y)| ((x - cx) * s, (y - cy) * s)).collect();
    Some((t, normed))
}

/// Normalized DLT for n >= 4 correspondences. Returns H with h33 = 1
/// (Frobenius-normalized if h33 is degenerate).
pub fn dlt(src: &[(f64, f64)], dst: &[(f64, f64)]) -> Option<Matrix3<f64>> {
    assert_eq!(src.len(), dst.len());
    let n = src.len();
    if n < 4 {
        return None;
    }
    let (t_src, src_n) = normalize_points(src)?;
    let (t_dst, dst_n) = normalize_points(dst)?;
    // Pad to at least 9 rows so the thin SVD still exposes all 9 right
    // singular vectors (the null vector) for the minimal 4-point case.
    let rows = (2 * n).max(9);
    let mut a = DMatrix::<f64>::zeros(rows, 9);
    for i in 0..n {
        let (x, y) = src_n[i];
        let (u, v) = dst_n[i];
        a[(2 * i, 0)] = -x;
        a[(2 * i, 1)] = -y;
        a[(2 * i, 2)] = -1.0;
        a[(2 * i, 6)] = u * x;
        a[(2 * i, 7)] = u * y;
        a[(2 * i, 8)] = u;
        a[(2 * i + 1, 3)] = -x;
        a[(2 * i + 1, 4)] = -y;
        a[(2 * i + 1, 5)] = -1.0;
        a[(2 * i + 1, 6)] = v * x;
        a[(2 * i + 1, 7)] = v * y;
        a[(2 * i + 1, 8)] = v;
    }
    let svd = a.svd(false, true);
    let v_t = svd.v_t?;
    let sv = &svd.singular_values;
    let mut min_i = 0;
    for i in 1..sv.len() {
        if sv[i] < sv[min_i] {
            min_i = i;
        }
    }
    let hv = v_t.row(min_i);
    let hn = Matrix3::new(hv[0], hv[1], hv[2], hv[3], hv[4], hv[5], hv[6], hv[7], hv[8]);
    let t_dst_inv = t_dst.try_inverse()?;
    let mut h = t_dst_inv * hn * t_src;
    let h33 = h[(2, 2)];
    if h33.abs() > 1e-9 {
        h /= h33;
    } else {
        let f = h.norm();
        if f < 1e-12 {
            return None;
        }
        h /= f;
    }
    Some(h)
}

fn collinear(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> bool {
    let area2 = (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0);
    area2.abs() < 1.0 // pixel-scale threshold
}

fn sample_degenerate(pts: &[(f64, f64); 4]) -> bool {
    for i in 0..4 {
        for j in i + 1..4 {
            for k in j + 1..4 {
                if collinear(pts[i], pts[j], pts[k]) {
                    return true;
                }
            }
        }
    }
    false
}

fn collect_inliers(h: &Matrix3<f64>, src: &[(f64, f64)], dst: &[(f64, f64)], t2: f64) -> Vec<usize> {
    let mut v = Vec::new();
    for i in 0..src.len() {
        let p = h * Vector3::new(src[i].0, src[i].1, 1.0);
        if p.z.abs() < 1e-9 {
            continue;
        }
        let dx = p.x / p.z - dst[i].0;
        let dy = p.y / p.z - dst[i].1;
        if dx * dx + dy * dy < t2 {
            v.push(i);
        }
    }
    v
}

pub struct RansacResult {
    pub h: Matrix3<f64>,
    pub inliers: Vec<usize>,
}

pub fn ransac(
    src: &[(f64, f64)],
    dst: &[(f64, f64)],
    thresh_px: f64,
    max_iters: usize,
    seed: u64,
) -> Option<RansacResult> {
    let n = src.len();
    assert_eq!(n, dst.len());
    if n < 4 {
        return None;
    }
    let mut rng = XorShift64::new(seed);
    let t2 = thresh_px * thresh_px;
    let mut best_inliers: Vec<usize> = Vec::new();
    let mut iters = max_iters;
    let mut iter = 0;
    while iter < iters {
        iter += 1;
        let mut idx = [0usize; 4];
        let mut k = 0;
        while k < 4 {
            let c = rng.gen_range(n);
            if !idx[..k].contains(&c) {
                idx[k] = c;
                k += 1;
            }
        }
        let s = [src[idx[0]], src[idx[1]], src[idx[2]], src[idx[3]]];
        let d = [dst[idx[0]], dst[idx[1]], dst[idx[2]], dst[idx[3]]];
        if sample_degenerate(&s) || sample_degenerate(&d) {
            continue;
        }
        let Some(h) = dlt(&s, &d) else { continue };
        let inliers = collect_inliers(&h, src, dst, t2);
        if inliers.len() > best_inliers.len() {
            best_inliers = inliers;
            // Adaptive termination (99% confidence).
            let w = best_inliers.len() as f64 / n as f64;
            let denom = (1.0 - w.powi(4)).max(1e-12).ln();
            if denom < 0.0 {
                let needed = ((1.0f64 - 0.99).ln() / denom).ceil() as usize;
                iters = iters.min(needed.max(iter));
            }
        }
    }
    if best_inliers.len() < 4 {
        return None;
    }
    // Refit on inliers (twice: refit -> re-collect -> refit).
    let mut best: Option<RansacResult> = None;
    let mut inliers = best_inliers;
    for _ in 0..2 {
        let s: Vec<_> = inliers.iter().map(|&i| src[i]).collect();
        let d: Vec<_> = inliers.iter().map(|&i| dst[i]).collect();
        let Some(h) = dlt(&s, &d) else { break };
        let new_inliers = collect_inliers(&h, src, dst, t2);
        if new_inliers.len() < 4 {
            break;
        }
        inliers = new_inliers.clone();
        best = Some(RansacResult { h, inliers: new_inliers });
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_h() -> Matrix3<f64> {
        Matrix3::new(1.2, 0.1, 5.0, -0.05, 0.95, -3.0, 1e-4, -2e-4, 1.0)
    }

    fn grid(n: usize, span: f64) -> Vec<(f64, f64)> {
        let mut v = Vec::new();
        for i in 0..n {
            for j in 0..n {
                v.push((i as f64 * span / (n - 1) as f64, j as f64 * span / (n - 1) as f64));
            }
        }
        v
    }

    fn max_transfer_error(a: &Matrix3<f64>, b: &Matrix3<f64>, pts: &[(f64, f64)]) -> f64 {
        pts.iter()
            .map(|&(x, y)| {
                let pa = project(a, x, y);
                let pb = project(b, x, y);
                ((pa.0 - pb.0).powi(2) + (pa.1 - pb.1).powi(2)).sqrt()
            })
            .fold(0.0, f64::max)
    }

    #[test]
    fn dlt_recovers_exact_homography() {
        let h_gt = sample_h();
        let src = grid(5, 100.0);
        let dst: Vec<_> = src.iter().map(|&(x, y)| project(&h_gt, x, y)).collect();
        let h = dlt(&src, &dst).unwrap();
        assert!(max_transfer_error(&h, &h_gt, &src) < 1e-6);
    }

    #[test]
    fn dlt_minimal_four_points() {
        let h_gt = sample_h();
        let src = vec![(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)];
        let dst: Vec<_> = src.iter().map(|&(x, y)| project(&h_gt, x, y)).collect();
        let h = dlt(&src, &dst).unwrap();
        assert!(max_transfer_error(&h, &h_gt, &src) < 1e-6);
    }

    #[test]
    fn ransac_survives_forty_percent_outliers() {
        let h_gt = sample_h();
        let src = grid(9, 300.0); // 81 points
        let mut rng = XorShift64::new(77);
        let mut dst: Vec<_> = src
            .iter()
            .map(|&(x, y)| {
                let (px, py) = project(&h_gt, x, y);
                // ~N(0, 0.3) noise
                let (nx, ny) = rng.next_gaussian_pair();
                (px + nx * 0.3, py + ny * 0.3)
            })
            .collect();
        // Corrupt 40% with random positions.
        let n_out = (dst.len() * 2) / 5;
        for i in 0..n_out {
            let idx = (i * 2) % dst.len();
            dst[idx] = (rng.next_f64() * 400.0, rng.next_f64() * 400.0);
        }
        let res = ransac(&src, &dst, 3.0, 1000, 42).unwrap();
        let corners = [(0.0, 0.0), (300.0, 0.0), (300.0, 300.0), (0.0, 300.0)];
        let err = max_transfer_error(&res.h, &h_gt, &corners);
        assert!(err < 1.0, "corner transfer error = {err}");
        assert!(res.inliers.len() >= 40, "inliers = {}", res.inliers.len());
    }

    #[test]
    fn ransac_rejects_pure_noise() {
        let mut rng = XorShift64::new(5);
        let src: Vec<_> = (0..40).map(|_| (rng.next_f64() * 300.0, rng.next_f64() * 300.0)).collect();
        let dst: Vec<_> = (0..40).map(|_| (rng.next_f64() * 300.0, rng.next_f64() * 300.0)).collect();
        // With pure noise, RANSAC may fit *something*, but never with many inliers.
        if let Some(res) = ransac(&src, &dst, 3.0, 500, 42) {
            assert!(res.inliers.len() < 12, "noise produced {} inliers", res.inliers.len());
        }
    }
}
