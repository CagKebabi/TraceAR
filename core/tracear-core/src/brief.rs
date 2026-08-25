//! 256-bit steered BRIEF descriptor.
//!
//! The sampling pattern is generated once from a fixed seed (isotropic
//! Gaussian, sigma = 6, clamped to [-13, 13]), so marker compiler and runtime
//! always agree bit-for-bit. The pattern is rotated by the keypoint
//! orientation before sampling ("steered" BRIEF). Images must be pre-smoothed
//! (~= Gaussian sigma 2) before sampling or the comparisons are noise-dominated.

use crate::image::GrayImage;
use crate::rng::XorShift64;
use std::sync::OnceLock;

pub const DESC_BITS: usize = 256;
pub type Descriptor = [u64; 4];

/// Border (px) a keypoint needs from the image edge so that both the
/// orientation patch (radius 15) and the rotated pattern (|pt| <= 13, rotated
/// norm <= 13*sqrt(2) ~= 18.4, rounded <= 19) stay inside the image.
pub const FEATURE_BORDER: usize = 20;

const PATTERN_SIGMA: f64 = 6.0;
const PATTERN_CLAMP: f64 = 13.0;

struct Pattern {
    pts: [(i8, i8, i8, i8); DESC_BITS],
}

fn pattern() -> &'static Pattern {
    static P: OnceLock<Pattern> = OnceLock::new();
    P.get_or_init(|| {
        let mut rng = XorShift64::new(0x7472_6163_6561_7231); // "tracear1"
        let mut pts = [(0i8, 0i8, 0i8, 0i8); DESC_BITS];
        let mut i = 0;
        while i < DESC_BITS {
            let (a, b) = rng.next_gaussian_pair();
            let (c, d) = rng.next_gaussian_pair();
            let px = (a * PATTERN_SIGMA).round().clamp(-PATTERN_CLAMP, PATTERN_CLAMP) as i8;
            let py = (b * PATTERN_SIGMA).round().clamp(-PATTERN_CLAMP, PATTERN_CLAMP) as i8;
            let qx = (c * PATTERN_SIGMA).round().clamp(-PATTERN_CLAMP, PATTERN_CLAMP) as i8;
            let qy = (d * PATTERN_SIGMA).round().clamp(-PATTERN_CLAMP, PATTERN_CLAMP) as i8;
            if px == qx && py == qy {
                continue;
            }
            pts[i] = (px, py, qx, qy);
            i += 1;
        }
        Pattern { pts }
    })
}

/// `img` must be pre-smoothed. Caller guarantees FEATURE_BORDER around (cx, cy).
pub fn compute(img: &GrayImage, cx: f32, cy: f32, angle: f32) -> Descriptor {
    let (sin, cos) = angle.sin_cos();
    let p = pattern();
    let mut desc = [0u64; 4];
    for (i, &(px, py, qx, qy)) in p.pts.iter().enumerate() {
        let (px, py) = (px as f32, py as f32);
        let (qx, qy) = (qx as f32, qy as f32);
        let ax = (cx + cos * px - sin * py).round() as usize;
        let ay = (cy + sin * px + cos * py).round() as usize;
        let bx = (cx + cos * qx - sin * qy).round() as usize;
        let by = (cy + sin * qx + cos * qy).round() as usize;
        if img.at(ax, ay) < img.at(bx, by) {
            desc[i >> 6] |= 1u64 << (i & 63);
        }
    }
    desc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matcher::hamming;
    use crate::synthetic;

    #[test]
    fn pattern_is_stable_and_in_bounds() {
        let p1 = pattern();
        for &(px, py, qx, qy) in p1.pts.iter() {
            for v in [px, py, qx, qy] {
                assert!((-13..=13).contains(&(v as i32)));
            }
        }
    }

    #[test]
    fn descriptor_survives_noise() {
        let img = synthetic::textured_image(128, 128, 11).box_blur(2).box_blur(2);
        let mut noisy_src = synthetic::textured_image(128, 128, 11);
        synthetic::add_gaussian_noise(&mut noisy_src, 4.0, 3);
        let noisy = noisy_src.box_blur(2).box_blur(2);
        let d1 = compute(&img, 64.0, 64.0, 0.3);
        let d2 = compute(&noisy, 64.0, 64.0, 0.3);
        let dist = hamming(&d1, &d2);
        assert!(dist < 40, "distance under noise = {dist}");
    }

    #[test]
    fn different_locations_differ() {
        let img = synthetic::textured_image(128, 128, 11).box_blur(2).box_blur(2);
        let d1 = compute(&img, 40.0, 40.0, 0.0);
        let d2 = compute(&img, 90.0, 85.0, 0.0);
        let dist = hamming(&d1, &d2);
        assert!(dist > 60, "unrelated descriptors too close: {dist}");
    }
}
