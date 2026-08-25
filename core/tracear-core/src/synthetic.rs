//! Deterministic synthetic image generation for tests and the bench harness.

use crate::image::GrayImage;
use crate::rng::XorShift64;

/// Corner-rich texture: overlapping random rectangles on a mid-gray base.
pub fn textured_image(w: usize, h: usize, seed: u64) -> GrayImage {
    let mut rng = XorShift64::new(seed);
    let mut img = GrayImage::new(w, h);
    for p in img.data.iter_mut() {
        *p = 128;
    }
    let n = (w * h / 1500).max(40);
    for _ in 0..n {
        let rw = 8 + rng.gen_range(32);
        let rh = 8 + rng.gen_range(32);
        if w <= rw + 1 || h <= rh + 1 {
            continue;
        }
        let x0 = rng.gen_range(w - rw);
        let y0 = rng.gen_range(h - rh);
        let v = (30 + rng.gen_range(196)) as u8;
        for y in y0..y0 + rh {
            for x in x0..x0 + rw {
                img.set(x, y, v);
            }
        }
    }
    img
}

pub fn add_gaussian_noise(img: &mut GrayImage, sigma: f64, seed: u64) {
    let mut rng = XorShift64::new(seed);
    let len = img.data.len();
    let mut i = 0;
    while i < len {
        let (a, b) = rng.next_gaussian_pair();
        img.data[i] = (img.data[i] as f64 + a * sigma).round().clamp(0.0, 255.0) as u8;
        if i + 1 < len {
            img.data[i + 1] = (img.data[i + 1] as f64 + b * sigma).round().clamp(0.0, 255.0) as u8;
        }
        i += 2;
    }
}

/// v' = v * alpha + beta, clamped.
pub fn brightness_contrast(img: &mut GrayImage, alpha: f64, beta: f64) {
    for p in img.data.iter_mut() {
        *p = ((*p as f64) * alpha + beta).round().clamp(0.0, 255.0) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texture_is_deterministic() {
        let a = textured_image(64, 64, 9);
        let b = textured_image(64, 64, 9);
        assert_eq!(a.data, b.data);
        let c = textured_image(64, 64, 10);
        assert_ne!(a.data, c.data);
    }

    #[test]
    fn noise_stays_close() {
        let mut img = textured_image(64, 64, 9);
        let orig = img.clone();
        add_gaussian_noise(&mut img, 3.0, 4);
        let mean_abs: f64 = img
            .data
            .iter()
            .zip(orig.data.iter())
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .sum::<f64>()
            / img.data.len() as f64;
        assert!(mean_abs > 0.5 && mean_abs < 6.0, "mean abs diff = {mean_abs}");
    }
}
