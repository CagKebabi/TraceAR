//! Shared feature-extraction path used by both the marker compiler and the
//! runtime detector — the two sides must produce comparable descriptors, so
//! this is deliberately one code path.

use crate::brief::{self, Descriptor, FEATURE_BORDER};
use crate::fast;
use crate::image::GrayImage;
use crate::keypoint::{select_uniform, Keypoint};
use crate::orientation;

pub const SELECT_CELL_PX: usize = 16;
pub const SELECT_PER_CELL: usize = 3;

/// FAST -> uniform selection -> orientation + steered BRIEF on a smoothed copy.
/// Returned positions are in `img`'s own pixel space.
pub fn extract_features(
    img: &GrayImage,
    fast_threshold: u8,
    max_features: usize,
) -> (Vec<(f32, f32)>, Vec<Descriptor>) {
    let corners = fast::detect(img, fast_threshold, FEATURE_BORDER);
    let kps: Vec<Keypoint> = corners
        .iter()
        .map(|c| Keypoint { x: c.x as f32, y: c.y as f32, score: c.score, angle: 0.0, level: 0 })
        .collect();
    let kps = select_uniform(kps, img.w, img.h, SELECT_CELL_PX, SELECT_PER_CELL, max_features);
    // ~= Gaussian sigma 2; used for both orientation and descriptor sampling.
    let blurred = img.box_blur(2).box_blur(2);
    let mut pos = Vec::with_capacity(kps.len());
    let mut desc = Vec::with_capacity(kps.len());
    for kp in &kps {
        let angle = orientation::compute(&blurred, kp.x, kp.y);
        desc.push(brief::compute(&blurred, kp.x, kp.y, angle));
        pos.push((kp.x, kp.y));
    }
    (pos, desc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic;

    #[test]
    fn extracts_features_from_textured_image() {
        let img = synthetic::textured_image(256, 256, 21);
        let (pos, desc) = extract_features(&img, 20, 500);
        assert_eq!(pos.len(), desc.len());
        assert!(pos.len() > 50, "only {} features", pos.len());
        for &(x, y) in &pos {
            assert!(x >= FEATURE_BORDER as f32 && x < (256 - FEATURE_BORDER) as f32);
            assert!(y >= FEATURE_BORDER as f32 && y < (256 - FEATURE_BORDER) as f32);
        }
    }
}
