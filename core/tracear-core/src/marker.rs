//! Marker compiler: precomputes everything the runtime needs from a target
//! image, so no marker-side feature extraction happens at runtime.
//!
//! The marker is described at a series of scales (factor 1/1.26 per step).
//! The runtime camera pyramid uses factor-2 steps; together any observed
//! scale lands within ~13% of a compiled scale, which BRIEF tolerates.

use crate::brief::Descriptor;
use crate::features::extract_features;
use crate::image::GrayImage;

pub struct CompileConfig {
    pub fast_threshold: u8,
    pub max_features_per_scale: usize,
    pub min_side: usize,
    pub scale_step: f32,
}

impl Default for CompileConfig {
    fn default() -> Self {
        Self {
            fast_threshold: 20,
            max_features_per_scale: 200,
            min_side: 64,
            scale_step: 1.26,
        }
    }
}

pub struct CompiledMarker {
    pub width: u32,
    pub height: u32,
    /// Feature positions in marker level-0 pixels (parallel to `descriptors`).
    pub positions: Vec<(f32, f32)>,
    pub descriptors: Vec<Descriptor>,
}

pub fn compile_marker(img: &GrayImage, cfg: &CompileConfig) -> CompiledMarker {
    let mut positions = Vec::new();
    let mut descriptors = Vec::new();
    let mut scale = 1.0f32;
    loop {
        let sw = (img.w as f32 * scale).round() as usize;
        let sh = (img.h as f32 * scale).round() as usize;
        if sw.min(sh) < cfg.min_side {
            break;
        }
        let level = if scale >= 0.999 { img.clone() } else { img.resize_area(sw, sh) };
        let (pos, desc) = extract_features(&level, cfg.fast_threshold, cfg.max_features_per_scale);
        for (i, &(x, y)) in pos.iter().enumerate() {
            positions.push((x / scale, y / scale));
            descriptors.push(desc[i]);
        }
        scale /= cfg.scale_step;
    }
    CompiledMarker {
        width: img.w as u32,
        height: img.h as u32,
        positions,
        descriptors,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic;

    #[test]
    fn compiles_multi_scale_features() {
        let img = synthetic::textured_image(320, 320, 7);
        let m = compile_marker(&img, &CompileConfig::default());
        assert_eq!((m.width, m.height), (320, 320));
        assert!(m.descriptors.len() > 300, "only {} features", m.descriptors.len());
        assert_eq!(m.descriptors.len(), m.positions.len());
        // all positions inside marker bounds (level-0 coords)
        for &(x, y) in &m.positions {
            assert!(x >= 0.0 && x <= 320.0 && y >= 0.0 && y <= 320.0, "({x},{y}) out of bounds");
        }
    }
}
