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

/// `.tracear` binary format v1 (little-endian):
/// magic "TRCR" | version u32 | width u32 | height u32 | count u32 |
/// then per feature: x f32, y f32, descriptor 4 x u64  (40 bytes each).
pub const MAGIC: [u8; 4] = *b"TRCR";
pub const FORMAT_VERSION: u32 = 1;
const HEADER_LEN: usize = 20;
const FEATURE_LEN: usize = 40;

impl CompiledMarker {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_LEN + self.positions.len() * FEATURE_LEN);
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&self.width.to_le_bytes());
        out.extend_from_slice(&self.height.to_le_bytes());
        out.extend_from_slice(&(self.positions.len() as u32).to_le_bytes());
        for (&(x, y), desc) in self.positions.iter().zip(&self.descriptors) {
            out.extend_from_slice(&x.to_le_bytes());
            out.extend_from_slice(&y.to_le_bytes());
            for word in desc {
                out.extend_from_slice(&word.to_le_bytes());
            }
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < HEADER_LEN {
            return Err("marker data too short".into());
        }
        if bytes[0..4] != MAGIC {
            return Err("not a .tracear file (bad magic)".into());
        }
        let u32_at = |off: usize| u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        let version = u32_at(4);
        if version != FORMAT_VERSION {
            return Err(format!("unsupported .tracear version {version}"));
        }
        let width = u32_at(8);
        let height = u32_at(12);
        let count = u32_at(16) as usize;
        if bytes.len() != HEADER_LEN + count * FEATURE_LEN {
            return Err("marker data length mismatch".into());
        }
        let mut positions = Vec::with_capacity(count);
        let mut descriptors = Vec::with_capacity(count);
        for i in 0..count {
            let base = HEADER_LEN + i * FEATURE_LEN;
            let x = f32::from_le_bytes(bytes[base..base + 4].try_into().unwrap());
            let y = f32::from_le_bytes(bytes[base + 4..base + 8].try_into().unwrap());
            let mut desc: Descriptor = [0u64; 4];
            for (w, word) in desc.iter_mut().enumerate() {
                let off = base + 8 + w * 8;
                *word = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
            }
            positions.push((x, y));
            descriptors.push(desc);
        }
        Ok(CompiledMarker { width, height, positions, descriptors })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic;

    #[test]
    fn serialization_roundtrip() {
        let img = synthetic::textured_image(128, 128, 3);
        let m = compile_marker(&img, &CompileConfig::default());
        let bytes = m.to_bytes();
        let back = CompiledMarker::from_bytes(&bytes).unwrap();
        assert_eq!((back.width, back.height), (m.width, m.height));
        assert_eq!(back.positions, m.positions);
        assert_eq!(back.descriptors, m.descriptors);
    }

    #[test]
    fn rejects_corrupt_data() {
        let img = synthetic::textured_image(128, 128, 3);
        let mut bytes = compile_marker(&img, &CompileConfig::default()).to_bytes();
        assert!(CompiledMarker::from_bytes(&bytes[..10]).is_err()); // truncated header
        assert!(CompiledMarker::from_bytes(&bytes[..bytes.len() - 1]).is_err()); // truncated body
        bytes[0] = b'X';
        assert!(CompiledMarker::from_bytes(&bytes).is_err()); // bad magic
    }

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
