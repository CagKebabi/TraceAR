//! Marker compiler: precomputes everything the runtime needs from a target
//! image, so no marker-side feature extraction happens at runtime.
//!
//! Two kinds of data are compiled:
//! - **Detection features** at a series of scales (factor 1/1.26 per step —
//!   combined with the runtime's factor-2 camera pyramid, any observed scale
//!   lands within ~13% of a compiled scale, which BRIEF tolerates).
//! - **Tracking patches** at factor-2 levels: small grayscale templates around
//!   strong corners, used by the frame-to-frame sub-pixel tracker (M2). The
//!   21x21 support is large enough to re-warp a 9x9 (+gradient ring) frame
//!   window under 45 deg rotation and the worst-case sqrt(2) scale residual.

use crate::brief::Descriptor;
use crate::fast;
use crate::features::extract_features;
use crate::image::GrayImage;
use crate::keypoint::{select_uniform, Keypoint};

/// Side of a stored tracking template.
pub const PATCH_SIZE: usize = 21;
/// Center offset inside a stored template.
pub const PATCH_CENTER: f32 = (PATCH_SIZE as f32 - 1.0) / 2.0;
const PATCH_AREA: usize = PATCH_SIZE * PATCH_SIZE;

pub struct CompileConfig {
    pub fast_threshold: u8,
    pub max_features_per_scale: usize,
    pub min_side: usize,
    pub scale_step: f32,
    pub track_fast_threshold: u8,
    pub track_patches_per_level: usize,
}

impl Default for CompileConfig {
    fn default() -> Self {
        Self {
            fast_threshold: 20,
            max_features_per_scale: 200,
            min_side: 64,
            scale_step: 1.26,
            track_fast_threshold: 15,
            track_patches_per_level: 80,
        }
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct TrackPatch {
    /// Patch center in marker level-0 pixels.
    pub x: f32,
    pub y: f32,
    pub score: f32,
    /// PATCH_SIZE x PATCH_SIZE grayscale template (row-major), sampled from
    /// the lightly-blurred marker at this level's resolution.
    pub template: Vec<u8>,
}

#[derive(Clone, PartialEq, Debug)]
pub struct TrackingLevel {
    /// Level resolution relative to marker level 0 (1.0, 0.5, 0.25, ...).
    pub scale: f32,
    pub patches: Vec<TrackPatch>,
}

pub struct CompiledMarker {
    pub width: u32,
    pub height: u32,
    /// Detection feature positions in marker level-0 pixels (parallel to `descriptors`).
    pub positions: Vec<(f32, f32)>,
    pub descriptors: Vec<Descriptor>,
    pub tracking_levels: Vec<TrackingLevel>,
}

fn extract_track_patches(level: &GrayImage, level_scale: f32, cfg: &CompileConfig) -> Vec<TrackPatch> {
    let half = PATCH_SIZE / 2;
    let blurred = level.box_blur(1);
    let corners = fast::detect(&blurred, cfg.track_fast_threshold, half + 1);
    let kps: Vec<Keypoint> = corners
        .iter()
        .map(|c| Keypoint { x: c.x as f32, y: c.y as f32, score: c.score, angle: 0.0, level: 0 })
        .collect();
    let kps = select_uniform(kps, level.w, level.h, 12, 1, cfg.track_patches_per_level);
    kps.iter()
        .map(|kp| {
            let cx = kp.x as usize;
            let cy = kp.y as usize;
            let mut template = Vec::with_capacity(PATCH_AREA);
            for dy in 0..PATCH_SIZE {
                for dx in 0..PATCH_SIZE {
                    template.push(blurred.at(cx + dx - half, cy + dy - half));
                }
            }
            TrackPatch {
                x: kp.x / level_scale,
                y: kp.y / level_scale,
                score: kp.score,
                template,
            }
        })
        .collect()
}

pub fn compile_marker(img: &GrayImage, cfg: &CompileConfig) -> CompiledMarker {
    // Detection features across the 1.26-step scale series.
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

    // Tracking patches across factor-2 levels.
    let mut tracking_levels = Vec::new();
    let mut level_img = img.clone();
    let mut level_scale = 1.0f32;
    while level_img.w.min(level_img.h) >= cfg.min_side {
        tracking_levels.push(TrackingLevel {
            scale: level_scale,
            patches: extract_track_patches(&level_img, level_scale, cfg),
        });
        level_img = level_img.downsample_half();
        level_scale /= 2.0;
    }

    CompiledMarker {
        width: img.w as u32,
        height: img.h as u32,
        positions,
        descriptors,
        tracking_levels,
    }
}

/// `.tracear` binary format v2 (little-endian):
/// magic "TRCR" | version u32 | width u32 | height u32 | feature count u32 |
/// per feature: x f32, y f32, descriptor 4 x u64  (40 bytes each) |
/// tracking level count u32 |
/// per level: scale f32, patch count u32,
///   per patch: x f32, y f32, score f32, template PATCH_SIZE^2 bytes.
pub const MAGIC: [u8; 4] = *b"TRCR";
pub const FORMAT_VERSION: u32 = 2;
const HEADER_LEN: usize = 20;
const FEATURE_LEN: usize = 40;
const PATCH_LEN: usize = 12 + PATCH_AREA;

impl CompiledMarker {
    pub fn to_bytes(&self) -> Vec<u8> {
        let track_len: usize = 4 + self
            .tracking_levels
            .iter()
            .map(|l| 8 + l.patches.len() * PATCH_LEN)
            .sum::<usize>();
        let mut out = Vec::with_capacity(HEADER_LEN + self.positions.len() * FEATURE_LEN + track_len);
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
        out.extend_from_slice(&(self.tracking_levels.len() as u32).to_le_bytes());
        for level in &self.tracking_levels {
            out.extend_from_slice(&level.scale.to_le_bytes());
            out.extend_from_slice(&(level.patches.len() as u32).to_le_bytes());
            for p in &level.patches {
                out.extend_from_slice(&p.x.to_le_bytes());
                out.extend_from_slice(&p.y.to_le_bytes());
                out.extend_from_slice(&p.score.to_le_bytes());
                debug_assert_eq!(p.template.len(), PATCH_AREA);
                out.extend_from_slice(&p.template);
            }
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        struct Cursor<'a> {
            b: &'a [u8],
            pos: usize,
        }
        impl<'a> Cursor<'a> {
            fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
                if self.pos + n > self.b.len() {
                    return Err("marker data truncated".into());
                }
                let s = &self.b[self.pos..self.pos + n];
                self.pos += n;
                Ok(s)
            }
            fn u32(&mut self) -> Result<u32, String> {
                Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
            }
            fn f32(&mut self) -> Result<f32, String> {
                Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
            }
            fn u64(&mut self) -> Result<u64, String> {
                Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
            }
        }
        let mut c = Cursor { b: bytes, pos: 0 };
        if c.take(4)? != MAGIC {
            return Err("not a .tracear file (bad magic)".into());
        }
        let version = c.u32()?;
        if version != FORMAT_VERSION {
            return Err(format!(
                "unsupported .tracear version {version} (expected {FORMAT_VERSION}) — recompile the marker"
            ));
        }
        let width = c.u32()?;
        let height = c.u32()?;
        let count = c.u32()? as usize;
        let mut positions = Vec::with_capacity(count);
        let mut descriptors = Vec::with_capacity(count);
        for _ in 0..count {
            let x = c.f32()?;
            let y = c.f32()?;
            let mut desc: Descriptor = [0u64; 4];
            for word in desc.iter_mut() {
                *word = c.u64()?;
            }
            positions.push((x, y));
            descriptors.push(desc);
        }
        let level_count = c.u32()? as usize;
        let mut tracking_levels = Vec::with_capacity(level_count);
        for _ in 0..level_count {
            let scale = c.f32()?;
            let patch_count = c.u32()? as usize;
            let mut patches = Vec::with_capacity(patch_count);
            for _ in 0..patch_count {
                let x = c.f32()?;
                let y = c.f32()?;
                let score = c.f32()?;
                let template = c.take(PATCH_AREA)?.to_vec();
                patches.push(TrackPatch { x, y, score, template });
            }
            tracking_levels.push(TrackingLevel { scale, patches });
        }
        if c.pos != bytes.len() {
            return Err("marker data has trailing bytes".into());
        }
        Ok(CompiledMarker { width, height, positions, descriptors, tracking_levels })
    }
}

/// Multi-marker pack container ("album" file), format v1 (little-endian):
/// magic "TRPK" | version u32 | marker count u32 | per marker: byte length u32 |
/// then the markers' complete single-marker `.tracear` blobs back to back.
///
/// Deliberately a dumb container: each entry is an unmodified single-marker
/// file, so packing is byte concatenation — no recompilation. Adding or
/// removing one target never touches the others' compiled data.
pub const PACK_MAGIC: [u8; 4] = *b"TRPK";
pub const PACK_VERSION: u32 = 1;

/// Bundle single-marker `.tracear` blobs into one pack file.
pub fn pack_markers<T: AsRef<[u8]>>(blobs: &[T]) -> Vec<u8> {
    let total: usize = blobs.iter().map(|b| b.as_ref().len()).sum();
    let mut out = Vec::with_capacity(12 + blobs.len() * 4 + total);
    out.extend_from_slice(&PACK_MAGIC);
    out.extend_from_slice(&PACK_VERSION.to_le_bytes());
    out.extend_from_slice(&(blobs.len() as u32).to_le_bytes());
    for b in blobs {
        out.extend_from_slice(&(b.as_ref().len() as u32).to_le_bytes());
    }
    for b in blobs {
        out.extend_from_slice(b.as_ref());
    }
    out
}

/// Load a `.tracear` file that may be either a single marker or a pack.
/// Returns the contained markers in file order.
pub fn load_all(bytes: &[u8]) -> Result<Vec<CompiledMarker>, String> {
    if bytes.len() >= 4 && bytes[..4] == MAGIC {
        return Ok(vec![CompiledMarker::from_bytes(bytes)?]);
    }
    if bytes.len() < 12 || bytes[..4] != PACK_MAGIC {
        return Err("not a .tracear file (bad magic)".into());
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != PACK_VERSION {
        return Err(format!(
            "unsupported .tracear pack version {version} (expected {PACK_VERSION}) — repack the targets"
        ));
    }
    let count = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let table_end = 12 + count * 4;
    if bytes.len() < table_end {
        return Err("pack data truncated (length table)".into());
    }
    let mut lengths = Vec::with_capacity(count);
    for i in 0..count {
        let off = 12 + i * 4;
        lengths.push(u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize);
    }
    let body_len: usize = lengths.iter().sum();
    if bytes.len() != table_end + body_len {
        return Err("pack data truncated or has trailing bytes".into());
    }
    let mut markers = Vec::with_capacity(count);
    let mut pos = table_end;
    for (i, len) in lengths.into_iter().enumerate() {
        let m = CompiledMarker::from_bytes(&bytes[pos..pos + len])
            .map_err(|e| format!("pack entry {i}: {e}"))?;
        markers.push(m);
        pos += len;
    }
    Ok(markers)
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
        assert_eq!(back.tracking_levels, m.tracking_levels);
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
    fn pack_roundtrip() {
        let a = compile_marker(&synthetic::textured_image(128, 128, 3), &CompileConfig::default());
        let b = compile_marker(&synthetic::textured_image(128, 96, 8), &CompileConfig::default());
        let pack = pack_markers(&[a.to_bytes(), b.to_bytes()]);
        let loaded = load_all(&pack).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!((loaded[0].width, loaded[0].height), (128, 128));
        assert_eq!((loaded[1].width, loaded[1].height), (128, 96));
        assert_eq!(loaded[0].descriptors, a.descriptors);
        assert_eq!(loaded[1].tracking_levels, b.tracking_levels);
    }

    #[test]
    fn load_all_accepts_single_marker_file() {
        let m = compile_marker(&synthetic::textured_image(128, 128, 3), &CompileConfig::default());
        let loaded = load_all(&m.to_bytes()).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].positions, m.positions);
    }

    #[test]
    fn pack_rejects_corrupt_data() {
        let m = compile_marker(&synthetic::textured_image(128, 128, 3), &CompileConfig::default());
        let pack = pack_markers(&[m.to_bytes()]);
        assert!(load_all(&pack[..8]).is_err()); // truncated header
        assert!(load_all(&pack[..pack.len() - 1]).is_err()); // truncated body
        let mut bad = pack.clone();
        bad[0] = b'X';
        assert!(load_all(&bad).is_err()); // bad magic
        let mut extra = pack.clone();
        extra.push(0);
        assert!(load_all(&extra).is_err()); // trailing bytes
    }

    #[test]
    fn compiles_multi_scale_features() {
        let img = synthetic::textured_image(320, 320, 7);
        let m = compile_marker(&img, &CompileConfig::default());
        assert_eq!((m.width, m.height), (320, 320));
        assert!(m.descriptors.len() > 300, "only {} features", m.descriptors.len());
        assert_eq!(m.descriptors.len(), m.positions.len());
        for &(x, y) in &m.positions {
            assert!(x >= 0.0 && x <= 320.0 && y >= 0.0 && y <= 320.0, "({x},{y}) out of bounds");
        }
    }

    #[test]
    fn compiles_tracking_patches() {
        let img = synthetic::textured_image(320, 320, 7);
        let m = compile_marker(&img, &CompileConfig::default());
        // 320 -> 160 -> 80 (min side 64 stops the 40px level)
        assert_eq!(m.tracking_levels.len(), 3);
        let scales: Vec<f32> = m.tracking_levels.iter().map(|l| l.scale).collect();
        assert_eq!(scales, vec![1.0, 0.5, 0.25]);
        for level in &m.tracking_levels {
            assert!(level.patches.len() >= 20, "level {} has {} patches", level.scale, level.patches.len());
            for p in &level.patches {
                assert_eq!(p.template.len(), PATCH_SIZE * PATCH_SIZE);
                assert!(p.x >= 0.0 && p.x <= 320.0 && p.y >= 0.0 && p.y <= 320.0);
            }
        }
    }
}
