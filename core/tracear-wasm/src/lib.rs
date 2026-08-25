//! WASM bindings: a thin, allocation-conscious layer over tracear-core.
//! All heavy lifting stays in the core crate; this file only converts types.

use tracear_core::detector::{detect_marker, DetectorConfig};
use tracear_core::image::GrayImage;
use tracear_core::marker::{compile_marker, CompileConfig, CompiledMarker};
use wasm_bindgen::prelude::*;

fn rgba_to_gray(rgba: &[u8], w: usize, h: usize) -> Result<GrayImage, JsValue> {
    if rgba.len() != w * h * 4 {
        return Err(JsValue::from_str("rgba buffer size does not match width*height*4"));
    }
    let mut data = Vec::with_capacity(w * h);
    for px in rgba.chunks_exact(4) {
        // Rec. 601 luma, integer approximation.
        let g = (px[0] as u32 * 77 + px[1] as u32 * 150 + px[2] as u32 * 29) >> 8;
        data.push(g as u8);
    }
    Ok(GrayImage::from_vec(w, h, data))
}

/// Values per marker in the `detect_rgba` result:
/// [found, h00,h01,h02,h10,h11,h12,h20,h21,h22, inliers, matches]
pub const RESULT_STRIDE: usize = 12;

#[wasm_bindgen]
pub struct Engine {
    markers: Vec<CompiledMarker>,
    cfg: DetectorConfig,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl Engine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Engine {
        Engine { markers: Vec::new(), cfg: DetectorConfig::default() }
    }

    /// Add a compiled `.tracear` marker; returns its index.
    pub fn add_marker(&mut self, bytes: &[u8]) -> Result<u32, JsValue> {
        let m = CompiledMarker::from_bytes(bytes).map_err(|e| JsValue::from_str(&e))?;
        self.markers.push(m);
        Ok((self.markers.len() - 1) as u32)
    }

    pub fn marker_count(&self) -> u32 {
        self.markers.len() as u32
    }

    /// Marker native size as [width, height] for overlay drawing.
    pub fn marker_size(&self, index: u32) -> Vec<u32> {
        match self.markers.get(index as usize) {
            Some(m) => vec![m.width, m.height],
            None => vec![],
        }
    }

    /// Run detection for every added marker on an RGBA frame.
    /// Returns RESULT_STRIDE f64 values per marker (see constant above);
    /// homographies map marker px -> frame px.
    pub fn detect_rgba(&self, rgba: &[u8], width: u32, height: u32) -> Result<Vec<f64>, JsValue> {
        let gray = rgba_to_gray(rgba, width as usize, height as usize)?;
        let mut out = Vec::with_capacity(self.markers.len() * RESULT_STRIDE);
        for m in &self.markers {
            match detect_marker(m, &gray, &self.cfg) {
                Some(d) => {
                    out.push(1.0);
                    for r in 0..3 {
                        for c in 0..3 {
                            out.push(d.homography[(r, c)]);
                        }
                    }
                    out.push(d.inliers as f64);
                    out.push(d.matches as f64);
                }
                None => out.extend_from_slice(&[0.0; RESULT_STRIDE]),
            }
        }
        Ok(out)
    }
}

/// Compile an RGBA image into `.tracear` marker bytes.
#[wasm_bindgen]
pub fn compile_marker_rgba(rgba: &[u8], width: u32, height: u32) -> Result<Vec<u8>, JsValue> {
    let gray = rgba_to_gray(rgba, width as usize, height as usize)?;
    Ok(compile_marker(&gray, &CompileConfig::default()).to_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba_conversion_matches_luma() {
        let rgba = [255u8, 255, 255, 255, 0, 0, 0, 255];
        let g = rgba_to_gray(&rgba, 2, 1).unwrap();
        assert_eq!(g.at(0, 0), 255);
        assert_eq!(g.at(1, 0), 0);
    }
}
