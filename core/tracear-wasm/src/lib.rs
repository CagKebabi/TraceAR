//! WASM bindings: a thin, allocation-conscious layer over tracear-core.
//! All heavy lifting stays in the core crate; this file only converts types.

use tracear_core::image::GrayImage;
use tracear_core::marker::{compile_marker, CompileConfig, CompiledMarker};
use tracear_core::pipeline::{MarkerStatus, Pipeline, PipelineResult};
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

/// Values per marker in a result buffer:
/// [status (0 not found / 1 detected / 2 tracked), h00..h22, n_good, n_total, quality]
pub const RESULT_STRIDE: usize = 13;

fn encode_results(results: &[PipelineResult], out: &mut Vec<f64>) {
    for r in results {
        out.push(match r.status {
            MarkerStatus::NotFound => 0.0,
            MarkerStatus::Detected => 1.0,
            MarkerStatus::Tracked => 2.0,
        });
        match &r.homography {
            Some(h) => {
                for row in 0..3 {
                    for col in 0..3 {
                        out.push(h[(row, col)]);
                    }
                }
            }
            None => out.extend_from_slice(&[0.0; 9]),
        }
        out.push(r.n_good as f64);
        out.push(r.n_total as f64);
        out.push(r.quality as f64);
    }
}

#[wasm_bindgen]
pub struct Engine {
    pipeline: Pipeline,
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
        Engine { pipeline: Pipeline::new() }
    }

    /// Add a compiled `.tracear` marker; returns its index.
    pub fn add_marker(&mut self, bytes: &[u8]) -> Result<u32, JsValue> {
        let m = CompiledMarker::from_bytes(bytes).map_err(|e| JsValue::from_str(&e))?;
        Ok(self.pipeline.add_marker(m) as u32)
    }

    pub fn marker_count(&self) -> u32 {
        self.pipeline.marker_count() as u32
    }

    /// Marker native size as [width, height] for overlay drawing.
    pub fn marker_size(&self, index: u32) -> Vec<u32> {
        match self.pipeline.marker(index as usize) {
            Some(m) => vec![m.width, m.height],
            None => vec![],
        }
    }

    /// Drop all tracking state (e.g. when the camera stops).
    pub fn reset(&mut self) {
        self.pipeline.reset();
    }

    /// Stateful detect<->track processing of a live RGBA frame. `timestamp`
    /// is the frame's capture time in ms (performance.now() domain) — used to
    /// scale motion prediction to the real, non-uniform frame spacing.
    /// Returns RESULT_STRIDE f64 values per marker (see constant above);
    /// homographies map marker px -> frame px.
    pub fn process_rgba(&mut self, rgba: &[u8], width: u32, height: u32, timestamp: f64) -> Result<Vec<f64>, JsValue> {
        let gray = rgba_to_gray(rgba, width as usize, height as usize)?;
        let results = self.pipeline.process(&gray, timestamp);
        let mut out = Vec::with_capacity(results.len() * RESULT_STRIDE);
        encode_results(&results, &mut out);
        Ok(out)
    }

    /// Stateless one-shot detection (detectImage API). Same result layout.
    pub fn detect_rgba(&self, rgba: &[u8], width: u32, height: u32) -> Result<Vec<f64>, JsValue> {
        let gray = rgba_to_gray(rgba, width as usize, height as usize)?;
        let results = self.pipeline.detect_only(&gray);
        let mut out = Vec::with_capacity(results.len() * RESULT_STRIDE);
        encode_results(&results, &mut out);
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
