//! WASM bindings: a thin, allocation-conscious layer over tracear-core.
//! All heavy lifting stays in the core crate; this file only converts types.

use tracear_core::image::GrayImage;
use tracear_core::marker::{compile_marker, CompileConfig, CompiledMarker};
use tracear_core::pipeline::MarkerStatus;
use tracear_core::session::{Session, SessionConfig, SessionResult};
use wasm_bindgen::prelude::*;

/// Grayscale straight from a video frame's Y (luma) plane — no color
/// conversion, no canvas readback. `stride` is the plane's row pitch in
/// bytes (>= width). String errors: JsValue must not be constructed in
/// native unit tests (wasm-bindgen panics outside a JS host).
fn gray_from_y_plane(y: &[u8], stride: usize, w: usize, h: usize) -> Result<GrayImage, String> {
    if stride < w || h == 0 || y.len() < stride * (h - 1) + w {
        return Err("y-plane buffer/stride does not match dimensions".into());
    }
    if stride == w {
        return Ok(GrayImage::from_vec(w, h, y[..w * h].to_vec()));
    }
    let mut data = Vec::with_capacity(w * h);
    for row in 0..h {
        data.extend_from_slice(&y[row * stride..row * stride + w]);
    }
    Ok(GrayImage::from_vec(w, h, data))
}

fn rgba_to_gray(rgba: &[u8], w: usize, h: usize) -> Result<GrayImage, String> {
    if rgba.len() != w * h * 4 {
        return Err("rgba buffer size does not match width*height*4".into());
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
/// [status (0 not found / 1 detected / 2 tracked),
///  h00..h22 (marker px -> frame px),
///  nGood, nTotal, quality,
///  poseValid,
///  qx, qy, qz, qw            (marker-object -> OpenCV-camera rotation),
///  tx, ty, tz                (translation, physical units),
///  vx, vy, vz                (filtered linear velocity, units/s),
///  wx, wy, wz                (body-frame angular velocity, rad/s),
///  posLagS, rotLagS          (filter group delay, seconds),
///  rqx, rqy, rqz, rqw        (raw un-filtered rotation),
///  rtx, rty, rtz             (raw un-filtered translation),
///  focalRatio                (current f / long_side estimate)]
pub const RESULT_STRIDE: usize = 37;

fn encode_results(results: &[SessionResult], focal_ratio: f64, out: &mut Vec<f64>) {
    for r in results {
        out.push(match r.tracking.status {
            MarkerStatus::NotFound => 0.0,
            MarkerStatus::Detected => 1.0,
            MarkerStatus::Tracked => 2.0,
        });
        match &r.tracking.homography {
            Some(h) => {
                for row in 0..3 {
                    for col in 0..3 {
                        out.push(h[(row, col)]);
                    }
                }
            }
            None => out.extend_from_slice(&[0.0; 9]),
        }
        out.push(r.tracking.n_good as f64);
        out.push(r.tracking.n_total as f64);
        out.push(r.tracking.quality as f64);
        match (&r.pose, &r.raw) {
            (Some(p), Some(raw)) => {
                out.push(1.0);
                let q = p.rotation.quaternion();
                out.extend_from_slice(&[q.i, q.j, q.k, q.w]);
                out.extend_from_slice(&[p.translation.x, p.translation.y, p.translation.z]);
                out.extend_from_slice(&[p.velocity.x, p.velocity.y, p.velocity.z]);
                out.extend_from_slice(&[p.angular_velocity.x, p.angular_velocity.y, p.angular_velocity.z]);
                out.extend_from_slice(&[p.pos_lag_s, p.rot_lag_s]);
                let rq = raw.rotation.quaternion();
                out.extend_from_slice(&[rq.i, rq.j, rq.k, rq.w]);
                out.extend_from_slice(&[raw.translation.x, raw.translation.y, raw.translation.z]);
            }
            _ => out.extend_from_slice(&[0.0; 23]),
        }
        out.push(focal_ratio);
    }
}

#[wasm_bindgen]
pub struct Engine {
    session: Session,
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
        Engine { session: Session::new(SessionConfig::default()) }
    }

    /// Add a compiled `.tracear` marker with its physical width (meters, or
    /// any unit — poses come out in the same unit; pass 1.0 if unknown).
    /// Returns the marker index.
    pub fn add_marker(&mut self, bytes: &[u8], physical_width: f64) -> Result<u32, JsValue> {
        let m = CompiledMarker::from_bytes(bytes).map_err(|e| JsValue::from_str(&e))?;
        Ok(self.session.add_marker(m, physical_width) as u32)
    }

    /// Add every marker contained in a `.tracear` file — a single-marker file
    /// or a multi-marker pack. `physical_widths` supplies widths for the
    /// contained markers in order; missing entries default to 1.0. Returns
    /// the assigned marker indices.
    pub fn add_markers(&mut self, bytes: &[u8], physical_widths: &[f64]) -> Result<Vec<u32>, JsValue> {
        let markers = tracear_core::marker::load_all(bytes).map_err(|e| JsValue::from_str(&e))?;
        let mut indices = Vec::with_capacity(markers.len());
        for (i, m) in markers.into_iter().enumerate() {
            let w = physical_widths.get(i).copied().unwrap_or(1.0);
            indices.push(self.session.add_marker(m, w) as u32);
        }
        Ok(indices)
    }

    pub fn marker_count(&self) -> u32 {
        self.session.marker_count() as u32
    }

    /// Marker native size as [width, height] for overlay drawing.
    pub fn marker_size(&self, index: u32) -> Vec<u32> {
        match self.session.marker(index as usize) {
            Some(m) => vec![m.width, m.height],
            None => vec![],
        }
    }

    /// Cap the number of simultaneously tracked markers (0 = unlimited).
    /// With 1 the session is exclusive: once a target is acquired, no other
    /// detection runs at all until it is lost — the cheapest and steadiest
    /// mode for "one target at a time" experiences.
    pub fn set_max_tracked(&mut self, n: u32) {
        self.session.pipeline.schedule.max_tracked = n as usize;
    }

    /// Drop all tracking/filter state (e.g. when the camera stops).
    pub fn reset(&mut self) {
        self.session.reset();
    }

    /// Stateful processing of a live RGBA frame: detect<->track, pose
    /// estimation, SE(3) filtering. `timestamp` is the frame's capture time
    /// in ms (performance.now() domain). Returns RESULT_STRIDE f64 values
    /// per marker (see constant above).
    pub fn process_rgba(&mut self, rgba: &[u8], width: u32, height: u32, timestamp: f64) -> Result<Vec<f64>, JsValue> {
        let gray = rgba_to_gray(rgba, width as usize, height as usize).map_err(|e| JsValue::from_str(&e))?;
        let results = self.session.process(&gray, timestamp);
        let mut out = Vec::with_capacity(results.len() * RESULT_STRIDE);
        encode_results(&results, self.session.focal_ratio(), &mut out);
        Ok(out)
    }

    /// Stateful processing straight from a camera VideoFrame's Y plane
    /// (WebCodecs fast path): the luma plane IS the grayscale image. The
    /// frame is area-downscaled in here to exactly (out_width, out_height),
    /// so the returned homographies live in that pixel space.
    #[allow(clippy::too_many_arguments)]
    pub fn process_yplane(
        &mut self,
        y: &[u8],
        stride: u32,
        width: u32,
        height: u32,
        out_width: u32,
        out_height: u32,
        timestamp: f64,
    ) -> Result<Vec<f64>, JsValue> {
        let gray = gray_from_y_plane(y, stride as usize, width as usize, height as usize).map_err(|e| JsValue::from_str(&e))?;
        let gray = if gray.w == out_width as usize && gray.h == out_height as usize {
            gray
        } else {
            gray.resize_area(out_width as usize, out_height as usize)
        };
        let results = self.session.process(&gray, timestamp);
        let mut out = Vec::with_capacity(results.len() * RESULT_STRIDE);
        encode_results(&results, self.session.focal_ratio(), &mut out);
        Ok(out)
    }

    /// Stateless one-shot detection (detectImage API): raw un-filtered pose,
    /// zero velocities. Same result layout.
    pub fn detect_rgba(&self, rgba: &[u8], width: u32, height: u32) -> Result<Vec<f64>, JsValue> {
        let gray = rgba_to_gray(rgba, width as usize, height as usize).map_err(|e| JsValue::from_str(&e))?;
        let results = self.session.detect_only(&gray);
        let mut out = Vec::with_capacity(results.len() * RESULT_STRIDE);
        encode_results(&results, self.session.focal_ratio(), &mut out);
        Ok(out)
    }
}

/// Compile an RGBA image into `.tracear` marker bytes.
#[wasm_bindgen]
pub fn compile_marker_rgba(rgba: &[u8], width: u32, height: u32) -> Result<Vec<u8>, JsValue> {
    let gray = rgba_to_gray(rgba, width as usize, height as usize).map_err(|e| JsValue::from_str(&e))?;
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

    #[test]
    fn y_plane_stride_crop() {
        // 3x2 visible in a stride-4 buffer; last row has no tail padding.
        let y = [1u8, 2, 3, 99, 4, 5, 6];
        let g = gray_from_y_plane(&y, 4, 3, 2).unwrap();
        assert_eq!(g.data, vec![1, 2, 3, 4, 5, 6]);
        assert!(gray_from_y_plane(&y, 2, 3, 2).is_err()); // stride < width
        assert!(gray_from_y_plane(&y[..5], 4, 3, 2).is_err()); // too short
    }
}
