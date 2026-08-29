//! Multi-target session cost: 10 markers loaded, 1 visible — the UNICOAR
//! album shape. Run: cargo run --release --example bench_multi
//!
//! Compares the M6 scheduler (shared frame features + detection budget)
//! against the naive worst case (every lost marker fully detected per frame),
//! which is what pre-0.2 releases did.

use nalgebra::Matrix3;
use std::time::Instant;
use tracear_core::detector::{detect_marker, extract_frame_features, DetectorConfig};
use tracear_core::homography::dlt;
use tracear_core::image::warp_onto_aa;
use tracear_core::marker::{compile_marker, CompileConfig, CompiledMarker};
use tracear_core::pipeline::{MarkerStatus, Pipeline};
use tracear_core::synthetic;

fn main() {
    let hero_img = synthetic::textured_image(320, 320, 7);
    let mut markers: Vec<CompiledMarker> = Vec::new();
    for i in 0..10u64 {
        let img = if i == 0 { hero_img.clone() } else { synthetic::textured_image(320, 320, 100 + i) };
        markers.push(compile_marker(&img, &CompileConfig::default()));
    }

    let quad = [(180.0, 90.0), (470.0, 110.0), (450.0, 380.0), (160.0, 350.0)];
    let src = [(0.0, 0.0), (320.0, 0.0), (320.0, 320.0), (0.0, 320.0)];
    let h_gt: Matrix3<f64> = dlt(&src, &quad).unwrap();
    let bg = synthetic::textured_image(640, 480, 99);
    let mut scene = warp_onto_aa(&hero_img, &h_gt.try_inverse().unwrap(), &bg);
    synthetic::add_gaussian_noise(&mut scene, 2.0, 5);

    // Naive worst case: full per-marker detection (pyramid + features each time).
    let cfg = DetectorConfig::default();
    let t = Instant::now();
    let reps = 5;
    for _ in 0..reps {
        for m in &markers {
            let _ = detect_marker(m, &scene, &cfg);
        }
    }
    let naive_ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;

    // Shared features: one extraction, 10 matches (detectImage / detect_only path).
    let t = Instant::now();
    for _ in 0..reps {
        let feats = extract_frame_features(&scene, &cfg);
        for m in &markers {
            let _ = tracear_core::detector::detect_marker_in(m, &feats, &cfg);
        }
    }
    let shared_ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;

    // Scheduled pipeline: budgeted detection, hero tracked after acquire.
    let mut pipeline = Pipeline::new();
    for m in markers {
        pipeline.add_marker(m);
    }
    let mut acquire_frame = None;
    let mut t_ms = 0.0f64;
    for f in 0..8u64 {
        let res = pipeline.process(&scene, t_ms);
        t_ms += 33.0;
        if acquire_frame.is_none() && res[0].status != MarkerStatus::NotFound {
            acquire_frame = Some(f);
        }
    }
    let frames = 30;
    let t = Instant::now();
    let mut max_ms = 0.0f64;
    for _ in 0..frames {
        let ft = Instant::now();
        let res = pipeline.process(&scene, t_ms);
        max_ms = max_ms.max(ft.elapsed().as_secs_f64() * 1e3);
        t_ms += 33.0;
        assert_eq!(res[0].status, MarkerStatus::Tracked);
    }
    let steady_ms = t.elapsed().as_secs_f64() * 1e3 / frames as f64;

    println!("10 markers, 1 visible, 640x480 frame:");
    println!("  naive per-marker detection : {naive_ms:.2} ms/frame  (pre-0.2 behavior)");
    println!("  shared-feature detection   : {shared_ms:.2} ms/frame  (detect_only / detectImage)");
    println!(
        "  scheduled pipeline steady  : {steady_ms:.2} ms/frame avg, {max_ms:.2} ms max  (track + amortized cold scan)"
    );
    println!("  hero acquired on frame     : {:?}", acquire_frame.expect("hero never acquired"));
}
