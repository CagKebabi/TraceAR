//! Rough native timing of the detection path (compile once, detect many).
//! Run: cargo run --release --example bench_detect
//!
//! Native x86 timings don't transfer 1:1 to WASM on a phone (expect ~2-4x
//! slower there), but they catch order-of-magnitude regressions early.

use nalgebra::Matrix3;
use std::time::Instant;
use tracear_core::detector::{detect_marker, DetectorConfig};
use tracear_core::homography::dlt;
use tracear_core::image::warp_onto_aa;
use tracear_core::marker::{compile_marker, CompileConfig};
use tracear_core::synthetic;

fn main() {
    let marker_img = synthetic::textured_image(320, 320, 7);

    let t0 = Instant::now();
    let compiled = compile_marker(&marker_img, &CompileConfig::default());
    println!("marker compile: {:.1} ms ({} features)", t0.elapsed().as_secs_f64() * 1e3, compiled.descriptors.len());

    let quad = [(180.0, 90.0), (470.0, 110.0), (450.0, 380.0), (160.0, 350.0)];
    let src = [(0.0, 0.0), (320.0, 0.0), (320.0, 320.0), (0.0, 320.0)];
    let h_gt: Matrix3<f64> = dlt(&src, &quad).unwrap();
    let bg = synthetic::textured_image(640, 480, 99);
    let mut scene = warp_onto_aa(&marker_img, &h_gt.try_inverse().unwrap(), &bg);
    synthetic::add_gaussian_noise(&mut scene, 4.0, 5);

    let cfg = DetectorConfig::default();
    // warm-up + correctness
    let det = detect_marker(&compiled, &scene, &cfg).expect("detection");
    println!("warm-up: {} inliers / {} matches", det.inliers, det.matches);

    let n = 20;
    let t1 = Instant::now();
    for _ in 0..n {
        let d = detect_marker(&compiled, &scene, &cfg);
        assert!(d.is_some());
    }
    let ms = t1.elapsed().as_secs_f64() * 1e3 / n as f64;
    println!("detect (640x480): {ms:.2} ms/frame");
}
