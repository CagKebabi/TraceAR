//! End-to-end M0 acceptance tests: a compiled marker must be found in a
//! synthetic scene under perspective, scale, noise, and brightness change —
//! and must NOT be found where it isn't.

use nalgebra::Matrix3;
use tracear_core::detector::{detect_marker, DetectorConfig};
use tracear_core::homography::{dlt, project};
use tracear_core::image::{warp_onto_aa, GrayImage};
use tracear_core::marker::{compile_marker, CompileConfig, CompiledMarker};
use tracear_core::synthetic;

const MARKER_SIZE: f64 = 320.0;

fn compiled_marker() -> (GrayImage, CompiledMarker) {
    let img = synthetic::textured_image(320, 320, 7);
    let compiled = compile_marker(&img, &CompileConfig::default());
    (img, compiled)
}

/// Ground-truth homography sending the marker onto the given frame quad.
fn h_from_quad(quad: [(f64, f64); 4]) -> Matrix3<f64> {
    let src = [
        (0.0, 0.0),
        (MARKER_SIZE, 0.0),
        (MARKER_SIZE, MARKER_SIZE),
        (0.0, MARKER_SIZE),
    ];
    dlt(&src, &quad).expect("ground-truth homography")
}

fn make_scene(marker_img: &GrayImage, h_gt: &Matrix3<f64>, seed: u64) -> GrayImage {
    let bg = synthetic::textured_image(640, 480, seed);
    let h_inv = h_gt.try_inverse().expect("invertible ground truth");
    warp_onto_aa(marker_img, &h_inv, &bg)
}

fn mean_corner_error(h_est: &Matrix3<f64>, h_gt: &Matrix3<f64>) -> f64 {
    let corners = [
        (0.0, 0.0),
        (MARKER_SIZE, 0.0),
        (MARKER_SIZE, MARKER_SIZE),
        (0.0, MARKER_SIZE),
    ];
    corners
        .iter()
        .map(|&(x, y)| {
            let a = project(h_est, x, y);
            let b = project(h_gt, x, y);
            ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
        })
        .sum::<f64>()
        / 4.0
}

#[test]
fn detects_marker_under_perspective_noise_and_brightness() {
    let (marker_img, compiled) = compiled_marker();
    // Moderate perspective: ~0.9x scale, slight rotation + keystone.
    let quad = [(180.0, 90.0), (470.0, 110.0), (450.0, 380.0), (160.0, 350.0)];
    let h_gt = h_from_quad(quad);
    let mut scene = make_scene(&marker_img, &h_gt, 99);
    synthetic::brightness_contrast(&mut scene, 0.9, 10.0);
    synthetic::add_gaussian_noise(&mut scene, 4.0, 5);

    let det = detect_marker(&compiled, &scene, &DetectorConfig::default())
        .expect("marker should be detected");
    let err = mean_corner_error(&det.homography, &h_gt);
    assert!(err < 3.0, "mean corner error = {err:.2} px (inliers: {})", det.inliers);
}

#[test]
fn detects_small_marker() {
    let (marker_img, compiled) = compiled_marker();
    // Marker at roughly 0.4x scale (~130 px on screen).
    let quad = [(240.0, 160.0), (372.0, 166.0), (366.0, 292.0), (236.0, 286.0)];
    let h_gt = h_from_quad(quad);
    let mut scene = make_scene(&marker_img, &h_gt, 55);
    synthetic::add_gaussian_noise(&mut scene, 3.0, 6);

    let det = detect_marker(&compiled, &scene, &DetectorConfig::default())
        .expect("small marker should be detected");
    let err = mean_corner_error(&det.homography, &h_gt);
    assert!(err < 3.0, "mean corner error = {err:.2} px (inliers: {})", det.inliers);
}

#[test]
fn detects_rotated_marker() {
    let (marker_img, compiled) = compiled_marker();
    // ~35 degrees in-plane rotation, mild scale.
    let quad = [(230.0, 60.0), (460.0, 200.0), (330.0, 420.0), (110.0, 270.0)];
    let h_gt = h_from_quad(quad);
    let mut scene = make_scene(&marker_img, &h_gt, 31);
    synthetic::add_gaussian_noise(&mut scene, 3.0, 8);

    let det = detect_marker(&compiled, &scene, &DetectorConfig::default())
        .expect("rotated marker should be detected");
    let err = mean_corner_error(&det.homography, &h_gt);
    assert!(err < 3.0, "mean corner error = {err:.2} px (inliers: {})", det.inliers);
}

#[test]
fn no_false_positive_on_marker_free_scene() {
    let (_, compiled) = compiled_marker();
    let mut scene = synthetic::textured_image(640, 480, 1234);
    synthetic::add_gaussian_noise(&mut scene, 3.0, 9);
    assert!(
        detect_marker(&compiled, &scene, &DetectorConfig::default()).is_none(),
        "detector hallucinated a marker in a marker-free scene"
    );
}
