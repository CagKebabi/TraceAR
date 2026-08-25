//! Diagnostic for the small-marker detection case: prints per-stage counts.
//! Run: cargo run --example debug_small

use nalgebra::Matrix3;
use tracear_core::detector::DetectorConfig;
use tracear_core::features::extract_features;
use tracear_core::homography::{self, dlt};
use tracear_core::image::{build_pyramid, warp_onto_aa};
use tracear_core::marker::{compile_marker, CompileConfig};
use tracear_core::matcher;
use tracear_core::synthetic;

fn main() {
    let marker_img = synthetic::textured_image(320, 320, 7);
    let compiled = compile_marker(&marker_img, &CompileConfig::default());
    println!("marker features: {}", compiled.descriptors.len());

    let quad = [(240.0, 160.0), (372.0, 166.0), (366.0, 292.0), (236.0, 286.0)];
    let src = [(0.0, 0.0), (320.0, 0.0), (320.0, 320.0), (0.0, 320.0)];
    let h_gt: Matrix3<f64> = dlt(&src, &quad).unwrap();
    let bg = synthetic::textured_image(640, 480, 55);
    let mut scene = warp_onto_aa(&marker_img, &h_gt.try_inverse().unwrap(), &bg);
    synthetic::add_gaussian_noise(&mut scene, 3.0, 6);

    let cfg = DetectorConfig::default();
    let pyr = build_pyramid(&scene, cfg.pyramid_min_side, cfg.pyramid_max_levels);
    let mut fpos: Vec<(f64, f64)> = Vec::new();
    let mut fdesc = Vec::new();
    for (li, level) in pyr.levels.iter().enumerate() {
        let s = (1usize << li) as f64;
        let (pos, desc) = extract_features(level, cfg.fast_threshold, cfg.max_features_per_level);
        println!("frame level {li}: {} features", pos.len());
        for (i, &(x, y)) in pos.iter().enumerate() {
            fpos.push((x as f64 * s, y as f64 * s));
            fdesc.push(desc[i]);
        }
    }

    let matches = matcher::match_descriptors(
        &fdesc,
        &compiled.descriptors,
        &compiled.positions,
        cfg.match_max_dist,
        cfg.match_ratio,
        cfg.second_dist_px,
    );
    println!("matches: {}", matches.len());

    // How many matches agree with ground truth?
    let mut gt_consistent = 0;
    for m in &matches {
        let (mx, my) = compiled.positions[m.train as usize];
        let p = homography::project(&h_gt, mx as f64, my as f64);
        let q = fpos[m.query as usize];
        let err = ((p.0 - q.0).powi(2) + (p.1 - q.1).powi(2)).sqrt();
        if err < 3.0 {
            gt_consistent += 1;
        }
    }
    println!("ground-truth-consistent matches: {gt_consistent}");

    let src_pts: Vec<(f64, f64)> = matches
        .iter()
        .map(|m| {
            let p = compiled.positions[m.train as usize];
            (p.0 as f64, p.1 as f64)
        })
        .collect();
    let dst_pts: Vec<(f64, f64)> = matches.iter().map(|m| fpos[m.query as usize]).collect();
    match homography::ransac(&src_pts, &dst_pts, cfg.ransac_thresh_px, cfg.ransac_iters, cfg.seed) {
        Some(r) => println!("ransac inliers: {}", r.inliers.len()),
        None => println!("ransac: FAILED"),
    }
}
