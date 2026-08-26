//! Stage-level profile of the detection path — optimize what's measured,
//! not what's guessed. Run: cargo run --release --example bench_profile

use nalgebra::Matrix3;
use std::time::Instant;
use tracear_core::brief;
use tracear_core::detector::DetectorConfig;
use tracear_core::fast;
use tracear_core::features::extract_features;
use tracear_core::keypoint::{select_uniform, Keypoint};
use tracear_core::orientation;
use tracear_core::homography::{self, dlt};
use tracear_core::image::{build_pyramid, warp_onto_aa};
use tracear_core::marker::{compile_marker, CompileConfig};
use tracear_core::matcher;
use tracear_core::synthetic;

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn main() {
    let marker_img = synthetic::textured_image(320, 320, 7);
    let compiled = compile_marker(&marker_img, &CompileConfig::default());

    let quad = [(180.0, 90.0), (470.0, 110.0), (450.0, 380.0), (160.0, 350.0)];
    let src = [(0.0, 0.0), (320.0, 0.0), (320.0, 320.0), (0.0, 320.0)];
    let h_gt: Matrix3<f64> = dlt(&src, &quad).unwrap();
    let bg = synthetic::textured_image(640, 480, 99);
    let mut scene = warp_onto_aa(&marker_img, &h_gt.try_inverse().unwrap(), &bg);
    synthetic::add_gaussian_noise(&mut scene, 4.0, 5);

    let cfg = DetectorConfig::default();
    let reps = 30;

    // pyramid
    let t = Instant::now();
    for _ in 0..reps {
        let _ = build_pyramid(&scene, cfg.pyramid_min_side, cfg.pyramid_max_levels);
    }
    println!("pyramid:            {:6.2} ms", ms(t) / reps as f64);

    let pyr = build_pyramid(&scene, cfg.pyramid_min_side, cfg.pyramid_max_levels);

    // feature extraction (FAST + select + blur + orientation + BRIEF), per level
    let mut total_extract = 0.0;
    for (li, level) in pyr.levels.iter().enumerate() {
        let t = Instant::now();
        for _ in 0..reps {
            let _ = extract_features(level, cfg.fast_threshold, cfg.max_features_per_level);
        }
        let m = ms(t) / reps as f64;
        total_extract += m;
        println!("extract level {li}:    {m:6.2} ms  ({}x{})", level.w, level.h);
    }
    println!("extract total:      {total_extract:6.2} ms");

    // decompose level 0
    let level0 = &pyr.levels[0];
    let t = Instant::now();
    for _ in 0..reps {
        let _ = fast::detect(level0, cfg.fast_threshold, brief::FEATURE_BORDER);
    }
    println!("  fast level 0:     {:6.2} ms", ms(t) / reps as f64);
    let corners = fast::detect(level0, cfg.fast_threshold, brief::FEATURE_BORDER);
    println!("  corners found:    {}", corners.len());
    let t = Instant::now();
    for _ in 0..reps {
        let kps: Vec<Keypoint> = corners
            .iter()
            .map(|c| Keypoint { x: c.x as f32, y: c.y as f32, score: c.score, angle: 0.0, level: 0 })
            .collect();
        let _ = select_uniform(kps, level0.w, level0.h, 16, 3, cfg.max_features_per_level);
    }
    println!("  select level 0:   {:6.2} ms", ms(t) / reps as f64);
    let t = Instant::now();
    for _ in 0..reps {
        let _ = level0.box_blur(2).box_blur(2);
    }
    println!("  2x blur level 0:  {:6.2} ms", ms(t) / reps as f64);
    let blurred0 = level0.box_blur(2).box_blur(2);
    let kps: Vec<Keypoint> = corners
        .iter()
        .map(|c| Keypoint { x: c.x as f32, y: c.y as f32, score: c.score, angle: 0.0, level: 0 })
        .collect();
    let sel = select_uniform(kps, level0.w, level0.h, 16, 3, cfg.max_features_per_level);
    let t = Instant::now();
    for _ in 0..reps {
        for kp in &sel {
            let a = orientation::compute(&blurred0, kp.x, kp.y);
            let _ = brief::compute(&blurred0, kp.x, kp.y, a);
        }
    }
    println!("  orient+desc:      {:6.2} ms  ({} kps)", ms(t) / reps as f64, sel.len());

    // pooled features for matching
    let mut fdesc = Vec::new();
    let mut fpos = Vec::new();
    for (li, level) in pyr.levels.iter().enumerate() {
        let s = (1usize << li) as f64;
        let (pos, desc) = extract_features(level, cfg.fast_threshold, cfg.max_features_per_level);
        for (i, &(x, y)) in pos.iter().enumerate() {
            fpos.push((x as f64 * s, y as f64 * s));
            fdesc.push(desc[i]);
        }
    }
    println!(
        "features: {} frame x {} marker",
        fdesc.len(),
        compiled.descriptors.len()
    );

    // matching
    let t = Instant::now();
    for _ in 0..reps {
        let _ = matcher::match_descriptors(
            &fdesc,
            &compiled.descriptors,
            &compiled.positions,
            cfg.match_max_dist,
            cfg.match_ratio,
            cfg.second_dist_px,
        );
    }
    println!("matching:           {:6.2} ms", ms(t) / reps as f64);

    let matches = matcher::match_descriptors(
        &fdesc,
        &compiled.descriptors,
        &compiled.positions,
        cfg.match_max_dist,
        cfg.match_ratio,
        cfg.second_dist_px,
    );
    let srcp: Vec<(f64, f64)> = matches
        .iter()
        .map(|m| {
            let p = compiled.positions[m.train as usize];
            (p.0 as f64, p.1 as f64)
        })
        .collect();
    let dstp: Vec<(f64, f64)> = matches.iter().map(|m| fpos[m.query as usize]).collect();

    // ransac
    let t = Instant::now();
    for _ in 0..reps {
        let _ = homography::ransac(&srcp, &dstp, cfg.ransac_thresh_px, cfg.ransac_iters, cfg.seed);
    }
    println!("ransac:             {:6.2} ms  ({} matches)", ms(t) / reps as f64, matches.len());

    // tracking-side hot ops
    let t = Instant::now();
    for _ in 0..reps {
        let _ = scene.box_blur(1);
    }
    println!("box_blur(1) 640:    {:6.2} ms", ms(t) / reps as f64);
    let blurred = scene.box_blur(1);
    let t = Instant::now();
    for _ in 0..reps {
        let _ = blurred.downsample_half();
    }
    println!("downsample 640:     {:6.2} ms", ms(t) / reps as f64);
}
