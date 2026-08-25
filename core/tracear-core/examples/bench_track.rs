//! M2 metrics in release mode: unfiltered static jitter, 500-frame motion
//! robustness, and track-vs-detect speed ratio.
//! Run: cargo run --release --example bench_track

use nalgebra::Matrix3;
use std::time::Instant;
use tracear_core::homography::project;
use tracear_core::image::{warp_onto_aa, GrayImage};
use tracear_core::marker::{compile_marker, CompileConfig};
use tracear_core::pipeline::{MarkerStatus, Pipeline};
use tracear_core::synthetic;

fn render(marker_img: &GrayImage, bg: &GrayImage, h: &Matrix3<f64>, noise_seed: u64) -> GrayImage {
    let mut f = warp_onto_aa(marker_img, &h.try_inverse().unwrap(), bg);
    synthetic::add_gaussian_noise(&mut f, 2.0, noise_seed);
    f
}

fn corners(h: &Matrix3<f64>) -> [(f64, f64); 4] {
    let c = [(0.0, 0.0), (320.0, 0.0), (320.0, 320.0), (0.0, 320.0)];
    let mut out = [(0.0, 0.0); 4];
    for (i, &(x, y)) in c.iter().enumerate() {
        out[i] = project(h, x, y);
    }
    out
}

fn main() {
    let marker_img = synthetic::textured_image(320, 320, 7);
    let compiled = compile_marker(&marker_img, &CompileConfig::default());
    println!(
        "marker: {} detection features, {} tracking levels, {:.0} KB serialized",
        compiled.descriptors.len(),
        compiled.tracking_levels.len(),
        compiled.to_bytes().len() as f64 / 1024.0
    );

    // --- static jitter (100 frames) ---
    let mut pipeline = Pipeline::new();
    pipeline.add_marker(compiled);
    let h_gt = synthetic::trajectory_homography(320.0, 320.0, 640.0, 480.0, 0.4);
    let bg = synthetic::textured_image(640, 480, 42);
    let mut tracked: Vec<[(f64, f64); 4]> = Vec::new();
    let mut track_ms = Vec::new();
    for f in 0..100u64 {
        let frame = render(&marker_img, &bg, &h_gt, 1000 + f);
        let t0 = Instant::now();
        let res = &pipeline.process(&frame)[0];
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        if res.status == MarkerStatus::Tracked {
            tracked.push(corners(&res.homography.unwrap()));
            track_ms.push(ms);
        }
    }
    let n = tracked.len() as f64;
    let jitter = (0..4)
        .map(|c| {
            let mx = tracked.iter().map(|f| f[c].0).sum::<f64>() / n;
            let my = tracked.iter().map(|f| f[c].1).sum::<f64>() / n;
            (tracked.iter().map(|f| (f[c].0 - mx).powi(2) + (f[c].1 - my).powi(2)).sum::<f64>() / n).sqrt()
        })
        .sum::<f64>()
        / 4.0;
    println!("static jitter (unfiltered, {} frames): {:.3} px   [target < 0.3]", tracked.len(), jitter);

    // --- 500-frame motion sequence ---
    pipeline.reset();
    let bg = synthetic::textured_image(640, 480, 21);
    let mut losses = 0;
    let mut was_tracking = false;
    let mut tracked_frames = 0;
    let mut err_sum = 0.0;
    for f in 0..500u64 {
        let t = f as f64 / 500.0;
        let h_gt = synthetic::trajectory_homography(320.0, 320.0, 640.0, 480.0, t);
        let frame = render(&marker_img, &bg, &h_gt, 5000 + f);
        let res = &pipeline.process(&frame)[0];
        match res.status {
            MarkerStatus::Tracked => {
                was_tracking = true;
                tracked_frames += 1;
                let est = corners(&res.homography.unwrap());
                let gt = corners(&h_gt);
                err_sum += (0..4)
                    .map(|c| ((est[c].0 - gt[c].0).powi(2) + (est[c].1 - gt[c].1).powi(2)).sqrt())
                    .sum::<f64>()
                    / 4.0;
            }
            _ => {
                if was_tracking {
                    losses += 1;
                }
                was_tracking = false;
            }
        }
    }
    println!(
        "motion 500 frames: {} losses [target < 2], {}/500 tracked, mean corner err {:.2} px",
        losses,
        tracked_frames,
        err_sum / tracked_frames.max(1) as f64
    );

    // --- speed: track vs detect ---
    let mean_track = track_ms.iter().sum::<f64>() / track_ms.len() as f64;
    let frame = render(&marker_img, &bg, &h_gt, 77);
    pipeline.reset();
    let t0 = Instant::now();
    let reps = 20;
    for _ in 0..reps {
        pipeline.reset(); // force full detection every iteration
        let _ = pipeline.process(&frame);
    }
    let mean_detect = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    println!(
        "track {:.2} ms vs detect {:.2} ms -> {:.1}x faster   [target >= 5x]",
        mean_track,
        mean_detect,
        mean_detect / mean_track
    );
}
