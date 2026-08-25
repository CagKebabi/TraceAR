//! M2 acceptance tests: unfiltered static jitter < 0.3 px, and a moderate-
//! motion sequence survives with at most one tracking loss. Everything is
//! seeded — failures reproduce exactly.

use nalgebra::Matrix3;
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

fn project_corners(h: &Matrix3<f64>, mw: f64, mh: f64) -> [(f64, f64); 4] {
    let corners = [(0.0, 0.0), (mw, 0.0), (mw, mh), (0.0, mh)];
    let mut out = [(0.0, 0.0); 4];
    for (i, &(x, y)) in corners.iter().enumerate() {
        out[i] = project(h, x, y);
    }
    out
}

#[test]
fn static_scene_jitter_below_target() {
    let marker_img = synthetic::textured_image(320, 320, 7);
    let mut pipeline = Pipeline::new();
    pipeline.add_marker(compile_marker(&marker_img, &CompileConfig::default()));
    let h_gt = synthetic::trajectory_homography(320.0, 320.0, 640.0, 480.0, 0.4);
    let bg = synthetic::textured_image(640, 480, 42);

    let mut tracked: Vec<[(f64, f64); 4]> = Vec::new();
    for f in 0..40u64 {
        let frame = render(&marker_img, &bg, &h_gt, 1000 + f);
        let res = &pipeline.process(&frame)[0];
        if f == 0 {
            assert_eq!(res.status, MarkerStatus::Detected, "first frame should acquire");
        } else {
            assert_eq!(res.status, MarkerStatus::Tracked, "frame {f} was not tracked");
            tracked.push(project_corners(&res.homography.unwrap(), 320.0, 320.0));
        }
    }

    // Jitter = mean over corners of the positional std-dev across frames.
    let n = tracked.len() as f64;
    let mut jitter_sum = 0.0;
    let mut err_sum = 0.0;
    let gt = project_corners(&h_gt, 320.0, 320.0);
    for c in 0..4 {
        let mx = tracked.iter().map(|f| f[c].0).sum::<f64>() / n;
        let my = tracked.iter().map(|f| f[c].1).sum::<f64>() / n;
        let var = tracked
            .iter()
            .map(|f| (f[c].0 - mx).powi(2) + (f[c].1 - my).powi(2))
            .sum::<f64>()
            / n;
        jitter_sum += var.sqrt();
        err_sum += ((mx - gt[c].0).powi(2) + (my - gt[c].1).powi(2)).sqrt();
    }
    let jitter = jitter_sum / 4.0;
    let accuracy = err_sum / 4.0;
    assert!(jitter < 0.3, "unfiltered static jitter = {jitter:.3} px (target < 0.3)");
    assert!(accuracy < 1.5, "mean corner bias vs ground truth = {accuracy:.2} px");
}

#[test]
fn survives_moderate_motion_sequence() {
    let marker_img = synthetic::textured_image(320, 320, 7);
    let mut pipeline = Pipeline::new();
    pipeline.add_marker(compile_marker(&marker_img, &CompileConfig::default()));
    let bg = synthetic::textured_image(480, 360, 21);

    let frames = 150u64;
    let mut acquired = false;
    let mut losses = 0usize;
    let mut was_tracking = false;
    let mut tracked_frames = 0usize;
    let mut err_sum = 0.0f64;
    for f in 0..frames {
        // Same angular velocity as the 500-frame full bench.
        let t = f as f64 / 500.0;
        let h_gt = synthetic::trajectory_homography(320.0, 320.0, 480.0, 360.0, t);
        let frame = render(&marker_img, &bg, &h_gt, 5000 + f);
        let res = &pipeline.process(&frame)[0];
        match res.status {
            MarkerStatus::Tracked => {
                was_tracking = true;
                tracked_frames += 1;
                let est = project_corners(&res.homography.unwrap(), 320.0, 320.0);
                let gt = project_corners(&h_gt, 320.0, 320.0);
                err_sum += (0..4)
                    .map(|c| ((est[c].0 - gt[c].0).powi(2) + (est[c].1 - gt[c].1).powi(2)).sqrt())
                    .sum::<f64>()
                    / 4.0;
            }
            MarkerStatus::Detected => {
                if was_tracking {
                    losses += 1; // had to re-acquire
                }
                acquired = true;
                was_tracking = false;
            }
            MarkerStatus::NotFound => {
                if was_tracking {
                    losses += 1;
                }
                was_tracking = false;
            }
        }
    }
    assert!(acquired, "never acquired the marker");
    assert!(losses <= 1, "{losses} tracking losses over {frames} frames (target <= 1)");
    let mean_err = err_sum / tracked_frames.max(1) as f64;
    assert!(tracked_frames as f64 > frames as f64 * 0.9, "only {tracked_frames}/{frames} frames tracked");
    assert!(mean_err < 2.0, "mean corner error while tracking = {mean_err:.2} px");
}
