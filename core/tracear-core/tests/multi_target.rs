//! M6 acceptance tests: multi-target sessions must keep per-frame detection
//! cost flat (shared frame features + a per-frame detection budget) without
//! giving up same-frame re-acquire of a just-lost target.

use nalgebra::Matrix3;
use tracear_core::homography::dlt;
use tracear_core::image::{warp_onto_aa, GrayImage};
use tracear_core::marker::{compile_marker, CompileConfig};
use tracear_core::pipeline::{MarkerStatus, Pipeline};
use tracear_core::synthetic;

/// 9 cheap distractor markers + one 320px "hero" marker at the given index.
fn build_pipeline(hero_idx: usize) -> (Pipeline, GrayImage) {
    let mut pipeline = Pipeline::new();
    let hero_img = synthetic::textured_image(320, 320, 7);
    for i in 0..10 {
        let img = if i == hero_idx {
            hero_img.clone()
        } else {
            synthetic::textured_image(128, 128, 100 + i as u64)
        };
        pipeline.add_marker(compile_marker(&img, &CompileConfig::default()));
    }
    (pipeline, hero_img)
}

fn hero_scene(hero_img: &GrayImage) -> GrayImage {
    let quad = [(180.0, 90.0), (470.0, 110.0), (450.0, 380.0), (160.0, 350.0)];
    let src = [(0.0, 0.0), (320.0, 0.0), (320.0, 320.0), (0.0, 320.0)];
    let h: Matrix3<f64> = dlt(&src, &quad).unwrap();
    let bg = synthetic::textured_image(640, 480, 42);
    let mut f = warp_onto_aa(hero_img, &h.try_inverse().unwrap(), &bg);
    synthetic::add_gaussian_noise(&mut f, 2.0, 5);
    f
}

fn empty_scene(seed: u64) -> GrayImage {
    let mut f = synthetic::textured_image(640, 480, seed);
    synthetic::add_gaussian_noise(&mut f, 2.0, seed + 1);
    f
}

#[test]
fn detection_budget_is_capped_and_rotation_covers_all_markers() {
    let (mut pipeline, _) = build_pipeline(0);
    let mut seen: Vec<usize> = Vec::new();
    for f in 0..15u64 {
        let frame = empty_scene(900 + f);
        let results = pipeline.process(&frame, f as f64 * 33.0);
        assert!(
            pipeline.last_detect_indices.len() <= 2,
            "frame {f}: {} detections exceed the budget",
            pipeline.last_detect_indices.len()
        );
        if f % 3 != 0 {
            // All markers are cold: off-cadence frames must not scan at all.
            assert!(
                pipeline.last_detect_indices.is_empty(),
                "frame {f} scanned cold markers off cadence: {:?}",
                pipeline.last_detect_indices
            );
        }
        for &i in &pipeline.last_detect_indices {
            seen.push(i);
        }
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.status, MarkerStatus::NotFound, "marker {i} hallucinated on frame {f}");
        }
    }
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen, (0..10).collect::<Vec<_>>(), "rotation did not cover every marker in 15 frames");
}

#[test]
fn visible_marker_is_acquired_within_rotation_then_tracked() {
    let hero = 7;
    let (mut pipeline, hero_img) = build_pipeline(hero);
    let scene = hero_scene(&hero_img);

    let mut found_frame = None;
    for f in 0..16u64 {
        let results = pipeline.process(&scene, f as f64 * 33.0);
        for (i, r) in results.iter().enumerate() {
            if i != hero {
                assert_eq!(r.status, MarkerStatus::NotFound, "distractor {i} found on frame {f}");
            }
        }
        if results[hero].status == MarkerStatus::Detected {
            found_frame = Some(f);
            break;
        }
    }
    let found_frame = found_frame.expect("hero marker not acquired within 16 frames of rotation");

    // Next frame it must ride the cheap tracker, not detection.
    let results = pipeline.process(&scene, (found_frame + 1) as f64 * 33.0);
    assert_eq!(results[hero].status, MarkerStatus::Tracked, "hero did not hand off to tracking");
}

#[test]
fn recently_lost_marker_gets_priority_every_frame() {
    let hero = 0;
    let (mut pipeline, hero_img) = build_pipeline(hero);
    let scene = hero_scene(&hero_img);

    // Acquire and track the hero for a few frames.
    let mut t = 0.0;
    for _ in 0..4 {
        pipeline.process(&scene, t);
        t += 33.0;
    }

    // Occlude it: every subsequent frame must still attempt the hero
    // (priority window), spending the rest of the budget on cold markers.
    for f in 0..5u64 {
        let frame = empty_scene(500 + f);
        pipeline.process(&frame, t);
        t += 33.0;
        assert!(
            pipeline.last_detect_indices.contains(&hero),
            "frame {f} after loss did not attempt the just-lost marker (attempts: {:?})",
            pipeline.last_detect_indices
        );
        assert!(pipeline.last_detect_indices.len() <= 2);
    }

    // Show it again: priority means immediate re-acquire.
    let results = pipeline.process(&scene, t);
    assert!(
        matches!(results[hero].status, MarkerStatus::Detected | MarkerStatus::Tracked),
        "hero not re-acquired on the first frame it reappeared"
    );
}

#[test]
fn detect_only_shares_features_across_markers() {
    let hero = 7;
    let (pipeline, hero_img) = build_pipeline(hero);
    let scene = hero_scene(&hero_img);
    let results = pipeline.detect_only(&scene);
    assert_eq!(results.len(), 10);
    for (i, r) in results.iter().enumerate() {
        if i == hero {
            assert_eq!(r.status, MarkerStatus::Detected, "hero not found by detect_only");
        } else {
            assert_eq!(r.status, MarkerStatus::NotFound, "distractor {i} hallucinated");
        }
    }
}
