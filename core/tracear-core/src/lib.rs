//! Tracear core — portable CV pipeline for planar image tracking.
//!
//! Pure Rust, no I/O, fully deterministic (all randomness is seeded).
//! Compiled natively for tests/tools and to wasm32 for the browser runtime.
//!
//! Conventions (see also /CLAUDE.md):
//! - Pixel centers at integer coordinates, y grows downward.
//! - Homographies map marker level-0 px -> frame level-0 px unless named otherwise.
//! - Coordinates crossing module boundaries are level-0; per-level coordinates
//!   stay inside the module that produced them.

pub mod rng;
pub mod image;
pub mod keypoint;
pub mod fast;
pub mod orientation;
pub mod brief;
pub mod features;
pub mod matcher;
pub mod homography;
pub mod marker;
pub mod detector;
pub mod tracker;
pub mod pipeline;
pub mod synthetic;
