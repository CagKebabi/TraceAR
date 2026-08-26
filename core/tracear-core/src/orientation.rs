//! Keypoint orientation by intensity centroid (as in ORB): the angle of the
//! first-moment vector of a circular patch around the keypoint. Makes the
//! BRIEF descriptor rotation-invariant when the pattern is steered by it.

use crate::image::GrayImage;

pub const PATCH_RADIUS: i32 = 15;

fn u_max_table() -> [i32; 16] {
    let mut t = [0i32; 16];
    let r2 = (PATCH_RADIUS * PATCH_RADIUS) as f64;
    let mut v = 0usize;
    while v <= PATCH_RADIUS as usize {
        t[v] = ((r2 - (v * v) as f64).sqrt() + 0.5).floor() as i32;
        v += 1;
    }
    t
}

/// Caller must guarantee the patch fits: PATCH_RADIUS <= x,y < dim - PATCH_RADIUS.
pub fn compute(img: &GrayImage, cx: f32, cy: f32) -> f32 {
    let x0 = cx.round() as i32;
    let y0 = cy.round() as i32;
    debug_assert!(x0 >= PATCH_RADIUS && y0 >= PATCH_RADIUS);
    debug_assert!(x0 < img.w as i32 - PATCH_RADIUS && y0 < img.h as i32 - PATCH_RADIUS);
    let umax = u_max_table();
    let mut m01 = 0i64;
    let mut m10 = 0i64;
    // Row-slice inner loops: one bounds check per row instead of per pixel,
    // and LLVM can vectorize the running sums. Identical integer math:
    // m10 row contribution = sum(i*val) - bound*sum(val), with u = i - bound.
    for v in -PATCH_RADIUS..=PATCH_RADIUS {
        let bound = umax[v.unsigned_abs() as usize];
        let start = (y0 + v) as usize * img.w + (x0 - bound) as usize;
        let row = &img.data[start..start + (2 * bound + 1) as usize];
        let mut sum = 0i64;
        let mut wsum = 0i64;
        for (i, &px) in row.iter().enumerate() {
            let val = px as i64;
            sum += val;
            wsum += i as i64 * val;
        }
        m10 += wsum - bound as i64 * sum;
        m01 += v as i64 * sum;
    }
    (m01 as f32).atan2(m10 as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient_image(horizontal: bool) -> GrayImage {
        let mut img = GrayImage::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let v = if horizontal { x * 4 } else { y * 4 };
                img.set(x, y, v.min(255) as u8);
            }
        }
        img
    }

    #[test]
    fn horizontal_gradient_points_along_x() {
        let img = gradient_image(true);
        let a = compute(&img, 32.0, 32.0);
        assert!(a.abs() < 0.1, "angle = {a}");
    }

    #[test]
    fn vertical_gradient_points_along_y() {
        let img = gradient_image(false);
        let a = compute(&img, 32.0, 32.0);
        assert!((a - std::f32::consts::FRAC_PI_2).abs() < 0.1, "angle = {a}");
    }
}
