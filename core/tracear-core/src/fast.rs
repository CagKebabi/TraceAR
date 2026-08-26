//! FAST-9 corner detector with SAD score and 3x3 non-max suppression.
//!
//! The contiguous-arc test uses a 16-bit ring mask: AND-ing the mask with its
//! own 1-bit rotation k times leaves a bit set iff k+1 circularly-consecutive
//! bits were set, so 8 iterations test for an arc of length >= 9.

use crate::image::GrayImage;

/// Bresenham circle of radius 3, clockwise from 12 o'clock.
pub const CIRCLE: [(i32, i32); 16] = [
    (0, -3), (1, -3), (2, -2), (3, -1), (3, 0), (3, 1), (2, 2), (1, 3),
    (0, 3), (-1, 3), (-2, 2), (-3, 1), (-3, 0), (-3, -1), (-2, -2), (-1, -3),
];

#[derive(Clone, Copy, Debug)]
pub struct FastCorner {
    pub x: u16,
    pub y: u16,
    pub score: f32,
}

#[inline]
fn has_arc9(mask: u16) -> bool {
    let mut r = mask;
    for _ in 0..8 {
        r &= r.rotate_left(1);
        if r == 0 {
            return false;
        }
    }
    true
}

pub fn detect(img: &GrayImage, threshold: u8, border: usize) -> Vec<FastCorner> {
    let b = border.max(3);
    if img.w <= 2 * b || img.h <= 2 * b {
        return Vec::new();
    }
    let t = threshold as i32;
    let w = img.w;
    let data: &[u8] = &img.data;
    let offsets: [isize; 16] = {
        let mut o = [0isize; 16];
        let mut i = 0;
        while i < 16 {
            o[i] = CIRCLE[i].1 as isize * w as isize + CIRCLE[i].0 as isize;
            i += 1;
        }
        o
    };
    let mut scores = vec![0f32; w * img.h];
    let mut candidates: Vec<u32> = Vec::new();

    for y in b..img.h - b {
        let row = y * w;
        for x in b..img.w - b {
            let idx = row + x;
            // SAFETY: b >= 3 and every CIRCLE offset stays within +-3 rows
            // and columns, so idx + offsets[i] is in-bounds for all pixels
            // at least `b` away from every image edge (the loop range).
            let at = |i: usize| unsafe { *data.get_unchecked((idx as isize + offsets[i]) as usize) } as i32;
            let p = unsafe { *data.get_unchecked(idx) } as i32;

            // Quick compass rejection: any 9-arc contains >= 2 of the 4
            // compass points (indices 0, 4, 8, 12).
            let mut nb = 0;
            let mut nd = 0;
            for ci in [0usize, 4, 8, 12] {
                let v = at(ci);
                if v >= p + t {
                    nb += 1;
                } else if v <= p - t {
                    nd += 1;
                }
            }
            if nb < 2 && nd < 2 {
                continue;
            }

            let mut bright: u16 = 0;
            let mut dark: u16 = 0;
            let mut vals = [0i32; 16];
            for i in 0..16 {
                let v = at(i);
                vals[i] = v;
                if v >= p + t {
                    bright |= 1 << i;
                } else if v <= p - t {
                    dark |= 1 << i;
                }
            }
            let is_b = bright.count_ones() >= 9 && has_arc9(bright);
            let is_d = dark.count_ones() >= 9 && has_arc9(dark);
            if !is_b && !is_d {
                continue;
            }
            let mut sb = 0i32;
            let mut sd = 0i32;
            for i in 0..16 {
                if bright & (1 << i) != 0 {
                    sb += vals[i] - p - t;
                }
                if dark & (1 << i) != 0 {
                    sd += p - t - vals[i];
                }
            }
            let s = if is_b && is_d { sb.max(sd) } else if is_b { sb } else { sd };
            scores[idx] = s as f32 + 1.0; // +1 so a valid corner is always > 0
            candidates.push(idx as u32);
        }
    }

    // NMS only over actual corners (a few thousand) instead of re-scanning
    // the whole score image. Same predicate, same tie-breaks.
    let mut out = Vec::new();
    for &ci in &candidates {
        let idx = ci as usize;
        let s = scores[idx];
        let x = idx % w;
        let y = idx / w;
        let mut is_max = true;
        'nms: for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let ns = scores[(y as i32 + dy) as usize * w + (x as i32 + dx) as usize];
                // Ties broken toward the scan-order-first pixel.
                if ns > s || (ns == s && (dy < 0 || (dy == 0 && dx < 0))) {
                    is_max = false;
                    break 'nms;
                }
            }
        }
        if is_max {
            out.push(FastCorner { x: x as u16, y: y as u16, score: s });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arc9_mask_logic() {
        assert!(has_arc9(0b0000_0001_1111_1111)); // 9 consecutive
        assert!(!has_arc9(0b0000_0000_1111_1111)); // only 8
        assert!(has_arc9(0b1111_1000_0000_1111)); // 9 consecutive across the wrap
        assert!(!has_arc9(0b0101_0101_0101_0101));
        assert!(has_arc9(0xFFFF));
    }

    #[test]
    fn detects_square_corners_not_edges() {
        // Bright 20x20 square on dark background.
        let mut img = GrayImage::new(40, 40);
        for p in img.data.iter_mut() {
            *p = 20;
        }
        for y in 10..30 {
            for x in 10..30 {
                img.set(x, y, 220);
            }
        }
        let corners = detect(&img, 40, 3);
        assert!(corners.len() >= 4, "found {} corners", corners.len());
        let expected = [(10.0, 10.0), (29.0, 10.0), (10.0, 29.0), (29.0, 29.0)];
        for c in &corners {
            let (cx, cy) = (c.x as f32, c.y as f32);
            let near = expected
                .iter()
                .any(|&(ex, ey)| ((cx - ex).powi(2) + (cy - ey).powi(2)).sqrt() < 2.5);
            assert!(near, "corner at ({cx},{cy}) is not near a square corner");
        }
    }

    #[test]
    fn flat_image_has_no_corners() {
        let img = GrayImage::from_vec(32, 32, vec![128; 1024]);
        assert!(detect(&img, 20, 3).is_empty());
    }
}
