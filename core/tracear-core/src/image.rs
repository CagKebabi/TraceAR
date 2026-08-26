//! Grayscale image container and basic operations: bilinear sampling,
//! resize, box blur, pyramid, perspective warp.

use nalgebra::{Matrix3, Vector3};

#[derive(Clone)]
pub struct GrayImage {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>,
}

impl GrayImage {
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, data: vec![0; w * h] }
    }

    pub fn from_vec(w: usize, h: usize, data: Vec<u8>) -> Self {
        assert_eq!(data.len(), w * h);
        Self { w, h, data }
    }

    #[inline(always)]
    pub fn at(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.w + x]
    }

    #[inline(always)]
    pub fn set(&mut self, x: usize, y: usize, v: u8) {
        self.data[y * self.w + x] = v;
    }

    /// Bilinear sample with clamp-to-edge.
    pub fn bilinear(&self, x: f32, y: f32) -> f32 {
        let x = x.clamp(0.0, (self.w - 1) as f32);
        let y = y.clamp(0.0, (self.h - 1) as f32);
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(self.w - 1);
        let y1 = (y0 + 1).min(self.h - 1);
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;
        let a = self.at(x0, y0) as f32;
        let b = self.at(x1, y0) as f32;
        let c = self.at(x0, y1) as f32;
        let d = self.at(x1, y1) as f32;
        a * (1.0 - fx) * (1.0 - fy) + b * fx * (1.0 - fy) + c * (1.0 - fx) * fy + d * fx * fy
    }

    /// 2x2 average downsample (drops odd last row/column).
    pub fn downsample_half(&self) -> GrayImage {
        let nw = self.w / 2;
        let nh = self.h / 2;
        let mut out = GrayImage::new(nw, nh);
        for y in 0..nh {
            for x in 0..nw {
                let s = self.at(2 * x, 2 * y) as u16
                    + self.at(2 * x + 1, 2 * y) as u16
                    + self.at(2 * x, 2 * y + 1) as u16
                    + self.at(2 * x + 1, 2 * y + 1) as u16;
                out.set(x, y, ((s + 2) / 4) as u8);
            }
        }
        out
    }

    pub fn resize_bilinear(&self, nw: usize, nh: usize) -> GrayImage {
        assert!(nw > 0 && nh > 0);
        let mut out = GrayImage::new(nw, nh);
        let sx = self.w as f32 / nw as f32;
        let sy = self.h as f32 / nh as f32;
        for y in 0..nh {
            for x in 0..nw {
                // pixel-center mapping
                let src_x = (x as f32 + 0.5) * sx - 0.5;
                let src_y = (y as f32 + 0.5) * sy - 0.5;
                out.set(x, y, self.bilinear(src_x, src_y).round() as u8);
            }
        }
        out
    }

    /// Area-correct resize for minification: progressively 2x2-averages while
    /// the remaining factor is >= 2, then bilinear for the residual. Plain
    /// bilinear point-sampling aliases badly beyond 2x minification, which
    /// corrupts descriptors compiled at small marker scales.
    pub fn resize_area(&self, nw: usize, nh: usize) -> GrayImage {
        assert!(nw > 0 && nh > 0);
        let mut cur = self.clone();
        while cur.w >= 2 * nw && cur.h >= 2 * nh && cur.w >= 2 && cur.h >= 2 {
            cur = cur.downsample_half();
        }
        if cur.w == nw && cur.h == nh {
            cur
        } else {
            cur.resize_bilinear(nw, nh)
        }
    }

    /// Separable box blur with edge replication (constant window size, so a
    /// single normalization at the end — max radius 127). Running-sum
    /// implementation: O(1) per pixel regardless of radius, bit-identical to
    /// the naive windowed sum (pure integer arithmetic).
    pub fn box_blur(&self, radius: usize) -> GrayImage {
        if radius == 0 {
            return self.clone();
        }
        assert!(radius <= 127);
        let w = self.w;
        let h = self.h;
        let r = radius as isize;
        let win = (2 * radius + 1) as u32;
        let clamp_x = |x: isize| x.clamp(0, w as isize - 1) as usize;
        let clamp_y = |y: isize| y.clamp(0, h as isize - 1) as usize;

        // Horizontal pass: sliding window sum per row.
        let mut tmp = vec![0u16; w * h];
        for y in 0..h {
            let row = &self.data[y * w..(y + 1) * w];
            let orow = &mut tmp[y * w..(y + 1) * w];
            let mut s: u32 = 0;
            for i in -r..=r {
                s += row[clamp_x(i)] as u32;
            }
            orow[0] = s as u16;
            for x in 1..w {
                s += row[clamp_x(x as isize + r)] as u32;
                s -= row[clamp_x(x as isize - 1 - r)] as u32;
                orow[x] = s as u16;
            }
        }

        // Vertical pass: one running column-sum array swept down the image
        // (row-major access — cache friendly).
        let mut out = GrayImage::new(w, h);
        let norm = win * win;
        let half = norm / 2;
        let mut acc = vec![0u32; w];
        for i in -r..=r {
            let src = &tmp[clamp_y(i) * w..clamp_y(i) * w + w];
            for x in 0..w {
                acc[x] += src[x] as u32;
            }
        }
        for y in 0..h {
            if y > 0 {
                let add = &tmp[clamp_y(y as isize + r) * w..clamp_y(y as isize + r) * w + w];
                let sub = &tmp[clamp_y(y as isize - 1 - r) * w..clamp_y(y as isize - 1 - r) * w + w];
                for x in 0..w {
                    acc[x] += add[x] as u32;
                    acc[x] -= sub[x] as u32;
                }
            }
            let orow = &mut out.data[y * w..(y + 1) * w];
            for x in 0..w {
                orow[x] = ((acc[x] + half) / norm) as u8;
            }
        }
        out
    }
}

pub struct Pyramid {
    /// levels[i] has scale factor 2^i relative to level 0.
    pub levels: Vec<GrayImage>,
}

pub fn build_pyramid(img: &GrayImage, min_side: usize, max_levels: usize) -> Pyramid {
    let mut levels = vec![img.clone()];
    while levels.len() < max_levels {
        let last = levels.last().unwrap();
        if last.w / 2 < min_side || last.h / 2 < min_side {
            break;
        }
        levels.push(last.downsample_half());
    }
    Pyramid { levels }
}

fn warp_into(src: &GrayImage, h_dst_to_src: &Matrix3<f64>, out: &mut GrayImage, keep_background: bool, fill: u8) {
    let max_x = (src.w - 1) as f64;
    let max_y = (src.h - 1) as f64;
    for y in 0..out.h {
        for x in 0..out.w {
            let p = h_dst_to_src * Vector3::new(x as f64, y as f64, 1.0);
            if p.z.abs() < 1e-12 {
                if !keep_background {
                    out.set(x, y, fill);
                }
                continue;
            }
            let sx = p.x / p.z;
            let sy = p.y / p.z;
            if sx >= 0.0 && sy >= 0.0 && sx <= max_x && sy <= max_y {
                let v = src.bilinear(sx as f32, sy as f32).round() as u8;
                out.set(x, y, v);
            } else if !keep_background {
                out.set(x, y, fill);
            }
        }
    }
}

/// Render `src` into a new dst_w x dst_h image. `h_dst_to_src` maps destination
/// pixels to source pixels. Out-of-source pixels get `fill`.
pub fn warp_perspective(src: &GrayImage, h_dst_to_src: &Matrix3<f64>, dst_w: usize, dst_h: usize, fill: u8) -> GrayImage {
    let mut out = GrayImage::new(dst_w, dst_h);
    warp_into(src, h_dst_to_src, &mut out, false, fill);
    out
}

/// Render `src` on top of a background image; pixels mapping outside the
/// source keep the background value.
pub fn warp_onto(src: &GrayImage, h_dst_to_src: &Matrix3<f64>, background: &GrayImage) -> GrayImage {
    let mut out = background.clone();
    warp_into(src, h_dst_to_src, &mut out, true, 0);
    out
}

/// Like `warp_onto` but with 2x2 supersampling per destination pixel —
/// approximates the optical anti-aliasing a real camera applies when a target
/// is minified. Used by tests and the bench harness to render realistic
/// scenes; pixels only partially covered by the source keep the background.
pub fn warp_onto_aa(src: &GrayImage, h_dst_to_src: &Matrix3<f64>, background: &GrayImage) -> GrayImage {
    let mut out = background.clone();
    let max_x = (src.w - 1) as f64;
    let max_y = (src.h - 1) as f64;
    let offs = [(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)];
    for y in 0..out.h {
        for x in 0..out.w {
            let mut acc = 0.0f32;
            let mut cnt = 0u32;
            for &(ox, oy) in &offs {
                let p = h_dst_to_src * Vector3::new(x as f64 + ox, y as f64 + oy, 1.0);
                if p.z.abs() < 1e-12 {
                    cnt = 0;
                    break;
                }
                let sx = p.x / p.z;
                let sy = p.y / p.z;
                if sx >= 0.0 && sy >= 0.0 && sx <= max_x && sy <= max_y {
                    acc += src.bilinear(sx as f32, sy as f32);
                    cnt += 1;
                } else {
                    cnt = 0;
                    break;
                }
            }
            if cnt == 4 {
                out.set(x, y, (acc / 4.0).round() as u8);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_at_integer_coords_is_exact() {
        let img = GrayImage::from_vec(2, 2, vec![10, 20, 30, 40]);
        assert_eq!(img.bilinear(0.0, 0.0), 10.0);
        assert_eq!(img.bilinear(1.0, 0.0), 20.0);
        assert_eq!(img.bilinear(0.0, 1.0), 30.0);
        assert_eq!(img.bilinear(1.0, 1.0), 40.0);
        assert!((img.bilinear(0.5, 0.5) - 25.0).abs() < 1e-4);
    }

    #[test]
    fn downsample_averages() {
        let img = GrayImage::from_vec(2, 2, vec![10, 20, 30, 40]);
        let d = img.downsample_half();
        assert_eq!((d.w, d.h), (1, 1));
        assert_eq!(d.at(0, 0), 25);
    }

    #[test]
    fn box_blur_preserves_flat_image() {
        let img = GrayImage::from_vec(8, 8, vec![77; 64]);
        let b = img.box_blur(2);
        assert!(b.data.iter().all(|&v| v == 77));
    }

    /// The running-sum implementation must stay bit-identical to the naive
    /// windowed sum — compiled markers depend on exact blur values.
    #[test]
    fn box_blur_matches_naive_reference() {
        let mut img = GrayImage::new(23, 17); // odd sizes exercise the edges
        for (i, p) in img.data.iter_mut().enumerate() {
            *p = ((i * 37 + 11) % 256) as u8;
        }
        for radius in [1usize, 2, 3] {
            let fast = img.box_blur(radius);
            let r = radius as isize;
            let win = (2 * radius + 1) as u32;
            let norm = win * win;
            for y in 0..img.h as isize {
                for x in 0..img.w as isize {
                    let mut s: u32 = 0;
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let xx = (x + dx).clamp(0, img.w as isize - 1) as usize;
                            let yy = (y + dy).clamp(0, img.h as isize - 1) as usize;
                            s += img.at(xx, yy) as u32;
                        }
                    }
                    let expect = ((s + norm / 2) / norm) as u8;
                    assert_eq!(fast.at(x as usize, y as usize), expect, "r={radius} at ({x},{y})");
                }
            }
        }
    }

    #[test]
    fn identity_warp_reproduces_image() {
        let mut img = GrayImage::new(16, 16);
        for y in 0..16 {
            for x in 0..16 {
                img.set(x, y, ((x * 16 + y * 3) % 255) as u8);
            }
        }
        let id = Matrix3::identity();
        let out = warp_perspective(&img, &id, 16, 16, 0);
        assert_eq!(out.data, img.data);
    }

    #[test]
    fn pyramid_levels_halve() {
        let img = GrayImage::new(64, 48);
        let p = build_pyramid(&img, 12, 8);
        let dims: Vec<(usize, usize)> = p.levels.iter().map(|l| (l.w, l.h)).collect();
        assert_eq!(dims, vec![(64, 48), (32, 24), (16, 12)]);
    }
}
