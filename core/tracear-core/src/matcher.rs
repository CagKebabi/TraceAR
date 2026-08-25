//! Brute-force Hamming matcher with a scale-aware ratio test.
//!
//! Marker descriptors contain the same physical feature at several compiled
//! scales. A plain Lowe ratio test would let a feature's own adjacent-scale
//! twin veto the match, so the "second best" only counts if it sits at a
//! spatially distinct marker location.

use crate::brief::Descriptor;

#[inline]
pub fn hamming(a: &Descriptor, b: &Descriptor) -> u32 {
    (a[0] ^ b[0]).count_ones()
        + (a[1] ^ b[1]).count_ones()
        + (a[2] ^ b[2]).count_ones()
        + (a[3] ^ b[3]).count_ones()
}

#[derive(Clone, Copy, Debug)]
pub struct Match {
    pub query: u32,
    pub train: u32,
    pub distance: u32,
}

pub fn match_descriptors(
    query: &[Descriptor],
    train: &[Descriptor],
    train_pos: &[(f32, f32)],
    max_dist: u32,
    ratio: f32,
    min_second_dist_px: f32,
) -> Vec<Match> {
    assert_eq!(train.len(), train_pos.len());
    let mut out = Vec::new();
    let r2 = min_second_dist_px * min_second_dist_px;
    for (qi, qd) in query.iter().enumerate() {
        let mut best = u32::MAX;
        let mut best_i = 0usize;
        for (ti, td) in train.iter().enumerate() {
            let d = hamming(qd, td);
            if d < best {
                best = d;
                best_i = ti;
            }
        }
        if best > max_dist {
            continue;
        }
        let (bx, by) = train_pos[best_i];
        let mut second = u32::MAX;
        for (ti, td) in train.iter().enumerate() {
            let (tx, ty) = train_pos[ti];
            let dx = tx - bx;
            let dy = ty - by;
            if dx * dx + dy * dy <= r2 {
                continue; // same physical neighborhood (any compiled scale)
            }
            let d = hamming(qd, td);
            if d < second {
                second = d;
            }
        }
        if second != u32::MAX && (best as f32) >= ratio * (second as f32) {
            continue;
        }
        out.push(Match { query: qi as u32, train: best_i as u32, distance: best });
    }
    // Keep only the best query per train feature.
    out.sort_by_key(|m| m.distance);
    let mut used = vec![false; train.len()];
    let mut deduped = Vec::with_capacity(out.len());
    for m in out {
        if !used[m.train as usize] {
            used[m.train as usize] = true;
            deduped.push(m);
        }
    }
    deduped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hamming_counts_bits() {
        let a: Descriptor = [0, 0, 0, 0];
        let b: Descriptor = [0b1011, 0, 1 << 63, 0];
        assert_eq!(hamming(&a, &b), 4);
        assert_eq!(hamming(&b, &b), 0);
    }

    #[test]
    fn matches_identical_and_rejects_far() {
        let d0: Descriptor = [0xAAAA, 0x5555, 0xF0F0, 0x0F0F];
        let d1: Descriptor = [!0xAAAA, !0x5555, !0xF0F0, !0x0F0F];
        let train = vec![d0, d1];
        let pos = vec![(0.0, 0.0), (100.0, 100.0)];
        let query = vec![d0];
        let m = match_descriptors(&query, &train, &pos, 64, 0.8, 8.0);
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].train, 0);
        assert_eq!(m[0].distance, 0);
    }

    #[test]
    fn adjacent_scale_twin_does_not_veto() {
        let d0: Descriptor = [0xDEAD_BEEF, 1, 2, 3];
        let mut twin = d0;
        twin[1] ^= 0b111; // distance 3 — like the same corner at the next scale
        let far: Descriptor = [!0u64, !1, !2, !3];
        let train = vec![d0, twin, far];
        // twin sits at (1,1) in marker space — within min_second_dist of best
        let pos = vec![(50.0, 50.0), (51.0, 51.0), (200.0, 10.0)];
        let m = match_descriptors(&[d0], &train, &pos, 64, 0.8, 8.0);
        assert_eq!(m.len(), 1, "spatially-near twin must not act as ratio-test second");
        assert_eq!(m[0].train, 0);
    }
}
