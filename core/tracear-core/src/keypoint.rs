//! Keypoint type and spatially-uniform selection.
//!
//! Homography estimation conditions much better with features spread over the
//! whole target than with the same count clustered in one texture-rich area,
//! so selection is grid-bucketed top-K rather than a global score sort.

#[derive(Clone, Copy, Debug)]
pub struct Keypoint {
    /// Coordinates in the pixel space of the image the keypoint was detected in.
    pub x: f32,
    pub y: f32,
    pub score: f32,
    /// Orientation in radians (0 until computed).
    pub angle: f32,
    /// Pyramid level / scale index the keypoint came from.
    pub level: u8,
}

pub fn select_uniform(
    mut kps: Vec<Keypoint>,
    w: usize,
    h: usize,
    cell: usize,
    per_cell: usize,
    max_total: usize,
) -> Vec<Keypoint> {
    assert!(cell > 0);
    let cols = (w + cell - 1) / cell;
    let rows = (h + cell - 1) / cell;
    let mut buckets: Vec<Vec<Keypoint>> = vec![Vec::new(); cols * rows];
    for kp in kps.drain(..) {
        let cx = (kp.x as usize / cell).min(cols - 1);
        let cy = (kp.y as usize / cell).min(rows - 1);
        buckets[cy * cols + cx].push(kp);
    }
    let mut out = Vec::new();
    for b in buckets.iter_mut() {
        b.sort_by(|a, b| b.score.total_cmp(&a.score));
        out.extend(b.iter().take(per_cell).copied());
    }
    if out.len() > max_total {
        out.sort_by(|a, b| b.score.total_cmp(&a.score));
        out.truncate(max_total);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kp(x: f32, y: f32, score: f32) -> Keypoint {
        Keypoint { x, y, score, angle: 0.0, level: 0 }
    }

    #[test]
    fn keeps_top_per_cell() {
        // three keypoints in the same 16px cell, one elsewhere
        let kps = vec![kp(1.0, 1.0, 5.0), kp(2.0, 2.0, 9.0), kp(3.0, 3.0, 7.0), kp(40.0, 40.0, 1.0)];
        let sel = select_uniform(kps, 64, 64, 16, 2, 100);
        assert_eq!(sel.len(), 3);
        // the score-5 keypoint in the crowded cell was dropped
        assert!(sel.iter().all(|k| k.score != 5.0));
        // the lone low-score keypoint survives (spatial spread wins)
        assert!(sel.iter().any(|k| k.score == 1.0));
    }

    #[test]
    fn respects_max_total() {
        let kps: Vec<Keypoint> = (0..100).map(|i| kp((i * 20) as f32, 0.0, i as f32)).collect();
        let sel = select_uniform(kps, 2000, 32, 16, 3, 10);
        assert_eq!(sel.len(), 10);
        // highest scores kept
        assert!(sel.iter().all(|k| k.score >= 90.0));
    }
}
