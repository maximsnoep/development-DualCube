//! The sides of the loops (see `Solution::loop_frames`): the negative side of a loop lies toward the negative end of
//! its axis.

use dualcube::prelude::*;

mod common;
use common::blub;

#[test]
fn the_negative_side_of_a_loop_points_down_its_axis() {
    let mut solution = Solution::new(blub());
    solution.initialize();
    assert!(!solution.loops.is_empty());
    for (loop_id, lewp) in &solution.loops {
        let axis = match lewp.direction {
            Direction::X => Vector3D::x(),
            Direction::Y => Vector3D::y(),
            Direction::Z => Vector3D::z(),
        };
        let frames = solution.loop_frames(loop_id);
        // Along the loop, weighted by the length of its segments.
        let along: f64 = frames
            .iter()
            .zip(frames.iter().cycle().skip(1))
            .map(|(a, b)| (b.position - a.position).norm() * a.negative.try_normalize(1e-12).map_or(0., |v| v.dot(&axis)))
            .sum();
        assert!(along < 0., "loop {loop_id:?} ({:?}): {along}", lewp.direction);
    }
}
