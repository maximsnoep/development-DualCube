//! Jobs for manual loop edits (adding and removing dual loops).

use super::{Job, JobResult};
use crate::resources::Configuration;
use dualcube::prelude::*;

impl Job {
    /// Add a loop (into the given gaps, for valid loops). The loop is only added if the result is a valid loop
    /// structure, unless `force` is set (e.g., to construct an initial loop structure by hand).
    pub fn add_loop(
        solution: Solution,
        loop_to_add: Loop,
        gaps: Option<Vec<(usize, usize)>>,
        force: bool,
        configuration: Configuration,
    ) -> Self {
        Self::new("adding loop", move || {
            let mut candidate = solution.clone();
            match gaps.clone() {
                Some(gaps) => candidate.add_loop_in_gaps(loop_to_add.clone(), gaps),
                None => candidate.add_loop(loop_to_add.clone()),
            };
            match Dual::from(candidate.mesh_ref.clone(), &candidate.loops) {
                Ok(_) => {}
                Err(err) if force => {
                    warn!("Adding loop although the loop structure is not valid (forced): {err}");
                }
                Err(err) => {
                    warn!(
                        "Loop not added: the loop structure would not be valid ({err}). Hold Shift to add it anyway."
                    );
                    return None;
                }
            }
            Some(JobResult::AddedLoop {
                solution: candidate,
                configuration: configuration.clone(),
            })
        })
    }

    pub fn remove_loop(solution: Solution, loop_id: LoopID, configuration: Configuration) -> Self {
        Self::new("removing loop", move || {
            let mut candidate = solution.clone();
            candidate.del_loop(loop_id);
            Some(JobResult::RemovedLoop {
                solution: candidate,
                configuration: configuration.clone(),
            })
        })
    }
}
