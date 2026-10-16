//! Dual structures read from a loop map agree with those derived from the loops' geometry.

mod common;

use common::blub;
use dualcube::prelude::*;
use dualcube_dual::loop_map::LoopMap;

// Mesh vertices and segment labels of a region.
type Region = (Vec<slotmap::DefaultKey>, Vec<(LoopID, bool)>);

// Per region: its mesh vertices and the labels of its segments, both sorted.
fn regions(dual: &Dual) -> Vec<Region> {
    let mut result: Vec<_> = dual
        .loop_structure
        .face_ids()
        .into_iter()
        .map(|region| {
            let mut verts: Vec<_> = dual
                .region_to_verts(region)
                .into_iter()
                .map(|v| v.raw())
                .collect();
            verts.sort();
            let mut labels: Vec<_> = dual
                .loop_structure
                .edges(region)
                .map(|s| {
                    (
                        dual.segment_to_loop(s),
                        dual.segment_to_orientation(s) == Sign::Positive,
                    )
                })
                .collect();
            labels.sort();
            (verts, labels)
        })
        .collect();
    result.sort();
    result
}

fn pairs(dual: &Dual) -> Vec<[LoopID; 2]> {
    let mut result: Vec<_> = dual
        .loop_structure
        .vert_ids()
        .into_iter()
        .map(|x| {
            let mut pair = dual.intersection(x).loops;
            pair.sort();
            pair
        })
        .collect();
    result.sort();
    result
}

#[test]
fn map_dual_matches_geometric_dual() {
    let mut tested = 0;
    for _ in 0..10 {
        let mut solution = Solution::new(blub());
        solution.initialize();
        let mesh = solution.mesh_ref.clone();
        let Ok(geometric) = Dual::from(mesh.clone(), &solution.loops) else {
            continue;
        };
        let map = LoopMap::from_loops(&mesh, &solution.loops).unwrap();
        let exact = Dual::from_map(mesh.clone(), &map, &solution.loops).expect("map dual");
        let size = |d: &Dual| {
            let s = &d.loop_structure;
            (s.vert_ids().len(), s.edge_ids().len(), s.face_ids().len())
        };
        assert_eq!(size(&exact), size(&geometric));
        assert_eq!(pairs(&exact), pairs(&geometric));
        assert_eq!(regions(&exact), regions(&geometric));

        let refined = exact.refined_mesh().expect("refined mesh");
        assert_eq!(refined.face_regions.len(), refined.mesh.nr_faces());
        assert!(
            mesh.vert_ids()
                .iter()
                .all(|v| refined.vertex_map.contains_key(v))
        );

        // Reconstruction reuses the installed dual structure.
        solution.dual = Ok(exact);
        assert!(solution.dual_is_current());
        solution
            .reconstruct_solution(false)
            .expect("reconstruction");
        assert!(solution.layout.is_some());
        tested += 1;
    }
    assert!(tested > 0);
}
