use super::*;
use slotmap::SlotMap;

// The levels of a level graph on `count` zones with the given edges, zones listed in topological order.
fn levels(count: usize, edges: &[(usize, usize)]) -> (Vec<ZoneID>, HashMap<ZoneID, usize>) {
    let mut zones = SlotMap::<ZoneID, ()>::with_key();
    let mut loops = SlotMap::<LoopID, ()>::with_key();
    let ids: Vec<_> = (0..count).map(|_| zones.insert(())).collect();
    let loop_id = loops.insert(());
    let edges = edges
        .iter()
        .map(|&(a, b)| (ids[a], ids[b], loop_id))
        .collect();
    let graph = grapff::fixed::FixedGraph::from(ids.clone(), edges);
    (ids, zone_levels(&graph))
}

fn assert_leveled(count: usize, edges: &[(usize, usize)]) {
    let (ids, levels) = levels(count, edges);
    assert_eq!(levels.len(), count, "every zone has a level");
    for &(a, b) in edges {
        assert!(levels[&ids[b]] > levels[&ids[a]], "edge {a} -> {b} goes up");
    }
    assert_eq!(levels.values().min(), Some(&0));
}

#[test]
fn zones_reached_only_by_zigzags_get_a_level() {
    // Two minima (0, 2) and two maxima (1, 3) in a row: one sweep each way levels 0, 1, 2, but not 3.
    assert_leveled(5, &[(0, 1), (2, 1), (2, 3), (4, 3)]);
}

#[test]
fn every_zone_of_a_knot_gets_a_level() {
    // The Z level graph of knot_cgal (four minima and four maxima, around a cycle), zones in topological order, on
    // which one sweep each way left four zones without a level.
    let edges = [
        (0, 1),
        (1, 5),
        (1, 12),
        (2, 11),
        (3, 4),
        (4, 5),
        (4, 9),
        (5, 6),
        (7, 8),
        (8, 9),
        (8, 14),
        (9, 10),
        (11, 12),
        (11, 14),
        (12, 13),
        (14, 15),
    ];
    assert_leveled(16, &edges);
}

#[test]
fn a_chain_is_leveled_in_order() {
    let (ids, levels) = levels(3, &[(0, 1), (1, 2)]);
    assert_eq!(
        ids.iter().map(|id| levels[id]).collect::<Vec<_>>(),
        [0, 1, 2]
    );
}
