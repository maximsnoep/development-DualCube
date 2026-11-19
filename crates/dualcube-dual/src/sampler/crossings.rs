//! Where loops cross input edges, by occurrence, for loops that include explicit curves.
//!
//! A curve may cross an edge, and visit a triangle, several times. Every visit of a triangle runs
//! between two crossings on its sides, so for the gap graph it is a chord between them, exactly
//! like a legacy loop's single visit.

use super::*;

#[derive(Clone)]
pub(crate) struct Crossings {
    // Per half-edge: the loops crossing it from root to tip, and their positions along it.
    on_edge: HashMap<EdgeID, (Vec<LoopID>, Vec<f64>)>,
    // Per face: every visit of a loop.
    visits: HashMap<FaceID, Vec<FaceChord>>,
    // Whether every loop could be read (curves that do not alternate between faces cannot).
    pub complete: bool,
}

/// The half-edges through which a loop leaves its triangles, in order, with the position along
/// each (from root to tip).
pub(crate) fn exits(mesh: &Mesh<INPUT>, l: &Loop) -> Option<Vec<(EdgeID, f64)>> {
    if l.surface.is_empty() {
        let len = l.edges.len();
        return Some(
            (0..len)
                .filter(|&i| mesh.twin(l.edges[i]) == l.edges[(i + 1) % len])
                .map(|i| (l.edges[i], l.offsets.get(i).copied().unwrap_or(0.5)))
                .collect(),
        );
    }
    let arcs = &l.surface;
    let mut result = vec![];
    for (i, arc) in arcs.iter().enumerate() {
        if arcs[(i + 1) % arcs.len()].face == arc.face {
            continue;
        }
        let weights = [1. - arc.to[0] - arc.to[1], arc.to[0], arc.to[1]];
        // Side `k` runs from corner `k` to corner `k + 1`.
        let k = (0..3).find(|&k| weights[(k + 2) % 3].abs() < 1e-9)?;
        result.push((mesh.edges(arc.face).nth(k)?, weights[(k + 1) % 3]));
    }
    Some(result)
}

fn side_in_face(mesh: &Mesh<INPUT>, edge: EdgeID) -> usize {
    mesh.edges(mesh.face(edge)).position(|e| e == edge).unwrap()
}

impl Crossings {
    pub fn new(mesh: &Mesh<INPUT>, loops: &SlotMap<LoopID, Loop>) -> Self {
        let mut complete = true;
        // Every occurrence: its loop, and the canonical half-edge it crosses with its position.
        let mut occurrences: Vec<Vec<(EdgeID, f64)>> = vec![];
        let mut ids = vec![];
        for (id, l) in loops {
            match exits(mesh, l) {
                Some(exits) if !exits.is_empty() => {
                    ids.push(id);
                    occurrences.push(exits);
                }
                _ => complete = false,
            }
        }
        let canonical = |edge: EdgeID, t: f64| {
            let twin = mesh.twin(edge);
            if edge < twin {
                (edge, t)
            } else {
                (twin, 1. - t)
            }
        };
        let mut along: HashMap<EdgeID, Vec<(OrderedFloat<f64>, usize, usize)>> = HashMap::new();
        for (l, exits) in occurrences.iter().enumerate() {
            for (k, &(edge, t)) in exits.iter().enumerate() {
                let (edge, t) = canonical(edge, t);
                along.entry(edge).or_default().push((OrderedFloat(t), l, k));
            }
        }
        // The rank of every occurrence along its canonical half-edge, of how many.
        let mut rank: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
        let mut on_edge = HashMap::new();
        for (edge, mut list) in along {
            list.sort();
            let n = list.len();
            for (i, &(_, l, k)) in list.iter().enumerate() {
                rank.insert((l, k), (i, n));
            }
            let forward = (
                list.iter().map(|&(_, l, _)| ids[l]).collect_vec(),
                list.iter().map(|&(t, _, _)| *t).collect_vec(),
            );
            let backward = (
                forward.0.iter().rev().copied().collect_vec(),
                forward.1.iter().rev().map(|t| 1. - t).collect_vec(),
            );
            on_edge.insert(edge, forward);
            on_edge.insert(mesh.twin(edge), backward);
        }
        let rank_on = |l: usize, k: usize, half: EdgeID| {
            let (i, n) = rank[&(l, k)];
            let edge = occurrences[l][k].0;
            let twin = mesh.twin(edge);
            if half == if edge < twin { edge } else { twin } {
                i
            } else {
                n - 1 - i
            }
        };
        let mut visits: HashMap<FaceID, Vec<FaceChord>> = HashMap::new();
        for (l, exits) in occurrences.iter().enumerate() {
            let m = exits.len();
            for k in 0..m {
                // Entering through the twin of exit `k`, leaving through exit `k + 1`.
                let entry = mesh.twin(exits[k].0);
                let exit = exits[(k + 1) % m].0;
                if mesh.face(entry) != mesh.face(exit) {
                    complete = false;
                    continue;
                }
                let ends = [
                    boundary_coord(side_in_face(mesh, entry), 2 * rank_on(l, k, entry) + 1),
                    boundary_coord(
                        side_in_face(mesh, exit),
                        2 * rank_on(l, (k + 1) % m, exit) + 1,
                    ),
                ];
                visits.entry(mesh.face(exit)).or_default().push(FaceChord {
                    loop_id: ids[l],
                    direction: loops[ids[l]].direction,
                    ends,
                });
            }
        }
        Self {
            on_edge,
            visits,
            complete,
        }
    }

    pub fn loops_on(&self, edge: EdgeID) -> &[LoopID] {
        self.on_edge
            .get(&edge)
            .map_or(&[], |(ids, _)| ids.as_slice())
    }

    pub fn offset(&self, edge: EdgeID, rank: usize) -> Option<f64> {
        self.on_edge.get(&edge)?.1.get(rank).copied()
    }

    pub(super) fn visits(&self, face: FaceID) -> &[FaceChord] {
        self.visits.get(&face).map_or(&[], Vec::as_slice)
    }
}
