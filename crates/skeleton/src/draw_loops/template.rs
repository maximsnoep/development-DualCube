//! The loop structure of the labeled skeleton's cube surface, from the labels alone.
//!
//! Every node that is not a leaf is a cube and every edge a cuboid, capped at a leaf and
//! otherwise between two cube faces. Their free faces are
//! the quads of a closed quad mesh; corners are identified only across the connected faces, so
//! coincident solids stay distinct. The dual loops of that quad mesh are the template: every
//! quad holds one crossing. A loop of axis a crosses the quads whose normal is not a, turning
//! counterclockwise around +a seen from outside.

use super::*;
use crate::label::Port;
use crate::{LabeledSkeleton, SkeletonEdgeID};
use dualcube_dual::insertion;
use petgraph::unionfind::UnionFind;

pub(super) struct Template {
    /// One placeholder per template loop, with its axis.
    pub loops: DualLoops,
    /// The loop around every edge cuboid: its patch boundary.
    pub boundaries: HashMap<SkeletonEdgeID, LoopID>,
    pub target: insertion::Template,
    /// The free faces, in the order of the template's crossings.
    pub quads: Vec<Quad>,
    /// The outward normal of every quad, in the order given to the loop map.
    pub quad_normals: Vec<Vector3D>,
}

/// A free face of the template's cube surface.
pub(super) struct Quad {
    /// Its solid, in the order of [`crate::RealizedSkeleton::template_solids`].
    pub solid: usize,
    /// Corners, counterclockwise seen from outside: glued across solids, and as corner codes
    /// of the solid.
    pub corners: [usize; 4],
    pub codes: [usize; 4],
    /// The loop joining sides 0 and 2, running from 0 to 2, and the loop joining sides 1 and 3,
    /// with whether it runs from 1 to 3.
    pub loops: [(LoopID, bool); 2],
}

impl Quad {
    /// The side (from corner `i` to corner `i + 1`) through which loop `k` leaves.
    pub fn exit(&self, k: usize) -> usize {
        match (k, self.loops[1].1) {
            (0, _) => 2,
            (_, true) => 3,
            (_, false) => 1,
        }
    }

    pub fn side(&self, i: usize) -> (usize, usize) {
        (self.corners[i], self.corners[(i + 1) % 4])
    }
}

/// Corner `c` of a solid has bit `d` set on its positive side of axis `d`.
const CORNERS: usize = 8;

impl Template {
    pub fn build(labeled: &LabeledSkeleton) -> SkeletonResult<Self> {
        let skeleton = &labeled.skeleton;
        // Solids: cubes of the nodes that are not leaves, then edge cuboids. A leaf's cube
        // would only lengthen its cuboid, so the cuboid's end is the leaf's cap.
        let degree = skeleton.degrees();
        let nodes: HashMap<_, _> = skeleton
            .nodes
            .keys()
            .filter(|&n| degree[n] != 1)
            .zip(0..)
            .collect();
        let edges: HashMap<_, _> = skeleton.edges.keys().zip(nodes.len()..).collect();
        let solids = nodes.len() + edges.len();
        let mut axis_of = vec![None; solids];
        for (&edge, &solid) in &edges {
            axis_of[solid] = Some((edge, Port::from_label(labeled.labels[edge]).axis()));
        }
        let mut corners = UnionFind::new(CORNERS * solids);
        let mut hidden = HashSet::new();
        // Glue the face of `a` on `port` to the opposite face of `b`.
        let mut glue = |a: usize, b: usize, port: Port| {
            let (axis, positive) = (port.axis(), port.is_positive());
            hidden.insert((a, axis, positive));
            hidden.insert((b, axis, !positive));
            for c in (0..CORNERS).filter(|c| (c & (1 << axis) != 0) == positive) {
                corners.union(CORNERS * a + c, CORNERS * b + (c ^ (1 << axis)));
            }
        };
        for (id, edge) in &skeleton.edges {
            let port = Port::from_label(labeled.labels[id]);
            if let Some(&node) = nodes.get(&edge.nodes[0]) {
                glue(node, edges[&id], port);
            }
            if let Some(&node) = nodes.get(&edge.nodes[1]) {
                glue(edges[&id], node, port);
            }
        }
        // Free faces as quads, counterclockwise seen from outside, with the axis of the loop
        // joining sides 0 and 2, of the loop joining sides 1 and 3, and whether that one runs
        // from side 1 to side 3 (the first always runs from side 0 to side 2).
        let mut quads = Vec::new();
        let mut codes_of = Vec::new();
        let mut sections = Vec::new();
        for solid in 0..solids {
            for axis in 0..3 {
                for positive in [false, true] {
                    if hidden.contains(&(solid, axis, positive)) {
                        continue;
                    }
                    let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
                    let base = usize::from(positive) << axis;
                    let mut codes = [base, base | 1 << u, base | 1 << u | 1 << v, base | 1 << v];
                    if !positive {
                        codes.reverse();
                    }
                    codes_of.push(codes);
                    quads.push(codes.map(|c| corners.find(CORNERS * solid + c)));
                    sections.push((solid, [u, v], positive));
                }
            }
        }
        let mut numbering = HashMap::new();
        for quad in &mut quads {
            for v in quad {
                let next = numbering.len();
                *v = *numbering.entry(*v).or_insert(next);
            }
        }
        // Loops: sections continue through opposite sides of a quad, into the next quad.
        let mut side = HashMap::new();
        for (q, quad) in quads.iter().enumerate() {
            for i in 0..4 {
                side.insert((quad[i], quad[(i + 1) % 4]), (q, i));
            }
        }
        let mut sections_of = UnionFind::new(2 * quads.len());
        for (&(a, b), &(q, i)) in &side {
            let &(p, j) = side.get(&(b, a)).ok_or("Cube surface is not closed")?;
            sections_of.union(2 * q + i % 2, 2 * p + j % 2);
        }
        let mut loops = DualLoops::with_key();
        let mut boundaries = HashMap::new();
        let mut ids = HashMap::new();
        let mut chords = Vec::new();
        for (q, &(solid, axes, positive)) in sections.iter().enumerate() {
            let mut pair = [(LoopID::default(), true); 2];
            for k in 0..2 {
                let id = *ids.entry(sections_of.find(2 * q + k)).or_insert_with(|| {
                    loops.insert(Loop {
                        surface: vec![],
                        edges: vec![],
                        offsets: vec![],
                        direction: DIRECTIONS[axes[k]],
                    })
                });
                pair[k] = (id, k == 0 || positive);
                // The section of an edge cuboid across its own axis is its patch boundary.
                if let Some((edge, axis)) = axis_of[solid]
                    && axis == axes[k]
                {
                    boundaries.insert(edge, id);
                }
            }
            chords.push(pair);
        }
        let map = LoopMap::from_quads(numbering.len(), &quads, &chords)
            .map_err(|_| "Cube surface is not a manifold quad mesh")?;
        let quad_normals = sections
            .iter()
            .map(|&(_, [u, v], positive)| {
                let mut n = Vector3D::zeros();
                n[3 - u - v] = if positive { 1. } else { -1. };
                n
            })
            .collect();
        let quads = (0..quads.len())
            .map(|q| Quad {
                solid: sections[q].0,
                corners: quads[q],
                codes: codes_of[q],
                loops: chords[q],
            })
            .collect();
        Ok(Self {
            loops,
            boundaries,
            target: insertion::Template::new(map),
            quads,
            quad_normals,
        })
    }
}
