//! Connectivity surgery on patch boundaries.
//!
//! Patches start as the dual cells of the mesh vertices, so their boundaries form the dual
//! graph: one arc per mesh edge and a junction in every triangle. Surgery only deletes arcs,
//! one shared boundary at a time, guided by the collapse costs of Au et al.:
//! - An arc between distinct patches merges them. Two patches glued along an arc are
//!   contractible in the volume when both are, so this needs no check.
//! - An arc with one patch on both sides adds a loop crossing it once. It is dissolved only
//!   when that loop bounds in the volume (Z/2 homology modulo the handle subspace K).
//!
//! A junction of degree two joins its two arcs. Surgery ends when no junctions remain; the
//! closed arcs then become the skeleton edges.

use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap, VecDeque};

use mehsh::prelude::{HasNeighbors, HasPosition, Mesh, Vector3D, VertKey};
use nalgebra::{Matrix4, Vector4};
use slotmap::SlotMap;
use tracing::{info, warn};

mod f2_rref;
mod flip;
mod handle_subspace;
mod homology;
mod output;
mod tet_boundary;
mod tetrahedralize;

use super::deflate::CONTRACTION;
use crate::surface::canonical;
use crate::{BoundaryCrossing, BoundaryCrossingID, BoundarySurface, Skeleton, SkeletonNodeID};
use dualcube_types::prelude::*;
use homology::{Class, Homology};

/// Weights of collapse costs; topology checks are unchanged.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub shape_weight: f64,
    pub sampling_weight: f64,
    /// Subtracted once per handle cycle crossed by the dissolved arc. Not from Au et al.
    pub handle_bonus: f64,
    /// Flip sequences tried when surgery is stuck, before it fails.
    pub flip_budget: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            shape_weight: 1.0,
            sampling_weight: 0.1,
            handle_bonus: 1.0e6,
            flip_budget: 100_000,
        }
    }
}

impl Options {
    pub fn validate(self) -> crate::SkeletonResult<()> {
        if [self.shape_weight, self.sampling_weight, self.handle_bonus]
            .iter()
            .any(|x| !x.is_finite() || *x < 0.0)
        {
            return Err("Surgery weights must be finite and nonnegative".into());
        }
        Ok(())
    }
}

/// Patches and arcs left when surgery cannot continue.
#[derive(Clone, Debug)]
pub struct DiagnosticGraph {
    pub positions: slotmap::SecondaryMap<SkeletonNodeID, Vector3D>,
    pub edges: Vec<[SkeletonNodeID; 2]>,
}

/// Failed surgery, with the available partial state.
#[derive(Debug)]
pub struct SurgeryFailure {
    pub reason: String,
    /// Inspection only: not a validated surface partition.
    pub graph: DiagnosticGraph,
    /// Triangles holding the remaining junctions.
    pub remaining_face_positions: Vec<[Vector3D; 3]>,
}

impl std::fmt::Display for SurgeryFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.reason)
    }
}

impl std::error::Error for SurgeryFailure {}

/// Reduce the deflated mesh to a skeleton whose patches lie on the original surface.
pub fn run(
    contracted: &Mesh<CONTRACTION>,
    original: &Mesh<INPUT>,
) -> crate::SkeletonResult<Skeleton> {
    run_with_options(contracted, original, Options::default())
}

pub fn run_with_options(
    contracted: &Mesh<CONTRACTION>,
    original: &Mesh<INPUT>,
    options: Options,
) -> crate::SkeletonResult<Skeleton> {
    options.validate()?;
    let genus = crate::validate::input_genus(original)?;
    let converted = Mesh::<CONTRACTION>::convert(original);
    if converted.vert_ids() != contracted.vert_ids()
        || converted.edge_ids() != contracted.edge_ids()
        || converted.face_ids() != contracted.face_ids()
        || converted.edge_ids().into_iter().any(|e| {
            converted.root(e) != contracted.root(e)
                || converted.twin(e) != contracted.twin(e)
                || converted.next(e) != contracted.next(e)
                || converted.face(e) != contracted.face(e)
        })
    {
        return Err("Deflation must preserve mesh topology and vertex IDs".into());
    }
    if contracted
        .vert_ids()
        .iter()
        .any(|&v| !contracted.position(v).iter().all(|x| x.is_finite()))
    {
        return Err("Deflated positions must be finite".into());
    }
    let mut surgery = Surgery::new(contracted, original, genus, options)?;
    surgery.run()?;
    let mut skeleton = surgery.skeleton();
    output::untangle(original, &mut skeleton);
    output::simplify(original, &mut skeleton);
    let result = skeleton.validate(original).and_then(|()| {
        if skeleton.edges.len() + 1 == skeleton.nodes.len() + genus {
            Ok(())
        } else {
            Err("Extracted graph does not preserve input genus".into())
        }
    });
    if let Err(error) = result {
        return Err(Box::new(SurgeryFailure {
            reason: error.to_string(),
            graph: surgery.diagnostic().0,
            remaining_face_positions: Vec::new(),
        }));
    }
    Ok(skeleton)
}

slotmap::new_key_type! {
    struct ArcID;
    struct JunctionID;
}

/// A boundary curve between junctions, or a closed one.
#[derive(Clone, Debug)]
struct Arc {
    /// In traversal order; see [`crate::BoundaryCrossing`].
    crossings: VecDeque<BoundaryCrossingID>,
    /// Start and end junction; none when closed.
    ends: Option<[JunctionID; 2]>,
    /// Left and right patch.
    sides: [usize; 2],
    /// Class of a loop crossing it once, modulo K; set once one patch is on both sides.
    class: Option<Class>,
    /// Handle cycles it crosses.
    handles: Class,
}

/// A point inside a triangle where three arc ends meet.
#[derive(Clone, Debug)]
struct Junction {
    face: FaceID,
    /// Arc ends (0 = start, 1 = end); an arc ending here twice appears twice.
    spokes: Vec<(ArcID, usize)>,
}

#[derive(Clone, Debug)]
struct Patch {
    /// Normalized contracted position of the surviving vertex.
    position: Vector3D,
    quadric: Matrix4<f64>,
    vertices: Vec<usize>,
    arcs: BTreeSet<ArcID>,
}

/// Dissolving `arc` merges patch `removed` into `kept`, which keeps its position.
#[derive(Debug)]
struct Candidate {
    arc: ArcID,
    removed: usize,
    kept: usize,
    cost: f64,
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed for a min-heap.
        other
            .cost
            .total_cmp(&self.cost)
            .then_with(|| other.arc.cmp(&self.arc))
            .then_with(|| other.removed.cmp(&self.removed))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Candidate {}

struct Surgery<'a> {
    mesh: &'a Mesh<INPUT>,
    options: Options,
    center: Vector3D,
    scale: f64,
    index: HashMap<VertID, usize>,
    owner: Vec<usize>,
    /// Class of a path from a fixed vertex of the owner patch, modulo K. Paths within a
    /// patch differ by loops of that patch, which lie in K.
    potential: Vec<Class>,
    patches: Vec<Option<Patch>>,
    /// Crossings of all arcs. Until the first flip each mesh edge has at most one.
    surface: BoundarySurface,
    arcs: SlotMap<ArcID, Arc>,
    junctions: SlotMap<JunctionID, Junction>,
    homology: Homology,
    heap: BinaryHeap<Candidate>,
    /// Arcs with one patch on both sides, to dissolve if their loop bounds.
    pending: Vec<ArcID>,
}

impl<'a> Surgery<'a> {
    fn new(
        contracted: &Mesh<CONTRACTION>,
        mesh: &'a Mesh<INPUT>,
        genus: usize,
        options: Options,
    ) -> crate::SkeletonResult<Self> {
        let vertices = mesh.vert_ids();
        let index: HashMap<_, _> = vertices.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        let contracted_key = |v: VertID| VertKey::<CONTRACTION>::new(v.raw());
        let position = |v: VertID| contracted.position(contracted_key(v));
        // Normalize so the cost terms balance.
        let (mut min, mut max) = (Vector3D::repeat(f64::MAX), Vector3D::repeat(f64::MIN));
        for &v in &vertices {
            min = min.inf(&position(v));
            max = max.sup(&position(v));
        }
        let center = (min + max) * 0.5;
        let extent = (max - min).max();
        let scale = if extent > 1e-8 { 1.0 / extent } else { 1.0 };
        let mut patches: Vec<_> = vertices
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                Some(Patch {
                    position: (position(v) - center) * scale,
                    quadric: Matrix4::zeros(),
                    vertices: vec![i],
                    arcs: BTreeSet::new(),
                })
            })
            .collect();
        for &u in &vertices {
            for w in contracted.neighbors(contracted_key(u)) {
                let w = VertKey::<INPUT>::new(w.raw());
                let (p, q) = (
                    patches[index[&u]].as_ref().unwrap().position,
                    patches[index[&w]].as_ref().unwrap().position,
                );
                let length = (q - p).norm();
                if index[&u] < index[&w] && length >= 1e-12 {
                    let quadric = edge_quadric(p, (q - p) / length);
                    patches[index[&u]].as_mut().unwrap().quadric += quadric;
                    patches[index[&w]].as_mut().unwrap().quadric += quadric;
                }
            }
        }
        let homology = Homology::new(mesh, &index, genus)?;
        let mut surgery = Self {
            mesh,
            options,
            center,
            scale,
            owner: (0..vertices.len()).collect(),
            potential: vec![Class::default(); vertices.len()],
            index,
            patches,
            surface: BoundarySurface::default(),
            arcs: SlotMap::with_key(),
            junctions: SlotMap::with_key(),
            homology,
            heap: BinaryHeap::new(),
            pending: Vec::new(),
        };
        let junctions: HashMap<_, _> = mesh
            .face_ids()
            .into_iter()
            .map(|face| {
                let id = surgery.junctions.insert(Junction {
                    face,
                    spokes: Vec::new(),
                });
                (face, id)
            })
            .collect();
        for e in mesh.edge_ids() {
            if e > mesh.twin(e) {
                continue;
            }
            let crossing = surgery
                .surface
                .crossings
                .insert(BoundaryCrossing { exit: e });
            surgery.surface.edge_order.insert(e, vec![crossing]);
            let ends = [
                junctions[&mesh.face(e)],
                junctions[&mesh.face(mesh.twin(e))],
            ];
            let arc = surgery.arcs.insert(Arc {
                crossings: VecDeque::from([crossing]),
                ends: Some(ends),
                sides: surgery.ends(e),
                class: None,
                handles: surgery.homology.handles(surgery.ends(e)),
            });
            for (end, j) in ends.into_iter().enumerate() {
                surgery.junctions[j].spokes.push((arc, end));
            }
            for patch in surgery.arcs[arc].sides {
                surgery.patch_mut(patch).arcs.insert(arc);
            }
        }
        let arcs: Vec<_> = surgery.arcs.keys().collect();
        for arc in arcs {
            surgery.push(arc);
        }
        Ok(surgery)
    }

    fn run(&mut self) -> Result<(), SurgeryFailure> {
        let start = std::time::Instant::now();
        let (mut merges, mut dissolved, mut flips) = (0, 0, 0);
        loop {
            while let Some(arc) = self.pending.pop() {
                if self.arcs.contains_key(arc) && self.bounds(arc) {
                    self.remove(arc);
                    dissolved += 1;
                }
            }
            if self.junctions.is_empty() {
                break;
            }
            let Some(candidate) = self.heap.pop() else {
                // Every arc at a junction has one patch on both sides and a loop that does
                // not bound. Flip arcs until one bounds.
                if let Some(sequence) = self.search() {
                    flips += sequence.len();
                    for &arc in &sequence {
                        self.flip(arc);
                    }
                    self.pending.push(*sequence.last().unwrap());
                    continue;
                }
                let (graph, remaining) = self.diagnostic();
                warn!(
                    "Connectivity surgery stuck: no flip sequence within the budget frees one of {} junctions ({} arcs, {} patches, after {merges} merges)",
                    self.junctions.len(),
                    self.arcs.len(),
                    self.patches.iter().flatten().count()
                );
                return Err(SurgeryFailure {
                    reason: "Connectivity surgery stopped with remaining junctions".into(),
                    graph,
                    remaining_face_positions: remaining,
                });
            };
            let Some(arc) = self.arcs.get(candidate.arc) else {
                continue;
            };
            let mut sides = arc.sides;
            sides.sort();
            let mut expected = [candidate.removed, candidate.kept];
            expected.sort();
            if arc.ends.is_none() || sides != expected || sides[0] == sides[1] {
                continue;
            }
            // Refresh stale entries instead of dropping them.
            let cost = self.cost(candidate.arc, candidate.removed, candidate.kept);
            if (cost - candidate.cost).abs() > 1e-8 {
                self.heap.push(Candidate { cost, ..candidate });
                continue;
            }
            self.merge(candidate.arc, candidate.removed, candidate.kept);
            merges += 1;
        }
        info!(
            "Connectivity surgery complete in {:.2}s: {merges} merges, {dissolved} self-arcs dissolved, {flips} flips, {} loops",
            start.elapsed().as_secs_f64(),
            self.arcs.len()
        );
        Ok(())
    }

    fn patch(&self, patch: usize) -> &Patch {
        self.patches[patch].as_ref().unwrap()
    }

    fn patch_mut(&mut self, patch: usize) -> &mut Patch {
        self.patches[patch].as_mut().unwrap()
    }

    fn exit(&self, crossing: BoundaryCrossingID) -> EdgeID {
        self.surface.crossings[crossing].exit
    }

    /// Vertex indices of the mesh edge crossed by an exit, left first.
    fn ends(&self, exit: EdgeID) -> [usize; 2] {
        [
            self.index[&self.mesh.toor(exit)],
            self.index[&self.mesh.root(exit)],
        ]
    }

    /// Whether the loop crossing a self-arc once bounds in the volume. The class is taken
    /// from the potentials when the arc first has one patch on both sides; mesh edges are
    /// then crossed at most once.
    fn bounds(&mut self, arc: ArcID) -> bool {
        let [left, right] = self.arcs[arc].sides;
        if left != right {
            return false;
        }
        if self.arcs[arc].class.is_none() {
            let [x, y] = self.ends(self.exit(self.arcs[arc].crossings[0]));
            let class = &(&self.potential[x] ^ &self.potential[y]) ^ self.homology.edge([x, y]);
            self.arcs[arc].class = Some(class);
        }
        self.homology.bounds(self.arcs[arc].class.clone().unwrap())
    }

    /// Collapse cost of Au et al. (shape and sampling) with the handle bias.
    fn cost(&self, arc: ArcID, removed: usize, kept: usize) -> f64 {
        let (u, v) = (self.patch(removed), self.patch(kept));
        let p = Vector4::new(v.position.x, v.position.y, v.position.z, 1.0);
        let shape = p.dot(&((u.quadric + v.quadric) * p));
        let neighbors: BTreeSet<_> = u
            .arcs
            .iter()
            .flat_map(|&a| self.arcs[a].sides)
            .filter(|&n| n != removed && n != kept)
            .collect();
        let sampling: f64 = neighbors
            .into_iter()
            .map(|n| (v.position - self.patch(n).position).norm_squared())
            .sum();
        self.options.shape_weight * shape + self.options.sampling_weight * sampling
            - self.options.handle_bonus * f64::from(self.arcs[arc].handles.count_ones())
    }

    /// Queue an arc: a merge in both directions, or a self-arc test.
    fn push(&mut self, arc: ArcID) {
        let [a, b] = self.arcs[arc].sides;
        if a == b {
            self.pending.push(arc);
        } else if self.arcs[arc].ends.is_some() {
            for (removed, kept) in [(a, b), (b, a)] {
                let cost = self.cost(arc, removed, kept);
                self.heap.push(Candidate {
                    arc,
                    removed,
                    kept,
                    cost,
                });
            }
        }
    }

    fn merge(&mut self, arc: ArcID, removed: usize, kept: usize) {
        let [x, y] = self.ends(self.exit(self.arcs[arc].crossings[0]));
        let (u, v) = (
            self.patches[removed].take().unwrap(),
            self.patches[kept].take().unwrap(),
        );
        // Relabel the smaller patch, keeping potentials consistent across the arc.
        let (mut large, small, id, gone) = if u.vertices.len() > v.vertices.len() {
            (u.clone(), v.clone(), removed, kept)
        } else {
            (v.clone(), u.clone(), kept, removed)
        };
        let shift = &(&self.potential[x] ^ &self.potential[y]) ^ self.homology.edge([x, y]);
        for &w in &small.vertices {
            self.owner[w] = id;
            self.potential[w] ^= &shift;
        }
        large.vertices.extend(&small.vertices);
        for &a in &small.arcs {
            for side in &mut self.arcs[a].sides {
                if *side == gone {
                    *side = id;
                }
            }
        }
        // Arcs the two shared besides this one now have the merged patch on both sides.
        let shared: Vec<_> = small
            .arcs
            .iter()
            .filter(|&&a| a != arc && large.arcs.contains(&a))
            .copied()
            .collect();
        large.arcs.extend(&small.arcs);
        large.position = v.position;
        large.quadric = u.quadric + v.quadric;
        self.patches[id] = Some(large);
        self.remove(arc);
        self.pending.extend(shared);
        let arcs: Vec<_> = self.patch(id).arcs.iter().copied().collect();
        for a in arcs {
            let [left, right] = self.arcs[a].sides;
            if left != right {
                self.push(a);
            }
        }
    }

    /// Delete an arc and settle its junctions.
    fn remove(&mut self, arc: ArcID) {
        let removed = self.arcs.remove(arc).unwrap();
        for patch in removed.sides {
            self.patch_mut(patch).arcs.remove(&arc);
        }
        for c in removed.crossings {
            let exit = self.surface.crossings.remove(c).unwrap().exit;
            let key = canonical(self.mesh, exit);
            let order = self.surface.edge_order.get_mut(&key).unwrap();
            order.retain(|&o| o != c);
            if order.is_empty() {
                self.surface.edge_order.remove(&key);
            }
        }
        if let Some([f, g]) = removed.ends {
            for j in [f, g] {
                self.junctions[j].spokes.retain(|s| s.0 != arc);
            }
            self.settle(f);
            if g != f {
                self.settle(g);
            }
        }
    }

    fn settle(&mut self, junction: JunctionID) {
        match self.junctions[junction].spokes[..] {
            [] => {
                self.junctions.remove(junction);
            }
            // A spur lies inside one patch; its loop always bounds.
            [(spur, _)] => self.pending.push(spur),
            [a, b] => {
                self.junctions.remove(junction);
                if a.0 == b.0 {
                    self.arcs[a.0].ends = None;
                    self.push(a.0);
                } else {
                    self.join(a, b);
                }
            }
            _ => {}
        }
    }

    /// Join two arcs meeting at a removed degree-two junction.
    fn join(&mut self, (a, end_a): (ArcID, usize), (b, end_b): (ArcID, usize)) {
        if end_a != 1 {
            self.reverse(a);
        }
        if end_b != 0 {
            self.reverse(b);
        }
        let (kept, killed, end) = if self.arcs[a].crossings.len() >= self.arcs[b].crossings.len() {
            let crossings = std::mem::take(&mut self.arcs[b].crossings);
            self.arcs[a].crossings.extend(crossings);
            (a, b, 1)
        } else {
            let crossings = std::mem::take(&mut self.arcs[a].crossings);
            for c in crossings.into_iter().rev() {
                self.arcs[b].crossings.push_front(c);
            }
            (b, a, 0)
        };
        let gone = self.arcs.remove(killed).unwrap();
        let other = gone.ends.unwrap()[end];
        self.arcs[kept].ends.as_mut().unwrap()[end] = other;
        for spoke in &mut self.junctions[other].spokes {
            if *spoke == (killed, end) {
                *spoke = (kept, end);
            }
        }
        self.arcs[kept].handles |= &gone.handles;
        if self.arcs[kept].class.is_none() {
            self.arcs[kept].class = gone.class;
        }
        for patch in gone.sides {
            self.patch_mut(patch).arcs.remove(&killed);
        }
        self.push(kept);
    }

    fn reverse(&mut self, arc: ArcID) {
        let mesh = self.mesh;
        let a = &mut self.arcs[arc];
        a.crossings.make_contiguous().reverse();
        for &c in &a.crossings {
            let crossing = &mut self.surface.crossings[c];
            crossing.exit = mesh.twin(crossing.exit);
        }
        a.sides.swap(0, 1);
        if let Some(ends) = &mut a.ends {
            ends.swap(0, 1);
            // The junction being joined at is already gone.
            for j in ends.iter().copied().collect::<BTreeSet<_>>() {
                let Some(junction) = self.junctions.get_mut(j) else {
                    continue;
                };
                for spoke in &mut junction.spokes {
                    if spoke.0 == arc {
                        spoke.1 = 1 - spoke.1;
                    }
                }
            }
        }
    }

    fn input_position(&self, patch: usize) -> Vector3D {
        self.patch(patch).position / self.scale + self.center
    }

    /// Remaining patches and arcs, and the junction triangles.
    fn diagnostic(&self) -> (DiagnosticGraph, Vec<[Vector3D; 3]>) {
        let mut ids = SlotMap::<SkeletonNodeID, ()>::with_key();
        let mut nodes = HashMap::new();
        let mut positions = slotmap::SecondaryMap::new();
        for (patch, p) in self.patches.iter().enumerate() {
            if p.is_some() {
                let id = ids.insert(());
                nodes.insert(patch, id);
                positions.insert(id, self.input_position(patch));
            }
        }
        let edges = self
            .arcs
            .values()
            .map(|a| a.sides.map(|p| nodes[&p]))
            .collect();
        let triangles = self
            .junctions
            .values()
            .map(|j| {
                let mut corners = self
                    .mesh
                    .edges(j.face)
                    .map(|e| self.mesh.position(self.mesh.root(e)));
                [(); 3].map(|()| corners.next().unwrap())
            })
            .collect();
        (DiagnosticGraph { positions, edges }, triangles)
    }
}

fn edge_quadric(p: Vector3D, a: Vector3D) -> Matrix4<f64> {
    let b = a.cross(&p);
    let k = Matrix4::new(
        0.0, -a.z, a.y, -b.x, //
        a.z, 0.0, -a.x, -b.y, //
        -a.y, a.x, 0.0, -b.z, //
        0.0, 0.0, 0.0, 0.0,
    );
    k.transpose() * k
}

#[cfg(test)]
#[path = "test/test_mod.rs"]
mod test_mod;
