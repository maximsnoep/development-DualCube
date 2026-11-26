//! Connectivity surgery on patch boundaries.
//!
//! Patches start as the dual cells of the mesh vertices, so their boundaries form the dual
//! graph: one arc per mesh edge and a junction in every triangle. Surgery only deletes arcs,
//! one shared boundary at a time, guided by the collapse costs of Au et al.:
//! - An arc between distinct patches merges them. Two patches glued along an arc are
//!   contractible in the volume when both are, so this needs no check.
//! - An arc with one patch on both sides adds a loop crossing it once. It is dissolved only
//!   when that loop bounds in the volume (integer homology, see [`homology`]). With
//!   `certify_each_step`, DiskScissors must also find a disk for it (homology is blind to
//!   commutators); a loop that is not null-homologous has none, so homology still rejects.
//!
//! A junction of degree two joins its two arcs. Surgery ends when no junctions remain; the
//! closed arcs then become the skeleton edges.

use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap, VecDeque};

use mehsh::prelude::{HasNeighbors, HasPosition, Mesh, Vector3D, VertKey};
use nalgebra::{Matrix4, Vector4};
use slotmap::SlotMap;
use tracing::{info, warn};

mod flip;
mod homology;
mod output;
mod tetrahedralize;

pub(crate) use output::simplify;

use super::deflate::CONTRACTION;
use crate::surface::canonical;
use crate::{BoundaryCrossing, BoundaryCrossingID, BoundarySurface, Skeleton};
use dualcube_types::prelude::*;
use homology::{Bits, Class, Homology};

/// Weights of collapse costs; topology checks are unchanged.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub shape_weight: f64,
    pub sampling_weight: f64,
    /// Subtracted once per volume cocycle that is nonzero on the dissolved arc. Not from Au et al.
    pub handle_bonus: f64,
    /// Rotation systems tried when surgery is stuck, before it fails (about 24 bytes each).
    pub flip_budget: u32,
    /// Require a DiskScissors disk for every self-arc loop before dissolving it.
    pub certify_each_step: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            shape_weight: 1.0,
            sampling_weight: 0.1,
            handle_bonus: 1.0e6,
            flip_budget: 200_000_000,
            certify_each_step: false,
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

/// Failed surgery.
#[derive(Debug)]
pub struct SurgeryFailure {
    pub reason: String,
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
    // Genus 1: integer homology is taken as exact, as in `certify::run`.
    let options = Options {
        certify_each_step: options.certify_each_step && genus > 1,
        ..options
    };
    let mut surgery = Surgery::new(contracted, original, genus, options)?;
    surgery.run()?;
    let mut skeleton = surgery.skeleton();
    output::untangle(original, &mut skeleton);
    simplify(original, &mut skeleton);
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
    /// Class of a loop crossing it once from left to right; set once one patch is on both
    /// sides. Reversing the arc negates it.
    class: Option<Class>,
    /// Volume cocycles nonzero on its crossings.
    handles: Bits,
    /// Whether DiskScissors found a disk for its loop, once checked.
    disk: Option<bool>,
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
    /// Class of a path from a fixed vertex of the owner patch. Paths within a patch differ
    /// by loops of that patch, which bound.
    potential: Vec<Class>,
    patches: Vec<Option<Patch>>,
    /// Crossings of all arcs. Until the first flip each mesh edge has at most one.
    surface: BoundarySurface,
    arcs: SlotMap<ArcID, Arc>,
    junctions: SlotMap<JunctionID, Junction>,
    homology: Homology,
    /// DiskScissors for `certify_each_step`, made on the first check.
    disks: Option<crate::certify::LoopCheck>,
    vertices: Vec<VertID>,
    genus: usize,
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
            disks: None,
            vertices: vertices.clone(),
            genus,
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
                disk: None,
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
        let unavailable = |reason: String| SurgeryFailure {
            reason: format!("Connectivity surgery cannot certify a step: {reason}"),
        };
        loop {
            while let Some(arc) = self.pending.pop() {
                if self.arcs.contains_key(arc)
                    && self.dissolvable(arc, None).map_err(&unavailable)?
                {
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
                let stuck = match self.search() {
                    Ok(sequence) => {
                        flips += sequence.len();
                        let (&last, earlier) = sequence.split_last().unwrap();
                        for &arc in earlier {
                            self.flip(arc);
                        }
                        // The flipped arc runs beside others, so its loop is found before.
                        let walk = self.flip_walk(last);
                        self.flip(last);
                        // Searching again could undo these flips forever.
                        if !self.dissolvable(last, walk).map_err(&unavailable)? {
                            return Err(SurgeryFailure {
                                reason: "Connectivity surgery stuck: flips freed a null-homologous loop without a disk".into(),
                            });
                        }
                        self.pending.push(last);
                        continue;
                    }
                    Err(stuck) => stuck,
                };
                warn!(
                    "Connectivity surgery stuck: {stuck}; {} junctions remain ({} arcs, {} patches, after {merges} merges)",
                    self.junctions.len(),
                    self.arcs.len(),
                    self.patches.iter().flatten().count()
                );
                return Err(SurgeryFailure {
                    reason: format!(
                        "Connectivity surgery stopped with remaining junctions: {stuck}"
                    ),
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
            "Connectivity surgery complete in {:.2}s: {merges} merges, {dissolved} self-arcs dissolved, {flips} flips, {} loops, {} disks checked",
            start.elapsed().as_secs_f64(),
            self.arcs.len(),
            self.disks.as_ref().map_or(0, |d| d.checked())
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
            let class = self.crossing_class(x, y);
            self.arcs[arc].class = Some(class);
        }
        self.arcs[arc].class.as_ref().unwrap().is_zero()
    }

    /// Whether a self-arc may be dissolved: its loop bounds, and with `certify_each_step` also
    /// has a disk. Spur loops bound on the surface and need none.
    /// `walk` replaces [`Self::walk`] for the loop, as after a flip.
    fn dissolvable(&mut self, arc: ArcID, walk: Option<Vec<VertID>>) -> Result<bool, String> {
        if !self.bounds(arc) {
            return Ok(false);
        }
        let spur = self.arcs[arc].ends.is_some_and(|ends| {
            ends.iter()
                .any(|&j| self.junctions.get(j).is_some_and(|j| j.spokes.len() == 1))
        });
        if !self.options.certify_each_step || self.genus == 0 || spur {
            return Ok(true);
        }
        if let Some(disk) = self.arcs[arc].disk {
            return Ok(disk);
        }
        let Some(walk) = walk.or_else(|| self.walk(arc)) else {
            warn!("Connectivity surgery: no vertex loop through a self-arc; kept");
            self.arcs[arc].disk = Some(false);
            return Ok(false);
        };
        if self.disks.is_none() {
            let check = crate::certify::LoopCheck::new(self.mesh, Default::default())
                .map_err(|e| e.to_string())?;
            self.disks = Some(check);
        }
        let disk = self
            .disks
            .as_mut()
            .unwrap()
            .disk(&walk)
            .map_err(|e| e.to_string())?;
        if let Err(miss) = disk {
            info!(
                "Connectivity surgery: self-arc loop bounds in homology but has no disk ({miss:?}); kept"
            );
        }
        self.arcs[arc].disk = Some(disk.is_ok());
        Ok(disk.is_ok())
    }

    /// A simple vertex cycle crossing the arc once and otherwise inside its patch: a mesh edge
    /// crossed only by the arc, closed by a shortest path over uncrossed edges of the patch.
    fn walk(&self, arc: ArcID) -> Option<Vec<VertID>> {
        let patch = self.arcs[arc].sides[0];
        self.crossing_edges(arc, patch, false)
            .into_iter()
            .find_map(|edge| self.cycle(patch, &[edge]))
    }

    /// Mesh edges crossed only by the arc with both ends in `patch`, left vertex first (right
    /// first when `reversed`).
    fn crossing_edges(&self, arc: ArcID, patch: usize, reversed: bool) -> Vec<[usize; 2]> {
        self.arcs[arc]
            .crossings
            .iter()
            .map(|&c| self.exit(c))
            .filter(|&exit| self.surface.edge_order[&canonical(self.mesh, exit)].len() == 1)
            .map(|exit| self.ends(exit))
            .filter(|ends| ends.iter().all(|&v| self.owner[v] == patch))
            .map(|[x, y]| if reversed { [y, x] } else { [x, y] })
            .collect()
    }

    /// A simple vertex cycle in `patch` that crosses the given mesh edges in order, each from
    /// its first vertex to its second, joined by shortest paths over uncrossed edges. It ends
    /// with the first vertex of the first edge.
    fn cycle(&self, patch: usize, edges: &[[usize; 2]]) -> Option<Vec<VertID>> {
        let mut used: HashSet<usize> = edges.iter().flatten().copied().collect();
        if used.len() != 2 * edges.len() {
            return None;
        }
        let mut walk = Vec::new();
        for (i, edge) in edges.iter().enumerate() {
            let next = edges[(i + 1) % edges.len()][0];
            let path = self.path(edge[1], next, patch, &used)?;
            used.extend(&path);
            walk.extend(path.into_iter().map(|v| self.vertices[v]));
        }
        Some(walk)
    }

    /// Shortest path from `from` to `to` over uncrossed edges of the patch, avoiding the other
    /// vertices of `avoid`.
    fn path(
        &self,
        from: usize,
        to: usize,
        patch: usize,
        avoid: &HashSet<usize>,
    ) -> Option<Vec<usize>> {
        let uncrossed = |e: EdgeID| {
            !self
                .surface
                .edge_order
                .contains_key(&canonical(self.mesh, e))
        };
        let mut parent = HashMap::from([(from, from)]);
        let mut queue = VecDeque::from([from]);
        while let Some(u) = queue.pop_front() {
            if u == to {
                let mut path = vec![to];
                while *path.last().unwrap() != from {
                    path.push(parent[path.last().unwrap()]);
                }
                path.reverse();
                return Some(path);
            }
            for e in self.mesh.edges(self.vertices[u]) {
                let w = self.index[&self.mesh.toor(e)];
                let free = w == to || !avoid.contains(&w);
                if free && self.owner[w] == patch && uncrossed(e) && !parent.contains_key(&w) {
                    parent.insert(w, u);
                    queue.push_back(w);
                }
            }
        }
        None
    }

    /// Class of a loop through the mesh edge from x to y, closed inside one patch.
    fn crossing_class(&self, x: usize, y: usize) -> Class {
        &(&self.potential[x] - &self.potential[y]) + &self.homology.edge(x, y)
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
        // Afterwards the potential steps by the edge's value from x to y.
        let mut shift = self.crossing_class(x, y);
        if self.owner[x] == gone {
            shift = -&shift;
        }
        for &w in &small.vertices {
            self.owner[w] = id;
            self.potential[w] = &self.potential[w] + &shift;
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
            self.surface.remove(self.mesh, c);
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
        if self.arcs[kept].disk.is_none() {
            self.arcs[kept].disk = gone.disk;
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
        a.class = a.class.as_ref().map(|c| -c);
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
