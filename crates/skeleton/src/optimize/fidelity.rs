//! Restricted-label fidelity: every surface cell takes the best label its patch's star allows.
//!
//! A patch's star is its centroid plus one arm to the centroid of each boundary loop. A cell
//! beyond the cube around the centroid along some arm is tube wall of the nearest such arm (the
//! four labels orthogonal to its axis); any other cell is cube face (the node's free ports).
//! For fixed geometry the score splits into per-node tables over free-port masks and per-arm
//! tables over axes, so labels are compared without touching the surface.

use crate::label::Port;
use crate::surface::{Regions, Vertex};
use crate::{BoundaryCrossingID, EdgeLabel, Skeleton, SkeletonEdgeID, SkeletonNodeID};
use dualcube_types::prelude::*;
use slotmap::SecondaryMap;
use std::collections::HashMap;

/// Which per-cell value is summed: the label alignment n·a (higher is better) or the
/// flattening distortion (1/c − c)² of the reconstructed polycube's fidelity (lower is better).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Measure {
    Alignment,
    Distortion,
}

/// Area-weighted means over the surface.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Fidelity {
    pub alignment: f64,
    pub distortion: f64,
}

impl Fidelity {
    /// Higher is better for both measures.
    pub fn gain(&self, measure: Measure) -> f64 {
        match measure {
            Measure::Alignment => self.alignment,
            Measure::Distortion => -self.distortion,
        }
    }
}

/// Distortion of flattening a cell whose best label has cosine `c`; capped near folding as in
/// the polycube quality.
pub fn distortion(c: f64) -> f64 {
    let floor = 80f64.to_radians().cos();
    let c = c.max(floor);
    (1. / c - c).powi(2)
}

/// One piece of a triangle between boundary arcs.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CellGeometry {
    pub owner: SkeletonNodeID,
    pub area: f64,
    pub centroid: Vector3D,
    pub normal: Vector3D,
}

/// Uniform offsets along each edge, as `surface::position`, for all crossings at once.
pub(crate) fn crossing_positions(
    mesh: &Mesh<INPUT>,
    skeleton: &Skeleton,
) -> HashMap<BoundaryCrossingID, Vector3D> {
    let mut positions = HashMap::new();
    for (&edge, order) in &skeleton.surface.edge_order {
        let (a, b) = (
            mesh.position(mesh.root(edge)),
            mesh.position(mesh.toor(edge)),
        );
        for (rank, &id) in order.iter().enumerate() {
            let t = (rank + 1) as f64 / (order.len() + 1) as f64;
            positions.insert(id, a * (1. - t) + b * t);
        }
    }
    positions
}

pub(crate) fn cells(
    mesh: &Mesh<INPUT>,
    regions: &Regions,
    positions: &HashMap<BoundaryCrossingID, Vector3D>,
) -> Vec<CellGeometry> {
    regions
        .cells
        .iter()
        .zip(&regions.owners)
        .map(|(cell, &owner)| {
            let points: Vec<_> = cell
                .vertices
                .iter()
                .map(|v| match *v {
                    Vertex::Input(v) => mesh.position(v),
                    Vertex::Crossing(c) => positions[&c],
                })
                .collect();
            let normal = mesh.normal(cell.face);
            let mut area = 0.;
            for i in 1..points.len().saturating_sub(1) {
                area += (points[i] - points[0])
                    .cross(&(points[i + 1] - points[0]))
                    .dot(&normal)
                    / 2.;
            }
            CellGeometry {
                owner,
                area: area.max(0.),
                centroid: points.iter().sum::<Vector3D>() / points.len() as f64,
                normal,
            }
        })
        .collect()
}

/// A patch's centroid, cube half-size, and one arm per incident boundary.
#[derive(Clone, Debug)]
pub(crate) struct Star {
    pub centre: Vector3D,
    pub radius: f64,
    pub arms: Vec<(SkeletonEdgeID, Vector3D)>,
}

/// Where a cell lies in its star.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Element {
    Cube,
    Arm(SkeletonEdgeID),
}

impl Star {
    pub fn element(&self, point: Vector3D) -> Element {
        let mut best = (f64::INFINITY, Element::Cube);
        for &(edge, end) in &self.arms {
            let axis = end - self.centre;
            let length = axis.norm();
            if length == 0. {
                continue;
            }
            let along = (point - self.centre).dot(&axis) / length;
            if along <= self.radius {
                continue;
            }
            let t = (along / length).min(1.);
            let distance = (point - self.centre - axis * t).norm();
            if distance < best.0 {
                best = (distance, Element::Arm(edge));
            }
        }
        best.1
    }
}

/// Centroid and perimeter of each loop through its crossings.
pub(crate) fn loops(
    skeleton: &Skeleton,
    positions: &HashMap<BoundaryCrossingID, Vector3D>,
) -> SecondaryMap<SkeletonEdgeID, (Vector3D, f64)> {
    skeleton
        .edges
        .iter()
        .map(|(id, edge)| {
            let points: Vec<_> = edge
                .boundary
                .crossings
                .iter()
                .map(|c| positions[c])
                .collect();
            let perimeter = (0..points.len())
                .map(|i| (points[(i + 1) % points.len()] - points[i]).norm())
                .sum();
            let centroid = points.iter().sum::<Vector3D>() / points.len().max(1) as f64;
            (id, (centroid, perimeter))
        })
        .collect()
}

pub(crate) fn stars(
    skeleton: &Skeleton,
    cells: &[CellGeometry],
    loops: &SecondaryMap<SkeletonEdgeID, (Vector3D, f64)>,
) -> SecondaryMap<SkeletonNodeID, Star> {
    let mut sums = SecondaryMap::<SkeletonNodeID, (Vector3D, f64)>::new();
    for node in skeleton.nodes.keys() {
        sums.insert(node, (Vector3D::zeros(), 0.));
    }
    for cell in cells {
        let sum = &mut sums[cell.owner];
        sum.0 += cell.centroid * cell.area;
        sum.1 += cell.area;
    }
    let mut arms = SecondaryMap::<SkeletonNodeID, Vec<_>>::new();
    for node in skeleton.nodes.keys() {
        arms.insert(node, Vec::new());
    }
    for (id, edge) in &skeleton.edges {
        for node in edge.nodes {
            arms[node].push((id, loops[id].0));
        }
    }
    skeleton
        .nodes
        .iter()
        .map(|(node, data)| {
            let arms = std::mem::take(&mut arms[node]);
            let (sum, area) = sums[node];
            let centre = if area > 0. {
                sum / area
            } else if !arms.is_empty() {
                arms.iter().map(|a| a.1).sum::<Vector3D>() / arms.len() as f64
            } else {
                data.position
            };
            // A square cross-section of side w has perimeter 4w and half-size w/2.
            let radius = if arms.is_empty() {
                0.
            } else {
                arms.iter().map(|&(e, _)| loops[e].1 / 8.).sum::<f64>() / arms.len() as f64
            };
            (
                node,
                Star {
                    centre,
                    radius,
                    arms,
                },
            )
        })
        .collect()
}

/// Cosines of a normal with the six signed axes, indexed by `Port`.
pub(crate) fn cosines(normal: Vector3D) -> [f64; 6] {
    std::array::from_fn(|i| normal[i / 2] * if i % 2 == 0 { 1. } else { -1. })
}

/// Best cosine among the walls of an arm along `axis`.
pub(crate) fn wall(normal: Vector3D, axis: usize) -> f64 {
    normal[(axis + 1) % 3]
        .abs()
        .max(normal[(axis + 2) % 3].abs())
}

/// Best cosine for every set of allowed ports; −1 when none is allowed.
pub(crate) fn best_per_mask(normal: Vector3D) -> [f64; 64] {
    let c = cosines(normal);
    let mut best = [-1f64; 64];
    for mask in 1..64usize {
        let low = mask.trailing_zeros() as usize;
        best[mask] = best[mask & (mask - 1)].max(c[low]);
    }
    best
}

/// Per-cell value of a best cosine, as [alignment, distortion].
fn value(c: f64) -> [f64; 2] {
    [c, distortion(c)]
}

/// Sums for fixed geometry; evaluating labels is O(nodes + edges).
#[derive(Clone, Debug)]
pub struct Tables {
    pub area: f64,
    /// Unrestricted closest-axis labeling.
    pub naive: Fidelity,
    /// Cube cells of a node, per free-port mask.
    nodes: SecondaryMap<SkeletonNodeID, Box<[[f64; 2]; 64]>>,
    /// Wall cells of an arm, per axis.
    arms: HashMap<(SkeletonNodeID, SkeletonEdgeID), [[f64; 2]; 3]>,
}

impl Tables {
    pub fn build(mesh: &Mesh<INPUT>, skeleton: &Skeleton) -> crate::SkeletonResult<Self> {
        let regions = Regions::build(mesh, skeleton)?;
        Ok(Self::from_regions(mesh, skeleton, &regions))
    }

    pub(crate) fn from_regions(mesh: &Mesh<INPUT>, skeleton: &Skeleton, regions: &Regions) -> Self {
        let positions = crossing_positions(mesh, skeleton);
        let cells = cells(mesh, regions, &positions);
        let stars = stars(skeleton, &cells, &loops(skeleton, &positions));
        let mut nodes: SecondaryMap<_, _> = skeleton
            .nodes
            .keys()
            .map(|n| (n, Box::new([[0.; 2]; 64])))
            .collect();
        let mut arms = HashMap::new();
        let mut area = 0.;
        let mut naive = [0.; 2];
        for cell in &cells {
            if cell.area == 0. {
                continue;
            }
            area += cell.area;
            let best = best_per_mask(cell.normal);
            let [a, d] = value(best[63]);
            naive[0] += cell.area * a;
            naive[1] += cell.area * d;
            match stars[cell.owner].element(cell.centroid) {
                Element::Cube => {
                    let table = &mut nodes[cell.owner];
                    for (entry, &c) in table.iter_mut().zip(&best) {
                        let [a, d] = value(c);
                        entry[0] += cell.area * a;
                        entry[1] += cell.area * d;
                    }
                }
                Element::Arm(edge) => {
                    let table: &mut [[f64; 2]; 3] = arms.entry((cell.owner, edge)).or_default();
                    for (axis, entry) in table.iter_mut().enumerate() {
                        let [a, d] = value(wall(cell.normal, axis));
                        entry[0] += cell.area * a;
                        entry[1] += cell.area * d;
                    }
                }
            }
        }
        let scale = if area > 0. { 1. / area } else { 0. };
        Self {
            area,
            naive: Fidelity {
                alignment: naive[0] * scale,
                distortion: naive[1] * scale,
            },
            nodes,
            arms,
        }
    }

    /// Cube part of a node with the given free ports, unnormalized.
    pub(crate) fn node(&self, node: SkeletonNodeID, free: u8) -> [f64; 2] {
        self.nodes.get(node).map_or([0.; 2], |t| t[free as usize])
    }

    /// Wall part of an arm along an axis, unnormalized.
    pub(crate) fn arm(&self, node: SkeletonNodeID, edge: SkeletonEdgeID, axis: usize) -> [f64; 2] {
        self.arms.get(&(node, edge)).map_or([0.; 2], |t| t[axis])
    }

    pub(crate) fn normalize(&self, sum: [f64; 2]) -> Fidelity {
        let scale = if self.area > 0. { 1. / self.area } else { 0. };
        Fidelity {
            alignment: sum[0] * scale,
            distortion: sum[1] * scale,
        }
    }

    pub fn evaluate(
        &self,
        skeleton: &Skeleton,
        labels: &SecondaryMap<SkeletonEdgeID, EdgeLabel>,
    ) -> Fidelity {
        let mut used = SecondaryMap::<SkeletonNodeID, u8>::new();
        for node in skeleton.nodes.keys() {
            used.insert(node, 0);
        }
        let mut sum = [0.; 2];
        for (id, edge) in &skeleton.edges {
            let port = Port::from_label(labels[id]);
            used[edge.nodes[0]] |= port.bit();
            used[edge.nodes[1]] |= port.opposite().bit();
            for node in edge.nodes {
                let [a, d] = self.arm(node, id, port.axis());
                sum[0] += a;
                sum[1] += d;
            }
        }
        for (node, &mask) in &used {
            let [a, d] = self.node(node, 63 & !mask);
            sum[0] += a;
            sum[1] += d;
        }
        self.normalize(sum)
    }
}
