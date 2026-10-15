//! Marked topology of incomplete loop arrangements, including regions with holes.

use crate::{Loop, LoopID, PropertyViolationError, arrangement::Arrangement};
use dualcube_types::prelude::*;
use slotmap::SlotMap;

/// Identifies another topology's regions and crossings, indexed by this topology's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Correspondence {
    pub regions: Vec<usize>,
    pub crossings: Vec<usize>,
}

/// One crossing of a loop inserted into an arrangement (see `insertion::Plan`).
#[derive(Clone, Copy, Debug)]
pub struct Insertion {
    pub partner: LoopID,
    /// Index in the partner's previous `along` of the crossing preceding this one, if any.
    pub after: Option<usize>,
    /// Previous region entered by this crossing.
    pub region: usize,
}

#[derive(Clone, Debug)]
pub struct PartialTopology {
    pub along: HashMap<LoopID, Vec<usize>>,
    pub crossings: Vec<[LoopID; 2]>,
    pub regions: usize,
    sides: HashMap<(LoopID, usize), [usize; 2]>,
    positive: Vec<bool>,
    chi: Vec<i64>,
    /// The loop-map vertex of every crossing, when built from a map.
    pub vertices: Vec<usize>,
}

impl PartialTopology {
    pub fn new(
        mesh: &Mesh<INPUT>,
        loops: &SlotMap<LoopID, Loop>,
    ) -> Result<Self, PropertyViolationError> {
        Self::from_arrangement(mesh, &Arrangement::new(mesh, loops)?)
    }

    pub(crate) fn from_arrangement(
        mesh: &Mesh<INPUT>,
        arrangement: &Arrangement,
    ) -> Result<Self, PropertyViolationError> {
        let (components, regions, _) = arrangement.components(mesh)?;
        let chi = arrangement.region_euler(mesh, &components, regions)?;
        let mut sides = HashMap::new();
        let mut crossings = Vec::new();
        let mut positive = Vec::new();
        for crossing in &arrangement.crossings {
            let face = &arrangement.faces[&crossing.face];
            let [a, b] = crossing.chords.map(|i| &face.chords[i]);
            crossings.push([a.loop_id, b.loop_id]);
            let da = [a.to[0] - a.from[0], a.to[1] - a.from[1]];
            let db = [b.to[0] - b.from[0], b.to[1] - b.from[1]];
            positive.push(da[0] * db[1] - da[1] * db[0] > 0.);
        }
        for (&id, along) in &arrangement.loop_crossings {
            if along.is_empty() {
                let cells = arrangement
                    .isolated_sides(id)
                    .ok_or(PropertyViolationError::UnknownError)?;
                sides.insert((id, 0), cells.map(|c| components[c]));
            } else {
                for (i, &crossing) in along.iter().enumerate() {
                    let cells = arrangement
                        .cells_after_crossing(id, crossing)
                        .ok_or(PropertyViolationError::UnknownError)?;
                    sides.insert((id, i), cells.map(|c| components[c]));
                }
            }
        }
        Ok(Self {
            along: arrangement.loop_crossings.clone(),
            crossings,
            positive,
            sides,
            regions,
            chi,
            vertices: vec![],
        })
    }

    /// Exact topology of closed loops in a loop map.
    pub fn from_map(map: &crate::loop_map::LoopMap) -> Self {
        use crate::loop_map::{Piece, Point};
        let (region, regions) = map.regions();
        let chi = map.euler(&region, regions);
        let chains = map.chains();
        let mut index = HashMap::new();
        let mut crossings = Vec::new();
        let mut positive = Vec::new();
        let mut along = HashMap::new();
        let mut sides = HashMap::new();
        let mut vertices = Vec::new();
        let mut ids: Vec<_> = chains.keys().copied().collect();
        ids.sort();
        for &id in &ids {
            let chain = &chains[&id];
            let mut list = vec![];
            for &h in chain {
                let v = map.halves[h].origin;
                if map.points[v] != Point::Crossing {
                    continue;
                }
                let c = *index.entry(v).or_insert_with(|| {
                    // The other loop's direction, counterclockwise from this one's.
                    let out = map.outgoing(v);
                    let p = out.iter().position(|&o| o == h).unwrap();
                    let (other, forward) = (1..out.len())
                        .find_map(|d| match map.halves[out[(p + d) % out.len()]].piece {
                            Piece::Loop(l, f) if l != id => Some((l, f)),
                            _ => None,
                        })
                        .unwrap();
                    crossings.push([id, other]);
                    positive.push(forward);
                    vertices.push(v);
                    crossings.len() - 1
                });
                sides.insert(
                    (id, list.len()),
                    [
                        region[map.halves[h].face],
                        region[map.halves[map.halves[h].twin].face],
                    ],
                );
                list.push(c);
            }
            if list.is_empty() {
                let h = chain[0];
                sides.insert(
                    (id, 0),
                    [
                        region[map.halves[h].face],
                        region[map.halves[map.halves[h].twin].face],
                    ],
                );
            }
            along.insert(id, list);
        }
        Self {
            along,
            crossings,
            regions,
            sides,
            positive,
            chi,
            vertices,
        }
    }

    /// Reverse one loop's traversal without changing its geometry or rebuilding the arrangement.
    pub fn reverse_loop(&mut self, id: LoopID) {
        let along = self
            .along
            .get_mut(&id)
            .expect("loop belongs to this topology");
        if along.is_empty() {
            self.sides.get_mut(&(id, 0)).unwrap().swap(0, 1);
            return;
        }
        let count = along.len();
        let sides: Vec<_> = (0..count).map(|i| self.sides[&(id, i)]).collect();
        for &crossing in along.iter() {
            self.positive[crossing] = !self.positive[crossing];
        }
        along.reverse();
        for i in 0..count {
            // After a reversed crossing is the interval before its original occurrence.
            let [left, right] = sides[(2 * count - 2 - i) % count];
            self.sides.insert((id, i), [right, left]);
        }
    }

    /// Exact correspondence preserving loop IDs, orientation, boundary cycles, and region genus.
    /// Cyclic starting points along loops and unmarked crossing IDs may differ.
    pub fn equivalent(&self, other: &Self) -> bool {
        self.region_correspondence(other).is_some()
    }

    /// Region IDs in `other`, indexed by this arrangement's regions.
    pub fn region_correspondence(&self, other: &Self) -> Option<Vec<usize>> {
        self.correspondences(other, 1)
            .into_iter()
            .next()
            .map(|c| c.regions)
    }

    /// Bounded alternatives for symmetric partial arrangements.
    pub fn correspondences(&self, other: &Self, limit: usize) -> Vec<Correspondence> {
        if self.regions != other.regions
            || self.crossings.len() != other.crossings.len()
            || self.along.len() != other.along.len()
            || self
                .along
                .iter()
                .any(|(id, a)| other.along.get(id).is_none_or(|b| a.len() != b.len()))
        {
            return vec![];
        }
        let mut regions = vec![None; self.regions];
        for (&id, along) in &self.along {
            if along.is_empty()
                && !self.match_sides(
                    other,
                    self.sides[&(id, 0)],
                    other.sides[&(id, 0)],
                    &mut regions,
                )
            {
                return vec![];
            }
        }
        let mut result = Vec::new();
        self.match_crossings(
            other,
            vec![None; self.crossings.len()],
            regions,
            limit,
            &mut result,
        );
        result
    }

    /// Region at every mesh-edge gap, for constraining incremental routing.
    pub fn gap_regions(
        mesh: &Mesh<INPUT>,
        loops: &SlotMap<LoopID, Loop>,
    ) -> Result<HashMap<crate::GapNode, usize>, PropertyViolationError> {
        let arrangement = Arrangement::new(mesh, loops)?;
        let (components, _, _) = arrangement.components(mesh)?;
        let mut offsets = HashMap::<EdgeID, Vec<f64>>::new();
        for l in loops.values() {
            for (&edge, &offset) in l.edges.iter().zip(&l.offsets) {
                offsets.entry(edge).or_default().push(offset);
            }
        }
        let mut result = HashMap::new();
        for edge in mesh.edge_ids() {
            let mut bounds = offsets.remove(&edge).unwrap_or_default();
            bounds.sort_by(f64::total_cmp);
            bounds.insert(0, 0.);
            bounds.push(1.);
            for (gap, pair) in bounds.windows(2).enumerate() {
                let cell = arrangement
                    .locate_on_edge(mesh, edge, (pair[0] + pair[1]) / 2.)
                    .ok_or(PropertyViolationError::UnknownError)?;
                result.insert((edge, gap), components[cell]);
            }
        }
        Ok(result)
    }

    fn match_sides(
        &self,
        other: &Self,
        a: [usize; 2],
        b: [usize; 2],
        map: &mut [Option<usize>],
    ) -> bool {
        for (a, b) in a.into_iter().zip(b) {
            if self.chi[a] != other.chi[b] {
                return false;
            }
            if let Some(old) = map[a] {
                if old != b {
                    return false;
                }
            } else {
                if map.contains(&Some(b)) {
                    return false;
                }
                map[a] = Some(b);
            }
        }
        true
    }

    fn match_crossings(
        &self,
        other: &Self,
        map: Vec<Option<usize>>,
        regions: Vec<Option<usize>>,
        limit: usize,
        results: &mut Vec<Correspondence>,
    ) {
        if results.len() >= limit {
            return;
        }
        let Some(start) = map.iter().position(Option::is_none) else {
            let mut result = regions;
            for i in 0..result.len() {
                if result[i].is_none() {
                    let Some(j) = (0..other.regions)
                        .find(|&j| !result.contains(&Some(j)) && self.chi[i] == other.chi[j])
                    else {
                        return;
                    };
                    result[i] = Some(j);
                }
            }
            let result = Correspondence {
                regions: result.into_iter().map(Option::unwrap).collect(),
                crossings: map.into_iter().map(Option::unwrap).collect(),
            };
            if !results.contains(&result) {
                results.push(result);
            }
            return;
        };
        for target in 0..other.crossings.len() {
            if map.contains(&Some(target)) {
                continue;
            }
            let mut mapping = map.clone();
            let mut region_mapping = regions.clone();
            let mut pending = vec![(start, target)];
            let mut valid = true;
            while let Some((a, b)) = pending.pop() {
                if let Some(old) = mapping[a] {
                    if old != b {
                        valid = false;
                        break;
                    }
                    continue;
                }
                if mapping.contains(&Some(b)) {
                    valid = false;
                    break;
                }
                let pair = self.crossings[a];
                let target_pair = other.crossings[b];
                if !(pair == target_pair && self.positive[a] == other.positive[b]
                    || pair == [target_pair[1], target_pair[0]]
                        && self.positive[a] != other.positive[b])
                {
                    valid = false;
                    break;
                }
                mapping[a] = Some(b);
                for id in pair {
                    let from = &self.along[&id];
                    let to = &other.along[&id];
                    let i = from.iter().position(|&c| c == a).unwrap();
                    let j = to.iter().position(|&c| c == b).unwrap();
                    if !self.match_sides(
                        other,
                        self.sides[&(id, i)],
                        other.sides[&(id, j)],
                        &mut region_mapping,
                    ) {
                        valid = false;
                        break;
                    }
                    pending.push((from[(i + 1) % from.len()], to[(j + 1) % to.len()]));
                }
                if !valid {
                    break;
                }
            }
            if valid {
                self.match_crossings(other, mapping, region_mapping, limit, results);
            }
            if results.len() >= limit {
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests;
