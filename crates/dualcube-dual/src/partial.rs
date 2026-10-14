//! Marked topology of incomplete loop arrangements, including regions with holes.

use crate::{Loop, LoopID, PropertyViolationError, arrangement::Arrangement};
use dualcube_types::prelude::*;
use slotmap::SlotMap;

#[derive(Clone, Debug)]
pub struct PartialTopology {
    pub along: HashMap<LoopID, Vec<usize>>,
    pub crossings: Vec<[LoopID; 2]>,
    pub regions: usize,
    sides: HashMap<(LoopID, usize), [usize; 2]>,
    positive: Vec<bool>,
    chi: Vec<i64>,
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
        })
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
        self.region_correspondences(other, 1).into_iter().next()
    }

    /// Bounded alternatives for symmetric partial arrangements.
    pub fn region_correspondences(&self, other: &Self, limit: usize) -> Vec<Vec<usize>> {
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

    /// Crossing partners and outgoing old regions along one newly inserted loop.
    pub fn insertion_itinerary(
        mesh: &Mesh<INPUT>,
        before: &SlotMap<LoopID, Loop>,
        after: &SlotMap<LoopID, Loop>,
        id: LoopID,
    ) -> Result<Vec<(LoopID, usize)>, PropertyViolationError> {
        let old = Arrangement::new(mesh, before)?;
        let (components, _, _) = old.components(mesh)?;
        let new = Arrangement::new(mesh, after)?;
        new.loop_crossings[&id]
            .iter()
            .map(|&crossing| {
                let c = &new.crossings[crossing];
                let face = &new.faces[&c.face];
                let partner = c
                    .chords
                    .iter()
                    .map(|&i| face.chords[i].loop_id)
                    .find(|&i| i != id)
                    .unwrap();
                let region = new
                    .region_after_in(&old, &components, id, crossing)
                    .ok_or(PropertyViolationError::UnknownError)?;
                Ok((partner, region))
            })
            .collect()
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
        results: &mut Vec<Vec<usize>>,
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
            let result = result.into_iter().map(Option::unwrap).collect();
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
