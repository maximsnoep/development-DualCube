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
        let arrangement = Arrangement::new(mesh, loops)?;
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
            along: arrangement.loop_crossings,
            crossings,
            positive,
            sides,
            regions,
            chi,
        })
    }

    /// Exact correspondence preserving loop IDs, orientation, boundary cycles, and region genus.
    /// Cyclic starting points along loops and unmarked crossing IDs may differ.
    pub fn equivalent(&self, other: &Self) -> bool {
        if self.regions != other.regions
            || self.crossings.len() != other.crossings.len()
            || self.along.len() != other.along.len()
            || self
                .along
                .iter()
                .any(|(id, a)| other.along.get(id).is_none_or(|b| a.len() != b.len()))
        {
            return false;
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
                return false;
            }
        }
        self.match_crossings(other, vec![None; self.crossings.len()], regions)
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
    ) -> bool {
        let Some(start) = map.iter().position(Option::is_none) else {
            let mut a: Vec<_> = self
                .chi
                .iter()
                .enumerate()
                .filter(|(i, _)| regions[*i].is_none())
                .map(|(_, &x)| x)
                .collect();
            let mut b: Vec<_> = other
                .chi
                .iter()
                .enumerate()
                .filter(|(i, _)| !regions.contains(&Some(*i)))
                .map(|(_, &x)| x)
                .collect();
            a.sort();
            b.sort();
            return a == b;
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
            if valid && self.match_crossings(other, mapping, region_mapping) {
                return true;
            }
        }
        false
    }
}
