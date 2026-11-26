//! Simplify a labeled skeleton chain by chain for the [`quality`] score, then recompute target
//! placement.
//!
//! On every chain (a maximal path through degree-two patches) a DP keeps the boundaries and
//! labels with the best score, all other labels fixed; dropped boundaries merge their patches.
//! A chain from a junction to a leaf may drop all of them. This repeats until no chain improves.
//! Alignment is a mean over the boundary length, so the DP maximizes its linearization at the
//! current mean (Dinkelbach), and a move is applied only when the score itself improves.

use crate::label::{self, Port, constraints, path};
use crate::quality::{self, Alignment, BETA, Visibility};
use crate::{
    LabeledSkeleton, RealizedSkeleton, Skeleton, SkeletonEdgeID, SkeletonNodeID, SkeletonResult,
    place_cubes,
};
use dualcube_types::prelude::*;
use slotmap::SecondaryMap;

/// Linearizations tried per chain.
const ROUNDS: usize = 4;

pub fn run(mesh: &Mesh<INPUT>, placed: &RealizedSkeleton) -> SkeletonResult<RealizedSkeleton> {
    placed.labeled.skeleton.debug_validate(mesh);
    let reduced = reduce(mesh, &placed.labeled)?;
    place_cubes::run(mesh, &reduced)
}

/// Reduce every chain until none improves; the result is validated.
pub fn reduce(mesh: &Mesh<INPUT>, labeled: &LabeledSkeleton) -> SkeletonResult<LabeledSkeleton> {
    let start = std::time::Instant::now();
    let mut labeled = labeled.clone();
    let mut state = State::new(mesh, &labeled)?;
    let (patches, before) = (labeled.skeleton.nodes.len(), state.score());
    // Every move raises the score; the bound is a guard.
    'improve: for _ in 0..10 * (labeled.skeleton.edges.len() + 10) {
        for chain in chains(&labeled.skeleton) {
            if let Some(next) = reduce_chain(mesh, &labeled, &chain, &mut state) {
                labeled = next;
                continue 'improve;
            }
        }
        break;
    }
    labeled.validate(mesh)?;
    tracing::info!(
        "Optimization: {patches} → {} patches, score {before:.4} → {:.4}, {:.2}s",
        labeled.skeleton.nodes.len(),
        state.score(),
        start.elapsed().as_secs_f64()
    );
    Ok(labeled)
}

/// The score's sums, kept up to date with the moves.
struct State {
    /// Of the boundaries as extracted; a kept boundary keeps its loop.
    alignments: SecondaryMap<SkeletonEdgeID, Alignment>,
    visibility: Visibility,
    /// The patches of the input skeleton merged into every patch, sorted.
    members: SecondaryMap<SkeletonNodeID, Vec<SkeletonNodeID>>,
    /// Visible area of unions of input patches.
    seen: HashMap<Vec<SkeletonNodeID>, f64>,
    /// Over the boundaries: Σ length · alignment with the label, and Σ length.
    aligned: f64,
    length: f64,
    /// Σ visible area over the patches.
    visible: f64,
    boundaries: usize,
}

impl State {
    fn new(mesh: &Mesh<INPUT>, labeled: &LabeledSkeleton) -> SkeletonResult<Self> {
        let skeleton = &labeled.skeleton;
        let alignments = quality::alignments(mesh, skeleton);
        let (mut aligned, mut length) = (0., 0.);
        for (e, a) in &alignments {
            aligned += a.length * a.with(Port::from_label(labeled.labels[e]));
            length += a.length;
        }
        let mut state = Self {
            alignments,
            visibility: Visibility::build(mesh, skeleton)?,
            members: skeleton.nodes.keys().map(|n| (n, vec![n])).collect(),
            seen: HashMap::new(),
            aligned,
            length,
            visible: 0.,
            boundaries: skeleton.edges.len(),
        };
        let unions: Vec<_> = skeleton.nodes.keys().map(|n| vec![n]).collect();
        state.compute(unions.clone());
        state.visible = unions.iter().map(|u| state.seen[u]).sum();
        Ok(state)
    }

    fn score(&self) -> f64 {
        let alignment = if self.length > 0. {
            self.aligned / self.length
        } else {
            1.
        };
        quality::score(
            alignment,
            self.visible / self.visibility.area,
            self.boundaries,
        )
    }

    /// The input patches of a union of patches.
    fn union(&self, nodes: impl IntoIterator<Item = SkeletonNodeID>) -> Vec<SkeletonNodeID> {
        let mut union: Vec<_> = nodes
            .into_iter()
            .flat_map(|n| self.members[n].iter().copied())
            .collect();
        union.sort();
        union.dedup();
        union
    }

    /// Fill `seen` for the unions it lacks.
    fn compute(&mut self, mut unions: Vec<Vec<SkeletonNodeID>>) {
        unions.retain(|u| !self.seen.contains_key(u));
        unions.sort();
        unions.dedup();
        let values = self.visibility.visible_all(&unions);
        self.seen.extend(unions.into_iter().zip(values));
    }
}

/// A maximal path whose interior nodes have degree two, from `nodes[0]` to its last node; a
/// component that is one ring runs from a node back to itself.
struct Chain {
    nodes: Vec<SkeletonNodeID>,
    edges: Vec<SkeletonEdgeID>,
    /// Whether edge `i` is stored in chain direction.
    forward: Vec<bool>,
}

fn chains(skeleton: &Skeleton) -> Vec<Chain> {
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    for edge in skeleton.edges.keys() {
        if seen.contains(&edge) {
            continue;
        }
        let (steps, [start, _]) = label::chain(skeleton, edge);
        let mut nodes = vec![start];
        for &(e, forward) in &steps {
            seen.insert(e);
            nodes.push(skeleton.edges[e].nodes[usize::from(forward)]);
        }
        result.push(Chain {
            nodes,
            edges: steps.iter().map(|s| s.0).collect(),
            forward: steps.iter().map(|s| s.1).collect(),
        });
    }
    result
}

/// Kept boundaries (chain indices) with their ports in chain direction.
struct Choice {
    kept: Vec<usize>,
    ports: Vec<Port>,
}

/// Chain nodes `a..=b` as input patches.
fn nodes(state: &State, chain: &Chain, a: usize, b: usize) -> Vec<SkeletonNodeID> {
    state.union(chain.nodes[a..=b].iter().copied())
}

/// For a ring, the patch through its node: after kept `last`, up to kept `first`.
fn wrap(state: &State, chain: &Chain, last: usize, first: usize) -> Vec<SkeletonNodeID> {
    let n = chain.edges.len();
    state.union(
        chain.nodes[last + 1..n]
            .iter()
            .chain(&chain.nodes[..=first])
            .copied(),
    )
}

/// The chain relabelled and reduced if that raises the score and stays valid.
fn reduce_chain(
    mesh: &Mesh<INPUT>,
    labeled: &LabeledSkeleton,
    chain: &Chain,
    state: &mut State,
) -> Option<LabeledSkeleton> {
    let n = chain.edges.len();
    let ring = chain.nodes[0] == chain.nodes[n];
    let mut unions = Vec::new();
    for i in 0..n {
        for j in 0..i {
            unions.push(nodes(state, chain, j + 1, i));
        }
        if ring {
            unions.extend((0..n).map(|first| wrap(state, chain, i, first)));
        } else {
            unions.push(nodes(state, chain, 0, i));
            unions.push(nodes(state, chain, i + 1, n));
        }
    }
    unions.push(nodes(state, chain, 0, n));
    unions.extend(chain.nodes.iter().map(|&v| state.union([v])));
    state.compute(unions);
    let (next, update) = choose(mesh, labeled, chain, state)?;
    state.aligned = update.aligned;
    state.length = update.length;
    state.visible = update.visible;
    state.boundaries = update.boundaries;
    state.members = update.members;
    Some(next)
}

/// The sums of [`State`] after a move.
struct Update {
    aligned: f64,
    length: f64,
    visible: f64,
    boundaries: usize,
    members: SecondaryMap<SkeletonNodeID, Vec<SkeletonNodeID>>,
}

/// The best valid improving choice on the chain, applied; `seen` covers its unions.
fn choose(
    mesh: &Mesh<INPUT>,
    labeled: &LabeledSkeleton,
    chain: &Chain,
    state: &State,
) -> Option<(LabeledSkeleton, Update)> {
    let skeleton = &labeled.skeleton;
    let degree = skeleton.degrees();
    let n = chain.edges.len();
    let (start, end) = (chain.nodes[0], chain.nodes[n]);
    let ring = start == end;
    let run = |a: usize, b: usize| nodes(state, chain, a, b);
    let wrap = |last: usize, first: usize| wrap(state, chain, last, first);
    let seen = |union: Vec<SkeletonNodeID>| state.seen[&union];

    // Port `q` in chain direction as stored, and the alignment of edge `i` with it.
    let stored = |i: usize, q: Port| if chain.forward[i] { q } else { q.opposite() };
    let fit = |i: usize, q: Port| state.alignments[chain.edges[i]].with(stored(i, q));
    let length = |i: usize| state.alignments[chain.edges[i]].length;
    let current: Vec<Port> = (0..n)
        .map(|i| stored(i, Port::from_label(labeled.labels[chain.edges[i]])))
        .collect();
    let chain_aligned: f64 = (0..n).map(|i| length(i) * fit(i, current[i])).sum();
    let chain_length: f64 = (0..n).map(length).sum();
    let mut chain_visible: f64 = chain.nodes[..n]
        .iter()
        .map(|&v| seen(state.union([v])))
        .sum();
    if !ring {
        chain_visible += seen(state.union([end]));
    }

    // The score after a choice; keeping nothing merges the whole chain.
    let score = |choice: &Choice| {
        let kept = &choice.kept;
        let (mut aligned, mut total) = (state.aligned - chain_aligned, state.length - chain_length);
        for (&i, &q) in kept.iter().zip(&choice.ports) {
            aligned += length(i) * fit(i, q);
            total += length(i);
        }
        let mut visible = state.visible - chain_visible;
        if kept.is_empty() {
            visible += seen(run(0, n));
        } else {
            for pair in kept.windows(2) {
                visible += seen(run(pair[0] + 1, pair[1]));
            }
            let (first, last) = (kept[0], kept[kept.len() - 1]);
            visible += if ring {
                seen(wrap(last, first))
            } else {
                seen(run(0, first)) + seen(run(last + 1, n))
            };
        }
        let alignment = if total > 0. { aligned / total } else { 1. };
        let boundaries = state.boundaries - n + kept.len();
        (
            quality::score(alignment, visible / state.visibility.area, boundaries),
            aligned,
            total,
            visible,
        )
    };
    let now = score(&Choice {
        kept: (0..n).collect(),
        ports: current.clone(),
    })
    .0;
    debug_assert!(
        (now - state.score()).abs() < 1e-9,
        "{now} vs {}",
        state.score()
    );

    let mut others = labeled.labels.clone();
    for &e in &chain.edges {
        others.remove(e);
    }
    let used = constraints::used_ports(skeleton, &others).ok()?;
    let masks: [bool; 64] = std::array::from_fn(|mask| {
        constraints::feasible(skeleton, &others, Some(([start, end], mask as u8)))
    });
    // Ends joined outside the chain would get parallel edges from a single kept boundary.
    let joined = skeleton.edges.iter().any(|(e, data)| {
        !chain.edges.contains(&e) && data.nodes.contains(&start) && data.nodes.contains(&end)
    });
    let half = 0.5 / state.visibility.area;
    let patch = |j: usize, i: usize| -half * seen(run(j + 1, i));
    // The choices maximizing the score linearized at alignment `lambda` over boundary length
    // `total`, at least `min_kept` boundaries.
    let select = |lambda: f64, total: f64, min_kept: usize| -> Option<Choice> {
        let costs: Vec<[f64; 6]> = (0..n)
            .map(|i| {
                std::array::from_fn(|q| {
                    let gain = 0.5 * length(i) * (fit(i, Port::ALL[q]) - lambda) / total;
                    BETA - gain
                })
            })
            .collect();
        let solve = |first: &dyn Fn(usize, usize) -> f64,
                     last: &dyn Fn(usize, usize) -> f64,
                     limits: path::Constraints,
                     min_kept: usize| {
            let ends = path::Ends {
                first,
                last,
                patch: &patch,
            };
            path::select(&costs, &limits, &ends, min_kept)
        };
        let selection = if !ring {
            let first = |i: usize, _: usize| -half * seen(run(0, i));
            let last = |i: usize, _: usize| -half * seen(run(i + 1, n));
            let limits = path::Constraints {
                start_ports: 63 & !used[start],
                end_ports: 63 & !used[end],
                masks,
            };
            solve(&first, &last, limits, min_kept)
        } else {
            // A ring leaves and enters its node through two different ports; one or two kept
            // boundaries close a self-loop or parallel edges. Its first kept boundary is fixed
            // per solve, as the patch through the node depends on it.
            let mut best: Option<path::Selection> = None;
            for p in Port::ALL.into_iter().filter(|p| used[start] & p.bit() == 0) {
                for kept_first in 0..n {
                    let first = |i: usize, _: usize| {
                        if i == kept_first { 0. } else { f64::INFINITY }
                    };
                    let last = |i: usize, _: usize| -half * seen(wrap(i, kept_first));
                    let limits = path::Constraints {
                        start_ports: p.bit(),
                        end_ports: 63 & !used[start] & !p.bit(),
                        masks,
                    };
                    if let Some(s) = solve(&first, &last, limits, min_kept.max(3))
                        && best.as_ref().is_none_or(|b| s.cost < b.cost)
                    {
                        best = Some(s);
                    }
                }
            }
            best
        };
        selection.map(|s| Choice {
            kept: s.kept,
            ports: s.ports,
        })
    };

    // Candidates by improving score; Dinkelbach rounds move the linearization to the best so far.
    let mut candidates: Vec<(f64, Choice)> = Vec::new();
    let mut at = (
        state.aligned / state.length.max(f64::MIN_POSITIVE),
        state.length,
    );
    for _ in 0..ROUNDS {
        let mut moved = false;
        let found: Vec<Choice> = [1 + usize::from(joined), 2, 3]
            .into_iter()
            .filter_map(|k| select(at.0, at.1, k))
            .collect();
        for choice in found {
            let (value, aligned, total, _) = score(&choice);
            let new = candidates
                .iter()
                .all(|(_, c)| c.kept != choice.kept || c.ports != choice.ports);
            if value <= now + 1e-12 || !new {
                continue;
            }
            if candidates.iter().all(|(v, _)| value > *v) && total > 0. {
                at = (aligned / total, total);
                moved = true;
            }
            candidates.push((value, choice));
        }
        if !moved {
            break;
        }
    }
    // No boundary kept: a branch merges into its junction.
    let into = match (degree[start] == 1, degree[end] == 1) {
        _ if ring => None,
        (true, false) if degree[end] >= 3 => Some(end),
        (false, true) if degree[start] >= 3 => Some(start),
        _ => None,
    };
    if into.is_some() {
        let choice = Choice {
            kept: Vec::new(),
            ports: Vec::new(),
        };
        let value = score(&choice).0;
        if value > now + 1e-12 {
            candidates.push((value, choice));
        }
    }
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));

    // A choice that fails validation (parallel edges) gives way to the next best.
    candidates.iter().find_map(|(_, choice)| {
        let into = if choice.kept.is_empty() { into } else { None };
        let (next, members) = apply(mesh, labeled, chain, choice, into, &state.members)?;
        let (_, aligned, length, visible) = score(choice);
        let update = Update {
            aligned,
            length,
            visible,
            boundaries: state.boundaries - n + choice.kept.len(),
            members,
        };
        Some((next, update))
    })
}

/// The chain with `choice` applied: kept boundaries relabelled, the others dissolved (into
/// `into` for a whole branch); `None` unless valid.
fn apply(
    mesh: &Mesh<INPUT>,
    labeled: &LabeledSkeleton,
    chain: &Chain,
    choice: &Choice,
    into: Option<SkeletonNodeID>,
    members: &SecondaryMap<SkeletonNodeID, Vec<SkeletonNodeID>>,
) -> Option<(
    LabeledSkeleton,
    SecondaryMap<SkeletonNodeID, Vec<SkeletonNodeID>>,
)> {
    let n = chain.edges.len();
    let (start, end) = (chain.nodes[0], chain.nodes[n]);
    let mut next = labeled.clone();
    let mut members = members.clone();
    for (&i, &port) in choice.kept.iter().zip(&choice.ports) {
        let port = if chain.forward[i] {
            port
        } else {
            port.opposite()
        };
        next.labels[chain.edges[i]] = port.label();
    }
    // A whole branch merges outward from the junction, one neighbour at a time.
    let order: Vec<usize> = if into == Some(end) {
        (0..n).rev().collect()
    } else {
        (0..n).collect()
    };
    for i in order.into_iter().filter(|i| !choice.kept.contains(i)) {
        let edge = chain.edges[i];
        let nodes = next.skeleton.edges[edge].nodes;
        let keep = into.unwrap_or_else(|| {
            nodes
                .into_iter()
                .find(|v| *v == start || *v == end)
                .unwrap_or(nodes[0])
        });
        let other = next.skeleton.edges[edge].other(keep);
        dissolve(mesh, &mut next, edge, keep).ok()?;
        let mut merged = members.remove(other).unwrap_or_default();
        merged.extend(members[keep].iter().copied());
        merged.sort();
        members[keep] = merged;
    }
    (next.skeleton.validate(mesh).is_ok() && next.validate_labels().is_ok())
        .then_some((next, members))
}

/// Delete a boundary and merge its other patch into `keep` (merge lemma). Fails when the
/// graph would get parallel edges; labels may need repair afterwards.
fn dissolve(
    mesh: &Mesh<INPUT>,
    labeled: &mut LabeledSkeleton,
    edge: SkeletonEdgeID,
    keep: SkeletonNodeID,
) -> SkeletonResult<()> {
    let skeleton = &labeled.skeleton;
    let other = skeleton.edges.get(edge).ok_or("Missing edge")?.other(keep);
    if other == keep {
        return Err("Edge does not join `keep` to another node".into());
    }
    let mut neighbors = HashSet::new();
    for (e, data) in &skeleton.edges {
        if e == edge || !data.nodes.iter().any(|&n| n == keep || n == other) {
            continue;
        }
        let nodes = data.nodes.map(|n| if n == other { keep } else { n });
        if nodes[0] == nodes[1] {
            return Err("Merging would create a self-loop".into());
        }
        if !neighbors.insert(nodes[usize::from(nodes[0] == keep)]) {
            return Err("Merging would create parallel edges".into());
        }
    }
    crate::edit::merge(mesh, &mut labeled.skeleton, edge, keep);
    labeled.labels.remove(edge);
    Ok(())
}

#[cfg(test)]
mod tests;
