//! Adaptive large neighbourhood search over valid moves.
//!
//! Each step proposes a batch of candidates in parallel: every candidate picks an operator by
//! roulette over adaptive weights (never below a floor) and a random target. The best candidate
//! is kept if it improves the score; operators whose candidates improve gain weight. The score
//! is an external objective (e.g. the primalized quality) or, without one, restricted-label
//! fidelity, which also guides the proposals. Validity never depends on the score.

use super::fidelity::{Fidelity, Measure, Tables};
use super::moves::{self, Chain};
use super::redraw::{self, RedrawParams};
use super::relax::{self, RelaxParams};
use crate::{LabeledSkeleton, SkeletonResult};
use dualcube_types::prelude::*;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operator {
    /// Move one boundary within the union of its two patches.
    Redraw,
    /// Best labels for one chain under fixed geometry.
    Relabel,
    /// Dissolve one boundary of a chain, then relabel it.
    Shorten,
    /// Add a boundary parallel to one of a chain, redraw and relabel.
    Lengthen,
    /// Merge a leaf into its neighbour.
    RemoveLeaf,
    /// Split a disk off a patch as a new leaf at its best free port.
    AddLeaf,
}

impl Operator {
    pub const ALL: [Self; 6] = [
        Self::Redraw,
        Self::Relabel,
        Self::Shorten,
        Self::Lengthen,
        Self::RemoveLeaf,
        Self::AddLeaf,
    ];
}

/// What runs once before the search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitialPass {
    None,
    /// Redraw every boundary once, one at a time.
    Redraw,
    /// Move all boundaries at once by isotopies.
    Relax,
}

#[derive(Clone, Debug)]
pub struct Params {
    pub measure: Measure,
    /// Score cost of every node, in units of the area-weighted gain.
    pub node_cost: f64,
    pub redraw: RedrawParams,
    pub relax: RelaxParams,
    pub initial: InitialPass,
    /// Candidates proposed and scored in parallel per step.
    pub parallel: usize,
    pub iterations: usize,
    /// Stop after this many steps without improvement.
    pub patience: usize,
    pub seconds: f64,
    pub seed: u64,
    /// Lower bound on an operator's selection weight.
    pub floor: f64,
    /// How fast weights follow recent rewards.
    pub reaction: f64,
    pub operators: Vec<Operator>,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            measure: Measure::Alignment,
            node_cost: 0.002,
            redraw: RedrawParams {
                smoothness: 2.,
                measure: Measure::Alignment,
                rounds: 2,
            },
            relax: RelaxParams::default(),
            initial: InitialPass::Redraw,
            parallel: 6,
            iterations: 400,
            patience: 150,
            seconds: 120.,
            seed: 1,
            floor: 0.1,
            reaction: 0.2,
            operators: Operator::ALL.to_vec(),
        }
    }
}

/// Higher is better; `None` rejects the candidate (e.g. it could not be primalized).
pub type Objective<'a> = &'a (dyn Fn(&LabeledSkeleton) -> Option<f64> + Sync);

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub score: [f64; 2],
    pub initial: Fidelity,
    pub after_sweep: Fidelity,
    /// Score of the relaxed skeleton, kept or not.
    pub relaxed: Option<f64>,
    pub result: Fidelity,
    pub naive: Fidelity,
    pub nodes: [usize; 2],
    pub steps: usize,
    /// Per operator in `Operator::ALL` order: attempts, valid candidates, and candidates that
    /// were the best improvement of their batch so far.
    pub operators: [[usize; 3]; 6],
    pub seconds: f64,
}

/// Small deterministic generator (`SplitMix64`).
pub(crate) struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    pub fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

struct State {
    labeled: LabeledSkeleton,
    tables: Tables,
    fidelity: Fidelity,
    score: f64,
}

fn state(
    mesh: &Mesh<INPUT>,
    labeled: LabeledSkeleton,
    params: &Params,
    objective: Option<Objective<'_>>,
) -> Option<State> {
    labeled.validate_labels().ok()?;
    let tables = Tables::build(mesh, &labeled.skeleton).ok()?;
    let fidelity = tables.evaluate(&labeled.skeleton, &labeled.labels);
    let score = match objective {
        Some(objective) => objective(&labeled)?,
        None => {
            fidelity.gain(params.measure) - params.node_cost * labeled.skeleton.nodes.len() as f64
        }
    };
    Some(State {
        labeled,
        tables,
        fidelity,
        score,
    })
}

/// `f(0..n)` on scoped threads, in order.
fn parallel<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    if n <= 1 {
        return (0..n).map(&f).collect();
    }
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..n)
            .map(|k| {
                let f = &f;
                scope.spawn(move || f(k))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

/// Relabel a chain with fresh tables for the candidate's geometry.
fn relabeled(
    mesh: &Mesh<INPUT>,
    mut labeled: LabeledSkeleton,
    chain: &Chain,
    measure: Measure,
) -> Option<LabeledSkeleton> {
    let tables = Tables::build(mesh, &labeled.skeleton).ok()?;
    labeled.labels = moves::relabel(&tables, &labeled, chain, measure)?.0;
    Some(labeled)
}

/// A random valid labeling of a chain.
fn random_labels(
    labeled: &LabeledSkeleton,
    chain: &Chain,
    rng: &mut Rng,
) -> Option<LabeledSkeleton> {
    let noise: Vec<[f64; 6]> = (0..chain.edges.len())
        .map(|_| std::array::from_fn(|_| rng.unit()))
        .collect();
    let (labels, _) = moves::relabel_by(labeled, chain, |i, p| noise[i][p], |_, _| 0.)?;
    Some(LabeledSkeleton {
        skeleton: labeled.skeleton.clone(),
        labels,
    })
}

fn propose(
    mesh: &Mesh<INPUT>,
    current: &State,
    operator: Operator,
    params: &Params,
    rng: &mut Rng,
) -> Option<LabeledSkeleton> {
    let labeled = &current.labeled;
    let skeleton = &labeled.skeleton;
    let pick = |rng: &mut Rng, chains: Vec<Chain>| {
        (!chains.is_empty()).then(|| {
            let i = rng.below(chains.len());
            chains.into_iter().nth(i).unwrap()
        })
    };
    match operator {
        Operator::Redraw => {
            let edges: Vec<_> = skeleton.edges.keys().collect();
            let edge = *edges.get(rng.below(edges.len().max(1)))?;
            redraw::redraw(mesh, labeled, edge, &params.redraw)
        }
        Operator::Relabel => {
            let chain = pick(rng, moves::chains(skeleton))?;
            // Half of the proposals ignore the fidelity tables, which misjudge labels.
            let candidate = if rng.below(2) == 0 {
                random_labels(labeled, &chain, rng)?
            } else {
                let (labels, _) = moves::relabel(&current.tables, labeled, &chain, params.measure)?;
                LabeledSkeleton {
                    skeleton: skeleton.clone(),
                    labels,
                }
            };
            let changed = chain
                .edges
                .iter()
                .any(|&e| !moves::same_label(candidate.labels[e], labeled.labels[e]));
            changed.then_some(candidate)
        }
        Operator::Shorten => {
            let chains: Vec<_> = moves::chains(skeleton)
                .into_iter()
                .filter(|c| c.edges.len() >= 2)
                .collect();
            let mut chain = pick(rng, chains)?;
            let i = rng.below(chain.edges.len());
            // Merge an interior node into its neighbour along the chain.
            let (keep, gone) = if i == 0 {
                (chain.nodes[0], chain.nodes[1])
            } else {
                (chain.nodes[i + 1], chain.nodes[i])
            };
            let mut candidate = labeled.clone();
            moves::dissolve(mesh, &mut candidate, chain.edges[i], keep).ok()?;
            chain.edges.remove(i);
            chain.nodes.retain(|&n| n != gone);
            relabeled(mesh, candidate, &chain, params.measure)
        }
        Operator::Lengthen => {
            let mut chain = pick(rng, moves::chains(skeleton))?;
            let i = rng.below(chain.edges.len());
            let edge = chain.edges[i];
            let side = rng.below(2);
            let (split, added) =
                crate::label::subdivide::collar(mesh, skeleton, edge, side).ok()?;
            // `collar` keeps `edge` at nodes[0] and adds `added` towards nodes[1].
            let middle = split.edges[added].nodes[0];
            let towards_end = split.edges[edge].nodes[0] == chain.nodes[i];
            let (first, second) = if towards_end {
                (edge, added)
            } else {
                (added, edge)
            };
            chain.edges.splice(i..=i, [first, second]);
            chain.nodes.insert(i + 1, middle);
            let mut labels = labeled.labels.clone();
            labels.insert(added, labeled.labels[edge]);
            let candidate = LabeledSkeleton {
                skeleton: split,
                labels,
            };
            let mut candidate = relabeled(mesh, candidate, &chain, params.measure)?;
            for e in [first, second] {
                if let Some(next) = redraw::redraw(mesh, &candidate, e, &params.redraw) {
                    candidate = next;
                }
            }
            relabeled(mesh, candidate, &chain, params.measure)
        }
        Operator::RemoveLeaf => {
            let degree = moves::degrees(skeleton);
            let leaves: Vec<_> = skeleton
                .edges
                .iter()
                .filter_map(|(id, e)| {
                    let [a, b] = e.nodes;
                    match (degree[a], degree[b]) {
                        (1, d) if d > 1 => Some((id, b)),
                        (d, 1) if d > 1 => Some((id, a)),
                        _ => None,
                    }
                })
                .collect();
            let &(edge, keep) = leaves.get(rng.below(leaves.len().max(1)))?;
            let mut candidate = labeled.clone();
            moves::dissolve(mesh, &mut candidate, edge, keep).ok()?;
            Some(candidate)
        }
        Operator::AddLeaf => {
            let degree = moves::degrees(skeleton);
            let nodes: Vec<_> = degree
                .iter()
                .filter(|(_, d)| **d < 6)
                .map(|(n, _)| n)
                .collect();
            let node = *nodes.get(rng.below(nodes.len().max(1)))?;
            // Best port by restricted fidelity among the free ones.
            crate::label::Port::ALL
                .into_iter()
                .filter_map(|port| redraw::add_leaf(mesh, labeled, node, port, &params.redraw))
                .filter_map(|c| state(mesh, c, params, None))
                .max_by(|a, b| a.score.total_cmp(&b.score))
                .map(|s| s.labeled)
        }
    }
}

/// Initial pass: redraw every boundary once, in parallel batches, keeping the best
/// improvement of each batch.
fn sweep(
    mesh: &Mesh<INPUT>,
    mut current: State,
    params: &Params,
    objective: Option<Objective<'_>>,
) -> State {
    let edges: Vec<_> = current.labeled.skeleton.edges.keys().collect();
    for batch in edges.chunks(params.parallel.max(1)) {
        let candidates = parallel(batch.len(), |k| {
            let edge = batch[k];
            if !current.labeled.skeleton.edges.contains_key(edge) {
                return None;
            }
            let candidate = redraw::redraw(mesh, &current.labeled, edge, &params.redraw)?;
            state(mesh, candidate, params, objective)
        });
        if let Some(best) = candidates
            .into_iter()
            .flatten()
            .max_by(|a, b| a.score.total_cmp(&b.score))
            && best.score > current.score
        {
            current = best;
        }
    }
    current
}

fn roulette(weights: &[f64], floor: f64, rng: &mut Rng) -> usize {
    let total: f64 = weights.iter().map(|w| w.max(floor)).sum();
    let mut r = rng.unit() * total;
    for (i, w) in weights.iter().enumerate() {
        r -= w.max(floor);
        if r <= 0. {
            return i;
        }
    }
    weights.len() - 1
}

pub fn search(
    mesh: &Mesh<INPUT>,
    labeled: &LabeledSkeleton,
    params: &Params,
    objective: Option<Objective<'_>>,
) -> SkeletonResult<(LabeledSkeleton, Stats)> {
    let start = Instant::now();
    let mut stats = Stats::default();
    let mut current =
        state(mesh, labeled.clone(), params, objective).ok_or("The start cannot be scored")?;
    stats.score[0] = current.score;
    stats.initial = current.fidelity;
    stats.naive = current.tables.naive;
    stats.nodes[0] = current.labeled.skeleton.nodes.len();
    match params.initial {
        InitialPass::None => {}
        InitialPass::Redraw => current = sweep(mesh, current, params, objective),
        InitialPass::Relax => {
            if let Some(relaxed) = relax::relax(mesh, &current.labeled, &params.relax)
                .and_then(|l| state(mesh, l, params, objective))
            {
                stats.relaxed = Some(relaxed.score);
                if relaxed.score > current.score {
                    current = relaxed;
                }
            }
        }
    }
    stats.after_sweep = current.fidelity;
    let mut weights = vec![1.; params.operators.len()];
    let mut stale = 0;
    for step in 0..params.iterations {
        if stale >= params.patience
            || start.elapsed().as_secs_f64() > params.seconds
            || params.operators.is_empty()
        {
            break;
        }
        stats.steps += 1;
        let results = parallel(params.parallel.max(1), |k| {
            let mut rng =
                Rng::new(params.seed ^ ((step as u64) << 16 | k as u64).wrapping_mul(0x9e37_79b9));
            let chosen = roulette(&weights, params.floor, &mut rng);
            let candidate = propose(mesh, &current, params.operators[chosen], params, &mut rng)
                .and_then(|c| state(mesh, c, params, objective));
            (chosen, candidate)
        });
        let mut best: Option<State> = None;
        let reward = |w: &mut f64, value: f64| {
            *w = (1. - params.reaction) * *w + params.reaction * value;
        };
        for (chosen, candidate) in results {
            let slot = Operator::ALL
                .iter()
                .position(|&o| o == params.operators[chosen])
                .unwrap();
            stats.operators[slot][0] += 1;
            let Some(candidate) = candidate else {
                reward(&mut weights[chosen], 0.);
                continue;
            };
            stats.operators[slot][1] += 1;
            let improved = candidate.score > current.score + 1e-12;
            reward(&mut weights[chosen], f64::from(u8::from(improved)));
            if improved && best.as_ref().is_none_or(|b| candidate.score > b.score) {
                stats.operators[slot][2] += 1;
                best = Some(candidate);
            }
        }
        if let Some(best) = best {
            current = best;
            stale = 0;
        } else {
            stale += 1;
        }
    }
    stats.score[1] = current.score;
    stats.result = current.fidelity;
    stats.nodes[1] = current.labeled.skeleton.nodes.len();
    stats.seconds = start.elapsed().as_secs_f64();
    Ok((current.labeled, stats))
}

/// Random valid moves regardless of score, to sample labelings around a start. Relabeling
/// picks a random valid labeling of a chain instead of the best one.
pub fn perturb(
    mesh: &Mesh<INPUT>,
    labeled: &LabeledSkeleton,
    steps: usize,
    params: &Params,
    seed: u64,
) -> SkeletonResult<LabeledSkeleton> {
    let mut current = state(mesh, labeled.clone(), params, None).ok_or("Invalid start")?;
    let mut rng = Rng::new(seed);
    let (mut done, mut tries) = (0, 0);
    while done < steps && tries < 20 * steps && !params.operators.is_empty() {
        tries += 1;
        let operator = params.operators[rng.below(params.operators.len())];
        let candidate = if operator == Operator::Relabel {
            let chains = moves::chains(&current.labeled.skeleton);
            if chains.is_empty() {
                continue;
            }
            let chain = &chains[rng.below(chains.len())];
            random_labels(&current.labeled, chain, &mut rng)
        } else {
            propose(mesh, &current, operator, params, &mut rng)
        };
        if let Some(candidate) = candidate
            && let Some(next) = state(mesh, candidate, params, None)
        {
            current = next;
            done += 1;
        }
    }
    Ok(current.labeled)
}
