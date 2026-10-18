//! Adaptive choice of the mutations (of the loops, see `Solution::anneal`, and of the layout, see
//! `Solution::optimize_layout`).

use crate::prelude::*;
use std::collections::VecDeque;

/// The number of past generations whose improvements determine the chances of the mutations.
pub const SELECTION_WINDOW: usize = 30;
/// The smallest chance of an enabled mutation.
pub const MIN_CHANCE: f64 = 0.01;

/// Pseudo-tries (at the mean success rate) per mutation: mutations with few tries get about the mean success rate.
const SELECTION_PSEUDO_TRIES: f64 = 10.;

/// Adaptive choice of the mutations (adaptive operator selection, as in probability matching): the chance of a
/// mutation is proportional to its prior weight (e.g., of the temperature) times its success rate (the fraction of its
/// tries that improved the best solution) in the last `SELECTION_WINDOW` generations, but at least `MIN_CHANCE`. The
/// rate does not depend on how often a mutation was tried; mutations with few tries are pulled towards the mean rate
/// (`SELECTION_PSEUDO_TRIES`). Without successes in the window, the chances follow the prior weights.
#[derive(Default)]
pub struct MutationSelection {
    // Per generation: the tries and the improvements of the best solution per mutation.
    window: VecDeque<HashMap<&'static str, (usize, usize)>>,
}

impl MutationSelection {
    /// Record the tries and improvements of the best solution per mutation in a generation.
    pub fn record(&mut self, generation: HashMap<&'static str, (usize, usize)>) {
        self.window.push_back(generation);
        while self.window.len() > SELECTION_WINDOW {
            self.window.pop_front();
        }
    }

    /// The chances of the given mutations, with their tries and improvements in the window (for reporting).
    #[must_use]
    pub fn report(&self, chances: &[(&'static str, f64)]) -> Vec<(&'static str, MutationChance)> {
        chances
            .iter()
            .map(|&(name, chance)| {
                let (tries, successes) = self
                    .window
                    .iter()
                    .filter_map(|generation| generation.get(name))
                    .fold((0, 0), |(t, s), &(tries, successes)| {
                        (t + tries, s + successes)
                    });
                (
                    name,
                    MutationChance {
                        chance,
                        tries,
                        successes,
                    },
                )
            })
            .collect()
    }

    /// The chances of the mutations with a positive prior weight (the others are switched off).
    #[must_use]
    pub fn chances(&self, prior: &[(&'static str, f64)]) -> Vec<(&'static str, f64)> {
        let enabled = prior
            .iter()
            .filter(|(_, weight)| *weight > 0.)
            .copied()
            .collect_vec();
        if enabled.is_empty() {
            return vec![];
        }
        let counts = enabled
            .iter()
            .map(|&(name, _)| {
                self.window
                    .iter()
                    .filter_map(|generation| generation.get(name))
                    .fold((0., 0.), |(t, s), &(tries, successes)| {
                        (t + tries as f64, s + successes as f64)
                    })
            })
            .collect_vec();
        let (tries, successes) = counts.iter().fold((0., 0.), |(t, s), &(tries, successes)| {
            (t + tries, s + successes)
        });
        let weights = if successes > 0. {
            let mean = successes / tries;
            enabled
                .iter()
                .zip(&counts)
                .map(|(&(_, prior), &(t, s))| {
                    prior * (s + SELECTION_PSEUDO_TRIES * mean) / (t + SELECTION_PSEUDO_TRIES)
                })
                .collect_vec()
        } else {
            enabled.iter().map(|&(_, weight)| weight).collect_vec()
        };
        let total: f64 = weights.iter().sum();
        // At least `MIN_CHANCE` each (if that fits), the rest in proportion to the weights.
        let floor = MIN_CHANCE.min(1. / enabled.len() as f64);
        let rest = 1. - floor * enabled.len() as f64;
        enabled
            .iter()
            .zip(weights)
            .map(|(&(name, _), weight)| (name, floor + rest * weight / total))
            .collect()
    }
}

/// The chance of a mutation to be chosen, and its tries and improvements of the best in the last `SELECTION_WINDOW`
/// generations (on which the chance is based, see `MutationSelection`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MutationChance {
    pub chance: f64,
    pub tries: usize,
    pub successes: usize,
}

/// A random element, with a chance proportional to its weight (its penalty), plus a small floor
/// (`ELEMENT_FLOOR` of the mean weight) so that every element can still be chosen, but good ones hardly ever are.
pub(crate) fn pick_weighted<T: Copy>(items: impl IntoIterator<Item = (T, f64)>) -> Option<T> {
    const ELEMENT_FLOOR: f64 = 0.02;
    let items = items
        .into_iter()
        .map(|(item, weight)| {
            (
                item,
                if weight.is_finite() {
                    weight.max(0.)
                } else {
                    0.
                },
            )
        })
        .collect_vec();
    if items.is_empty() {
        return None;
    }
    let mean = items.iter().map(|(_, w)| w).sum::<f64>() / items.len() as f64;
    let floor = if mean > 0. { ELEMENT_FLOOR * mean } else { 1. };
    let total: f64 = items.iter().map(|(_, w)| w + floor).sum();
    let mut pick = rand::random::<f64>() * total;
    for &(item, weight) in &items {
        pick -= weight + floor;
        if pick <= 0. {
            return Some(item);
        }
    }
    items.last().map(|(item, _)| *item)
}

/// A random mutation with the given chances.
#[must_use]
pub fn choose_mutation(chances: &[(&'static str, f64)]) -> Option<&'static str> {
    let total: f64 = chances.iter().map(|(_, c)| c).sum();
    let mut pick = rand::random::<f64>() * total;
    for &(name, chance) in chances {
        pick -= chance;
        if pick <= 0. {
            return Some(name);
        }
    }
    chances.last().map(|(name, _)| *name)
}

#[cfg(test)]
mod selection_tests {
    use super::{MIN_CHANCE, MutationSelection};
    use rustc_hash::FxHashMap as HashMap;

    #[test]
    fn chances_follow_success_rates_with_a_floor() {
        let prior = [("a", 0.5), ("b", 0.3), ("c", 0.2), ("off", 0.)];
        let mut selection = MutationSelection::default();
        // Without improvements: the prior.
        let chances = selection.chances(&prior);
        assert_eq!(chances.len(), 3);
        assert!(chances[0].1 > chances[1].1 && chances[1].1 > chances[2].1);
        // `a` tried often with few successes, `b` rarely but with a high rate, `c` never succeeds.
        for _ in 0..30 {
            selection.record(HashMap::from_iter([
                ("a", (20, 1)),
                ("b", (2, 1)),
                ("c", (5, 0)),
            ]));
        }
        let chances: HashMap<_, _> = selection.chances(&prior).into_iter().collect();
        assert!(chances["b"] > chances["a"], "{chances:?}");
        assert!(chances["c"] >= MIN_CHANCE - 1e-12, "{chances:?}");
        assert!((chances.values().sum::<f64>() - 1.).abs() < 1e-9);
        assert!(!chances.contains_key("off"));
    }
}
