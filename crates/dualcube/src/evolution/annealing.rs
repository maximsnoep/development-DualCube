//! Simulated annealing of the loop structure (see `Solution::anneal`): an alternative to the evolution
//! (`Solution::evolve`) with one solution instead of a population.
//!
//! Every step, a batch of children of the current solution is made with the mutations of the evolution (in parallel,
//! screened and embedded as there). The best child replaces the current solution if it is at least as good, or else
//! with probability exp(difference / temperature). The temperature falls linearly from `temperature` to 0, and the
//! mix of the mutations follows it: from that of growth (hot: mostly additions) to that of shaping and pruning (cold:
//! moving, merging and removing loops), with the adaptive choice of the evolution (see `MutationSelection`). After
//! `restart` steps without a new best, the current solution goes back to the best. The result is the best solution
//! seen, embedded as in the evolution, and never worse than the input.

use super::*;

/// Parameters of the simulated annealing of the loops.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnnealParams {
    /// Children per step (made and evaluated in parallel; the best is the candidate).
    pub offspring: usize,
    /// Number of steps.
    pub steps: usize,
    /// Starting temperature (a difference of quality).
    pub temperature: f64,
    /// Go back to the best solution after this many steps without a new best (0: never).
    pub restart: usize,
    /// The mutations: `targeted`, `interest`, `screen` and `enabled` (see `EvolutionParams`).
    pub mutations: EvolutionParams,
}

impl Default for AnnealParams {
    fn default() -> Self {
        Self {
            offspring: 15,
            steps: 120,
            temperature: 0.03,
            restart: 40,
            mutations: EvolutionParams::default(),
        }
    }
}

// Children whose acceptance is less likely than this are not accepted (and not prepared to be parents).
const MIN_ACCEPTANCE: f64 = 0.01;

impl Solution {
    /// Simulated annealing of the loop structure (see the module documentation), reporting to (and stoppable through)
    /// the given monitor.
    pub fn anneal(
        &self,
        params: &AnnealParams,
        monitor: &EvolutionMonitor,
    ) -> Result<Self, SolutionError> {
        let result = self
            .anneal_phase(params, monitor)
            .map(|result| self.better_of(result));
        monitor.finish();
        result
    }

    // The annealing (see `anneal`): its best solution, embedded, even if it is worse than the input.
    pub(crate) fn anneal_phase(
        &self,
        params: &AnnealParams,
        monitor: &EvolutionMonitor,
    ) -> Result<Self, SolutionError> {
        let initial_quality = self.get_quality().ok_or(SolutionError::NoPrimal)?;
        let started_at = Instant::now();
        profile::reset();
        monitor.begin_phase(initial_quality, || self.clone());
        monitor.note_best(initial_quality);
        let (seed, seed_quality) = self.evolution_seed(initial_quality);
        let context = MutationContext::new(&seed);
        let mut seen = HashSet::from([seed.loop_signature()]);
        let mut selection = monitor.take_selection("loops");
        let mut stats = monitor.mutation_stats_map();
        let mut current = (seed, seed_quality);
        let mut best = current.clone();
        let mut stale = 0;

        for step in 0..params.steps {
            let hot = 1. - step as f64 / params.steps as f64;
            let temperature = params.temperature * hot;
            // (The mutations that are switched on can change while running.)
            let (mutations, _, _) = params.mutations.live(monitor);
            let mutations = &mutations;
            // The mix of the mutations, from that of growth (hot) to that of shaping and pruning (cold).
            let [growth, shaping, pruning] =
                [LoopPhase::Growth, LoopPhase::Optimization, LoopPhase::Pruning]
                    .map(|phase| mutations.mutation_prior(phase));
            let prior = growth
                .iter()
                .zip(&shaping)
                .zip(&pruning)
                .map(|((&(name, g), &(_, s)), &(_, p))| (name, hot * g + (1. - hot) * (s + p) / 2.))
                .collect_vec();
            let chances = selection.chances(&prior);
            report_chances(monitor, &selection, &chances);

            let attempts = (0..params.offspring.max(1))
                .collect_vec()
                .into_par()
                .filter_map(|_| {
                    let op = choose_mutation(&chances)?;
                    Some((op, profile::timed(3, || current.0.mutate(mutations, &context, op))))
                })
                .collect::<Vec<_>>();
            // Tries and improvements of the best per mutation (see `MutationSelection`); failed tries count too.
            let mut tries: HashMap<&'static str, (usize, usize)> = HashMap::new();
            for (op, child) in &attempts {
                tries.entry(child.as_ref().map_or(*op, |(_, op)| *op)).or_default().0 += 1;
            }
            let children = attempts.into_iter().filter_map(|(_, child)| child).collect_vec();
            for (_, op) in &children {
                stats.entry(op).or_default()[0] += 1;
            }
            let threshold = current.1 + temperature * MIN_ACCEPTANCE.ln();
            let offspring = Self::evaluate_offspring(children, mutations, &mut seen, threshold);
            for (_, quality, op) in &offspring {
                let entry = stats.entry(op).or_default();
                entry[1] += 1;
                if *quality > current.1 {
                    entry[2] += 1;
                }
                if *quality > best.1 + 1e-12 {
                    entry[3] += 1;
                    tries.entry(op).or_default().1 += 1;
                }
            }
            selection.record(tries);

            if let Some((child, quality, _)) = offspring
                .into_iter()
                .max_by(|(_, a, _), (_, b, _)| a.total_cmp(b))
                && quality >= threshold
                && (quality >= current.1
                    || rand::random::<f64>() < ((quality - current.1) / temperature).exp())
            {
                current = (child, quality);
            }
            if current.1 > best.1 + 1e-12 {
                best = current.clone();
                stale = 0;
                monitor.offer_best(best.1, || self.embedded_copy(&best.0));
            } else {
                stale += 1;
                if params.restart > 0 && stale >= params.restart {
                    current = best.clone();
                    stale = 0;
                }
            }
            monitor.record_mutations(&stats);
            monitor.record(GenerationStats {
                generation: step + 1,
                best: best.1,
                mean: current.1,
                loops: current.0.loops.len(),
                phase: if hot > 0.5 {
                    LoopPhase::Growth
                } else {
                    LoopPhase::Optimization
                },
                layout: false,
                seconds: started_at.elapsed().as_secs_f64(),
            });
            info!(
                "anneal: step {} temperature={temperature:.4} current={} best={} loops={}",
                step + 1,
                current.1,
                best.1,
                current.0.loops.len()
            );
            if monitor.stopping() {
                break;
            }
        }
        monitor.put_selection("loops", selection);
        info!("anneal: mutation statistics (generated, evaluated, better than current, new best): {stats:?}");
        info!(
            "anneal: best solution with quality {} and {} loops after {:?}",
            best.1,
            best.0.loops.len(),
            started_at.elapsed()
        );
        info!("anneal: CPU time per phase: {}", profile::report());
        Ok(self.embed_result([best.0]))
    }
}
