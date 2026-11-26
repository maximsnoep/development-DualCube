//! Fixed-boundary path labeling. Costs are indexed by +X, -X, +Y, -Y, +Z, -Z.

use super::Port;

/// Available outward ports at the endpoints, and allowed displacement masks.
/// All ports/masks allowed gives a free path; constrain either or both ends.
#[derive(Clone, Debug)]
pub struct Constraints {
    pub start_ports: u8,
    pub end_ports: u8,
    pub masks: [bool; 64],
}

impl Default for Constraints {
    fn default() -> Self {
        Self {
            start_ports: 63,
            end_ports: 63,
            masks: [true; 64],
        }
    }
}

/// Relative coordinate order from the start to the end of a path.
#[derive(Clone, Copy, Debug)]
pub enum Order {
    Any,
    Equal,
    Increasing,
    Decreasing,
}

impl Constraints {
    pub fn with_orders(mut self, orders: [Order; 3]) -> Self {
        for mask in 0..64 {
            self.masks[mask] &= orders.iter().enumerate().all(|(axis, order)| {
                match ((mask >> (axis * 2)) & 3, order) {
                    (_, Order::Any) | (3, _) => true,
                    (0, Order::Equal) | (1, Order::Increasing) | (2, Order::Decreasing) => true,
                    _ => false,
                }
            });
        }
        self
    }
}

#[derive(Clone, Debug)]
pub struct Solution {
    pub ports: Vec<Port>,
    pub cost: f64,
    pub mask: u8,
}

/// Minimize additive angle costs without removing existing boundaries.
/// The final port is reversed when checking the end junction's available ports.
/// Complexity: O(n * 6 * 64 * 6), with 384 states per boundary.
pub fn optimize(costs: &[[f64; 6]], constraints: &Constraints) -> Option<Solution> {
    optimize_through(costs, constraints, &[])
}

/// As [`optimize`], for a path through junctions: `taken[i]` are the ports already in use at
/// the node before boundary `i` (missing entries and `taken[0]` count as none).
pub fn optimize_through(
    costs: &[[f64; 6]],
    constraints: &Constraints,
    taken: &[u8],
) -> Option<Solution> {
    if costs.is_empty() {
        return None;
    }
    let mut previous = [f64::INFINITY; 384];
    let mut back = vec![[usize::MAX; 384]; costs.len()];
    for d in 0..6 {
        if constraints.start_ports & (1 << d) != 0 && costs[0][d].is_finite() {
            previous[d * 64 + (1 << d)] = costs[0][d];
        }
    }
    for (i, cost) in costs.iter().enumerate().skip(1) {
        let mut next = [f64::INFINITY; 384];
        for (state, &value) in previous.iter().enumerate() {
            if !value.is_finite() {
                continue;
            }
            let (d, mask) = (state / 64, state % 64);
            let taken = taken.get(i).copied().unwrap_or(0);
            if taken & (1 << (d ^ 1)) != 0 {
                continue;
            }
            for (out, &angle) in cost.iter().enumerate() {
                if out == (d ^ 1) || taken & (1 << out) != 0 || !angle.is_finite() {
                    continue;
                }
                let target = out * 64 + (mask | (1 << out));
                if value + angle < next[target] {
                    next[target] = value + angle;
                    back[i][target] = state;
                }
            }
        }
        previous = next;
    }
    let (mut state, &cost) = previous
        .iter()
        .enumerate()
        .filter(|(s, c)| {
            c.is_finite()
                && constraints.end_ports & (1 << ((s / 64) ^ 1)) != 0
                && constraints.masks[s % 64]
        })
        .min_by(|a, b| a.1.total_cmp(b.1))?;
    let mask = (state % 64) as u8;
    let mut ports = vec![Port::PositiveX; costs.len()];
    for i in (0..costs.len()).rev() {
        ports[i] = Port::ALL[state / 64];
        if i > 0 {
            state = back[i][state];
        }
    }
    Some(Solution { ports, cost, mask })
}

/// Kept boundaries (indices into the path) with their ports.
#[derive(Clone, Debug)]
pub struct Selection {
    pub kept: Vec<usize>,
    pub ports: Vec<Port>,
    pub cost: f64,
}

/// End and patch costs of [`select`].
pub struct Ends<'a> {
    /// The patch before the first kept boundary `i` with port `q`: `first(i, q)`.
    pub first: &'a dyn Fn(usize, usize) -> f64,
    /// The patch after the last kept boundary: `last(i, q)`.
    pub last: &'a dyn Fn(usize, usize) -> f64,
    /// The merged patch between kept boundaries `j < i`: `patch(j, i)`.
    pub patch: &'a dyn Fn(usize, usize) -> f64,
}

/// Keep at least one of the boundaries of a path and give them ports, for the least cost:
/// `costs[i][q]` per boundary `i` kept with port `q`, plus the patches of `ends`. The patch
/// between two kept boundaries cannot use one port twice, so the path never reverses. Infinite
/// costs forbid. Complexity: O(n² · 384 · 6).
pub fn select(costs: &[[f64; 6]], constraints: &Constraints, ends: &Ends<'_>) -> Option<Selection> {
    let n = costs.len();
    if n == 0 || costs.iter().flatten().any(|c| c.is_nan()) {
        return None;
    }
    // Indexed by the last kept boundary and its port * 64 + the mask of kept ports.
    let mut value = vec![[f64::INFINITY; 384]; n];
    let mut back = vec![[(usize::MAX, 0); 384]; n];
    for i in 0..n {
        for q in 0..6 {
            if constraints.start_ports & (1 << q) != 0 {
                value[i][q * 64 + (1 << q)] = (ends.first)(i, q) + costs[i][q];
            }
        }
        for j in 0..i {
            let shape = (ends.patch)(j, i);
            for state in 0..384 {
                let current = value[j][state];
                if !current.is_finite() {
                    continue;
                }
                let (p, mask) = (state / 64, state % 64);
                for q in 0..6 {
                    if q == p ^ 1 {
                        continue;
                    }
                    let next = current + shape + costs[i][q];
                    let target = q * 64 + (mask | (1 << q));
                    if next < value[i][target] {
                        value[i][target] = next;
                        back[i][target] = (j, state);
                    }
                }
            }
        }
    }
    let (mut i, mut state, cost) = (0..n)
        .flat_map(|i| (0..384).map(move |s| (i, s)))
        .filter(|&(i, s)| {
            value[i][s].is_finite()
                && constraints.end_ports & (1 << ((s / 64) ^ 1)) != 0
                && constraints.masks[s % 64]
        })
        .map(|(i, s)| (i, s, value[i][s] + (ends.last)(i, s / 64)))
        .filter(|s| s.2.is_finite())
        .min_by(|a, b| a.2.total_cmp(&b.2))?;
    let (mut kept, mut ports) = (Vec::new(), Vec::new());
    loop {
        kept.push(i);
        ports.push(Port::ALL[state / 64]);
        let (j, previous) = back[i][state];
        if j == usize::MAX {
            break;
        }
        (i, state) = (j, previous);
    }
    kept.reverse();
    ports.reverse();
    Some(Selection { kept, ports, cost })
}
