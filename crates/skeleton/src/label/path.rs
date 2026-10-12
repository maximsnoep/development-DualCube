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
    // TODO: Jointly select boundaries and labels over bounded candidate intervals, including
    // boundary removal; precompute each candidate patch/label cost once before the DP.
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
            for (out, &angle) in cost.iter().enumerate() {
                if out == (d ^ 1) || !angle.is_finite() {
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
