use std::collections::HashSet;

pub mod fixed;
pub mod fluid;

pub type Float = ordered_float::OrderedFloat<f64>;
pub const ZERO: Float = ordered_float::OrderedFloat(0.0);

pub trait Grapff<V, E> {
    #[must_use]
    fn neighbors(&self, v: V) -> Vec<V>;

    #[must_use]
    fn shortest_path(&self, a: V, b: V, w: impl Fn(E) -> Float) -> Option<(Vec<V>, Float)>;

    #[must_use]
    fn shortest_path_heuristic(&self, a: V, b: V, w: impl Fn(E) -> Float, h: impl Fn((V, V)) -> Float) -> Option<(Vec<V>, Float)>;

    #[must_use]
    fn connected_component(&self, v: V) -> HashSet<V>;

    #[must_use]
    fn connected_components(&self, vs: &[V]) -> Vec<HashSet<V>>;
}
