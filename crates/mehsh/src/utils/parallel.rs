//! One thread pool for all parallel computations (of orx-parallel, with `.with_pool(pool())`): its threads are
//! started once, instead of new threads for every parallel computation (orx-parallel's default). Parallel
//! computations inside parallel computations share the pool (rayon's work stealing). Its number of threads is that of
//! `ORX_NUM_THREADS` if set, else the number of logical processors.

use std::sync::LazyLock;

static POOL: LazyLock<rayon_core::ThreadPool> = LazyLock::new(|| {
    let threads = std::env::var("ORX_NUM_THREADS")
        .ok()
        .and_then(|n| n.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, usize::from));
    rayon_core::ThreadPoolBuilder::new()
        .num_threads(threads)
        .thread_name(|i| format!("parallel {i}"))
        .build()
        .expect("a thread pool")
});

/// The thread pool for the parallel computations (see the module documentation).
#[must_use]
pub fn pool() -> &'static rayon_core::ThreadPool {
    &POOL
}

/// Drops the value on a thread of the pool, such that the caller does not wait for it (freeing large structures,
/// e.g. the solutions that are not kept, takes long).
pub fn drop_later<T: Send + 'static>(value: T) {
    pool().spawn(move || drop(value));
}
