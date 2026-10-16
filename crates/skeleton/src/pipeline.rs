//! Sequential initialization, retaining completed stages when a later one fails.

use crate::{
    DualLoops, LabeledSkeleton, RealizedSkeleton, Skeleton, draw_loops, label, optimize,
    place_cubes, skeletonize,
};
use dualcube_types::prelude::{INPUT, Mesh};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Extraction,
    Labeling,
    CubePlacement,
    Optimization,
    LoopDrawing,
}

impl std::fmt::Display for Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Extraction => "skeleton extraction",
            Self::Labeling => "labeling",
            Self::CubePlacement => "cube placement",
            Self::Optimization => "optimization",
            Self::LoopDrawing => "loop drawing",
        })
    }
}

#[derive(Debug)]
pub struct Failure {
    pub stage: Stage,
    pub error: Box<dyn std::error::Error + Send + Sync>,
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.stage, self.error)
    }
}
impl std::error::Error for Failure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.error.as_ref())
    }
}

/// Successful stages and any partial labeling. Updated only after each stage returns.
#[derive(Clone, Debug, Default)]
pub struct Progress {
    pub skeleton: Option<Skeleton>,
    pub labeling: Option<label::Attempt>,
    pub labeled: Option<LabeledSkeleton>,
    pub realized: Option<RealizedSkeleton>,
    pub optimized: Option<RealizedSkeleton>,
    pub drawing: Option<draw_loops::Attempt>,
    pub loops: Option<DualLoops>,
}

impl Progress {
    pub fn from_skeleton(skeleton: Skeleton) -> Self {
        Self {
            skeleton: Some(skeleton),
            ..Self::default()
        }
    }

    pub fn label(&mut self, mesh: &Mesh<INPUT>) -> Result<(), Failure> {
        self.labeling = None;
        self.labeled = None;
        self.realized = None;
        self.optimized = None;
        self.drawing = None;
        self.loops = None;
        let fail = |error| Failure {
            stage: Stage::Labeling,
            error,
        };
        let skeleton = self
            .skeleton
            .as_ref()
            .ok_or_else(|| fail("Missing extracted skeleton".into()))?;
        self.labeling = Some(label::attempt(mesh, skeleton).map_err(fail)?);
        let attempt = self.labeling.as_ref().unwrap();
        self.labeled = attempt.completed();
        if self.labeled.is_none() {
            return Err(fail(
                attempt
                    .failure
                    .clone()
                    .unwrap_or(label::Failure::IncompleteLabels)
                    .into(),
            ));
        }
        Ok(())
    }

    /// Recompute placement after labels change, invalidating all dependent results.
    pub fn place(&mut self, mesh: &Mesh<INPUT>) -> Result<(), Failure> {
        self.realized = None;
        self.optimized = None;
        self.drawing = None;
        self.loops = None;
        if self.labeled.is_none() {
            self.label(mesh)?;
        }
        self.realized = Some(
            place_cubes::run(mesh, self.labeled.as_ref().unwrap()).map_err(|error| Failure {
                stage: Stage::CubePlacement,
                error,
            })?,
        );
        Ok(())
    }

    /// Ensure placement and optimization are complete before drawing loops.
    pub fn optimize(&mut self, mesh: &Mesh<INPUT>) -> Result<(), Failure> {
        if self.labeled.is_none() {
            self.label(mesh)?;
        }
        if self.realized.is_none() {
            self.place(mesh)?;
        }
        if self.optimized.is_none() {
            self.optimized = Some(
                optimize::run(mesh, self.realized.as_ref().unwrap()).map_err(|error| Failure {
                    stage: Stage::Optimization,
                    error,
                })?,
            );
        }
        Ok(())
    }

    /// Continue the actual stage sequence; errors leave all earlier results available.
    pub fn finish(&mut self, mesh: &Mesh<INPUT>) -> Result<(), Failure> {
        self.optimize(mesh)?;
        if self.loops.is_none() {
            self.drawing = None;
            self.drawing = Some(
                draw_loops::attempt(mesh, self.optimized.as_ref().unwrap()).map_err(|error| {
                    Failure {
                        stage: Stage::LoopDrawing,
                        error,
                    }
                })?,
            );
            let attempt = self.drawing.as_ref().unwrap();
            if let Some(error) = &attempt.failure {
                return Err(Failure {
                    stage: Stage::LoopDrawing,
                    error: error.clone().into(),
                });
            }
            self.loops = Some(attempt.loops.clone());
        }
        Ok(())
    }
}

/// Intermediate states and final loops from a completed initialization run.
#[derive(Clone, Debug)]
pub struct SkeletonRun {
    pub skeleton: Skeleton,
    pub labeled: LabeledSkeleton,
    pub realized: RealizedSkeleton,
    pub optimized: RealizedSkeleton,
    pub loops: DualLoops,
    /// The dual structure of the loops, read from the drawing's loop map.
    pub dual: Option<dualcube_dual::Dual>,
}

/// The original stage error plus all results obtained before it.
#[derive(Debug)]
pub struct RunError {
    pub progress: Progress,
    pub failure: Failure,
}
impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.failure.fmt(f)
    }
}
impl std::error::Error for RunError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.failure)
    }
}

/// Run every stage. The error owns the completed and partial results for inspection.
pub fn run(mesh: &Mesh<INPUT>) -> Result<SkeletonRun, RunError> {
    let skeleton = skeletonize::run(mesh).map_err(|error| RunError {
        progress: Progress::default(),
        failure: Failure {
            stage: Stage::Extraction,
            error,
        },
    })?;
    let mut progress = Progress::from_skeleton(skeleton);
    if let Err(failure) = progress.finish(mesh) {
        return Err(RunError { progress, failure });
    }
    Ok(SkeletonRun {
        skeleton: progress.skeleton.unwrap(),
        labeled: progress.labeled.unwrap(),
        realized: progress.realized.unwrap(),
        optimized: progress.optimized.unwrap(),
        loops: progress.loops.unwrap(),
        dual: progress.drawing.and_then(|a| a.dual),
    })
}
