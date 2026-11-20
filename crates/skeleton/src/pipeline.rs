//! Sequential initialization, retaining completed stages when a later one fails.

use crate::{
    DualLoops, LabeledSkeleton, RealizedSkeleton, Skeleton, certify, draw_loops, label, optimize,
    place_cubes,
};
use dualcube_types::prelude::{INPUT, Mesh};

/// The stages of initialization, in order. [`Progress`] runs certification through loop
/// drawing; callers deflate and extract (`skeletonize`), and `dualcube` reconstructs.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum Stage {
    Deflation,
    Extraction,
    Certification,
    Labeling,
    CubePlacement,
    Optimization,
    LoopDrawing,
    Reconstruction,
}

impl Stage {
    pub const ALL: [Self; 8] = [
        Self::Deflation,
        Self::Extraction,
        Self::Certification,
        Self::Labeling,
        Self::CubePlacement,
        Self::Optimization,
        Self::LoopDrawing,
        Self::Reconstruction,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Deflation => "deflation",
            Self::Extraction => "extraction",
            Self::Certification => "certification",
            Self::Labeling => "labeling",
            Self::CubePlacement => "placement",
            Self::Optimization => "optimization",
            Self::LoopDrawing => "drawing",
            Self::Reconstruction => "reconstruction",
        }
    }

    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|s| s.name() == name)
    }
}

impl std::fmt::Display for Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
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

/// The result of every completed stage, and any partial drawing. Updated only after each
/// stage returns.
#[derive(Clone, Debug, Default)]
pub struct Progress {
    pub skeleton: Option<Skeleton>,
    pub certificate: Option<certify::Certificate>,
    pub labeling: Option<label::Attempt>,
    pub labeled: Option<LabeledSkeleton>,
    pub realized: Option<RealizedSkeleton>,
    pub optimized: Option<RealizedSkeleton>,
    /// Kept when drawing fails, without `loops`.
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

    /// Whether `stage` has its result.
    pub fn done(&self, stage: Stage) -> bool {
        match stage {
            Stage::Deflation | Stage::Extraction => self.skeleton.is_some(),
            Stage::Certification => self.certificate.is_some(),
            Stage::Labeling => self.labeled.is_some(),
            Stage::CubePlacement => self.realized.is_some(),
            Stage::Optimization => self.optimized.is_some(),
            Stage::LoopDrawing => self.loops.is_some(),
            Stage::Reconstruction => false,
        }
    }

    /// The first stage without its result, up to loop drawing.
    pub fn next(&self) -> Option<Stage> {
        Stage::ALL
            .into_iter()
            .take_while(|&s| s <= Stage::LoopDrawing)
            .find(|&s| !self.done(s))
    }

    /// Drop the results of `stage` and of every later stage.
    pub fn invalidate_from(&mut self, stage: Stage) {
        if stage <= Stage::Extraction {
            self.skeleton = None;
        }
        if stage <= Stage::Certification {
            self.certificate = None;
        }
        if stage <= Stage::Labeling {
            self.labeling = None;
            self.labeled = None;
        }
        if stage <= Stage::CubePlacement {
            self.realized = None;
        }
        if stage <= Stage::Optimization {
            self.optimized = None;
        }
        if stage <= Stage::LoopDrawing {
            self.drawing = None;
            self.loops = None;
        }
    }

    /// Run the next stage; `None` when loops are drawn. A failure leaves earlier results.
    pub fn advance(&mut self, mesh: &Mesh<INPUT>) -> Result<Option<Stage>, Failure> {
        let Some(stage) = self.next() else {
            return Ok(None);
        };
        let fail = |error| Failure { stage, error };
        // Stages run in order, so every earlier result is present.
        match stage {
            Stage::Deflation | Stage::Extraction => {
                return Err(fail("Missing extracted skeleton".into()));
            }
            Stage::Reconstruction => unreachable!("`next` stops at loop drawing"),
            Stage::Certification => {
                let skeleton = self.skeleton.as_ref().unwrap();
                self.certificate = Some(certify::run(mesh, skeleton).map_err(|e| fail(e.into()))?);
            }
            Stage::Labeling => {
                let attempt =
                    label::attempt(mesh, self.skeleton.as_ref().unwrap()).map_err(fail)?;
                self.labeled = Some(attempt.labeled());
                self.labeling = Some(attempt);
            }
            Stage::CubePlacement => {
                let labeled = self.labeled.as_ref().unwrap();
                self.realized = Some(place_cubes::run(mesh, labeled).map_err(fail)?);
            }
            Stage::Optimization => {
                let realized = self.realized.as_ref().unwrap();
                self.optimized = Some(optimize::run(mesh, realized).map_err(fail)?);
            }
            Stage::LoopDrawing => {
                let optimized = self.optimized.as_ref().unwrap();
                let attempt = draw_loops::attempt(mesh, optimized).map_err(fail)?;
                let failure = attempt.failure.clone();
                let loops = attempt.loops.clone();
                self.drawing = Some(attempt);
                if let Some(error) = failure {
                    return Err(fail(error.into()));
                }
                self.loops = Some(loops);
            }
        }
        Ok(Some(stage))
    }

    /// Run stages until `last` has its result.
    pub fn run_through(&mut self, mesh: &Mesh<INPUT>, last: Stage) -> Result<(), Failure> {
        while self.next().is_some_and(|s| s <= last) {
            self.advance(mesh)?;
        }
        Ok(())
    }

    /// Run `stage` again (after its input changed), and any missing stage before it.
    pub fn rerun(&mut self, mesh: &Mesh<INPUT>, stage: Stage) -> Result<(), Failure> {
        self.invalidate_from(stage);
        self.run_through(mesh, stage)
    }
}
