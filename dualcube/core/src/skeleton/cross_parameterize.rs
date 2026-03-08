use crate::skeleton::orthogonalize::LabeledCurveSkeleton;

/// Performs cross-parameterization between two labeled curve skeletons,
/// by mapping all regions individually and stitching at the borders.
pub fn cross_parameterize(s1: LabeledCurveSkeleton, s2: LabeledCurveSkeleton) -> () {

    // TODO: per region call parameterize
}

fn parameterize_region() -> () {
    // TODO:
    // Identity canonical domain
    // Map each to canonical domain, save mappings in some data structure
    // TODO: what data structure to return?
}
