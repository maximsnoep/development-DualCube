use crate::{layout::Layout, skeleton::SkeletonData};

pub fn populate_layout_from_skeleton(skeleton_data: &SkeletonData) -> Option<Layout> {
    let ortho_skeleton = if skeleton_data.labeled_skeleton().is_some() {
        skeleton_data.labeled_skeleton().unwrap()
    } else {
        return None;
    };


    // For each region, project a cube from its position outwards
    for node in ortho_skeleton.node_weights() {
        // We need to place the X,Y,Z planes in some way around this node.
        // If only zero edges, just align to global axes, else:
        // TODO: degree 0 case


        // The first degree of freedom we can lock by aligning one of the intersection lines with the longest edge
        // The second the degree of freedom, align with the second longest edge, or if there is only one, align globally for this degree.
        // TODO: degree 1, so 1 edge
        // TODO: degree 2, can use longest 2 axis to align


        // Project a point from centroid into quadrant (8thant?)
        // TODO: per quadrant project a point. Pick the point furthest from the centroid, where the closer the angle is to the optimal angle to that quadrant, the farther we pretend it is for cost selection

        // TODO later: Iteratively improve the 8 points together into more of a cube

        // Find edge-disjointpaths between adjacent quadrant points. The result will be a cube. 
        // TODO: Find paths, paths must be edge disjoint!!! Just do this iteratively, find sort by euclidean distance, then do some restricted dijkstra.
    }


    // For each edge, find the 4 paths between cube corners. The result will be a tube.
    // TODO later: ...
    

    // Result now has a layout of the corner vertices, and paths.
    // TODO make new PartialLayout struct with the info we currently have, corner placements and paths between corners.

    None
}
