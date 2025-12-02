use crate::prelude::*;
use itertools::Itertools;
use mehsh::prelude::*;
use std::collections::HashSet;

pub fn feature_extraction(mesh: &Mesh<INPUT>, angle_threshold: f64, min_connectedness: usize) -> [HashSet<EdgeID>; 3] {
    let mut intermediate_features = HashSet::new();

    // Add all edges with dihedral angle > angle_threshold
    for edge_id in mesh.edge_ids() {
        let angle = mesh.dihedral(edge_id);
        if angle > angle_threshold {
            intermediate_features.insert(edge_id);
        }
    }
    println!("Found {} intermediate feature edges", intermediate_features.len());

    let mut filtered = HashSet::new();
    let mut features = HashSet::new();
    for &edge_id in &intermediate_features {
        if features.contains(&edge_id) || filtered.contains(&edge_id) {
            continue;
        }

        let mut component = HashSet::new();
        let mut queue = mesh.neighbors2(edge_id).into_iter().collect_vec();
        while let Some(neighbor) = queue.pop() {
            if !intermediate_features.contains(&neighbor) {
                continue;
            }

            component.insert(neighbor);
            for next in mesh.neighbors2(neighbor) {
                if !queue.contains(&next) && !component.contains(&next) {
                    queue.push(next);
                }
            }
        }

        if component.len() >= min_connectedness {
            features.extend(component);
        } else {
            filtered.extend(component);
        }
    }

    println!("Found {} filtered feature edges", features.len());

    // Label each feature edge
    let mut labeled_features = [HashSet::new(), HashSet::new(), HashSet::new()];

    for edge_id in features {
        let [f1, f2] = [mesh.faces(edge_id)[0], mesh.faces(edge_id)[1]];
        let [n1, n2] = [mesh.normal(f1), mesh.normal(f2)];
        let [l1, l2] = [to_principal_direction(n1).0, to_principal_direction(n2).0];
        match (l1, l2) {
            (PrincipalDirection::X, PrincipalDirection::Y) | (PrincipalDirection::Y, PrincipalDirection::X) => {
                labeled_features[2].insert(edge_id);
            }
            (PrincipalDirection::X, PrincipalDirection::Z) | (PrincipalDirection::Z, PrincipalDirection::X) => {
                labeled_features[1].insert(edge_id);
            }
            (PrincipalDirection::Y, PrincipalDirection::Z) | (PrincipalDirection::Z, PrincipalDirection::Y) => {
                labeled_features[0].insert(edge_id);
            }
            _ => {}
        }
    }

    labeled_features
}
