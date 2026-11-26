use super::*;
use crate::surface::tests::bands;

fn ring() -> (Mesh<INPUT>, Skeleton) {
    let positions: Vec<_> = (0..4)
        .map(|i| {
            let angle = std::f64::consts::TAU * (i as f64 + 0.5) / 4.;
            [3. * angle.cos(), 3. * angle.sin(), 0.]
        })
        .collect();
    bands(&positions)
}

#[test]
fn boundary_alignment_runs_along_the_tube_from_first_to_second_patch() {
    let (mesh, skeleton) = ring();
    let centroids = SurfaceGeometry::build(&mesh, &skeleton)
        .unwrap()
        .centroids()
        .unwrap();
    for (edge, alignment) in alignments(&mesh, &skeleton) {
        let [a, b] = skeleton.edges[edge].nodes;
        let towards = (centroids[b] - centroids[a]).normalize();
        // A meridian: d × n is the tube's direction, up to the discretization.
        assert!(alignment.direction.norm() > 0.9, "{alignment:?}");
        assert!(alignment.direction.dot(&towards) > 0.9, "{alignment:?}");
        assert!(alignment.length > 0.);
        let best = Port::ALL
            .into_iter()
            .max_by(|p, q| alignment.with(*p).total_cmp(&alignment.with(*q)))
            .unwrap();
        assert_eq!(
            alignment.costs()[best as usize],
            alignment.costs().into_iter().fold(f64::INFINITY, f64::min)
        );
        assert!((alignment.with(best) + alignment.reversed().with(best)).abs() < 1e-12);
    }
}

#[test]
fn capped_patches_enclose_positive_volumes_summing_to_the_solid() {
    let (mesh, skeleton) = ring();
    let visibility = Visibility::build(&mesh, &skeleton).unwrap();
    let volumes = visibility.volumes();
    assert_eq!(volumes.len(), skeleton.nodes.len());
    let mut solid = 0.;
    for face in mesh.face_ids() {
        let p: Vec<_> = mesh.vertices(face).map(|v| mesh.position(v)).collect();
        solid += p[0].dot(&p[1].cross(&p[2])) / 6.;
    }
    for (_, &volume) in &volumes {
        assert!(volume > 0.2 * solid / 4., "{volume} of {solid}");
    }
    let total: f64 = volumes.values().sum();
    assert!(
        (total - solid).abs() < 1e-9 * solid.abs(),
        "{total} vs {solid}"
    );
}

#[test]
fn a_tube_piece_sees_more_of_itself_than_the_whole_ring() {
    let (mesh, skeleton) = ring();
    let visibility = Visibility::build(&mesh, &skeleton).unwrap();
    let nodes: Vec<_> = skeleton.nodes.keys().collect();
    let area = |n: &[SkeletonNodeID]| n.iter().map(|&n| visibility.patches[n].area).sum::<f64>();
    let piece = visibility.visible(&nodes[..1]) / area(&nodes[..1]);
    let whole = visibility.visible(&nodes) / area(&nodes);
    assert!(piece > 0.8, "{piece}");
    assert!(whole < piece - 0.2, "{whole} vs {piece}");
    let all = visibility.visible_all(&[nodes[..1].to_vec(), nodes.clone()]);
    assert_eq!(
        all,
        vec![visibility.visible(&nodes[..1]), visibility.visible(&nodes)]
    );
}

#[test]
fn evaluation_combines_the_terms() {
    let (mesh, skeleton) = ring();
    let labeled = crate::label::attempt(&mesh, &skeleton).unwrap().labeled();
    let quality = evaluate(&mesh, &labeled).unwrap();
    assert_eq!(quality.boundaries, labeled.skeleton.edges.len());
    assert!(quality.alignment > 0.5 && quality.alignment <= 1.);
    assert!(quality.visibility > 0.5 && quality.visibility <= 1.);
    let expected = score(quality.alignment, quality.visibility, quality.boundaries);
    assert_eq!(quality.score, expected);
}
