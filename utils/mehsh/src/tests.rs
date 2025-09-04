use crate::prelude::*;
use std::path::PathBuf;
define_tag!(TestMesh);

#[test]
fn from_manual() {
    let faces = vec![vec![0, 2, 1], vec![0, 1, 3], vec![1, 2, 3], vec![0, 3, 2]];
    let douconel = Mesh::<TestMesh>::from(&faces, &[Vector3D::new(0., 0., 0.); 4]);
    assert!(douconel.is_ok(), "{douconel:?}");
    if let Ok((douconel, _, _)) = douconel {
        assert!(douconel.nr_verts() == 4);
        assert!(douconel.nr_edges() == 6 * 2);
        assert!(douconel.nr_faces() == 4);

        for face_id in douconel.faces.ids() {
            assert!(douconel.vertices(face_id).len() == 3);
        }
    }
}

#[cfg(feature = "stl")]
#[test]
fn from_blub_stl() {
    let douconel = Mesh::<TestMesh>::from_stl(&PathBuf::from("assets/blub001k.stl"));
    assert!(douconel.is_ok(), "{douconel:?}");
    if let Ok((douconel, _, _)) = douconel {
        assert!(douconel.nr_verts() == 945);
        assert!(douconel.nr_edges() == 2829 * 2);
        assert!(douconel.nr_faces() == 1886);

        for face_id in douconel.faces.ids() {
            assert!(douconel.vertices(face_id).len() == 3);
        }
    }
}

#[cfg(feature = "obj")]
#[test]
fn from_blub_obj() {
    let douconel = Mesh::<TestMesh>::from_obj(&PathBuf::from("assets/blub001k.obj"));
    assert!(douconel.is_ok(), "{douconel:?}");
    if let Ok((douconel, _, _)) = douconel {
        assert!(douconel.nr_verts() == 945);
        assert!(douconel.nr_edges() == 2829 * 2);
        assert!(douconel.nr_faces() == 1886);

        for face_id in douconel.faces.ids() {
            assert!(douconel.vertices(face_id).len() == 3);
        }
    }
}

#[cfg(feature = "stl")]
#[test]
fn from_nefertiti_stl() {
    let douconel = Mesh::<TestMesh>::from_stl(&PathBuf::from("assets/nefertiti099k.stl"));
    assert!(douconel.is_ok(), "{douconel:?}");
    if let Ok((douconel, _, _)) = douconel {
        assert!(douconel.nr_verts() == 49971);
        assert!(douconel.nr_edges() == 149_907 * 2);
        assert!(douconel.nr_faces() == 99938);

        for face_id in douconel.faces.ids() {
            assert!(douconel.vertices(face_id).len() == 3);
        }
    }
}

#[cfg(feature = "obj")]
#[test]
fn from_hexahedron_obj() {
    let douconel = Mesh::<TestMesh>::from_obj(&PathBuf::from("assets/hexahedron.obj"));
    assert!(douconel.is_ok(), "{douconel:?}");
    if let Ok((douconel, _, _)) = douconel {
        assert!(douconel.nr_verts() == 8);
        assert!(douconel.nr_edges() == 4 * 6);
        assert!(douconel.nr_faces() == 6);

        for face_id in douconel.faces.ids() {
            assert!(douconel.vertices(face_id).len() == 4);
        }
    }
}

#[cfg(feature = "obj")]
#[test]
fn from_tetrahedron_obj() {
    let douconel = Mesh::<TestMesh>::from_obj(&PathBuf::from("assets/tetrahedron.obj"));
    assert!(douconel.is_ok(), "{douconel:?}");
    if let Ok((douconel, _, _)) = douconel {
        assert!(douconel.nr_verts() == 4);
        assert!(douconel.nr_edges() == 3 * 4);
        assert!(douconel.nr_faces() == 4);

        for face_id in douconel.faces.ids() {
            assert!(douconel.vertices(face_id).len() == 3);
        }
    }
}
