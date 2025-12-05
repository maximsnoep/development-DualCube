use crate::prelude::*;

impl<M: Tag> Mesh<M> {
    pub fn split_edge(&mut self, edge_id: EdgeKey<M>) -> (VertKey<M>, [FaceKey<M>; 4]) {
        // First face
        let e_ab = edge_id;
        let e_b0 = self.next(e_ab);
        let e_0a = self.next(e_b0);
        assert!(self.next(e_0a) == e_ab);

        let v_a = self.root(e_ab);
        let v_b = self.root(e_b0);
        let v_0 = self.root(e_0a);

        // Second face
        let e_ba = self.twin(edge_id);
        let e_a1 = self.next(e_ba);
        let e_1b = self.next(e_a1);
        assert!(self.next(e_1b) == e_ba);

        assert!(self.root(e_ba) == v_b);
        assert!(self.root(e_a1) == v_a);
        let v_1 = self.root(e_1b);

        // Four new faces (re-use original id for first 2)
        let f_0 = self.face(e_ab);
        self.face_repr.insert(f_0, e_0a);

        let f_1 = self.face(e_ba);
        self.face_repr.insert(f_1, e_a1);

        let f_2 = self.add_face();
        self.face_repr.insert(f_2, e_b0);

        let f_3 = self.add_face();
        self.face_repr.insert(f_3, e_1b);

        // Six new edges (with next six available ids)

        // f_0
        let e_ax = e_ab;
        let e_x0 = self.add_edge();

        // f_1
        let e_xa = e_ba;
        let e_1x = self.add_edge();

        // f_2
        let e_xb = self.add_edge();
        let e_0x = self.add_edge();

        // f_3
        let e_bx = self.add_edge();
        let e_x1 = self.add_edge();

        // One new vertex (with next available id)
        let v_x = self.add_vertex([0.0, 0.0, 0.0].into()); // Position is not important here`
        self.vert_repr.insert(v_x, e_xa);

        self.vert_repr.insert(v_b, e_b0);
        self.vert_repr.insert(v_a, e_a1);

        // Set the edges correctly
        self.edge_root.insert(e_ax, v_a);
        self.edge_face.insert(e_ax, f_0);
        self.edge_next.insert(e_ax, e_x0);
        self.edge_twin.insert(e_ax, e_xa);

        self.edge_root.insert(e_xa, v_x);
        self.edge_face.insert(e_xa, f_1);
        self.edge_next.insert(e_xa, e_a1);
        self.edge_twin.insert(e_xa, e_ax);

        self.edge_root.insert(e_bx, v_b);
        self.edge_face.insert(e_bx, f_3);
        self.edge_next.insert(e_bx, e_x1);
        self.edge_twin.insert(e_bx, e_xb);

        self.edge_root.insert(e_xb, v_x);
        self.edge_face.insert(e_xb, f_2);
        self.edge_next.insert(e_xb, e_b0);
        self.edge_twin.insert(e_xb, e_bx);

        self.edge_root.insert(e_0x, v_0);
        self.edge_face.insert(e_0x, f_2);
        self.edge_next.insert(e_0x, e_xb);
        self.edge_twin.insert(e_0x, e_x0);

        self.edge_root.insert(e_x0, v_x);
        self.edge_face.insert(e_x0, f_0);
        self.edge_next.insert(e_x0, e_0a);
        self.edge_twin.insert(e_x0, e_0x);

        self.edge_root.insert(e_1x, v_1);
        self.edge_face.insert(e_1x, f_1);
        self.edge_next.insert(e_1x, e_xa);
        self.edge_twin.insert(e_1x, e_x1);

        self.edge_root.insert(e_x1, v_x);
        self.edge_face.insert(e_x1, f_3);
        self.edge_next.insert(e_x1, e_1b);
        self.edge_twin.insert(e_x1, e_1x);

        self.edge_face.insert(e_a1, f_1);
        self.edge_next.insert(e_a1, e_1x);

        self.edge_face.insert(e_1b, f_3);
        self.edge_next.insert(e_1b, e_bx);

        self.edge_face.insert(e_b0, f_2);
        self.edge_next.insert(e_b0, e_0x);

        self.edge_face.insert(e_0a, f_0);
        self.edge_next.insert(e_0a, e_ax);

        (v_x, [f_0, f_1, f_2, f_3])
    }

    pub fn split_face(&mut self, face_id: FaceKey<M>) -> (VertKey<M>, [FaceKey<M>; 3]) {
        let edges = self.edges(face_id).collect_vec();
        // let centroid = self.centroid(face_id);

        // Original face
        let e_01 = edges[0];
        let v_0 = self.root(e_01);

        let e_12 = edges[1];
        let v_1 = self.root(e_12);

        let e_20 = edges[2];
        let v_2 = self.root(e_20);

        // Two new faces (original face stays the same)
        let f_0 = face_id;

        let f_1 = self.add_face();
        let f_2 = self.add_face();

        self.face_repr.insert(f_1, e_12);
        self.face_repr.insert(f_2, e_20);

        // Six new edges (with next six available ids)
        let e_x0 = self.add_edge();
        let e_x1 = self.add_edge();
        let e_x2 = self.add_edge();
        let e_0x = self.add_edge();
        let e_1x = self.add_edge();
        let e_2x = self.add_edge();

        let v_x = self.add_vertex([0., 0., 0.].into());
        self.vert_repr.insert(v_x, e_x0);

        self.edge_root.insert(e_x0, v_x);
        self.edge_face.insert(e_x0, f_0);
        self.edge_next.insert(e_x0, e_01);
        self.edge_twin.insert(e_x0, e_0x);

        self.edge_root.insert(e_x1, v_x);
        self.edge_face.insert(e_x1, f_1);
        self.edge_next.insert(e_x1, e_12);
        self.edge_twin.insert(e_x1, e_1x);

        self.edge_root.insert(e_x2, v_x);
        self.edge_face.insert(e_x2, f_2);
        self.edge_next.insert(e_x2, e_20);
        self.edge_twin.insert(e_x2, e_2x);

        self.edge_root.insert(e_0x, v_0);
        self.edge_face.insert(e_0x, f_2);
        self.edge_next.insert(e_0x, e_x2);
        self.edge_twin.insert(e_0x, e_x0);

        self.edge_root.insert(e_1x, v_1);
        self.edge_face.insert(e_1x, f_0);
        self.edge_next.insert(e_1x, e_x0);
        self.edge_twin.insert(e_1x, e_x1);

        self.edge_root.insert(e_2x, v_2);
        self.edge_face.insert(e_2x, f_1);
        self.edge_next.insert(e_2x, e_x1);
        self.edge_twin.insert(e_2x, e_x2);

        self.edge_face.insert(e_01, f_0);
        self.edge_face.insert(e_12, f_1);
        self.edge_face.insert(e_20, f_2);
        self.edge_next.insert(e_01, e_1x);
        self.edge_next.insert(e_12, e_2x);
        self.edge_next.insert(e_20, e_0x);

        (v_x, [f_0, f_1, f_2])
    }

    pub fn splip_edge(&mut self, a: VertKey<M>, b: VertKey<M>) -> Option<VertKey<M>> {
        // Make sure the edge exists
        let edge = self.edge_between_verts(a, b).unwrap().0;

        // Get the two faces adjacent to the two edges
        let Some([f1, f2]) = self.faces(edge).collect_array::<2>() else {
            panic!("Expected exactly two faces for edge {edge:?}");
        };

        // Get the anchor vertex of f1 (the vertex that is not a or b)
        let c1 = self.vertices(f1).find(|&v| v != a && v != b).unwrap();
        // Get the anchor vertex of f2 (the vertex that is not a or b)
        let c2 = self.vertices(f2).find(|&v| v != a && v != b).unwrap();

        // Get all required edges
        let a_c1 = self.edge_between_verts(a, c1).unwrap().0;
        let b_c1 = self.edge_between_verts(b, c1).unwrap().0;
        let a_c2 = self.edge_between_verts(a, c2).unwrap().0;
        let b_c2 = self.edge_between_verts(b, c2).unwrap().0;
        let a_b = edge;

        // Construct planar embedding respecting all edge lengths
        let a_c1_distance = self.size(a_c1);
        let b_c1_distance = self.size(b_c1);
        let a_c2_distance = self.size(a_c2);
        let b_c2_distance = self.size(b_c2);
        let a_b_distance = self.size(a_b);

        // if a_c1_distance < 1e-6 || b_c1_distance < 1e-6 || a_c2_distance < 1e-6 || b_c2_distance < 1e-6 || a_b_distance < 1e-6 {
        //     println!("oopsie ");
        //     return None;
        // }

        let a_position = Vector2D::new(0., 0.);
        let b_position = Vector2D::new(a_b_distance, 0.);

        // Calculate the position of c1 (under a_b)
        // Draw circle with radius a_c1_distance and center a_position
        // Draw circle with radius b_c1_distance and center b_position
        // Find intersection point with negative y: this is the position of c1
        let r1 = a_c1_distance;
        let r2 = b_c1_distance;
        let d = a_b_distance;

        let x = (d * d - r2 * r2 + r1 * r1) / (2. * d);
        let yy = r1 * r1 - x * x;
        let y = if yy < 0. { 0. } else { -(yy.sqrt()) };

        let c1_position = Vector2D::new(x, y);
        assert!(c1_position[1] <= 0., "c1_position: {:?}", c1_position);

        // Calculate the position of c2
        // Draw circle with radius a_c2_distance and center a_position
        // Draw circle with radius b_c2_distance and center b_position
        // Find intersection point with positive y: this is the position of c2
        let r1 = a_c2_distance;
        let r2 = b_c2_distance;
        let d = a_b_distance;

        let x = (d * d - r2 * r2 + r1 * r1) / (2. * d);
        let yy = r1 * r1 - x * x;
        let y = if yy < 0. { 0. } else { yy.sqrt() };
        let c2_position = Vector2D::new(x, y);
        assert!(c2_position[1] >= 0., "c2_position: {:?}", c2_position);

        // println!("a_position: {a_position:?}");
        // println!("b_position: {b_position:?}");
        // println!("c1_position: {c1_position:?}");
        // println!("c2_position: {c2_position:?}");

        // Find intersection of a_b and c1_c2
        // Calculate the intersection of the lines a_b and c1_c2
        let intersection = geom::calculate_2d_lineseg_intersection(a_position, b_position, c1_position, c2_position)?;

        // assert!(intersection[1].abs() == 0., "{intersection:?}");

        // The portion of the edge a_b that is before the intersection
        let t = intersection.0[0] / a_b_distance;

        // println!("t: {}", t);

        if t < 0.001 {
            return Some(a);
        }

        if t > 0.999 {
            return Some(b);
        }

        // Calculate the position of the split vertex in 3D
        let split_position = self.position(a) + (self.position(b) - self.position(a)) * t;

        // Split edge a_b
        let (split_vertex, _) = self.split_edge(a_b);

        // There exists an edge between c1 and split_vertex and c2 and split_vertex
        assert!(self.edge_between_verts(c1, split_vertex).is_some());
        assert!(self.edge_between_verts(c2, split_vertex).is_some());

        // Move the split vertex to the correct position
        self.set_position(split_vertex, split_position);

        Some(split_vertex)
    }

    // pub fn refine(&mut self, n: usize) {
    //     for _ in 0..n {
    //         // find the longest edge
    //         let longest_edge = self.edges.keys().max_by_key(|&edge_id| OrderedFloat(self.length(edge_id))).unwrap();
    //         let (a, b) = self.endpoints(longest_edge);
    //         self.splip_edge(a, b);
    //     }
    // }
}
