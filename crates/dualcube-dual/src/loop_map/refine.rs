//! The input mesh refined along a loop map, for consumers that need a triangle mesh whose edges
//! carry the loops. Its faces follow the barycentric drawing of `realize`, so it is planar.

use super::*;
use crate::SurfaceArc;
use crate::arrangement::Refinement;

impl LoopMap {
    /// Every map face fanned from its center through its corners and the midpoints of its edges
    /// off input edges, or of input-edge pieces between two loop points. A face that is already a
    /// triangle with a corner off the loops stays. `face_cells` holds map faces.
    pub(crate) fn refine(&self, mesh: &Mesh<INPUT>, embedding: &Embedding) -> Refinement {
        let mut out = Refinement::default();
        let on_loop: Vec<bool> = {
            let mut on = vec![false; self.points.len()];
            for h in &self.halves {
                if h.piece.wall().is_some() {
                    on[h.origin] = true;
                }
            }
            on
        };
        let triangle_of = |h: usize| self.faces[self.halves[h].face].triangle;
        // Map vertices keep their index.
        for (v, &point) in self.points.iter().enumerate() {
            let position = match point {
                Point::Input(input) => mesh.position(input),
                _ => {
                    let triangle = self.outgoing(v).first().map(|&h| triangle_of(h));
                    triangle
                        .and_then(|t| {
                            Some(SurfaceArc::position(
                                mesh,
                                t,
                                embedding.vertex.get(&(t, v)).copied()?,
                            ))
                        })
                        .unwrap_or_else(|| self.positions[v])
                }
            };
            out.positions.push(position);
            out.on_loop.push(on_loop[v]);
            out.vertex_input.push(match point {
                Point::Input(input) => Some(input),
                _ => None,
            });
        }
        let node = |out: &mut Refinement, position: Vector3D, on: bool| {
            out.positions.push(position);
            out.on_loop.push(on);
            out.vertex_input.push(None);
            out.positions.len() - 1
        };
        // Midpoints, by the lesser half-edge of their edge.
        let mut middles = HashMap::new();
        for (h, half) in self.halves.iter().enumerate() {
            if h > half.twin {
                continue;
            }
            let (a, b) = (half.origin, self.target(h));
            let position = match (half.piece, embedding.middle.get(&h)) {
                (Piece::Mesh(_), _) if !(on_loop[a] && on_loop[b]) => continue,
                (Piece::Mesh(_), _) | (_, None) => (out.positions[a] + out.positions[b]) / 2.,
                (_, Some(&p)) => SurfaceArc::position(mesh, triangle_of(h), p),
            };
            let m = node(&mut out, position, half.piece.wall().is_some());
            middles.insert(h, m);
        }
        for (f, face) in self.faces.iter().enumerate() {
            let mut perimeter = vec![];
            let start = face.half;
            let mut h = start;
            loop {
                perimeter.push(self.halves[h].origin);
                if let Some(&m) = middles.get(&h.min(self.halves[h].twin)) {
                    perimeter.push(m);
                }
                h = self.halves[h].next;
                if h == start {
                    break;
                }
            }
            let repeated = |q: usize| perimeter.iter().filter(|&&x| x == q).count() > 1;
            if perimeter.len() == 3
                && !perimeter.iter().all(|&q| out.on_loop[q])
                && !perimeter.iter().any(|&q| repeated(q))
            {
                out.faces.push(perimeter);
                out.face_cells.push(f);
                out.face_input.push(face.triangle);
                continue;
            }
            let center = embedding.center.get(&f).map_or_else(
                || {
                    perimeter
                        .iter()
                        .map(|&q| out.positions[q])
                        .sum::<Vector3D>()
                        / perimeter.len() as f64
                },
                |&p| SurfaceArc::position(mesh, face.triangle, p),
            );
            let c = node(&mut out, center, false);
            // A node met twice is joined to the center through its own point on the spoke.
            let inner: Vec<_> = perimeter
                .iter()
                .map(|&q| {
                    if repeated(q) {
                        let position = (out.positions[q] + center) / 2.;
                        node(&mut out, position, false)
                    } else {
                        q
                    }
                })
                .collect();
            let n = perimeter.len();
            let mut push = |triangle: [usize; 3]| {
                out.faces.push(triangle.to_vec());
                out.face_cells.push(f);
                out.face_input.push(face.triangle);
            };
            for j in 0..n {
                let k = (j + 1) % n;
                let (q, r, a, b) = (perimeter[j], perimeter[k], inner[j], inner[k]);
                match (a == q, b == r) {
                    (true, true) => {}
                    (false, true) => push([q, r, a]),
                    (true, false) => push([q, r, b]),
                    (false, false) => {
                        push([q, r, b]);
                        push([q, b, a]);
                    }
                }
                push([a, b, c]);
            }
        }
        out
    }
}
