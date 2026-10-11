#[cfg(test)]
mod tests {
    use std::collections::{BinaryHeap, HashSet};

    use mehsh::mesh::connectivity::Mesh;
    use mehsh::prelude::{Tag, VERT};
    use mehsh::utils::ids::IdMap;
    use mehsh::utils::primitives::Vector3D;

    use crate::skeletonize::deflate::CONTRACTION;
    use crate::skeletonize::surgery::{CollapseCandidate, SurgeryContext, VIdx, sort_face};
    use dualcube_types::prelude::INPUT;

    const RINGS: usize = 20;
    const SLOTS: usize = 6;
    const N_SIDE: usize = RINGS * SLOTS;
    const BOT_APEX: usize = N_SIDE;
    const TOP_APEX: usize = N_SIDE + 1;
    const N_VERTS: usize = N_SIDE + 2;
    // 19 strips * 6 slots * 2 tris + 2 caps * 6 = 240
    const N_FACES: usize = (RINGS - 1) * SLOTS * 2 + 2 * SLOTS;

    /// Build a closed triangulated cylinder mesh as described by `RINGS`/`SLOTS`,
    /// closed at top and bottom by triangle fans to two apex vertices. Returns the
    /// mesh along with the IdMap from original integer vertex index → mesh `VertKey`.
    fn build_cylinder<M: Tag>() -> (Mesh<M>, IdMap<VERT, M>) {
        let radius: f64 = 1.0;
        let height_step: f64 = 0.5;

        let mut positions = Vec::with_capacity(N_VERTS);
        for r in 0..RINGS {
            for s in 0..SLOTS {
                let theta = 2.0 * std::f64::consts::PI * (s as f64) / (SLOTS as f64);
                let x = radius * theta.cos();
                let y = radius * theta.sin();
                let z = (r as f64) * height_step;
                positions.push(Vector3D::new(x, y, z));
            }
        }
        positions.push(Vector3D::new(0.0, 0.0, -height_step));
        positions.push(Vector3D::new(0.0, 0.0, (RINGS as f64) * height_step));

        let v = |r: usize, s: usize| r * SLOTS + (s % SLOTS);
        let mut faces: Vec<Vec<usize>> = Vec::new();
        for r in 0..(RINGS - 1) {
            for s in 0..SLOTS {
                let s1 = (s + 1) % SLOTS;
                faces.push(vec![v(r, s), v(r + 1, s), v(r, s1)]);
                faces.push(vec![v(r, s1), v(r + 1, s), v(r + 1, s1)]);
            }
        }
        for s in 0..SLOTS {
            faces.push(vec![BOT_APEX, v(0, s), v(0, (s + 1) % SLOTS)]);
        }
        let last = RINGS - 1;
        for s in 0..SLOTS {
            faces.push(vec![TOP_APEX, v(last, (s + 1) % SLOTS), v(last, s)]);
        }

        let (mesh, vmap, _) =
            Mesh::<M>::from(&faces, &positions).expect("cylinder mesh build failed");
        (mesh, vmap)
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum EdgeKind {
        Vertical,
        Ring,
        Diagonal,
        Cap,
    }

    fn classify(a_idx: usize, b_idx: usize) -> EdgeKind {
        let is_apex = |idx: usize| idx == BOT_APEX || idx == TOP_APEX;
        if is_apex(a_idx) || is_apex(b_idx) {
            return EdgeKind::Cap;
        }
        let (ra, sa) = (a_idx / SLOTS, a_idx % SLOTS);
        let (rb, sb) = (b_idx / SLOTS, b_idx % SLOTS);
        if ra == rb {
            EdgeKind::Ring
        } else if sa == sb {
            EdgeKind::Vertical
        } else {
            EdgeKind::Diagonal
        }
    }

    /// Count undirected edges in `ctx.neighbors` (skipping dead vertices).
    fn edge_count(ctx: &SurgeryContext) -> usize {
        let mut sum: usize = 0;
        for (&u, nset) in &ctx.neighbors {
            if ctx.is_dead.contains(&u) {
                continue;
            }
            sum += nset.iter().filter(|n| !ctx.is_dead.contains(n)).count();
        }
        sum / 2
    }

    /// Reason a popped candidate was rejected.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum RejectReason {
        Dead,
        NotAdjacent,
        NoFaces,
        LinkFailed,
    }

    /// Try to classify why an `is_legal_collapse_candidate` returned false (or, if it
    /// passed, the cost is stale). Returns Some if a rejection reason applies.
    fn rejection_reason(ctx: &mut SurgeryContext, c: &CollapseCandidate) -> Option<RejectReason> {
        if ctx.is_dead.contains(&c.u) || ctx.is_dead.contains(&c.v) {
            return Some(RejectReason::Dead);
        }
        if !ctx.neighbors.get(&c.u).is_some_and(|s| s.contains(&c.v)) {
            return Some(RejectReason::NotAdjacent);
        }
        if !ctx.edge_has_faces(c.u, c.v) {
            return Some(RejectReason::NoFaces);
        }
        if !ctx.check_link_condition(c.u, c.v) {
            return Some(RejectReason::LinkFailed);
        }
        None
    }

    /// The reason the surgery loop terminated.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum ExitReason {
        FacesEmpty,
        HeapEmpty,
    }

    #[derive(Debug)]
    struct RunSummary {
        alive_verts: usize,
        active_faces: usize,
        edges: usize,
        collapses: usize,
        exit: ExitReason,
        last_kind: Option<EdgeKind>,
        /// Number of accepted collapses for which the strict (Dey-Edelsbrunner) edge-link
        /// condition would have rejected: i.e. there exist common neighbours w1, w2 with both
        /// (u, w1, w2) and (v, w1, w2) active faces (= edge (w1,w2) ∈ lk(u) ∩ lk(v) \ lk(uv)).
        strict_violations_accepted: usize,
    }

    /// Returns Some((w1, w2)) if the picked collapse u→v violates the strict 2-manifold
    /// edge-link condition: there exist common neighbours w1≠w2 of u and v such that
    /// BOTH (u, w1, w2) and (v, w1, w2) are currently active faces.
    fn strict_violation(ctx: &SurgeryContext, u: VIdx, v: VIdx) -> Option<(VIdx, VIdx)> {
        let common: Vec<_> = ctx.neighbors[&u]
            .iter()
            .copied()
            .filter(|w| *w != v && ctx.neighbors[&v].contains(w))
            .collect();
        for i in 0..common.len() {
            for j in (i + 1)..common.len() {
                let w1 = common[i];
                let w2 = common[j];
                if ctx.active_faces.contains(&sort_face(u, w1, w2))
                    && ctx.active_faces.contains(&sort_face(v, w1, w2))
                {
                    return Some((w1, w2));
                }
            }
        }
        None
    }

    /// Runs the surgery loop on a fresh context built from `mesh`.
    /// If `link_condition_enabled` is false, the link-condition guard inside
    /// `is_legal_collapse_candidate` is bypassed (we still require alive + adjacent + has-faces).
    /// `vmap` is used to translate `VIdx` back into original integer indices for classification.
    fn run_surgery(
        mesh: &Mesh<CONTRACTION>,
        vmap: &IdMap<VERT, CONTRACTION>,
        link_condition_enabled: bool,
        verbose: bool,
    ) -> RunSummary {
        // SurgeryContext::new now needs an INPUT counterpart for the
        // tetrahedralization-based handle-subspace computation. Build one
        // with the same geometry; vertex raw values match across tags.
        let (input_mesh, _vmap_input): (Mesh<INPUT>, _) = build_cylinder();
        let mut ctx = SurgeryContext::new(mesh, &input_mesh).expect("preprocessing failed in test");

        // Resolve original-index for a VIdx (relies on vmap being the build_cylinder output).
        let idx_of = |k: VIdx| -> usize { *vmap.id(&k).expect("vmap missing key") };

        // Seed the heap with all directed edges that currently have faces.
        let mut heap: BinaryHeap<CollapseCandidate> = BinaryHeap::new();
        let vert_ids: Vec<_> = ctx.neighbors.keys().copied().collect();
        for u in vert_ids {
            let nbrs: Vec<_> = ctx.neighbors[&u].iter().copied().collect();
            for v in nbrs {
                if ctx.edge_has_faces(u, v) {
                    let cost = ctx.compute_collapse_cost(u, v);
                    heap.push(CollapseCandidate { u, v, cost });
                }
            }
        }

        eprintln!(
            "[link={}] seed: V={}, E={}, F={}, heap={}",
            link_condition_enabled,
            ctx.positions.len() - ctx.is_dead.len(),
            edge_count(&ctx),
            ctx.active_faces.len(),
            heap.len()
        );

        let mut collapses: usize = 0;
        let mut last_kind: Option<EdgeKind> = None;
        let mut strict_violations_accepted: usize = 0;
        const COST_EPSILON: f64 = 1e-8;

        let exit = loop {
            if ctx.active_faces.is_empty() {
                break ExitReason::FacesEmpty;
            }

            // Per-iteration we may pop many stale/illegal candidates before doing one collapse.
            let mut rejected_dead: usize = 0;
            let mut rejected_not_adjacent: usize = 0;
            let mut rejected_no_faces: usize = 0;
            let mut rejected_link: usize = 0;
            let mut rejected_stale: usize = 0;
            let mut min_cost_seen = f64::INFINITY;

            let chosen = loop {
                let Some(c) = heap.pop() else {
                    break None;
                };
                min_cost_seen = min_cost_seen.min(c.cost);

                // Combined legality check (with link-condition optionally bypassed)
                let basic_ok = !ctx.is_dead.contains(&c.u)
                    && !ctx.is_dead.contains(&c.v)
                    && ctx.neighbors.get(&c.u).is_some_and(|s| s.contains(&c.v))
                    && ctx.edge_has_faces(c.u, c.v);

                let link_ok = if link_condition_enabled {
                    basic_ok && ctx.check_link_condition(c.u, c.v)
                } else {
                    basic_ok
                };

                if !link_ok {
                    match rejection_reason(&mut ctx, &c) {
                        Some(RejectReason::Dead) => rejected_dead += 1,
                        Some(RejectReason::NotAdjacent) => rejected_not_adjacent += 1,
                        Some(RejectReason::NoFaces) => rejected_no_faces += 1,
                        Some(RejectReason::LinkFailed) => rejected_link += 1,
                        _ => {}
                    }
                    continue;
                }

                let real_cost = ctx.compute_collapse_cost(c.u, c.v);
                if (c.cost - real_cost).abs() > COST_EPSILON {
                    heap.push(CollapseCandidate {
                        u: c.u,
                        v: c.v,
                        cost: real_cost,
                    });
                    rejected_stale += 1;
                    continue;
                }
                break Some(c);
            };

            let Some(c) = chosen else {
                if verbose {
                    eprintln!(
                        "[link={}] iter exhausted heap: rejected dead={}, !adj={}, !face={}, link={}, stale={}",
                        link_condition_enabled,
                        rejected_dead,
                        rejected_not_adjacent,
                        rejected_no_faces,
                        rejected_link,
                        rejected_stale,
                    );
                }
                break ExitReason::HeapEmpty;
            };

            let a_idx = idx_of(c.u);
            let b_idx = idx_of(c.v);
            let kind = classify(a_idx, b_idx);
            last_kind = Some(kind);

            let sv = strict_violation(&ctx, c.u, c.v);
            if sv.is_some() {
                strict_violations_accepted += 1;
            }

            if verbose {
                let sv_tag = match sv {
                    Some((w1, w2)) => {
                        format!(" STRICT-VIOLATION via ({},{})", idx_of(w1), idx_of(w2))
                    }
                    None => String::new(),
                };
                eprintln!(
                    "[link={}] it={:3}: V={:3} E={:3} F={:3} | pick {:3}->{:3} ({:?}) cost={:.4e} | rej dead={} !adj={} !face={} link={} stale={}{}",
                    link_condition_enabled,
                    collapses,
                    ctx.positions.len() - ctx.is_dead.len(),
                    edge_count(&ctx),
                    ctx.active_faces.len(),
                    a_idx,
                    b_idx,
                    kind,
                    c.cost,
                    rejected_dead,
                    rejected_not_adjacent,
                    rejected_no_faces,
                    rejected_link,
                    rejected_stale,
                    sv_tag,
                );
            }

            ctx.collapse_edge(c.u, c.v);
            collapses += 1;

            // Re-seed edges incident to the survivor
            let nbrs: Vec<_> = ctx.neighbors[&c.v].iter().copied().collect();
            for n in nbrs {
                if ctx.edge_has_faces(c.v, n) {
                    let cost = ctx.compute_collapse_cost(c.v, n);
                    heap.push(CollapseCandidate { u: c.v, v: n, cost });
                }
                if ctx.edge_has_faces(n, c.v) {
                    let cost = ctx.compute_collapse_cost(n, c.v);
                    heap.push(CollapseCandidate { u: n, v: c.v, cost });
                }
            }
        };

        let summary = RunSummary {
            alive_verts: ctx.positions.len() - ctx.is_dead.len(),
            active_faces: ctx.active_faces.len(),
            edges: edge_count(&ctx),
            collapses,
            exit,
            last_kind,
            strict_violations_accepted,
        };
        eprintln!(
            "[link={}] EXIT {:?}: collapses={}, V={}, E={}, F={}, last_edge_kind={:?}, strict_violations_accepted={}",
            link_condition_enabled,
            summary.exit,
            summary.collapses,
            summary.alive_verts,
            summary.edges,
            summary.active_faces,
            summary.last_kind,
            summary.strict_violations_accepted,
        );
        summary
    }

    /// Step 3: Direct probe of `check_link_condition` for a middle-ring edge.
    /// Ring 10 is the middle. The edge from slot 0 to slot 1 on ring 10 has a third
    /// ring-10 vertex (slot 5? slot 2?) as a common neighbor only if the triangulation
    /// connects it across — for our quad triangulation, common neighbors of an edge on a
    /// single ring are the two diagonal-vertex/vertical-vertex pairs from the strips on
    /// either side. The ring-10 vertices themselves are NOT mutual neighbors except via
    /// the strip diagonals, so this probe instead targets a *ring* edge directly to
    /// confirm the link-condition behaviour on edges within a single ring.
    #[test]
    fn link_condition_probe_middle_ring_edge() {
        let (mesh, vmap): (Mesh<CONTRACTION>, _) = build_cylinder();
        let (input_mesh, _): (Mesh<INPUT>, _) = build_cylinder();
        let mut ctx =
            SurgeryContext::new(&mesh, &input_mesh).expect("preprocessing failed in test");
        let idx_of = |k: VIdx| -> usize { *vmap.id(&k).expect("vmap missing key") };

        let r: usize = 10;
        let s0: usize = 0;
        let s1: usize = 1;
        let i = *vmap.key(r * SLOTS + s0).expect("missing ring 10 slot 0");
        let j = *vmap.key(r * SLOTS + s1).expect("missing ring 10 slot 1");
        eprintln!(
            "PROBE: middle-ring edge i={:?} (idx {}) -- j={:?} (idx {})",
            i,
            idx_of(i),
            j,
            idx_of(j)
        );

        let n_i: HashSet<_> = ctx.neighbors[&i].iter().copied().collect();
        let n_j: HashSet<_> = ctx.neighbors[&j].iter().copied().collect();
        let common: Vec<_> = n_i
            .intersection(&n_j)
            .copied()
            .filter(|k| *k != i && *k != j)
            .collect();
        eprintln!(
            "  N(i)={:?}",
            n_i.iter().map(|k| idx_of(*k)).collect::<Vec<_>>()
        );
        eprintln!(
            "  N(j)={:?}",
            n_j.iter().map(|k| idx_of(*k)).collect::<Vec<_>>()
        );
        eprintln!(
            "  common neighbors = {:?}",
            common.iter().map(|k| idx_of(*k)).collect::<Vec<_>>()
        );
        for &k in &common {
            let exists = ctx.active_faces.contains(&sort_face(i, j, k));
            eprintln!("    k={:>3}: (i,j,k) face exists? {}", idx_of(k), exists);
        }
        let verdict = ctx.check_link_condition(i, j);
        eprintln!("  check_link_condition(i, j) = {}", verdict);

        // Also: for any *pair* of common neighbors w1, w2, if both (i,w1,w2) and (j,w1,w2)
        // are faces, then w1-w2 ∈ lk(i) ∩ lk(j) but not in lk(ij). The classical 2-manifold
        // link condition would reject. The current implementation does NOT check this.
        let mut strict_violations: Vec<(usize, usize)> = Vec::new();
        for a in 0..common.len() {
            for b in (a + 1)..common.len() {
                let w1 = common[a];
                let w2 = common[b];
                if ctx.active_faces.contains(&sort_face(i, w1, w2))
                    && ctx.active_faces.contains(&sort_face(j, w1, w2))
                {
                    strict_violations.push((idx_of(w1), idx_of(w2)));
                }
            }
        }
        eprintln!(
            "  strict (Dey/Edelsbrunner) edge-link violations: {:?}",
            strict_violations
        );
    }

    /// Step 5: A/B experiment — run the surgery twice on the same cylinder, with the
    /// link-condition check enabled vs. disabled. If the implementation is functioning,
    /// (b) should leave V ≫ 2 while (a) collapses to V = 2.
    #[test]
    fn cylinder_a_b_link_condition() {
        let (mesh_a, vmap_a): (Mesh<CONTRACTION>, _) = build_cylinder();
        let (mesh_b, vmap_b): (Mesh<CONTRACTION>, _) = build_cylinder();

        eprintln!("--- Run (a): link condition DISABLED ---");
        let a = run_surgery(&mesh_a, &vmap_a, false, false);
        eprintln!("--- Run (b): link condition ENABLED  ---");
        let b = run_surgery(&mesh_b, &vmap_b, true, false);

        eprintln!("A/B summary:");
        eprintln!(
            "  (a) link disabled: V={}, E={}, F={}, collapses={}",
            a.alive_verts, a.edges, a.active_faces, a.collapses
        );
        eprintln!(
            "  (b) link enabled : V={}, E={}, F={}, collapses={}",
            b.alive_verts, b.edges, b.active_faces, b.collapses
        );
    }

    /// Step 2 + 4: Per-iteration instrumented run. Logs (V,E,F), the selected edge with
    /// classification, rejection counts, and the final exit reason.
    #[test]
    fn cylinder_surgery_per_iteration_trace() {
        let (mesh, vmap): (Mesh<CONTRACTION>, _) = build_cylinder();
        eprintln!(
            "Input mesh: verts={}, edges(half)={}, faces={}",
            mesh.nr_verts(),
            mesh.nr_edges(),
            mesh.nr_faces()
        );
        assert_eq!(mesh.nr_verts(), N_VERTS);
        assert_eq!(mesh.nr_faces(), N_FACES);

        let summary = run_surgery(&mesh, &vmap, true, true);

        eprintln!(
            "FINAL: V={}, E={}, F={}, exit={:?}, last_edge_kind={:?}",
            summary.alive_verts,
            summary.edges,
            summary.active_faces,
            summary.exit,
            summary.last_kind,
        );
    }
}
