//! Cheap, connectivity-preserving vertex relaxation for triangle meshes.
//!
//! The relaxer uses uniform Taubin smoothing: a positive Laplacian pass removes
//! high-frequency roughness and a following negative pass counteracts surface
//! shrinkage. Triangle connectivity and per-vertex attributes are unchanged.

use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use rayon::prelude::*;

use crate::Mesh3D;

/// Options controlling vertex relaxation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VertexRelaxOptions {
    /// Number of positive/negative Taubin pass pairs.
    pub iterations: usize,
    /// Strength of the smoothing pass, normally in `(0, 1]`.
    pub lambda: f32,
    /// Strength of the shrinkage-correction pass, normally slightly more
    /// negative than `-lambda`.
    pub mu: f32,
    /// Maximum movement in one pass as a fraction of the vertex's initial mean
    /// incident edge length. Set to zero to disable movement.
    pub max_step_ratio: f32,
    /// Keep vertices on edges incident to only one triangle fixed.
    pub lock_boundaries: bool,
}

impl Default for VertexRelaxOptions {
    fn default() -> Self {
        Self {
            iterations: 3,
            lambda: 0.5,
            mu: -0.53,
            max_step_ratio: 0.25,
            lock_boundaries: true,
        }
    }
}

/// Summary of a vertex-relaxation operation.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct VertexRelaxResult {
    /// Vertices eligible to move. Isolated and locked vertices are excluded.
    pub relaxed_vertex_count: usize,
    /// Vertices locked because they lie on an open or non-manifold edge.
    pub locked_vertex_count: usize,
    /// Timings for adjacency construction and relaxation.
    pub timings: VertexRelaxTimings,
}

/// Stage timings for [`relax_mesh_vertices`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VertexRelaxTimings {
    /// Input validation and construction of the one-ring adjacency.
    pub adjacency_build: Duration,
    /// Mean-edge calculation and all Taubin passes.
    pub relaxation: Duration,
    /// Complete wall-clock duration.
    pub total: Duration,
}

/// Smooth mesh positions in place without changing indices or attributes.
///
/// Each iteration performs a `lambda` Laplacian pass followed by a `mu` pass
/// that limits the shrinkage associated with ordinary Laplacian smoothing.
/// Open-boundary vertices are optionally fixed; vertices on non-manifold edges
/// are always fixed. `mesh.uv` is validated but deliberately left unchanged.
pub fn relax_mesh_vertices(
    mesh: &mut Mesh3D,
    options: VertexRelaxOptions,
) -> Result<VertexRelaxResult> {
    let total_start = Instant::now();
    validate_options(options)?;

    let stage_start = Instant::now();
    validate_mesh(mesh)?;
    let adjacency = MeshAdjacency::build(mesh, options.lock_boundaries)?;
    let adjacency_build = stage_start.elapsed();

    let locked_vertex_count = adjacency.locked.iter().filter(|&&locked| locked).count();
    let relaxed_vertex_count = (0..mesh.positions.len())
        .filter(|&vertex| !adjacency.locked[vertex] && adjacency.degree(vertex) > 0)
        .count();

    if options.iterations == 0 || relaxed_vertex_count == 0 || options.max_step_ratio == 0.0 {
        return Ok(VertexRelaxResult {
            relaxed_vertex_count,
            locked_vertex_count,
            timings: VertexRelaxTimings {
                adjacency_build,
                total: total_start.elapsed(),
                ..VertexRelaxTimings::default()
            },
        });
    }

    let stage_start = Instant::now();
    let mean_edge_lengths = mean_edge_lengths(&mesh.positions, &adjacency);
    let mut scratch = vec![[0.0_f32; 3]; mesh.positions.len()];
    for _ in 0..options.iterations {
        relaxation_pass(
            &mesh.positions,
            &mut scratch,
            &adjacency,
            &mean_edge_lengths,
            options.lambda,
            options.max_step_ratio,
        )?;
        relaxation_pass(
            &scratch,
            &mut mesh.positions,
            &adjacency,
            &mean_edge_lengths,
            options.mu,
            options.max_step_ratio,
        )?;
    }
    let relaxation = stage_start.elapsed();

    Ok(VertexRelaxResult {
        relaxed_vertex_count,
        locked_vertex_count,
        timings: VertexRelaxTimings {
            adjacency_build,
            relaxation,
            total: total_start.elapsed(),
        },
    })
}

struct MeshAdjacency {
    offsets: Vec<usize>,
    neighbors: Vec<u32>,
    locked: Vec<bool>,
}

impl MeshAdjacency {
    fn build(mesh: &Mesh3D, lock_boundaries: bool) -> Result<Self> {
        let edge_capacity = mesh
            .indices
            .len()
            .checked_mul(3)
            .context("mesh edge count overflow")?;
        let mut edges = Vec::with_capacity(edge_capacity);
        for (face_index, &[a, b, c]) in mesh.indices.iter().enumerate() {
            ensure!(
                a != b && b != c && c != a,
                "mesh face {face_index} has repeated vertex indices"
            );
            for index in [a, b, c] {
                ensure!(
                    (index as usize) < mesh.positions.len(),
                    "mesh face {face_index} index {index} is out of bounds"
                );
            }
            edges.push(ordered_edge(a, b));
            edges.push(ordered_edge(b, c));
            edges.push(ordered_edge(c, a));
        }
        edges.par_sort_unstable();

        let mut locked = vec![false; mesh.positions.len()];
        let mut read = 0;
        let mut write = 0;
        while read < edges.len() {
            let edge = edges[read];
            let mut end = read + 1;
            while end < edges.len() && edges[end] == edge {
                end += 1;
            }
            let incidence = end - read;
            if incidence > 2 || (lock_boundaries && incidence == 1) {
                locked[edge[0] as usize] = true;
                locked[edge[1] as usize] = true;
            }
            edges[write] = edge;
            write += 1;
            read = end;
        }
        edges.truncate(write);

        let mut degrees = vec![0_usize; mesh.positions.len()];
        for &[a, b] in &edges {
            degrees[a as usize] += 1;
            degrees[b as usize] += 1;
        }
        let mut offsets = Vec::with_capacity(mesh.positions.len() + 1);
        offsets.push(0_usize);
        for degree in degrees {
            let next = offsets
                .last()
                .copied()
                .unwrap_or(0)
                .checked_add(degree)
                .context("mesh adjacency size overflow")?;
            offsets.push(next);
        }

        let neighbor_count = offsets.last().copied().unwrap_or(0);
        let mut neighbors = vec![0_u32; neighbor_count];
        let mut cursors = offsets[..mesh.positions.len()].to_vec();
        for [a, b] in edges {
            neighbors[cursors[a as usize]] = b;
            cursors[a as usize] += 1;
            neighbors[cursors[b as usize]] = a;
            cursors[b as usize] += 1;
        }

        Ok(Self {
            offsets,
            neighbors,
            locked,
        })
    }

    fn neighbors(&self, vertex: usize) -> &[u32] {
        &self.neighbors[self.offsets[vertex]..self.offsets[vertex + 1]]
    }

    fn degree(&self, vertex: usize) -> usize {
        self.offsets[vertex + 1] - self.offsets[vertex]
    }
}

fn validate_options(options: VertexRelaxOptions) -> Result<()> {
    ensure!(
        options.lambda.is_finite() && (0.0..=1.0).contains(&options.lambda),
        "relaxation lambda must be finite and in [0, 1]"
    );
    ensure!(
        options.mu.is_finite() && (-1.0..=0.0).contains(&options.mu),
        "relaxation mu must be finite and in [-1, 0]"
    );
    ensure!(
        options.max_step_ratio.is_finite() && options.max_step_ratio >= 0.0,
        "maximum relaxation step ratio must be finite and non-negative"
    );
    Ok(())
}

fn validate_mesh(mesh: &Mesh3D) -> Result<()> {
    ensure!(
        mesh.positions.len() <= u32::MAX as usize,
        "mesh vertex count exceeds u32"
    );
    ensure!(
        mesh.uv.is_empty() || mesh.uv.len() == mesh.positions.len(),
        "mesh uv buffer must be empty or match position count"
    );
    for (index, position) in mesh.positions.iter().enumerate() {
        ensure!(
            position.iter().all(|component| component.is_finite()),
            "mesh position {index} contains a non-finite component"
        );
    }
    Ok(())
}

fn ordered_edge(a: u32, b: u32) -> [u32; 2] {
    if a < b { [a, b] } else { [b, a] }
}

fn mean_edge_lengths(positions: &[[f32; 3]], adjacency: &MeshAdjacency) -> Vec<f64> {
    let mut result = vec![0.0; positions.len()];
    result
        .par_iter_mut()
        .enumerate()
        .for_each(|(vertex, mean_length)| {
            let neighbors = adjacency.neighbors(vertex);
            if neighbors.is_empty() {
                return;
            }
            let position = positions[vertex];
            let sum = neighbors
                .iter()
                .map(|&neighbor| distance(position, positions[neighbor as usize]))
                .sum::<f64>();
            *mean_length = sum / neighbors.len() as f64;
        });
    result
}

fn relaxation_pass(
    source: &[[f32; 3]],
    destination: &mut [[f32; 3]],
    adjacency: &MeshAdjacency,
    mean_edge_lengths: &[f64],
    coefficient: f32,
    max_step_ratio: f32,
) -> Result<()> {
    destination
        .par_iter_mut()
        .enumerate()
        .try_for_each(|(vertex, output)| -> Result<()> {
            let position = source[vertex];
            let neighbors = adjacency.neighbors(vertex);
            if adjacency.locked[vertex] || neighbors.is_empty() || coefficient == 0.0 {
                *output = position;
                return Ok(());
            }

            let mut centroid = [0.0_f64; 3];
            for &neighbor in neighbors {
                let neighbor = source[neighbor as usize];
                centroid[0] += f64::from(neighbor[0]);
                centroid[1] += f64::from(neighbor[1]);
                centroid[2] += f64::from(neighbor[2]);
            }
            let inverse_degree = 1.0 / neighbors.len() as f64;
            centroid[0] *= inverse_degree;
            centroid[1] *= inverse_degree;
            centroid[2] *= inverse_degree;

            let coefficient = f64::from(coefficient);
            let mut delta = [
                (centroid[0] - f64::from(position[0])) * coefficient,
                (centroid[1] - f64::from(position[1])) * coefficient,
                (centroid[2] - f64::from(position[2])) * coefficient,
            ];
            let length = vector_length(delta);
            let max_length = mean_edge_lengths[vertex] * f64::from(max_step_ratio);
            if length > max_length && length > 0.0 {
                let scale = max_length / length;
                delta[0] *= scale;
                delta[1] *= scale;
                delta[2] *= scale;
            }

            let relaxed = [
                f64::from(position[0]) + delta[0],
                f64::from(position[1]) + delta[1],
                f64::from(position[2]) + delta[2],
            ];
            ensure!(
                relaxed.iter().all(|component| component.is_finite())
                    && relaxed
                        .iter()
                        .all(|&component| component >= f32::MIN as f64
                            && component <= f32::MAX as f64),
                "relaxation produced a non-representable vertex {vertex}"
            );
            *output = [relaxed[0] as f32, relaxed[1] as f32, relaxed[2] as f32];
            Ok(())
        })
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f64 {
    vector_length([
        f64::from(a[0]) - f64::from(b[0]),
        f64::from(a[1]) - f64::from(b[1]),
        f64::from(a[2]) - f64::from(b[2]),
    ])
}

fn vector_length(value: [f64; 3]) -> f64 {
    value[0]
        .mul_add(value[0], value[1].mul_add(value[1], value[2] * value[2]))
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open_fan() -> Mesh3D {
        Mesh3D {
            positions: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.2, 0.2, 0.0],
            ],
            uv: vec![[0.0, 0.0]; 5],
            indices: vec![[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        }
    }

    #[test]
    fn relaxes_interior_vertex_and_locks_open_boundary() -> Result<()> {
        let mut mesh = open_fan();
        let boundary = mesh.positions[..4].to_vec();
        let result = relax_mesh_vertices(
            &mut mesh,
            VertexRelaxOptions {
                iterations: 1,
                max_step_ratio: 1.0,
                ..VertexRelaxOptions::default()
            },
        )?;

        assert_eq!(result.relaxed_vertex_count, 1);
        assert_eq!(result.locked_vertex_count, 4);
        assert_eq!(&mesh.positions[..4], boundary);
        assert!(mesh.positions[4][0] > 0.2 && mesh.positions[4][0] < 0.5);
        assert!(mesh.positions[4][1] > 0.2 && mesh.positions[4][1] < 0.5);
        assert_eq!(mesh.positions[4][2], 0.0);
        Ok(())
    }

    #[test]
    fn preserves_indices_and_uvs() -> Result<()> {
        let mut mesh = open_fan();
        let indices = mesh.indices.clone();
        let uv = mesh.uv.clone();
        relax_mesh_vertices(&mut mesh, VertexRelaxOptions::default())?;
        assert_eq!(mesh.indices, indices);
        assert_eq!(mesh.uv, uv);
        Ok(())
    }

    #[test]
    fn rejects_repeated_face_indices() {
        let mut mesh = Mesh3D {
            positions: vec![[0.0; 3], [1.0, 0.0, 0.0]],
            uv: Vec::new(),
            indices: vec![[0, 0, 1]],
        };
        assert!(relax_mesh_vertices(&mut mesh, VertexRelaxOptions::default()).is_err());
    }
}
