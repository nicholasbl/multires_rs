//! Triangle-mesh decimation backed by meshoptimizer.
//!
//! This module keeps the native meshoptimizer interface private and exposes
//! operations directly on [`Mesh3D`].

use std::{
    mem, ptr,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use meshopt::{SimplifyOptions, VertexDataAdapter};
use rayon::prelude::*;

use crate::{Mesh3D, RemoveDegenerateTrianglesOptions, remove_degenerate_triangles};

/// Desired final mesh size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DecimateTarget {
    /// Attempt to reduce the mesh to this many triangles.
    FaceCount(usize),
    /// Attempt to retain this fraction of the input triangles, in `(0, 1]`.
    FaceRatio(f32),
}

/// Interpretation of [`DecimateOptions::max_error`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecimateErrorMode {
    /// Error relative to the extent of the mesh. `0.01` permits roughly 1%.
    Relative,
    /// Error in the same physical-space units as vertex positions.
    Absolute,
}

/// Controls topology-aware parallel simplification of large meshes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecimateParallelOptions {
    /// Minimum input triangle count that activates the grouped path.
    pub min_face_count: usize,
    /// Maximum vertex count of the meshlets used as partitioning seeds.
    pub meshlet_max_vertices: usize,
    /// Maximum triangle count of the meshlets used as partitioning seeds.
    pub meshlet_max_triangles: usize,
    /// Target number of meshlets in each independently simplified group.
    pub partition_size: usize,
    /// Fraction of the requested triangle removal reserved for the final pass.
    pub final_pass_reduction_fraction: f32,
    /// Fraction of the absolute error budget available to the grouped pass.
    pub group_error_fraction: f32,
}

impl Default for DecimateParallelOptions {
    fn default() -> Self {
        Self {
            min_face_count: 100_000,
            meshlet_max_vertices: 64,
            meshlet_max_triangles: 124,
            partition_size: 64,
            final_pass_reduction_fraction: 0.15,
            group_error_fraction: 0.5,
        }
    }
}

/// Options controlling mesh decimation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecimateOptions {
    /// Requested final mesh size.
    pub target: DecimateTarget,
    /// Maximum meshoptimizer simplification error.
    ///
    /// This is a quadric-derived error estimate, not a strict Hausdorff bound.
    pub max_error: f32,
    /// Whether `max_error` is relative to mesh extent or in physical units.
    pub error_mode: DecimateErrorMode,
    /// Relative importance of the two UV/scientific quantity channels.
    ///
    /// A zero weight excludes that channel from simplification and prevents
    /// meshoptimizer from updating it. Ignored when the mesh has no UV buffer.
    pub uv_weights: [f32; 2],
    /// Keep vertices on open mesh boundaries fixed.
    pub lock_boundaries: bool,
    /// Prefer more regular triangle shapes at some geometric-quality cost.
    pub regularize: bool,
    /// Grouped parallel simplification for large meshes. `None` always uses the
    /// single-pass serial simplifier.
    pub parallel: Option<DecimateParallelOptions>,
}

impl Default for DecimateOptions {
    fn default() -> Self {
        Self {
            target: DecimateTarget::FaceRatio(0.5),
            max_error: 0.01,
            error_mode: DecimateErrorMode::Relative,
            uv_weights: [1.0, 1.0],
            lock_boundaries: true,
            regularize: false,
            parallel: Some(DecimateParallelOptions::default()),
        }
    }
}

/// Summary of a decimation operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecimateResult {
    /// Triangle count before decimation.
    pub original_face_count: usize,
    /// Triangle count after decimation.
    pub final_face_count: usize,
    /// Vertex count before decimation.
    pub original_vertex_count: usize,
    /// Referenced vertex count after decimation and compaction.
    pub final_vertex_count: usize,
    /// Error reported by meshoptimizer, in the requested error mode.
    pub error: f32,
    /// True when the requested face count was reached.
    pub reached_target: bool,
    /// Whether topology-aware grouped simplification was used.
    pub used_parallel_path: bool,
    /// Number of independently simplified groups, or zero on the serial path.
    pub group_count: usize,
    /// Triangle count entering the final simplification pass.
    pub intermediate_face_count: usize,
    /// Time spent in each stage of this decimation call.
    pub timings: DecimateTimings,
}

/// Stage timings for [`decimate_mesh`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecimateTimings {
    /// Validation of the input mesh and decimation options.
    pub input_validation: Duration,
    /// Meshlet construction used to seed topology-aware groups.
    pub meshlet_build: Duration,
    /// Group construction with meshoptimizer's cluster partitioner.
    pub partitioning: Duration,
    /// Rayon wall-clock time for independent, border-locked group simplification.
    pub group_simplification: Duration,
    /// Concatenation of simplified group index buffers.
    pub group_merge: Duration,
    /// Final global simplification after group boundaries are reconnected.
    pub final_simplification: Duration,
    /// The meshoptimizer simplification call and result-size checks.
    ///
    /// This is zero when the input is empty or already meets the target.
    pub simplification: Duration,
    /// Validation of meshoptimizer's output.
    ///
    /// This is zero when meshoptimizer was not called.
    pub output_validation: Duration,
    /// Removal and remapping of unreferenced vertices after simplification.
    pub compaction: Duration,
    /// Wall-clock duration of the complete decimation call.
    pub total: Duration,
}

/// Options for cleaning and decimating a mesh in one call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecimatePipelineOptions {
    /// Threshold used to discard degenerate faces before simplification.
    pub remove_degenerate: RemoveDegenerateTrianglesOptions,
    /// Options passed to the decimation stage.
    pub decimate: DecimateOptions,
}

impl Default for DecimatePipelineOptions {
    fn default() -> Self {
        Self {
            remove_degenerate: RemoveDegenerateTrianglesOptions { area_epsilon: 0.0 },
            decimate: DecimateOptions::default(),
        }
    }
}

/// Summary of the complete cleanup and decimation pipeline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecimatePipelineResult {
    /// Triangle count before any cleanup.
    pub original_face_count: usize,
    /// Vertex count before any cleanup.
    pub original_vertex_count: usize,
    /// Number of degenerate faces removed before decimation.
    pub removed_degenerate_faces: usize,
    /// Number of unreferenced vertices removed before decimation.
    pub removed_unreferenced_vertices: usize,
    /// Result of the final decimation stage.
    pub decimation: DecimateResult,
    /// Time spent in each stage of the complete pipeline.
    pub timings: DecimatePipelineTimings,
}

/// Stage timings for [`decimate_mesh_pipeline`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DecimatePipelineTimings {
    /// Removal of repeated-index and small-area triangles.
    pub degenerate_removal: Duration,
    /// Removal and remapping of unreferenced vertices before decimation.
    pub pre_decimation_compaction: Duration,
    /// The complete [`decimate_mesh`] call.
    pub decimation: Duration,
    /// Wall-clock duration of the complete cleanup and decimation pipeline.
    pub total: Duration,
}

/// Decimate an indexed triangle mesh in place.
///
/// Meshoptimizer attempts to preserve topology and may stop before the target
/// when constrained by topology or `max_error`. Surviving vertices are compacted
/// after simplification, and UV values are updated alongside positions when
/// their corresponding weight is nonzero.
pub fn decimate_mesh(mesh: &mut Mesh3D, options: DecimateOptions) -> Result<DecimateResult> {
    let total_start = Instant::now();
    let stage_start = Instant::now();
    validate_mesh(mesh)?;
    validate_options(options)?;
    let input_validation = stage_start.elapsed();

    let original_face_count = mesh.indices.len();
    let original_vertex_count = mesh.positions.len();
    if original_face_count == 0 {
        let stage_start = Instant::now();
        compact_vertices(mesh)?;
        let compaction = stage_start.elapsed();
        let final_vertex_count = mesh.positions.len();
        return Ok(DecimateResult {
            original_face_count,
            final_face_count: 0,
            original_vertex_count,
            final_vertex_count,
            error: 0.0,
            reached_target: true,
            used_parallel_path: false,
            group_count: 0,
            intermediate_face_count: 0,
            timings: DecimateTimings {
                input_validation,
                compaction,
                total: total_start.elapsed(),
                ..DecimateTimings::default()
            },
        });
    }

    let target_face_count = resolve_target(options.target, original_face_count)?;
    if target_face_count >= original_face_count {
        let stage_start = Instant::now();
        compact_vertices(mesh)?;
        let compaction = stage_start.elapsed();
        return Ok(DecimateResult {
            original_face_count,
            final_face_count: original_face_count,
            original_vertex_count,
            final_vertex_count: mesh.positions.len(),
            error: 0.0,
            reached_target: true,
            used_parallel_path: false,
            group_count: 0,
            intermediate_face_count: original_face_count,
            timings: DecimateTimings {
                input_validation,
                compaction,
                total: total_start.elapsed(),
                ..DecimateTimings::default()
            },
        });
    }

    if let Some(parallel) = options
        .parallel
        .filter(|parallel| original_face_count >= parallel.min_face_count)
    {
        return decimate_mesh_grouped(
            mesh,
            options,
            parallel,
            target_face_count,
            original_face_count,
            original_vertex_count,
            input_validation,
            total_start,
        );
    }

    let mut simplify_options = SimplifyOptions::empty();
    if options.lock_boundaries {
        simplify_options.insert(SimplifyOptions::LockBorder);
    }
    if options.regularize {
        simplify_options.insert(SimplifyOptions::Regularize);
    }
    if options.error_mode == DecimateErrorMode::Absolute {
        simplify_options.insert(SimplifyOptions::ErrorAbsolute);
    }

    let use_uv = !mesh.uv.is_empty();
    let (attributes, attribute_stride, weights, attribute_count) = if use_uv {
        (
            mesh.uv.as_mut_ptr().cast::<f32>(),
            mem::size_of::<[f32; 2]>(),
            options.uv_weights.as_ptr(),
            options.uv_weights.len(),
        )
    } else {
        (ptr::null_mut(), 0, ptr::null(), 0)
    };

    let mut result_error = 0.0;
    let index_count = mesh.indices.len() * 3;
    let target_index_count = target_face_count
        .checked_mul(3)
        .context("target index count overflowed usize")?;

    // SAFETY: `validate_mesh` ensures that every index is in bounds and that
    // position/UV buffers have valid lengths and finite values. Arrays of
    // `[u32; 3]`, `[f32; 3]`, and `[f32; 2]` are contiguous, and all pointers
    // remain valid for the duration of this call. Counts and strides exactly
    // match their respective buffers. The optional lock pointer is null.
    let stage_start = Instant::now();
    let simplified_index_count = unsafe {
        meshopt::ffi::meshopt_simplifyWithUpdate(
            mesh.indices.as_mut_ptr().cast::<u32>(),
            index_count,
            mesh.positions.as_mut_ptr().cast::<f32>(),
            mesh.positions.len(),
            mem::size_of::<[f32; 3]>(),
            attributes,
            attribute_stride,
            weights,
            attribute_count,
            ptr::null(),
            target_index_count,
            options.max_error,
            simplify_options.bits(),
            &mut result_error,
        )
    };

    ensure!(
        simplified_index_count <= index_count && simplified_index_count % 3 == 0,
        "meshoptimizer returned an invalid index count {simplified_index_count}"
    );
    mesh.indices.truncate(simplified_index_count / 3);
    let simplification = stage_start.elapsed();

    let stage_start = Instant::now();
    validate_mesh(mesh).context("validating meshoptimizer output")?;
    let output_validation = stage_start.elapsed();

    let stage_start = Instant::now();
    compact_vertices(mesh)?;
    let compaction = stage_start.elapsed();

    let final_face_count = mesh.indices.len();
    Ok(DecimateResult {
        original_face_count,
        final_face_count,
        original_vertex_count,
        final_vertex_count: mesh.positions.len(),
        error: result_error,
        reached_target: final_face_count <= target_face_count,
        used_parallel_path: false,
        group_count: 0,
        intermediate_face_count: original_face_count,
        timings: DecimateTimings {
            input_validation,
            simplification,
            final_simplification: simplification,
            output_validation,
            compaction,
            total: total_start.elapsed(),
            ..DecimateTimings::default()
        },
    })
}

#[derive(Debug)]
struct GroupSimplification {
    indices: Vec<u32>,
    error: f32,
}

#[allow(clippy::too_many_arguments)]
fn decimate_mesh_grouped(
    mesh: &mut Mesh3D,
    options: DecimateOptions,
    parallel: DecimateParallelOptions,
    target_face_count: usize,
    original_face_count: usize,
    original_vertex_count: usize,
    input_validation: Duration,
    total_start: Instant,
) -> Result<DecimateResult> {
    let simplification_start = Instant::now();
    let adapter = position_adapter(&mesh.positions)?;
    let scale = meshopt::simplify_scale(&adapter);
    let absolute_error_limit = match options.error_mode {
        DecimateErrorMode::Relative => options.max_error * scale,
        DecimateErrorMode::Absolute => options.max_error,
    };

    let stage_start = Instant::now();
    let meshlets = meshopt::build_meshlets(
        faces_as_indices(&mesh.indices),
        &adapter,
        parallel.meshlet_max_vertices,
        parallel.meshlet_max_triangles,
        0.0,
    );
    let meshlet_build = stage_start.elapsed();
    ensure!(!meshlets.is_empty(), "meshlet builder returned no meshlets");

    let stage_start = Instant::now();
    let position_remap = meshopt::generate_position_remap(&adapter);
    // Partitioning only needs the unique vertices referenced by each meshlet,
    // not the full triangle stream. This keeps the temporary adjacency input
    // much smaller on the very large meshes this path is intended for.
    let mut cluster_indices = Vec::with_capacity(meshlets.vertices.len());
    let mut cluster_index_counts = Vec::with_capacity(meshlets.len());
    for meshlet in meshlets.iter() {
        cluster_index_counts.push(
            u32::try_from(meshlet.vertices.len()).context("meshlet vertex count exceeds u32")?,
        );
        for &global_index in meshlet.vertices {
            cluster_indices.push(position_remap[global_index as usize]);
        }
    }
    let mut partition_ids = vec![0; meshlets.len()];
    let group_count = meshopt::partition_clusters_with_positions(
        &mut partition_ids,
        &cluster_indices,
        &cluster_index_counts,
        &adapter,
        parallel.partition_size,
    );
    ensure!(group_count > 0, "cluster partitioner returned no groups");
    let mut groups = vec![Vec::new(); group_count];
    for (meshlet_id, &partition_id) in partition_ids.iter().enumerate() {
        let partition_id = partition_id as usize;
        ensure!(
            partition_id < groups.len(),
            "cluster partitioner returned invalid group {partition_id}"
        );
        groups[partition_id].push(meshlet_id);
    }
    drop(cluster_indices);
    drop(cluster_index_counts);
    drop(partition_ids);
    drop(position_remap);
    let partitioning = stage_start.elapsed();

    // The meshlets retain every original triangle and can reconstruct the input
    // if a parallel task fails. Releasing the original index allocation here
    // avoids carrying two full-sized index buffers through the grouped pass.
    mesh.indices = Vec::new();

    let removed_face_count = original_face_count - target_face_count;
    let reserved_for_final = ((removed_face_count as f64)
        * f64::from(parallel.final_pass_reduction_fraction))
    .ceil() as usize;
    let intermediate_target = target_face_count
        .saturating_add(reserved_for_final)
        .min(original_face_count);
    let intermediate_ratio = intermediate_target as f64 / original_face_count as f64;
    let group_error_limit = absolute_error_limit * parallel.group_error_fraction;

    let mut group_options =
        SimplifyOptions::LockBorder | SimplifyOptions::Sparse | SimplifyOptions::ErrorAbsolute;
    if options.regularize {
        group_options.insert(SimplifyOptions::Regularize);
    }

    let stage_start = Instant::now();
    let group_results = groups
        .par_iter()
        .map(|meshlet_ids| {
            let mut indices = Vec::new();
            for &meshlet_id in meshlet_ids {
                let meshlet = meshlets.get(meshlet_id);
                indices.reserve(meshlet.triangles.len());
                for &local_index in meshlet.triangles {
                    indices.push(meshlet.vertices[usize::from(local_index)]);
                }
            }
            let face_count = indices.len() / 3;
            let target_faces = ((face_count as f64) * intermediate_ratio).floor().max(1.0) as usize;
            simplify_group(
                &indices,
                &mesh.positions,
                &mesh.uv,
                options.uv_weights,
                target_faces,
                group_error_limit,
                group_options,
            )
        })
        .collect::<Result<Vec<_>>>();
    let group_simplification = stage_start.elapsed();
    let group_results = match group_results {
        Ok(results) => results,
        Err(error) => {
            mesh.indices = meshlet_faces(&meshlets);
            return Err(error.context("simplifying mesh groups"));
        }
    };

    let max_group_error = group_results
        .iter()
        .map(|group| group.error)
        .fold(0.0_f32, f32::max);
    let intermediate_face_count = group_results
        .iter()
        .map(|group| group.indices.len() / 3)
        .sum();

    let stage_start = Instant::now();
    mesh.indices = Vec::with_capacity(intermediate_face_count);
    for group in group_results {
        for triangle in group.indices.chunks_exact(3) {
            mesh.indices.push([triangle[0], triangle[1], triangle[2]]);
        }
    }
    let group_merge = stage_start.elapsed();
    drop(meshlets);

    let mut final_error = 0.0;
    let stage_start = Instant::now();
    if mesh.indices.len() > target_face_count {
        let mut final_options = SimplifyOptions::Sparse | SimplifyOptions::ErrorAbsolute;
        if options.lock_boundaries {
            final_options.insert(SimplifyOptions::LockBorder);
        }
        if options.regularize {
            final_options.insert(SimplifyOptions::Regularize);
        }
        let remaining_error = (absolute_error_limit - max_group_error).max(0.0);
        simplify_mesh_with_update(
            mesh,
            options.uv_weights,
            target_face_count,
            remaining_error,
            final_options,
            &mut final_error,
        )?;
    }
    let final_simplification = stage_start.elapsed();
    let simplification = simplification_start.elapsed();

    let stage_start = Instant::now();
    validate_mesh(mesh).context("validating grouped meshoptimizer output")?;
    let output_validation = stage_start.elapsed();

    let stage_start = Instant::now();
    compact_vertices(mesh)?;
    let compaction = stage_start.elapsed();

    let combined_absolute_error = max_group_error + final_error;
    let result_error = match options.error_mode {
        DecimateErrorMode::Relative if scale > 0.0 => combined_absolute_error / scale,
        DecimateErrorMode::Relative => 0.0,
        DecimateErrorMode::Absolute => combined_absolute_error,
    };
    let final_face_count = mesh.indices.len();
    Ok(DecimateResult {
        original_face_count,
        final_face_count,
        original_vertex_count,
        final_vertex_count: mesh.positions.len(),
        error: result_error,
        reached_target: final_face_count <= target_face_count,
        used_parallel_path: true,
        group_count,
        intermediate_face_count,
        timings: DecimateTimings {
            input_validation,
            meshlet_build,
            partitioning,
            group_simplification,
            group_merge,
            final_simplification,
            simplification,
            output_validation,
            compaction,
            total: total_start.elapsed(),
        },
    })
}

fn simplify_group(
    indices: &[u32],
    positions: &[[f32; 3]],
    uv: &[[f32; 2]],
    uv_weights: [f32; 2],
    target_face_count: usize,
    max_error: f32,
    options: SimplifyOptions,
) -> Result<GroupSimplification> {
    if target_face_count >= indices.len() / 3 {
        return Ok(GroupSimplification {
            indices: indices.to_vec(),
            error: 0.0,
        });
    }
    let mut output = vec![0_u32; indices.len()];
    let (attributes, attribute_stride, weights, attribute_count) = if uv.is_empty() {
        (ptr::null(), 0, ptr::null(), 0)
    } else {
        (
            uv.as_ptr().cast::<f32>(),
            mem::size_of::<[f32; 2]>(),
            uv_weights.as_ptr(),
            uv_weights.len(),
        )
    };
    let mut result_error = 0.0;
    let target_index_count = target_face_count
        .checked_mul(3)
        .context("group target index count overflowed usize")?;
    // SAFETY: validation performed before partitioning guarantees valid finite
    // positions, attributes, and indices. This call only reads the shared
    // vertex buffers and writes to this task's private output allocation.
    let output_count = unsafe {
        meshopt::ffi::meshopt_simplifyWithAttributes(
            output.as_mut_ptr(),
            indices.as_ptr(),
            indices.len(),
            positions.as_ptr().cast::<f32>(),
            positions.len(),
            mem::size_of::<[f32; 3]>(),
            attributes,
            attribute_stride,
            weights,
            attribute_count,
            ptr::null(),
            target_index_count,
            max_error,
            options.bits(),
            &mut result_error,
        )
    };
    ensure!(
        output_count <= indices.len() && output_count.is_multiple_of(3),
        "meshoptimizer returned invalid group index count {output_count}"
    );
    output.truncate(output_count);
    Ok(GroupSimplification {
        indices: output,
        error: result_error,
    })
}

fn simplify_mesh_with_update(
    mesh: &mut Mesh3D,
    uv_weights: [f32; 2],
    target_face_count: usize,
    max_error: f32,
    options: SimplifyOptions,
    result_error: &mut f32,
) -> Result<()> {
    let (attributes, attribute_stride, weights, attribute_count) = if mesh.uv.is_empty() {
        (ptr::null_mut(), 0, ptr::null(), 0)
    } else {
        (
            mesh.uv.as_mut_ptr().cast::<f32>(),
            mem::size_of::<[f32; 2]>(),
            uv_weights.as_ptr(),
            uv_weights.len(),
        )
    };
    let index_count = mesh.indices.len() * 3;
    let target_index_count = target_face_count
        .checked_mul(3)
        .context("final target index count overflowed usize")?;
    // SAFETY: the grouped path starts from a validated mesh, group outputs
    // retain valid global indices, and this final call is single-threaded.
    let output_count = unsafe {
        meshopt::ffi::meshopt_simplifyWithUpdate(
            mesh.indices.as_mut_ptr().cast::<u32>(),
            index_count,
            mesh.positions.as_mut_ptr().cast::<f32>(),
            mesh.positions.len(),
            mem::size_of::<[f32; 3]>(),
            attributes,
            attribute_stride,
            weights,
            attribute_count,
            ptr::null(),
            target_index_count,
            max_error,
            options.bits(),
            result_error,
        )
    };
    ensure!(
        output_count <= index_count && output_count.is_multiple_of(3),
        "meshoptimizer returned invalid final index count {output_count}"
    );
    mesh.indices.truncate(output_count / 3);
    Ok(())
}

fn position_adapter(positions: &[[f32; 3]]) -> Result<VertexDataAdapter<'_>> {
    let byte_len = positions
        .len()
        .checked_mul(mem::size_of::<[f32; 3]>())
        .context("position buffer byte length overflow")?;
    // SAFETY: `[f32; 3]` is contiguous and the returned bytes borrow the
    // original slice for exactly its initialized extent.
    let bytes = unsafe { std::slice::from_raw_parts(positions.as_ptr().cast::<u8>(), byte_len) };
    VertexDataAdapter::new(bytes, mem::size_of::<[f32; 3]>(), 0)
        .map_err(|error| anyhow::anyhow!("creating meshoptimizer vertex adapter: {error}"))
}

fn faces_as_indices(faces: &[[u32; 3]]) -> &[u32] {
    // SAFETY: `[u32; 3]` has the same alignment as `u32` and consists of three
    // contiguous initialized `u32` values with no inter-element padding.
    unsafe { std::slice::from_raw_parts(faces.as_ptr().cast::<u32>(), faces.len() * 3) }
}

fn meshlet_faces(meshlets: &meshopt::Meshlets) -> Vec<[u32; 3]> {
    let face_count = meshlets
        .iter()
        .map(|meshlet| meshlet.triangles.len() / 3)
        .sum();
    let mut faces = Vec::with_capacity(face_count);
    for meshlet in meshlets.iter() {
        for triangle in meshlet.triangles.chunks_exact(3) {
            faces.push([
                meshlet.vertices[usize::from(triangle[0])],
                meshlet.vertices[usize::from(triangle[1])],
                meshlet.vertices[usize::from(triangle[2])],
            ]);
        }
    }
    faces
}

/// Clean and decimate an indexed triangle mesh in place.
///
/// The pipeline removes degenerate triangles, removes vertices no longer
/// referenced by a triangle, and finally decimates and compacts the mesh.
pub fn decimate_mesh_pipeline(
    mesh: &mut Mesh3D,
    options: DecimatePipelineOptions,
) -> Result<DecimatePipelineResult> {
    let total_start = Instant::now();
    let original_face_count = mesh.indices.len();
    let original_vertex_count = mesh.positions.len();

    let stage_start = Instant::now();
    let removed_degenerate_faces = remove_degenerate_triangles(mesh, options.remove_degenerate)
        .context("removing degenerate triangles before decimation")?;
    let degenerate_removal = stage_start.elapsed();

    let stage_start = Instant::now();
    let removed_unreferenced_vertices = compact_vertices(mesh)?;
    let pre_decimation_compaction = stage_start.elapsed();

    let stage_start = Instant::now();
    let decimation = decimate_mesh(mesh, options.decimate)?;
    let decimation_time = stage_start.elapsed();

    Ok(DecimatePipelineResult {
        original_face_count,
        original_vertex_count,
        removed_degenerate_faces,
        removed_unreferenced_vertices,
        decimation,
        timings: DecimatePipelineTimings {
            degenerate_removal,
            pre_decimation_compaction,
            decimation: decimation_time,
            total: total_start.elapsed(),
        },
    })
}

fn validate_mesh(mesh: &Mesh3D) -> Result<()> {
    ensure!(
        mesh.uv.is_empty() || mesh.uv.len() == mesh.positions.len(),
        "mesh uv buffer must be empty or match position count"
    );
    for (index, position) in mesh.positions.iter().enumerate() {
        ensure!(
            position.iter().all(|value| value.is_finite()),
            "mesh position {index} contains a non-finite component"
        );
    }
    for (index, uv) in mesh.uv.iter().enumerate() {
        ensure!(
            uv.iter().all(|value| value.is_finite()),
            "mesh uv {index} contains a non-finite component"
        );
    }
    for (face_index, face) in mesh.indices.iter().enumerate() {
        for &vertex_index in face {
            ensure!(
                (vertex_index as usize) < mesh.positions.len(),
                "mesh face {face_index} index {vertex_index} is out of bounds"
            );
        }
    }
    Ok(())
}

fn validate_options(options: DecimateOptions) -> Result<()> {
    ensure!(
        options.max_error.is_finite() && options.max_error >= 0.0,
        "maximum decimation error must be finite and non-negative"
    );
    if options.error_mode == DecimateErrorMode::Relative {
        ensure!(
            options.max_error <= 1.0,
            "relative maximum decimation error must not exceed 1"
        );
    }
    ensure!(
        options
            .uv_weights
            .iter()
            .all(|weight| weight.is_finite() && *weight >= 0.0),
        "UV weights must be finite and non-negative"
    );
    match options.target {
        DecimateTarget::FaceCount(count) => {
            ensure!(count > 0, "target face count must be positive");
        }
        DecimateTarget::FaceRatio(ratio) => ensure!(
            ratio.is_finite() && ratio > 0.0 && ratio <= 1.0,
            "target face ratio must be finite and in (0, 1]"
        ),
    }
    if let Some(parallel) = options.parallel {
        ensure!(
            parallel.min_face_count > 0,
            "parallel decimation threshold must be positive"
        );
        ensure!(
            (3..=256).contains(&parallel.meshlet_max_vertices),
            "meshlet maximum vertices must be in [3, 256]"
        );
        ensure!(
            parallel.meshlet_max_triangles > 0
                && parallel.meshlet_max_triangles <= 512
                && parallel.meshlet_max_triangles.is_multiple_of(4),
            "meshlet maximum triangles must be a positive multiple of 4 no greater than 512"
        );
        ensure!(
            parallel.partition_size > 0,
            "meshlet partition size must be positive"
        );
        ensure!(
            parallel.final_pass_reduction_fraction.is_finite()
                && (0.0..=1.0).contains(&parallel.final_pass_reduction_fraction),
            "final-pass reduction fraction must be finite and in [0, 1]"
        );
        ensure!(
            parallel.group_error_fraction.is_finite()
                && (0.0..=1.0).contains(&parallel.group_error_fraction),
            "group error fraction must be finite and in [0, 1]"
        );
    }
    Ok(())
}

fn resolve_target(target: DecimateTarget, face_count: usize) -> Result<usize> {
    match target {
        DecimateTarget::FaceCount(count) => Ok(count),
        DecimateTarget::FaceRatio(ratio) => {
            let count = ((face_count as f64) * f64::from(ratio)).floor();
            ensure!(count <= usize::MAX as f64, "target face count is too large");
            Ok((count as usize).max(1))
        }
    }
}

fn compact_vertices(mesh: &mut Mesh3D) -> Result<usize> {
    let old_vertex_count = mesh.positions.len();
    if mesh.indices.is_empty() {
        mesh.positions.clear();
        mesh.uv.clear();
        return Ok(old_vertex_count);
    }

    let use_uv = !mesh.uv.is_empty();
    let mut remap = vec![u32::MAX; old_vertex_count];
    let mut positions = Vec::new();
    let mut uv = Vec::new();

    for (face_index, face) in mesh.indices.iter_mut().enumerate() {
        for vertex_index in face {
            let old_index = *vertex_index as usize;
            let mapped = remap.get_mut(old_index).with_context(|| {
                format!("mesh face {face_index} index {old_index} is out of bounds")
            })?;
            if *mapped == u32::MAX {
                *mapped =
                    u32::try_from(positions.len()).context("mesh vertex count exceeds u32")?;
                positions.push(mesh.positions[old_index]);
                if use_uv {
                    uv.push(mesh.uv[old_index]);
                }
            }
            *vertex_index = *mapped;
        }
    }

    let removed = old_vertex_count - positions.len();
    mesh.positions = positions;
    mesh.uv = uv;
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planar_grid() -> Mesh3D {
        planar_grid_with_size(4)
    }

    fn planar_grid_with_size(size: u32) -> Mesh3D {
        let mut positions = Vec::new();
        let mut uv = Vec::new();
        for y in 0..=size {
            for x in 0..=size {
                positions.push([x as f32, y as f32, 0.0]);
                uv.push([x as f32 / size as f32, y as f32 / size as f32]);
            }
        }

        let mut indices = Vec::new();
        for y in 0..size {
            for x in 0..size {
                let a = y * (size + 1) + x;
                let b = a + 1;
                let c = a + size + 1;
                let d = c + 1;
                indices.push([a, b, d]);
                indices.push([a, d, c]);
            }
        }

        Mesh3D {
            positions,
            uv,
            indices,
        }
    }

    #[test]
    fn decimates_and_compacts_mesh() -> Result<()> {
        let mut mesh = planar_grid();
        let result = decimate_mesh(
            &mut mesh,
            DecimateOptions {
                target: DecimateTarget::FaceRatio(0.5),
                max_error: 1.0,
                lock_boundaries: false,
                ..DecimateOptions::default()
            },
        )?;

        ensure!(result.final_face_count < result.original_face_count);
        ensure!(!result.used_parallel_path);
        ensure!(result.final_vertex_count <= result.original_vertex_count);
        ensure!(result.timings.total >= result.timings.input_validation);
        ensure!(result.timings.total >= result.timings.simplification);
        ensure!(result.timings.total >= result.timings.output_validation);
        ensure!(result.timings.total >= result.timings.compaction);
        ensure!(mesh.uv.len() == mesh.positions.len());
        ensure!(mesh.indices.iter().flatten().all(|&index| {
            usize::try_from(index).is_ok_and(|index| index < mesh.positions.len())
        }));
        Ok(())
    }

    #[test]
    fn grouped_path_simplifies_meshlet_partitions_in_parallel() -> Result<()> {
        let mut mesh = planar_grid_with_size(24);
        let result = decimate_mesh(
            &mut mesh,
            DecimateOptions {
                target: DecimateTarget::FaceRatio(0.4),
                max_error: 1.0,
                lock_boundaries: false,
                parallel: Some(DecimateParallelOptions {
                    min_face_count: 1,
                    meshlet_max_vertices: 16,
                    meshlet_max_triangles: 12,
                    partition_size: 2,
                    final_pass_reduction_fraction: 0.2,
                    group_error_fraction: 0.5,
                }),
                ..DecimateOptions::default()
            },
        )?;

        ensure!(result.used_parallel_path);
        ensure!(result.group_count > 1);
        ensure!(result.intermediate_face_count < result.original_face_count);
        ensure!(result.final_face_count <= result.intermediate_face_count);
        ensure!(result.timings.meshlet_build > Duration::ZERO);
        ensure!(result.timings.partitioning > Duration::ZERO);
        ensure!(result.timings.group_simplification > Duration::ZERO);
        validate_mesh(&mesh)?;
        Ok(())
    }

    #[test]
    fn grouped_path_supports_position_only_meshes() -> Result<()> {
        let mut mesh = planar_grid_with_size(16);
        mesh.uv.clear();
        let result = decimate_mesh(
            &mut mesh,
            DecimateOptions {
                target: DecimateTarget::FaceRatio(0.5),
                max_error: 1.0,
                lock_boundaries: true,
                parallel: Some(DecimateParallelOptions {
                    min_face_count: 1,
                    meshlet_max_vertices: 16,
                    meshlet_max_triangles: 12,
                    partition_size: 2,
                    ..DecimateParallelOptions::default()
                }),
                ..DecimateOptions::default()
            },
        )?;

        ensure!(result.used_parallel_path);
        ensure!(mesh.uv.is_empty());
        validate_mesh(&mesh)?;
        Ok(())
    }

    #[test]
    fn pipeline_removes_degenerate_faces_and_unreferenced_vertices() -> Result<()> {
        let mut mesh = Mesh3D {
            positions: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            uv: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            indices: vec![[0, 1, 2], [0, 3, 1]],
        };

        let result = decimate_mesh_pipeline(
            &mut mesh,
            DecimatePipelineOptions {
                remove_degenerate: RemoveDegenerateTrianglesOptions { area_epsilon: 0.0 },
                decimate: DecimateOptions {
                    target: DecimateTarget::FaceCount(1),
                    ..DecimateOptions::default()
                },
            },
        )?;

        ensure!(result.original_face_count == 2);
        ensure!(result.original_vertex_count == 4);
        ensure!(result.removed_degenerate_faces == 1);
        ensure!(result.removed_unreferenced_vertices == 1);
        ensure!(result.timings.total >= result.timings.degenerate_removal);
        ensure!(result.timings.total >= result.timings.pre_decimation_compaction);
        ensure!(result.timings.total >= result.timings.decimation);
        ensure!(result.timings.decimation >= result.decimation.timings.total);
        ensure!(mesh.positions.len() == 3);
        ensure!(mesh.indices == vec![[0, 1, 2]]);
        Ok(())
    }

    #[test]
    fn decimates_mesh_without_uvs() -> Result<()> {
        let mut mesh = planar_grid();
        mesh.uv.clear();
        decimate_mesh(
            &mut mesh,
            DecimateOptions {
                target: DecimateTarget::FaceCount(8),
                max_error: 1.0,
                lock_boundaries: false,
                ..DecimateOptions::default()
            },
        )?;

        ensure!(mesh.uv.is_empty());
        ensure!(mesh.indices.len() <= 8);
        Ok(())
    }

    #[test]
    fn empty_mesh_is_compacted_without_calling_meshoptimizer() -> Result<()> {
        let mut mesh = Mesh3D {
            positions: vec![[0.0, 0.0, 0.0]],
            uv: vec![[0.0, 0.0]],
            indices: Vec::new(),
        };
        let result = decimate_mesh(&mut mesh, DecimateOptions::default())?;

        ensure!(mesh.positions.is_empty());
        ensure!(mesh.uv.is_empty());
        ensure!(result.original_vertex_count == 1);
        ensure!(result.final_vertex_count == 0);
        ensure!(result.timings.simplification == Duration::ZERO);
        ensure!(result.timings.output_validation == Duration::ZERO);
        Ok(())
    }
}
