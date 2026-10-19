//! MC33 isosurface extraction from point-associated MSG scalar fields.

mod mc33;
mod sampling;

use std::{
    collections::HashMap,
    ops::RangeInclusive,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use glam::UVec3;
use rayon::prelude::*;

use crate::point_grid::{PointGridLevel, archive_point_grid_levels, point_grid_levels};
use crate::{
    Mesh3D, MsgArchive, MsgDataset,
    sparse::{Aabb3u, CHUNK_SIZE},
};

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Surface {
    pub field: String,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    pub field: String,
    pub range: RangeInclusive<f64>,
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct IsosurfaceOptions {
    pub surface: Surface,
    pub sampled_quantities: Vec<Sample>,
    pub levels: Option<RangeInclusive<usize>>,
    pub flip_winding: bool,
}

#[derive(Debug, Clone, Copy)]
struct SampleRange {
    min: f64,
    max: f64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PerLevelTimings {
    pub mesh_generation_time: Duration,
    pub merge_time: Duration,
    pub winding_time: Duration,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IsosurfaceTimings {
    pub preparation: Duration,
    pub per_level: Vec<PerLevelTimings>,
}

/// Extract an isosurface from one point-associated scalar field.
pub fn isosurface(
    dataset: &MsgDataset,
    options: IsosurfaceOptions,
) -> Result<(Mesh3D, IsosurfaceTimings)> {
    validate_options(dataset, &options)?;
    let start = Instant::now();
    let sampled_names = sampled_names(&options);
    let ranges = sample_ranges(&options);
    let levels = point_grid_levels(
        dataset,
        &options.surface.field,
        &sampled_names,
        options.levels.as_ref(),
    )?;
    mesh_levels(levels, &options, &ranges, start.elapsed())
}

/// Extract an isosurface directly from an archive without materializing a
/// [`MsgDataset`]. Scalar values are reconstructed on demand.
pub fn isosurface_archive(
    archive: &MsgArchive<'_>,
    options: IsosurfaceOptions,
) -> Result<(Mesh3D, IsosurfaceTimings)> {
    validate_archive_options(archive, &options)?;
    let start = Instant::now();
    let sampled_names = sampled_names(&options);
    let ranges = sample_ranges(&options);
    let levels = archive_point_grid_levels(
        archive,
        &options.surface.field,
        &sampled_names,
        options.levels.as_ref(),
        options.surface.value,
    )?;
    mesh_levels(levels, &options, &ranges, start.elapsed())
}

fn sampled_names(options: &IsosurfaceOptions) -> Vec<&str> {
    options
        .sampled_quantities
        .iter()
        .map(|sample| sample.field.as_str())
        .collect()
}

fn sample_ranges(options: &IsosurfaceOptions) -> Vec<SampleRange> {
    options
        .sampled_quantities
        .iter()
        .map(|sample| SampleRange {
            min: *sample.range.start(),
            max: *sample.range.end(),
        })
        .collect()
}

fn mesh_levels(
    levels: Vec<PointGridLevel<'_>>,
    options: &IsosurfaceOptions,
    ranges: &[SampleRange],
    preparation: Duration,
) -> Result<(Mesh3D, IsosurfaceTimings)> {
    let mut mesh = Mesh3D::default();
    let mut timings = Vec::with_capacity(levels.len());
    for level in &levels {
        let start = Instant::now();
        let level_mesh = mesh_level_parallel(level, &ranges, options.surface.value)
            .with_context(|| format!("extracting MSG level {}", level.level_index))?;
        let mesh_generation_time = start.elapsed();
        let start = Instant::now();
        let face_start = mesh.indices.len();
        merge_mesh(&mut mesh, level_mesh)?;
        let merge_time = start.elapsed();
        let start = Instant::now();
        if options.flip_winding {
            for face in &mut mesh.indices[face_start..] {
                face.swap(1, 2);
            }
        }
        timings.push(PerLevelTimings {
            mesh_generation_time,
            merge_time,
            winding_time: start.elapsed(),
        });
    }
    Ok((
        mesh,
        IsosurfaceTimings {
            preparation,
            per_level: timings,
        },
    ))
}

fn validate_archive_options(archive: &MsgArchive<'_>, options: &IsosurfaceOptions) -> Result<()> {
    ensure!(
        !options.surface.field.is_empty(),
        "surface field cannot be empty"
    );
    ensure!(
        archive.field(&options.surface.field).is_some(),
        "surface field {:?} is absent",
        options.surface.field
    );
    ensure!(options.surface.value.is_finite(), "isovalue must be finite");
    ensure!(
        options.sampled_quantities.len() <= 2,
        "at most two sampled fields are supported"
    );
    for sample in &options.sampled_quantities {
        ensure!(
            archive.field(&sample.field).is_some(),
            "sampled field {:?} is absent",
            sample.field
        );
        let (min, max) = (*sample.range.start(), *sample.range.end());
        ensure!(
            min.is_finite() && max.is_finite() && max > min,
            "invalid sampled field range"
        );
    }
    if let Some(levels) = &options.levels {
        ensure!(levels.start() <= levels.end(), "level range is reversed");
        ensure!(
            *levels.end() < archive.level_count(),
            "level range exceeds archive"
        );
    }
    Ok(())
}

fn validate_options(dataset: &MsgDataset, options: &IsosurfaceOptions) -> Result<()> {
    ensure!(
        !options.surface.field.is_empty(),
        "surface field cannot be empty"
    );
    ensure!(
        dataset.field(&options.surface.field).is_some(),
        "surface field {:?} is absent",
        options.surface.field
    );
    ensure!(options.surface.value.is_finite(), "isovalue must be finite");
    ensure!(
        options.sampled_quantities.len() <= 2,
        "at most two sampled fields are supported"
    );
    for sample in &options.sampled_quantities {
        ensure!(
            dataset.field(&sample.field).is_some(),
            "sampled field {:?} is absent",
            sample.field
        );
        let (min, max) = (*sample.range.start(), *sample.range.end());
        ensure!(
            min.is_finite() && max.is_finite() && max > min,
            "invalid sampled field range"
        );
    }
    if let Some(levels) = &options.levels {
        ensure!(levels.start() <= levels.end(), "level range is reversed");
        ensure!(
            *levels.end() < dataset.levels().len(),
            "level range exceeds dataset"
        );
    }
    Ok(())
}

fn mesh_level_parallel(
    level: &PointGridLevel<'_>,
    ranges: &[SampleRange],
    isovalue: f64,
) -> Result<Mesh3D> {
    let cube_bounds = level.active_cubes.bounds();
    if !level
        .value_ranges
        .iter()
        .any(|range| range.min <= isovalue && isovalue <= range.max)
    {
        return Ok(Mesh3D::default());
    }
    let mut chunks = level
        .value_ranges
        .par_iter()
        .filter(|range| range.min <= isovalue && isovalue <= range.max)
        .map(|range| {
            let max = (range.chunk_origin + UVec3::splat(CHUNK_SIZE)).min(cube_bounds);
            let active_aabb = Aabb3u::new(range.chunk_origin, max);
            let mut mesh = Mesh3D::default();
            let mut vertex_keys = Vec::new();
            mc33::mesh_level_aabb(
                level,
                active_aabb,
                ranges,
                isovalue,
                &mut mesh,
                &mut vertex_keys,
            )?;
            Ok((active_aabb, mesh, vertex_keys))
        })
        .collect::<Result<Vec<_>>>()?;
    chunks.sort_by_key(|(aabb, _, _)| (aabb.min.z, aabb.min.y, aabb.min.x));

    let mut mesh = Mesh3D::default();
    let vertex_capacity = chunks.iter().map(|(_, mesh, _)| mesh.positions.len()).sum();
    mesh.positions.reserve(vertex_capacity);
    mesh.uv.reserve(vertex_capacity);
    mesh.indices
        .reserve(chunks.iter().map(|(_, mesh, _)| mesh.indices.len()).sum());
    let mut vertex_ids = HashMap::with_capacity(vertex_capacity);
    for (_, chunk_mesh, keys) in chunks {
        merge_keyed_mesh(&mut mesh, chunk_mesh, keys, &mut vertex_ids)?;
    }
    Ok(mesh)
}

fn merge_keyed_mesh(
    dest: &mut Mesh3D,
    src: Mesh3D,
    keys: Vec<mc33::VertexKey>,
    ids: &mut HashMap<mc33::VertexKey, u32>,
) -> Result<()> {
    ensure!(
        src.positions.len() == src.uv.len() && src.positions.len() == keys.len(),
        "inconsistent keyed mesh buffers"
    );
    let mut remap = Vec::with_capacity(keys.len());
    for ((key, position), uv) in keys.into_iter().zip(src.positions).zip(src.uv) {
        let index = match ids.entry(key) {
            std::collections::hash_map::Entry::Occupied(entry) => {
                let index = *entry.get();
                ensure!(
                    dest.positions[index as usize] == position && dest.uv[index as usize] == uv,
                    "MC33 key produced inconsistent data"
                );
                index
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                let index =
                    u32::try_from(dest.positions.len()).context("mesh exceeds u32 vertices")?;
                dest.positions.push(position);
                dest.uv.push(uv);
                entry.insert(index);
                index
            }
        };
        remap.push(index);
    }
    for face in src.indices {
        dest.indices.push([
            *remap.get(face[0] as usize).context("invalid chunk face")?,
            *remap.get(face[1] as usize).context("invalid chunk face")?,
            *remap.get(face[2] as usize).context("invalid chunk face")?,
        ]);
    }
    Ok(())
}

fn merge_mesh(dest: &mut Mesh3D, src: Mesh3D) -> Result<()> {
    let base = u32::try_from(dest.positions.len()).context("mesh exceeds u32 vertices")?;
    ensure!(
        src.positions.len() == src.uv.len(),
        "mesh attribute lengths differ"
    );
    dest.positions.extend(src.positions);
    dest.uv.extend(src.uv);
    for face in src.indices {
        dest.indices.push([
            face[0].checked_add(base).context("mesh index overflow")?,
            face[1].checked_add(base).context("mesh index overflow")?,
            face[2].checked_add(base).context("mesh index overflow")?,
        ]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use glam::{DMat4, DVec3, IVec3};

    use super::*;
    use crate::{
        MsgField, MsgFieldLevel, MsgGeometry, MsgLevel, MsgScalarGrid, MsgScalarType,
        MsgWriteOptions, view_msg, write_msg,
    };

    #[test]
    fn extracts_point_field_through_global_transform() -> Result<()> {
        let transform = DMat4::from_translation(DVec3::new(10.0, 20.0, 30.0));
        let mut dataset = MsgDataset::new(
            0.0,
            MsgGeometry::point_centered(transform)?,
            vec![MsgLevel {
                index_origin: IVec3::ZERO,
                point_extent: UVec3::splat(2),
                scale: DVec3::ONE,
                refinement_ratio_from_coarse: None,
            }],
        )?;
        let mut values = MsgScalarGrid::new(UVec3::splat(2), MsgScalarType::F64);
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    values.set(UVec3::new(x, y, z), f64::from(x));
                }
            }
        }
        dataset.add_field(MsgField::new(
            "density",
            MsgScalarType::F64,
            vec![MsgFieldLevel::new(values)],
        ))?;
        dataset.rebuild_acceleration()?;

        let options = IsosurfaceOptions {
            surface: Surface {
                field: "density".into(),
                value: 0.5,
            },
            ..IsosurfaceOptions::default()
        };
        let (mesh, _) = isosurface(&dataset, options.clone())?;
        ensure!(!mesh.indices.is_empty(), "expected one planar surface");
        ensure!(
            mesh.positions
                .iter()
                .all(|position| (position[0] - 10.5).abs() <= f32::EPSILON),
            "global transform was not applied to surface positions"
        );
        let mut bytes = Vec::new();
        write_msg(&dataset, &MsgWriteOptions::default(), &mut bytes)?;
        let archive = view_msg(&bytes)?;
        let (archive_mesh, _) = isosurface_archive(&archive, options)?;
        assert_eq!(archive_mesh.positions, mesh.positions);
        assert_eq!(archive_mesh.uv, mesh.uv);
        assert_eq!(archive_mesh.indices, mesh.indices);
        Ok(())
    }
}
