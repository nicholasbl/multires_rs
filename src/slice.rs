//! Planar slices through point-associated MSG fields.

use std::{collections::HashMap, ops::RangeInclusive};

use anyhow::{Context, Result, ensure};
use glam::{DVec3, UVec3};

use crate::{
    Mesh3D, MsgDataset,
    isosurface::Sample,
    point_grid::{PointGridLevel, point_grid_levels},
    scalar::MsgScalarAccessor,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceAxis {
    X,
    Y,
    Z,
}

/// Plane in the canonical grid frame, before the dataset's global affine
/// transform is applied.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlicePlane {
    pub axis: SliceAxis,
    pub coordinate: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SliceOptions {
    /// Field whose point topology determines which slice quads exist.
    pub topology_field: String,
    pub plane: SlicePlane,
    /// One or two scalar fields sampled into mesh UV channels.
    pub sampled_quantities: Vec<Sample>,
    pub levels: Option<RangeInclusive<usize>>,
    pub flip_winding: bool,
}

#[derive(Debug, Clone, Copy)]
struct SampleRange {
    min: f64,
    max: f64,
}

/// Extract a grid-frame axis-aligned plane and transform it into world space.
pub fn slice(dataset: &MsgDataset, options: SliceOptions) -> Result<Mesh3D> {
    validate_options(dataset, &options)?;
    let sampled_names = options
        .sampled_quantities
        .iter()
        .map(|sample| sample.field.as_str())
        .collect::<Vec<_>>();
    let ranges = options
        .sampled_quantities
        .iter()
        .map(|sample| SampleRange {
            min: *sample.range.start(),
            max: *sample.range.end(),
        })
        .collect::<Vec<_>>();
    let levels = point_grid_levels(
        dataset,
        &options.topology_field,
        &sampled_names,
        options.levels.as_ref(),
    )?;

    let mut mesh = Mesh3D::default();
    for level in &levels {
        let Some(axis_coordinate) = local_axis_coordinate(level, options.plane)? else {
            continue;
        };
        let start = mesh.indices.len();
        let level_mesh = mesh_level(level, options.plane.axis, axis_coordinate, &ranges)
            .with_context(|| format!("extracting slice at level {}", level.level_index))?;
        merge_mesh(&mut mesh, level_mesh)?;
        if options.flip_winding {
            for face in &mut mesh.indices[start..] {
                face.swap(1, 2);
            }
        }
    }
    Ok(mesh)
}

fn validate_options(dataset: &MsgDataset, options: &SliceOptions) -> Result<()> {
    ensure!(
        options.plane.coordinate.is_finite(),
        "slice coordinate must be finite"
    );
    ensure!(
        dataset.field(&options.topology_field).is_some(),
        "topology field {:?} is absent",
        options.topology_field
    );
    ensure!(
        !options.sampled_quantities.is_empty(),
        "slice requires at least one sampled field"
    );
    ensure!(
        options.sampled_quantities.len() <= 2,
        "slice supports at most two sampled fields"
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

fn local_axis_coordinate(level: &PointGridLevel<'_>, plane: SlicePlane) -> Result<Option<f64>> {
    let axis = axis_index(plane.axis);
    let scale = component_d(level.scale, axis);
    let origin = f64::from(component_i(level.index_origin, axis));
    let offset = component_d(level.geometry.sample_offset(), axis);
    let coordinate = plane.coordinate / scale - origin - offset;
    let bound = f64::from(component(level.samples.bounds(), axis).saturating_sub(1));
    if !coordinate.is_finite() || coordinate < 0.0 || coordinate > bound {
        return Ok(None);
    }
    Ok(Some(coordinate))
}

fn mesh_level(
    level: &PointGridLevel<'_>,
    axis: SliceAxis,
    coordinate: f64,
    ranges: &[SampleRange],
) -> Result<Mesh3D> {
    let lower = coordinate.floor();
    let mut anchor_axis = lower as i64;
    if coordinate == lower && anchor_axis > 0 {
        anchor_axis -= 1;
    }
    let anchor_axis = u32::try_from(anchor_axis).context("slice anchor is negative")?;
    let t = coordinate - f64::from(anchor_axis);
    let mut mesh = Mesh3D::default();
    let mut vertex_ids = HashMap::<(UVec3, UVec3), u32>::new();
    let mut accessors = level
        .sampled_quantities
        .iter()
        .map(|field| field.accessor())
        .collect::<Vec<_>>();
    let mut error = None;
    level
        .active_cubes
        .for_each_present_in_aabb(level.active_cubes.bounds_aabb(), |anchor| {
            if error.is_none() && component(anchor, axis_index(axis)) == anchor_axis {
                error = emit_quad(
                    &mut mesh,
                    &mut vertex_ids,
                    &mut accessors,
                    ranges,
                    level,
                    axis,
                    anchor,
                    t,
                )
                .err();
            }
        });
    error.map_or(Ok(mesh), Err)
}

#[allow(clippy::too_many_arguments)]
fn emit_quad(
    mesh: &mut Mesh3D,
    vertex_ids: &mut HashMap<(UVec3, UVec3), u32>,
    accessors: &mut [MsgScalarAccessor<'_>],
    ranges: &[SampleRange],
    level: &PointGridLevel<'_>,
    axis: SliceAxis,
    anchor: UVec3,
    t: f64,
) -> Result<()> {
    let edges = slice_edges(axis, anchor);
    let ids = edges
        .map(|(start, end)| vertex(mesh, vertex_ids, accessors, ranges, level, start, end, t))
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    mesh.indices.push([ids[0], ids[1], ids[2]]);
    mesh.indices.push([ids[0], ids[2], ids[3]]);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn vertex(
    mesh: &mut Mesh3D,
    vertex_ids: &mut HashMap<(UVec3, UVec3), u32>,
    accessors: &mut [MsgScalarAccessor<'_>],
    ranges: &[SampleRange],
    level: &PointGridLevel<'_>,
    start: UVec3,
    end: UVec3,
    t: f64,
) -> Result<u32> {
    let key = (start, end);
    if let Some(&id) = vertex_ids.get(&key) {
        return Ok(id);
    }
    let local = start.as_dvec3().lerp(end.as_dvec3(), t);
    let position = level
        .geometry
        .point_position_to_world(level.index_origin, level.scale, local)
        .as_vec3()
        .to_array();
    let mut uv = [0.0_f32; 2];
    for (channel, (accessor, range)) in accessors.iter_mut().zip(ranges).enumerate() {
        let a = accessor
            .get(start)?
            .with_context(|| format!("slice sample is absent at {start:?}"))?;
        let b = accessor
            .get(end)?
            .with_context(|| format!("slice sample is absent at {end:?}"))?;
        let value = (b - a).mul_add(t, a);
        uv[channel] = ((value - range.min) / (range.max - range.min)).clamp(0.0, 1.0) as f32;
    }
    let id = u32::try_from(mesh.positions.len()).context("slice mesh exceeds u32 vertices")?;
    mesh.positions.push(position);
    mesh.uv.push(uv);
    vertex_ids.insert(key, id);
    Ok(id)
}

fn slice_edges(axis: SliceAxis, a: UVec3) -> [(UVec3, UVec3); 4] {
    match axis {
        SliceAxis::X => [
            (a, a + UVec3::X),
            (a + UVec3::Y, a + UVec3::Y + UVec3::X),
            (a + UVec3::Y + UVec3::Z, a + UVec3::Y + UVec3::Z + UVec3::X),
            (a + UVec3::Z, a + UVec3::Z + UVec3::X),
        ],
        SliceAxis::Y => [
            (a, a + UVec3::Y),
            (a + UVec3::Z, a + UVec3::Z + UVec3::Y),
            (a + UVec3::X + UVec3::Z, a + UVec3::X + UVec3::Z + UVec3::Y),
            (a + UVec3::X, a + UVec3::X + UVec3::Y),
        ],
        SliceAxis::Z => [
            (a, a + UVec3::Z),
            (a + UVec3::X, a + UVec3::X + UVec3::Z),
            (a + UVec3::X + UVec3::Y, a + UVec3::X + UVec3::Y + UVec3::Z),
            (a + UVec3::Y, a + UVec3::Y + UVec3::Z),
        ],
    }
}

fn merge_mesh(dest: &mut Mesh3D, src: Mesh3D) -> Result<()> {
    let base = u32::try_from(dest.positions.len()).context("slice mesh exceeds u32 vertices")?;
    dest.positions.extend(src.positions);
    dest.uv.extend(src.uv);
    for face in src.indices {
        dest.indices.push(face.map(|index| index + base));
    }
    Ok(())
}

fn axis_index(axis: SliceAxis) -> usize {
    match axis {
        SliceAxis::X => 0,
        SliceAxis::Y => 1,
        SliceAxis::Z => 2,
    }
}

fn component(value: UVec3, axis: usize) -> u32 {
    value.to_array()[axis]
}
fn component_i(value: glam::IVec3, axis: usize) -> i32 {
    value.to_array()[axis]
}
fn component_d(value: DVec3, axis: usize) -> f64 {
    value.to_array()[axis]
}

#[cfg(test)]
mod tests {
    use glam::{DMat4, IVec3};

    use super::*;
    use crate::{MsgField, MsgFieldLevel, MsgGeometry, MsgLevel, MsgScalarGrid, MsgScalarType};

    #[test]
    fn grid_frame_slice_is_transformed_to_world() -> Result<()> {
        let mut dataset = MsgDataset::new(
            0.0,
            MsgGeometry::point_centered(DMat4::from_translation(DVec3::new(10.0, 0.0, 0.0)))?,
            vec![MsgLevel {
                index_origin: IVec3::ZERO,
                point_extent: UVec3::splat(2),
                scale: DVec3::ONE,
                refinement_ratio_from_coarse: None,
            }],
        )?;
        let mut values = MsgScalarGrid::new(UVec3::splat(2), MsgScalarType::F32);
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    values.set(UVec3::new(x, y, z), f64::from(x));
                }
            }
        }
        dataset.add_field(MsgField::new(
            "density",
            MsgScalarType::F32,
            vec![MsgFieldLevel::new(values)],
        ))?;
        let mesh = slice(
            &dataset,
            SliceOptions {
                topology_field: "density".into(),
                plane: SlicePlane {
                    axis: SliceAxis::X,
                    coordinate: 0.5,
                },
                sampled_quantities: vec![Sample {
                    field: "density".into(),
                    range: 0.0..=1.0,
                }],
                levels: None,
                flip_winding: false,
            },
        )?;
        assert_eq!(mesh.indices.len(), 2);
        assert!(
            mesh.positions
                .iter()
                .all(|position| (position[0] - 10.5).abs() <= f32::EPSILON)
        );
        assert!(mesh.uv.iter().all(|uv| (uv[0] - 0.5).abs() <= f32::EPSILON));
        Ok(())
    }
}
