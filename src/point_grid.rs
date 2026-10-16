use std::ops::RangeInclusive;

use anyhow::{Context, Result, ensure};
use glam::{DVec3, I64Vec3, IVec3, UVec3};

use crate::{
    ChunkValueRange, MsgDataset, MsgGeometry, MsgScalarGrid,
    acceleration::{build_active_cubes, build_value_ranges},
    sparse::SparseGrid3,
};

pub(crate) struct PointGridLevel<'a> {
    pub(crate) level_index: usize,
    pub(crate) index_origin: IVec3,
    pub(crate) geometry: MsgGeometry,
    pub(crate) scale: DVec3,
    pub(crate) samples: &'a MsgScalarGrid,
    pub(crate) sampled_quantities: Vec<&'a MsgScalarGrid>,
    pub(crate) active_cubes: SparseGrid3<()>,
    pub(crate) value_ranges: Vec<ChunkValueRange>,
}

pub(crate) fn point_grid_levels<'a>(
    dataset: &'a MsgDataset,
    surface_field: &str,
    sampled_fields: &[&str],
    level_range: Option<&RangeInclusive<usize>>,
) -> Result<Vec<PointGridLevel<'a>>> {
    ensure!(
        sampled_fields.len() <= 2,
        "at most two sampled fields are supported"
    );
    let surface = dataset
        .field(surface_field)
        .with_context(|| format!("surface field {surface_field:?} is absent"))?;
    let sampled = sampled_fields
        .iter()
        .map(|name| {
            dataset
                .field(name)
                .with_context(|| format!("sampled field {name:?} is absent"))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut active = surface
        .levels()
        .iter()
        .map(|level| {
            level
                .acceleration()
                .map(|acceleration| acceleration.active_cubes.clone())
                .unwrap_or_else(|| build_active_cubes(level.values()))
        })
        .collect::<Vec<_>>();

    for fine_index in (1..active.len()).rev() {
        let coarse_index = fine_index - 1;
        let ratio = dataset.levels()[fine_index]
            .refinement_ratio_from_coarse
            .context("fine level is missing its refinement ratio")?;
        let translation = level_translation(
            dataset.levels()[coarse_index].index_origin,
            dataset.levels()[fine_index].index_origin,
            ratio,
        )?;
        surface.levels()[fine_index]
            .values()
            .mask_active_cubes_from_fine(&mut active[coarse_index], ratio, translation);
    }

    let mut output = Vec::new();
    for (index, ((level, surface_level), active_cubes)) in dataset
        .levels()
        .iter()
        .zip(surface.levels())
        .zip(active)
        .enumerate()
    {
        if level_range.is_some_and(|range| !range.contains(&index)) {
            continue;
        }
        let value_ranges = build_value_ranges(surface_level.values(), &active_cubes);
        output.push(PointGridLevel {
            level_index: index,
            index_origin: level.index_origin,
            geometry: dataset.geometry(),
            scale: level.scale,
            samples: surface_level.values(),
            sampled_quantities: sampled
                .iter()
                .map(|field| field.levels()[index].values())
                .collect(),
            active_cubes,
            value_ranges,
        });
    }
    Ok(output)
}

fn level_translation(coarse_origin: IVec3, fine_origin: IVec3, ratio: UVec3) -> Result<I64Vec3> {
    fn axis(coarse: i32, fine: i32, ratio: u32) -> Result<i64> {
        i64::try_from(i128::from(coarse) * i128::from(ratio) - i128::from(fine))
            .context("level-index translation exceeds i64")
    }
    Ok(I64Vec3::new(
        axis(coarse_origin.x, fine_origin.x, ratio.x)?,
        axis(coarse_origin.y, fine_origin.y, ratio.y)?,
        axis(coarse_origin.z, fine_origin.z, ratio.z)?,
    ))
}
