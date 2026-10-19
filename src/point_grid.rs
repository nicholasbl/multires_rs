use std::ops::RangeInclusive;

use anyhow::{Context, Result, ensure};
use glam::{DVec3, I64Vec3, IVec3, UVec3};

use crate::{
    ChunkValueRange, MsgArchive, MsgArchiveFieldLevel, MsgDataset, MsgGeometry, MsgScalarGrid,
    acceleration::{build_active_cubes, build_value_ranges},
    scalar::MsgScalarAccessor,
    sparse::{Aabb3u, SparseGrid3},
};

pub(crate) enum PointSamples<'a> {
    Owned(&'a MsgScalarGrid),
    Archived(MsgArchiveFieldLevel<'a>),
}

impl<'a> PointSamples<'a> {
    pub(crate) fn accessor(&self) -> MsgScalarAccessor<'a> {
        match self {
            Self::Owned(grid) => grid.accessor(),
            Self::Archived(level) => MsgScalarAccessor::Archived(level.accessor()),
        }
    }

    pub(crate) fn bounds(&self) -> UVec3 {
        match self {
            Self::Owned(grid) => grid.bounds(),
            Self::Archived(level) => level.bounds(),
        }
    }
}

pub(crate) enum ActiveCubes<'a> {
    Owned(SparseGrid3<()>),
    Archived(MsgArchiveFieldLevel<'a>),
}

impl ActiveCubes<'_> {
    pub(crate) fn bounds(&self) -> UVec3 {
        match self {
            Self::Owned(grid) => grid.bounds(),
            Self::Archived(level) => level.bounds().saturating_sub(UVec3::ONE),
        }
    }

    pub(crate) fn bounds_aabb(&self) -> Aabb3u {
        Aabb3u::full_grid(self.bounds())
    }

    pub(crate) fn for_each_present_in_aabb(&self, aabb: Aabb3u, mut visit: impl FnMut(UVec3)) {
        match self {
            Self::Owned(grid) => {
                grid.for_each_present_in_aabb(aabb, |position, ()| visit(position))
            }
            Self::Archived(level) => level.for_each_active_cube_in_aabb(aabb, visit),
        }
    }
}

pub(crate) struct PointGridLevel<'a> {
    pub(crate) level_index: usize,
    pub(crate) index_origin: IVec3,
    pub(crate) geometry: MsgGeometry,
    pub(crate) scale: DVec3,
    pub(crate) samples: PointSamples<'a>,
    pub(crate) sampled_quantities: Vec<PointSamples<'a>>,
    pub(crate) active_cubes: ActiveCubes<'a>,
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

    if surface
        .levels()
        .iter()
        .all(|level| level.acceleration().is_some())
    {
        let mut output = Vec::new();
        for (index, (level, surface_level)) in
            dataset.levels().iter().zip(surface.levels()).enumerate()
        {
            if level_range.is_some_and(|range| !range.contains(&index)) {
                continue;
            }
            let acceleration = surface_level
                .acceleration()
                .expect("checked acceleration presence");
            output.push(PointGridLevel {
                level_index: index,
                index_origin: level.index_origin,
                geometry: dataset.geometry(),
                scale: level.scale,
                samples: PointSamples::Owned(surface_level.values()),
                sampled_quantities: sampled
                    .iter()
                    .map(|field| PointSamples::Owned(field.levels()[index].values()))
                    .collect(),
                active_cubes: ActiveCubes::Owned(acceleration.active_cubes.clone()),
                value_ranges: acceleration.value_ranges.clone(),
            });
        }
        return Ok(output);
    }

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
            samples: PointSamples::Owned(surface_level.values()),
            sampled_quantities: sampled
                .iter()
                .map(|field| PointSamples::Owned(field.levels()[index].values()))
                .collect(),
            active_cubes: ActiveCubes::Owned(active_cubes),
            value_ranges,
        });
    }
    Ok(output)
}

pub(crate) fn archive_point_grid_levels<'a>(
    archive: &MsgArchive<'a>,
    surface_field: &str,
    sampled_fields: &[&str],
    level_range: Option<&RangeInclusive<usize>>,
    isovalue: f64,
) -> Result<Vec<PointGridLevel<'a>>> {
    ensure!(
        sampled_fields.len() <= 2,
        "at most two sampled fields are supported"
    );
    let surface = archive
        .field(surface_field)
        .with_context(|| format!("surface field {surface_field:?} is absent"))?;
    let sampled = sampled_fields
        .iter()
        .map(|name| {
            archive
                .field(name)
                .with_context(|| format!("sampled field {name:?} is absent"))
        })
        .collect::<Result<Vec<_>>>()?;
    let level_info = archive.levels().collect::<Vec<_>>();
    let geometry = archive.geometry();
    let mut output = Vec::new();
    for (index, level) in level_info.into_iter().enumerate() {
        if level_range.is_some_and(|range| !range.contains(&index)) {
            continue;
        }
        let Some(surface_level) = surface.level_if_value_may_be_present(index, isovalue)? else {
            continue;
        };
        let value_ranges = surface_level
            .value_ranges()
            .map(|range| ChunkValueRange {
                chunk_origin: range.chunk_origin,
                min: range.min,
                max: range.max,
            })
            .collect();
        output.push(PointGridLevel {
            level_index: index,
            index_origin: IVec3::from_array(level.index_origin),
            geometry,
            scale: DVec3::from_array(level.scale),
            samples: PointSamples::Archived(surface_level),
            sampled_quantities: sampled
                .iter()
                .map(|field| field.level(index).map(PointSamples::Archived))
                .collect::<Result<Vec<_>>>()?,
            active_cubes: ActiveCubes::Archived(surface_level),
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
