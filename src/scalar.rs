use glam::UVec3;

use crate::sparse::{Aabb3u, GridAccessor, GridError, SparseGrid3};

/// Logical precision exposed by an MSG scalar field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MsgScalarType {
    F32,
    F64,
}

/// Owned sparse scalar samples for one field on one level.
#[derive(Debug, Clone)]
pub enum MsgScalarGrid {
    F32(SparseGrid3<f32>),
    F64(SparseGrid3<f64>),
}

pub(crate) enum MsgScalarAccessor<'a> {
    F32(GridAccessor<'a, f32>),
    F64(GridAccessor<'a, f64>),
}

impl MsgScalarGrid {
    pub fn new(bounds: UVec3, scalar_type: MsgScalarType) -> Self {
        match scalar_type {
            MsgScalarType::F32 => Self::F32(SparseGrid3::new(bounds)),
            MsgScalarType::F64 => Self::F64(SparseGrid3::new(bounds)),
        }
    }

    pub fn scalar_type(&self) -> MsgScalarType {
        match self {
            Self::F32(_) => MsgScalarType::F32,
            Self::F64(_) => MsgScalarType::F64,
        }
    }

    pub fn bounds(&self) -> UVec3 {
        match self {
            Self::F32(grid) => grid.bounds(),
            Self::F64(grid) => grid.bounds(),
        }
    }

    pub fn present_point_count(&self) -> usize {
        match self {
            Self::F32(grid) => grid.present_voxel_count(),
            Self::F64(grid) => grid.present_voxel_count(),
        }
    }

    pub fn get(&self, point: UVec3) -> Option<f64> {
        match self {
            Self::F32(grid) => grid.get(point).map(f64::from),
            Self::F64(grid) => grid.get(point),
        }
    }

    pub(crate) fn accessor(&self) -> MsgScalarAccessor<'_> {
        match self {
            Self::F32(grid) => MsgScalarAccessor::F32(grid.accessor()),
            Self::F64(grid) => MsgScalarAccessor::F64(grid.accessor()),
        }
    }

    pub fn set(&mut self, point: UVec3, value: f64) -> bool {
        match self {
            Self::F32(grid) => grid.set(point, value as f32),
            Self::F64(grid) => grid.set(point, value),
        }
    }

    pub fn clear(&mut self, point: UVec3) -> bool {
        match self {
            Self::F32(grid) => grid.clear(point),
            Self::F64(grid) => grid.clear(point),
        }
    }

    pub fn write_f32_values(&mut self, aabb: Aabb3u, values: &[f32]) -> Result<(), GridError> {
        match self {
            Self::F32(grid) => grid.write_aabb_values(aabb, values),
            Self::F64(grid) => {
                let values = values.iter().copied().map(f64::from).collect::<Vec<_>>();
                grid.write_aabb_values(aabb, &values)
            }
        }
    }

    pub fn write_f64_values(&mut self, aabb: Aabb3u, values: &[f64]) -> Result<(), GridError> {
        match self {
            Self::F32(grid) => {
                let values = values
                    .iter()
                    .copied()
                    .map(|value| value as f32)
                    .collect::<Vec<_>>();
                grid.write_aabb_values(aabb, &values)
            }
            Self::F64(grid) => grid.write_aabb_values(aabb, values),
        }
    }

    pub(crate) fn contains_non_finite(&self) -> bool {
        let mut found = false;
        match self {
            Self::F32(grid) => grid.for_each_present_in_aabb(grid.bounds_aabb(), |_, value| {
                found |= !value.is_finite();
            }),
            Self::F64(grid) => grid.for_each_present_in_aabb(grid.bounds_aabb(), |_, value| {
                found |= !value.is_finite();
            }),
        }
        found
    }

    pub(crate) fn for_each_value_in_aabb(&self, aabb: Aabb3u, mut visitor: impl FnMut(UVec3, f64)) {
        match self {
            Self::F32(grid) => grid.for_each_present_in_aabb(aabb, |position, value| {
                visitor(position, f64::from(value));
            }),
            Self::F64(grid) => grid.for_each_present_in_aabb(aabb, visitor),
        }
    }

    pub(crate) fn mask_active_cubes_from_fine(
        &self,
        coarse_active_cubes: &mut SparseGrid3<()>,
        ratio: UVec3,
        translation: glam::I64Vec3,
    ) {
        match self {
            Self::F32(grid) => {
                coarse_active_cubes.mask_out_by_presence_scaled(grid, ratio, translation);
            }
            Self::F64(grid) => {
                coarse_active_cubes.mask_out_by_presence_scaled(grid, ratio, translation);
            }
        }
    }
}

impl MsgScalarAccessor<'_> {
    pub(crate) fn get(&mut self, point: UVec3) -> Option<f64> {
        match self {
            Self::F32(accessor) => accessor.get(point).map(f64::from),
            Self::F64(accessor) => accessor.get(point),
        }
    }
}
