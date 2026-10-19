use std::{error::Error, fmt};

use glam::{DVec3, IVec3, UVec3};

use crate::{MsgGeometry, MsgScalarGrid, MsgScalarType, sparse::SparseGrid3};

/// A validation failure for the in-memory MSG model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MsgValidationError {
    message: String,
}

impl MsgValidationError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for MsgValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for MsgValidationError {}

/// Shared geometry for one refinement level, ordered coarse to fine.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MsgLevel {
    pub index_origin: IVec3,
    pub point_extent: UVec3,
    pub scale: DVec3,
    /// Direction-wise ratio from the preceding coarser level. Level zero must
    /// use `None`; every finer level must use `Some`.
    pub refinement_ratio_from_coarse: Option<UVec3>,
}

/// Conservative value range for an encoded spatial chunk.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChunkValueRange {
    pub chunk_origin: UVec3,
    pub min: f64,
    pub max: f64,
}

/// Rebuildable isosurface acceleration for one field level.
#[derive(Debug, Clone)]
pub struct MsgFieldAcceleration {
    pub value_ranges: Vec<ChunkValueRange>,
    pub active_cubes: SparseGrid3<()>,
}

/// One field's independently sparse samples on one geometric level.
#[derive(Debug, Clone)]
pub struct MsgFieldLevel {
    values: MsgScalarGrid,
    acceleration: Option<MsgFieldAcceleration>,
}

impl MsgFieldLevel {
    pub fn new(values: MsgScalarGrid) -> Self {
        Self {
            values,
            acceleration: None,
        }
    }

    pub fn values(&self) -> &MsgScalarGrid {
        &self.values
    }

    /// Mutable access invalidates all derived acceleration for this level.
    pub fn values_mut(&mut self) -> &mut MsgScalarGrid {
        self.acceleration = None;
        &mut self.values
    }

    pub fn acceleration(&self) -> Option<&MsgFieldAcceleration> {
        self.acceleration.as_ref()
    }

    pub fn set_acceleration(&mut self, acceleration: MsgFieldAcceleration) {
        self.acceleration = Some(acceleration);
    }

    pub fn invalidate_acceleration(&mut self) {
        self.acceleration = None;
    }
}

/// A uniquely named, single-component scalar field with independent topology.
#[derive(Debug, Clone)]
pub struct MsgField {
    name: String,
    scalar_type: MsgScalarType,
    levels: Vec<MsgFieldLevel>,
}

impl MsgField {
    pub fn new(
        name: impl Into<String>,
        scalar_type: MsgScalarType,
        levels: Vec<MsgFieldLevel>,
    ) -> Self {
        Self {
            name: name.into(),
            scalar_type,
            levels,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn scalar_type(&self) -> MsgScalarType {
        self.scalar_type
    }

    pub fn levels(&self) -> &[MsgFieldLevel] {
        &self.levels
    }

    pub fn levels_mut(&mut self) -> &mut [MsgFieldLevel] {
        &mut self.levels
    }
}

/// Owned, mutable multiresolution sparse-grid dataset for one timestep.
#[derive(Debug, Clone)]
pub struct MsgDataset {
    time: f64,
    geometry: MsgGeometry,
    levels: Vec<MsgLevel>,
    fields: Vec<MsgField>,
}

impl MsgDataset {
    pub fn new(
        time: f64,
        geometry: MsgGeometry,
        levels: Vec<MsgLevel>,
    ) -> Result<Self, MsgValidationError> {
        let dataset = Self {
            time,
            geometry,
            levels,
            fields: Vec::new(),
        };
        dataset.validate()?;
        Ok(dataset)
    }

    /// Constructs a dataset from metadata already validated by an internal
    /// decoder.
    pub(crate) fn from_validated_parts(
        time: f64,
        geometry: MsgGeometry,
        levels: Vec<MsgLevel>,
    ) -> Self {
        Self {
            time,
            geometry,
            levels,
            fields: Vec::new(),
        }
    }

    pub fn time(&self) -> f64 {
        self.time
    }

    pub fn geometry(&self) -> MsgGeometry {
        self.geometry
    }

    pub fn levels(&self) -> &[MsgLevel] {
        &self.levels
    }

    pub fn fields(&self) -> &[MsgField] {
        &self.fields
    }

    pub fn field(&self, name: &str) -> Option<&MsgField> {
        self.fields.iter().find(|field| field.name == name)
    }

    pub fn field_mut(&mut self, name: &str) -> Option<&mut MsgField> {
        self.fields.iter_mut().find(|field| field.name == name)
    }

    pub fn add_field(&mut self, field: MsgField) -> Result<(), MsgValidationError> {
        if self
            .fields
            .iter()
            .any(|existing| existing.name == field.name)
        {
            return Err(MsgValidationError::new(format!(
                "field name {:?} is already present",
                field.name
            )));
        }
        validate_field(&field, &self.levels)?;
        self.fields.push(field);
        Ok(())
    }

    /// Adds a field whose name, level layout, bounds, precision, and values
    /// have already been validated by an internal constructor.
    pub(crate) fn push_validated_field(&mut self, field: MsgField) {
        self.fields.push(field);
    }

    pub fn validate(&self) -> Result<(), MsgValidationError> {
        if !self.time.is_finite() {
            return Err(MsgValidationError::new("simulation time must be finite"));
        }
        self.geometry.validate()?;
        validate_levels(&self.levels)?;

        for (index, field) in self.fields.iter().enumerate() {
            if self.fields[..index]
                .iter()
                .any(|other| other.name == field.name)
            {
                return Err(MsgValidationError::new(format!(
                    "field name {:?} is duplicated",
                    field.name
                )));
            }
            validate_field(field, &self.levels)?;
        }
        Ok(())
    }

    /// Rebuild field-local active-cube masks and conservative chunk ranges.
    /// Fine topology masks covered coarse cubes independently for every field.
    pub fn rebuild_acceleration(&mut self) -> Result<(), MsgValidationError> {
        self.validate()?;
        let rebuilt = self.derived_acceleration()?;
        for (field, levels) in self.fields.iter_mut().zip(rebuilt) {
            for (level, acceleration) in field.levels.iter_mut().zip(levels) {
                level.acceleration = Some(acceleration);
            }
        }
        Ok(())
    }

    /// Derives acceleration without mutating or trusting cached field state.
    pub(crate) fn derived_acceleration(
        &self,
    ) -> Result<Vec<Vec<MsgFieldAcceleration>>, MsgValidationError> {
        self.fields
            .iter()
            .map(|field| derive_field_acceleration(field, &self.levels))
            .collect()
    }
}

fn derive_field_acceleration(
    field: &MsgField,
    levels: &[MsgLevel],
) -> Result<Vec<MsgFieldAcceleration>, MsgValidationError> {
    let mut active = field
        .levels
        .iter()
        .map(|level| crate::acceleration::build_active_cubes(&level.values))
        .collect::<Vec<_>>();
    for fine_index in (1..active.len()).rev() {
        let coarse_index = fine_index - 1;
        let ratio = levels[fine_index]
            .refinement_ratio_from_coarse
            .expect("validated finer level ratio");
        let translation = level_translation(
            levels[coarse_index].index_origin,
            levels[fine_index].index_origin,
            ratio,
        )?;
        field.levels[fine_index].values.mask_active_cubes_from_fine(
            &mut active[coarse_index],
            ratio,
            translation,
        );
    }
    Ok(field
        .levels
        .iter()
        .zip(active)
        .map(|(level, active_cubes)| {
            crate::acceleration::finish_acceleration(&level.values, active_cubes)
        })
        .collect())
}

fn level_translation(
    coarse_origin: IVec3,
    fine_origin: IVec3,
    ratio: UVec3,
) -> Result<glam::I64Vec3, MsgValidationError> {
    fn axis(coarse: i32, fine: i32, ratio: u32) -> Result<i64, MsgValidationError> {
        let value = i128::from(coarse) * i128::from(ratio) - i128::from(fine);
        i64::try_from(value)
            .map_err(|_| MsgValidationError::new("level-index translation exceeds i64"))
    }
    Ok(glam::I64Vec3::new(
        axis(coarse_origin.x, fine_origin.x, ratio.x)?,
        axis(coarse_origin.y, fine_origin.y, ratio.y)?,
        axis(coarse_origin.z, fine_origin.z, ratio.z)?,
    ))
}

fn validate_levels(levels: &[MsgLevel]) -> Result<(), MsgValidationError> {
    if levels.is_empty() {
        return Err(MsgValidationError::new(
            "an MSG dataset must contain at least one level",
        ));
    }

    for (index, level) in levels.iter().enumerate() {
        if level.point_extent.cmpeq(UVec3::ZERO).any() {
            return Err(MsgValidationError::new(format!(
                "level {index} has an empty point extent"
            )));
        }
        if !level.scale.is_finite() || level.scale.cmple(DVec3::ZERO).any() {
            return Err(MsgValidationError::new(format!(
                "level {index} scale must be finite and strictly positive"
            )));
        }

        match (index, level.refinement_ratio_from_coarse) {
            (0, None) => {}
            (0, Some(_)) => {
                return Err(MsgValidationError::new(
                    "level zero cannot have a coarse refinement ratio",
                ));
            }
            (_, None) => {
                return Err(MsgValidationError::new(format!(
                    "level {index} is missing its coarse refinement ratio"
                )));
            }
            (_, Some(ratio)) => {
                if ratio.cmpeq(UVec3::ZERO).any() {
                    return Err(MsgValidationError::new(format!(
                        "level {index} refinement ratio must be nonzero"
                    )));
                }
                let expected = levels[index - 1].scale / ratio.as_dvec3();
                let relative = (level.scale - expected).abs() / expected;
                if relative.cmpgt(DVec3::splat(64.0 * f64::EPSILON)).any() {
                    return Err(MsgValidationError::new(format!(
                        "level {index} scale {:?} disagrees with refinement ratio {:?}; expected {:?}",
                        level.scale, ratio, expected
                    )));
                }
            }
        }
    }
    Ok(())
}

fn validate_field(field: &MsgField, levels: &[MsgLevel]) -> Result<(), MsgValidationError> {
    if field.name.is_empty() {
        return Err(MsgValidationError::new("field names cannot be empty"));
    }
    if field.levels.len() != levels.len() {
        return Err(MsgValidationError::new(format!(
            "field {:?} has {} levels but the dataset has {}",
            field.name,
            field.levels.len(),
            levels.len()
        )));
    }

    for (index, (field_level, level)) in field.levels.iter().zip(levels).enumerate() {
        if field_level.values.scalar_type() != field.scalar_type {
            return Err(MsgValidationError::new(format!(
                "field {:?} level {index} has inconsistent scalar precision",
                field.name
            )));
        }
        if field_level.values.bounds() != level.point_extent {
            return Err(MsgValidationError::new(format!(
                "field {:?} level {index} bounds {:?} differ from level extent {:?}",
                field.name,
                field_level.values.bounds(),
                level.point_extent
            )));
        }
        if field_level.values.contains_non_finite() {
            return Err(MsgValidationError::new(format!(
                "field {:?} level {index} contains a non-finite value",
                field.name
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use glam::DMat4;

    use super::*;

    fn levels() -> Vec<MsgLevel> {
        vec![
            MsgLevel {
                index_origin: IVec3::ZERO,
                point_extent: UVec3::splat(8),
                scale: DVec3::splat(2.0),
                refinement_ratio_from_coarse: None,
            },
            MsgLevel {
                index_origin: IVec3::ZERO,
                point_extent: UVec3::splat(16),
                scale: DVec3::ONE,
                refinement_ratio_from_coarse: Some(UVec3::splat(2)),
            },
        ]
    }

    #[test]
    fn fields_have_independent_topology() {
        let mut dataset = MsgDataset::new(
            1.25,
            MsgGeometry::point_centered(DMat4::IDENTITY).unwrap(),
            levels(),
        )
        .unwrap();
        let mut first = MsgScalarGrid::new(UVec3::splat(8), MsgScalarType::F64);
        first.set(UVec3::new(1, 2, 3), 4.0);
        let second = MsgScalarGrid::new(UVec3::splat(16), MsgScalarType::F64);
        dataset
            .add_field(MsgField::new(
                "density",
                MsgScalarType::F64,
                vec![MsgFieldLevel::new(first), MsgFieldLevel::new(second)],
            ))
            .unwrap();

        assert_eq!(
            dataset.fields()[0].levels()[0]
                .values()
                .present_point_count(),
            1
        );
        assert_eq!(
            dataset.fields()[0].levels()[1]
                .values()
                .present_point_count(),
            0
        );
    }

    #[test]
    fn rejects_non_finite_field_values() {
        let mut dataset = MsgDataset::new(
            0.0,
            MsgGeometry::point_centered(DMat4::IDENTITY).unwrap(),
            vec![levels()[0]],
        )
        .unwrap();
        let mut values = MsgScalarGrid::new(UVec3::splat(8), MsgScalarType::F32);
        values.set(UVec3::ZERO, f64::NAN);
        assert!(
            dataset
                .add_field(MsgField::new(
                    "bad",
                    MsgScalarType::F32,
                    vec![MsgFieldLevel::new(values)]
                ))
                .is_err()
        );
    }
}
