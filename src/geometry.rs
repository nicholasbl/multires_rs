use glam::{DMat4, DVec3, IVec3, UVec3};

use crate::MsgValidationError;

/// Stable column-major representation of a 4×4 matrix in an MSG archive.
///
/// Columns are concatenated and every column is ordered `x, y, z, w`. This is
/// deliberately an archive representation rather than a promise about the ABI
/// of [`DMat4`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct ArchivedMat4(pub [f64; 16]);

impl ArchivedMat4 {
    pub fn from_matrix(matrix: DMat4) -> Self {
        Self(matrix.to_cols_array())
    }

    pub fn to_matrix(self) -> DMat4 {
        DMat4::from_cols_array(&self.0)
    }
}

impl From<DMat4> for ArchivedMat4 {
    fn from(value: DMat4) -> Self {
        Self::from_matrix(value)
    }
}

impl From<ArchivedMat4> for DMat4 {
    fn from(value: ArchivedMat4) -> Self {
        value.to_matrix()
    }
}

/// Geometry shared by every field and refinement level in an MSG dataset.
///
/// `grid_to_world` supplies the one global affine frame. Level-dependent
/// spacing is kept separately on [`crate::MsgLevel`]. `sample_offset` is in
/// level-index units: native point data normally uses zero, while values
/// converted from cell centers use `(0.5, 0.5, 0.5)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MsgGeometry {
    grid_to_world: DMat4,
    sample_offset: DVec3,
}

impl MsgGeometry {
    pub fn point_centered(grid_to_world: DMat4) -> Result<Self, MsgValidationError> {
        Self::new(grid_to_world, DVec3::ZERO)
    }

    pub fn new(grid_to_world: DMat4, sample_offset: DVec3) -> Result<Self, MsgValidationError> {
        validate_affine(grid_to_world)?;
        if !sample_offset.is_finite() {
            return Err(MsgValidationError::new("sample offset must be finite"));
        }
        Ok(Self {
            grid_to_world,
            sample_offset,
        })
    }

    pub fn grid_to_world(self) -> DMat4 {
        self.grid_to_world
    }

    pub fn world_to_grid(self) -> DMat4 {
        self.grid_to_world.inverse()
    }

    pub fn sample_offset(self) -> DVec3 {
        self.sample_offset
    }

    /// Transform a local point index on a level into world space.
    pub fn point_to_world(
        self,
        index_origin: IVec3,
        level_scale: DVec3,
        local_point: UVec3,
    ) -> DVec3 {
        let index = index_origin.as_dvec3() + local_point.as_dvec3() + self.sample_offset;
        self.grid_to_world.transform_point3(index * level_scale)
    }

    /// Transform a possibly interpolated local point coordinate to world space.
    pub fn point_position_to_world(
        self,
        index_origin: IVec3,
        level_scale: DVec3,
        local_position: DVec3,
    ) -> DVec3 {
        let index = index_origin.as_dvec3() + local_position + self.sample_offset;
        self.grid_to_world.transform_point3(index * level_scale)
    }

    /// Transform a gradient expressed after division by level spacing into
    /// world space using the inverse transpose of the global affine frame.
    pub fn gradient_to_world(self, gradient: DVec3) -> DVec3 {
        self.grid_to_world
            .inverse()
            .transpose()
            .transform_vector3(gradient)
    }

    pub(crate) fn validate(self) -> Result<(), MsgValidationError> {
        validate_affine(self.grid_to_world)?;
        if !self.sample_offset.is_finite() {
            return Err(MsgValidationError::new("sample offset must be finite"));
        }
        Ok(())
    }
}

/// Create point-associated geometry for unchanged cell-center samples.
///
/// This does not interpolate or add boundary samples. It preserves the source
/// values and dimensions and locates index `(0, 0, 0)` half a level cell into
/// the source domain.
pub fn cell_center_to_point_geometry(
    grid_to_world: DMat4,
) -> Result<MsgGeometry, MsgValidationError> {
    MsgGeometry::new(grid_to_world, DVec3::splat(0.5))
}

fn validate_affine(matrix: DMat4) -> Result<(), MsgValidationError> {
    let values = matrix.to_cols_array();
    if !values.iter().all(|value| value.is_finite()) {
        return Err(MsgValidationError::new(
            "grid-to-world matrix must contain only finite values",
        ));
    }

    let tolerance = 32.0 * f64::EPSILON;
    let affine_row = [values[3], values[7], values[11], values[15]];
    let expected = [0.0, 0.0, 0.0, 1.0];
    if affine_row
        .iter()
        .zip(expected)
        .any(|(&actual, expected)| (actual - expected).abs() > tolerance)
    {
        return Err(MsgValidationError::new(format!(
            "grid-to-world matrix is not affine: final row is {affine_row:?}"
        )));
    }

    let determinant = matrix.determinant();
    let inverse_is_finite = matrix
        .inverse()
        .to_cols_array()
        .iter()
        .all(|value| value.is_finite());
    if !determinant.is_finite() || determinant == 0.0 || !inverse_is_finite {
        return Err(MsgValidationError::new(
            "grid-to-world matrix must be invertible",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archived_matrix_round_trips_by_columns() {
        let matrix = DMat4::from_cols_array(&[
            1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0, 7.0, 8.0, 10.0, 0.0, 11.0, 12.0, 13.0, 1.0,
        ]);
        let archived = ArchivedMat4::from(matrix);
        assert_eq!(archived.0, matrix.to_cols_array());
        assert_eq!(DMat4::from(archived), matrix);
    }

    #[test]
    fn cell_center_offset_scales_per_level() {
        let geometry = cell_center_to_point_geometry(DMat4::IDENTITY).unwrap();
        assert_eq!(
            geometry.point_to_world(IVec3::ZERO, DVec3::splat(2.0), UVec3::ZERO),
            DVec3::ONE
        );
        assert_eq!(
            geometry.point_to_world(IVec3::ZERO, DVec3::splat(0.5), UVec3::ZERO),
            DVec3::splat(0.25)
        );
    }

    #[test]
    fn rejects_non_affine_and_singular_matrices() {
        let perspective = DMat4::from_cols_array(&[
            1.0, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]);
        assert!(MsgGeometry::point_centered(perspective).is_err());
        assert!(MsgGeometry::point_centered(DMat4::ZERO).is_err());
    }
}
