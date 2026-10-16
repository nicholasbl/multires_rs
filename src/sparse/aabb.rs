use glam::UVec3;

/// Half-open unsigned 3D axis-aligned box: `[min, max)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Aabb3u {
    pub min: UVec3,
    pub max: UVec3,
}

impl Aabb3u {
    /// Construct a half-open box from minimum and maximum coordinates.
    #[inline]
    pub const fn new(min: UVec3, max: UVec3) -> Self {
        Self { min, max }
    }

    /// Construct a half-open box from minimum coordinate and extent.
    ///
    /// The maximum coordinate is saturating to avoid integer overflow.
    #[inline]
    pub fn from_min_extent(min: UVec3, extent: UVec3) -> Self {
        Self {
            min,
            max: UVec3::new(
                min.x.saturating_add(extent.x),
                min.y.saturating_add(extent.y),
                min.z.saturating_add(extent.z),
            ),
        }
    }

    /// Construct the full-grid box `[0, bounds)`.
    #[inline]
    pub fn full_grid(bounds: UVec3) -> Self {
        Self {
            min: UVec3::new(0, 0, 0),
            max: bounds,
        }
    }

    /// Return the saturating extent of the box.
    #[inline]
    pub fn extent(self) -> UVec3 {
        UVec3::new(
            self.max.x.saturating_sub(self.min.x),
            self.max.y.saturating_sub(self.min.y),
            self.max.z.saturating_sub(self.min.z),
        )
    }

    /// True when at least one axis has no positive width.
    #[inline]
    pub fn is_empty(self) -> bool {
        self.max.x <= self.min.x || self.max.y <= self.min.y || self.max.z <= self.min.z
    }

    /// True when `p` lies inside the half-open box.
    #[inline]
    pub fn contains_point(self, p: UVec3) -> bool {
        p.x >= self.min.x
            && p.y >= self.min.y
            && p.z >= self.min.z
            && p.x < self.max.x
            && p.y < self.max.y
            && p.z < self.max.z
    }

    /// Return the non-empty intersection of two boxes.
    #[inline]
    pub fn intersect(self, other: Self) -> Option<Self> {
        let min = UVec3::new(
            self.min.x.max(other.min.x),
            self.min.y.max(other.min.y),
            self.min.z.max(other.min.z),
        );
        let max = UVec3::new(
            self.max.x.min(other.max.x),
            self.max.y.min(other.max.y),
            self.max.z.min(other.max.z),
        );
        let out = Self { min, max };
        (!out.is_empty()).then_some(out)
    }

    /// Returns the AABB volume as `usize`, or `None` if it does not fit.
    #[inline]
    pub fn volume_usize(self) -> Option<usize> {
        if self.is_empty() {
            return Some(0);
        }

        let e = self.extent();
        let xy = (e.x as usize).checked_mul(e.y as usize)?;
        xy.checked_mul(e.z as usize)
    }
}
