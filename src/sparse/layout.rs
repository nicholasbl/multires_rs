use glam::{I64Vec3, UVec3};

use super::{Aabb3u, CHUNK_BITS, CHUNK_MASK, CHUNK_SIZE, CHUNK_VOLUME};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ChunkKey {
    pub(super) x: u32,
    pub(super) y: u32,
    pub(super) z: u32,
}

impl std::hash::Hash for ChunkKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let xy = ((self.x as u64) << 32) | self.y as u64;

        state.write_u64(xy);
        state.write_u32(self.z);
    }
}

impl ChunkKey {
    #[inline]
    pub(super) const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub(super) fn from_pos(pos: UVec3) -> Self {
        Self {
            x: pos.x >> CHUNK_BITS,
            y: pos.y >> CHUNK_BITS,
            z: pos.z >> CHUNK_BITS,
        }
    }

    #[inline]
    pub(super) fn world_min(self) -> UVec3 {
        UVec3::new(
            self.x << CHUNK_BITS,
            self.y << CHUNK_BITS,
            self.z << CHUNK_BITS,
        )
    }
}

#[inline]
pub(super) fn local_pos(pos: UVec3) -> UVec3 {
    UVec3::new(pos.x & CHUNK_MASK, pos.y & CHUNK_MASK, pos.z & CHUNK_MASK)
}

#[inline]
pub(super) fn local_index_vec(local: UVec3) -> usize {
    local_index(local.x, local.y, local.z)
}

#[inline]
pub(super) fn local_index(x: u32, y: u32, z: u32) -> usize {
    debug_assert!(x < CHUNK_SIZE && y < CHUNK_SIZE && z < CHUNK_SIZE);
    (x as usize)
        + (y as usize) * (CHUNK_SIZE as usize)
        + (z as usize) * (CHUNK_SIZE as usize) * (CHUNK_SIZE as usize)
}

#[inline]
pub(super) fn local_from_index(idx: usize) -> UVec3 {
    debug_assert!(idx < CHUNK_VOLUME);
    let x = (idx & ((CHUNK_SIZE as usize) - 1)) as u32;
    let y = ((idx >> CHUNK_BITS) & ((CHUNK_SIZE as usize) - 1)) as u32;
    let z = (idx >> (CHUNK_BITS * 2)) as u32;
    UVec3::new(x, y, z)
}

#[inline]
pub(super) fn source_index(local: UVec3, extent: UVec3) -> usize {
    debug_assert!(local.x < extent.x && local.y < extent.y && local.z < extent.z);
    (local.x as usize)
        + (local.y as usize) * (extent.x as usize)
        + (local.z as usize) * (extent.x as usize) * (extent.y as usize)
}

#[inline]
pub(super) fn valid_local_aabb(local_min: UVec3, local_max: UVec3) -> bool {
    local_min.x <= local_max.x
        && local_min.y <= local_max.y
        && local_min.z <= local_max.z
        && local_max.x <= CHUNK_SIZE
        && local_max.y <= CHUNK_SIZE
        && local_max.z <= CHUNK_SIZE
}

#[inline]
pub(super) fn is_full_chunk_local_aabb(local_min: UVec3, local_max: UVec3) -> bool {
    local_min.x == 0
        && local_min.y == 0
        && local_min.z == 0
        && local_max.x == CHUNK_SIZE
        && local_max.y == CHUNK_SIZE
        && local_max.z == CHUNK_SIZE
}

#[inline]
pub(super) fn local_extent(local_min: UVec3, local_max: UVec3) -> UVec3 {
    UVec3::new(
        local_max.x - local_min.x,
        local_max.y - local_min.y,
        local_max.z - local_min.z,
    )
}

#[inline]
pub(super) fn add_uvec3(a: UVec3, b: UVec3) -> UVec3 {
    UVec3::new(a.x + b.x, a.y + b.y, a.z + b.z)
}

#[inline]
pub(super) fn sub_uvec3(a: UVec3, b: UVec3) -> UVec3 {
    UVec3::new(a.x - b.x, a.y - b.y, a.z - b.z)
}

#[inline]
pub(super) fn saturating_add_chunk_size(v: UVec3) -> UVec3 {
    UVec3::new(
        v.x.saturating_add(CHUNK_SIZE),
        v.y.saturating_add(CHUNK_SIZE),
        v.z.saturating_add(CHUNK_SIZE),
    )
}

pub(super) fn chunk_key_range_for_aabb(aabb: Aabb3u) -> (ChunkKey, ChunkKey) {
    debug_assert!(!aabb.is_empty());

    let min_key = ChunkKey::from_pos(aabb.min);
    let last = UVec3::new(aabb.max.x - 1, aabb.max.y - 1, aabb.max.z - 1);
    let max_key = ChunkKey::from_pos(last);
    (min_key, max_key)
}

pub(super) fn chunk_local_intersection(key: ChunkKey, aabb: Aabb3u) -> Option<(UVec3, UVec3)> {
    let chunk_min = key.world_min();
    let chunk_max = saturating_add_chunk_size(chunk_min);
    let chunk_aabb = Aabb3u::new(chunk_min, chunk_max);
    let intersection = chunk_aabb.intersect(aabb)?;
    let local_min = sub_uvec3(intersection.min, chunk_min);
    let local_max = sub_uvec3(intersection.max, chunk_min);
    Some((local_min, local_max))
}

pub(super) fn fill_dense_aabb<T>(values: &mut [T], local_min: UVec3, local_max: UVec3, value: T)
where
    T: Copy,
{
    let len = (local_max.x - local_min.x) as usize;

    for z in local_min.z..local_max.z {
        for y in local_min.y..local_max.y {
            let start = local_index(local_min.x, y, z);
            let end = start + len;
            values[start..end].fill(value);
        }
    }
}

pub(super) fn copy_source_region_into_dense<T>(
    dst: &mut [T],
    local_min: UVec3,
    local_max: UVec3,
    source_base: UVec3,
    source_extent: UVec3,
    source: &[T],
) where
    T: Copy,
{
    let len = (local_max.x - local_min.x) as usize;

    for dz in 0..(local_max.z - local_min.z) {
        for dy in 0..(local_max.y - local_min.y) {
            let dst_start = local_index(local_min.x, local_min.y + dy, local_min.z + dz);
            let dst_end = dst_start + len;

            let src_local = UVec3::new(source_base.x, source_base.y + dy, source_base.z + dz);
            let src_start = source_index(src_local, source_extent);
            let src_end = src_start + len;

            dst[dst_start..dst_end].copy_from_slice(&source[src_start..src_end]);
        }
    }
}

pub(super) fn uniform_value_in_source_region<T>(
    source: &[T],
    source_base: UVec3,
    source_extent: UVec3,
    region_extent: UVec3,
) -> Option<T>
where
    T: Copy + PartialEq,
{
    if region_extent.x == 0 || region_extent.y == 0 || region_extent.z == 0 {
        return None;
    }

    let first = source[source_index(source_base, source_extent)];

    for z in 0..region_extent.z {
        for y in 0..region_extent.y {
            let src_local = UVec3::new(source_base.x, source_base.y + y, source_base.z + z);
            let row_start = source_index(src_local, source_extent);
            let row_end = row_start + region_extent.x as usize;
            for &v in &source[row_start..row_end] {
                if v != first {
                    return None;
                }
            }
        }
    }

    Some(first)
}

pub(super) fn scaled_voxel_aabb_in_fine_grid(
    coarse_pos: UVec3,
    scale: UVec3,
    translation: I64Vec3,
    fine_bounds: UVec3,
) -> Option<Aabb3u> {
    debug_assert!(scale.cmpgt(UVec3::ZERO).all());

    pub(super) fn transformed_axis(
        pos: u32,
        scale: u32,
        translation: i64,
        bound: u32,
    ) -> Option<(u32, u32)> {
        let scale = i128::from(scale);
        let min = i128::from(pos) * scale + i128::from(translation);
        let max = (i128::from(pos) + 1) * scale + i128::from(translation);
        let clipped_min = min.max(0);
        let clipped_max = max.min(i128::from(bound));
        (clipped_min < clipped_max).then(|| (clipped_min as u32, clipped_max as u32))
    }

    let (min_x, max_x) = transformed_axis(coarse_pos.x, scale.x, translation.x, fine_bounds.x)?;
    let (min_y, max_y) = transformed_axis(coarse_pos.y, scale.y, translation.y, fine_bounds.y)?;
    let (min_z, max_z) = transformed_axis(coarse_pos.z, scale.z, translation.z, fine_bounds.z)?;

    Some(Aabb3u::new(
        UVec3::new(min_x, min_y, min_z),
        UVec3::new(max_x, max_y, max_z),
    ))
}
