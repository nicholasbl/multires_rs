use std::collections::{HashMap, hash_map::Entry};

use glam::{I64Vec3, UVec3};

use super::{
    Aabb3u,
    chunk::{BitMask, BitMaskView, Chunk},
    layout::{
        ChunkKey, add_uvec3, chunk_key_range_for_aabb, chunk_local_intersection, local_from_index,
        local_index_vec, local_pos, scaled_voxel_aabb_in_fine_grid, sub_uvec3,
    },
};

/// Errors returned by fallible bulk operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GridError {
    /// The requested AABB volume does not fit in `usize`.
    AabbVolumeDoesNotFitUsize,
    /// The supplied dense value slice length did not match the AABB volume.
    ValuesLengthMismatch { expected: usize, actual: usize },
}

pub(crate) enum SparseGridChunkView<'a, T>
where
    T: Copy + PartialEq,
{
    Uniform {
        key: UVec3,
        value: T,
        mask: BitMaskView<'a>,
    },
    Dense {
        key: UVec3,
        values: &'a [T],
        mask: BitMaskView<'a>,
    },
}

/// Bounded, chunked, sparse 3D grid.
///
/// `T` is intentionally constrained to `Copy + PartialEq` to keep chunk promotion,
/// compaction, and bulk copies cheap. This is suitable for fundamental numeric types.
#[derive(Debug, Clone)]
pub struct SparseGrid3<T>
where
    T: Copy + PartialEq,
{
    bounds: UVec3,
    chunks: HashMap<ChunkKey, Chunk<T>>,
}

impl<T> SparseGrid3<T>
where
    T: Copy + PartialEq,
{
    /// Creates an empty bounded grid with coordinates in `[0, bounds)`.
    pub fn new(bounds: UVec3) -> Self {
        Self {
            bounds,
            chunks: HashMap::new(),
        }
    }

    /// Creates an empty grid with reserved chunk capacity.
    pub fn with_chunk_capacity(bounds: UVec3, chunk_capacity: usize) -> Self {
        Self {
            bounds,
            chunks: HashMap::with_capacity(chunk_capacity),
        }
    }

    #[inline]
    /// Grid bounds as exclusive coordinate limits.
    pub fn bounds(&self) -> UVec3 {
        self.bounds
    }

    #[inline]
    /// Grid bounds as the half-open AABB `[0, bounds)`.
    pub fn bounds_aabb(&self) -> Aabb3u {
        Aabb3u::full_grid(self.bounds)
    }

    #[inline]
    /// True when `p` lies inside `[0, bounds)`.
    pub fn is_in_bounds(&self, p: UVec3) -> bool {
        p.x < self.bounds.x && p.y < self.bounds.y && p.z < self.bounds.z
    }

    #[inline]
    /// True when the grid has no allocated chunks.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    #[inline]
    /// Number of allocated chunks.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub(crate) fn for_each_chunk<F>(&self, mut f: F)
    where
        F: FnMut(SparseGridChunkView<'_, T>),
    {
        let mut chunks = self.chunks.iter().collect::<Vec<_>>();
        chunks.sort_by_key(|(key, _)| (key.x, key.y, key.z));
        for (&key, chunk) in chunks {
            let key = UVec3::new(key.x, key.y, key.z);
            match chunk {
                Chunk::Uniform { value, mask } => f(SparseGridChunkView::Uniform {
                    key,
                    value: *value,
                    mask: mask.view(),
                }),
                Chunk::Dense { values, mask } => f(SparseGridChunkView::Dense {
                    key,
                    values,
                    mask: mask.view(),
                }),
            }
        }
    }

    pub(crate) fn par_map_chunks<R, F>(&self, f: F) -> Vec<R>
    where
        T: Sync,
        R: Send,
        F: Fn(SparseGridChunkView<'_, T>) -> R + Sync + Send,
    {
        use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

        self.chunks
            .par_iter()
            .map(|(&key, chunk)| {
                let key = UVec3::new(key.x, key.y, key.z);
                match chunk {
                    Chunk::Uniform { value, mask } => f(SparseGridChunkView::Uniform {
                        key,
                        value: *value,
                        mask: mask.view(),
                    }),
                    Chunk::Dense { values, mask } => f(SparseGridChunkView::Dense {
                        key,
                        values,
                        mask: mask.view(),
                    }),
                }
            })
            .collect()
    }

    pub(crate) fn chunk_aabbs(&self) -> Vec<Aabb3u> {
        let mut aabbs = self
            .chunks
            .keys()
            .filter_map(|&key| {
                let chunk_min = key.world_min();
                let chunk_max = super::layout::saturating_add_chunk_size(chunk_min);
                Aabb3u::new(chunk_min, chunk_max).intersect(self.bounds_aabb())
            })
            .collect::<Vec<_>>();
        aabbs.sort_by_key(|aabb| (aabb.min.z, aabb.min.y, aabb.min.x));
        aabbs
    }

    pub(crate) fn insert_uniform_chunk(
        &mut self,
        key: UVec3,
        value: T,
        mask: [u64; super::MASK_WORDS],
    ) {
        self.chunks.insert(
            ChunkKey::new(key.x, key.y, key.z),
            Chunk::Uniform {
                value,
                mask: BitMask::from_words(mask),
            },
        );
    }

    pub(crate) fn insert_dense_chunk(
        &mut self,
        key: UVec3,
        values: Box<[T]>,
        mask: [u64; super::MASK_WORDS],
    ) {
        self.chunks.insert(
            ChunkKey::new(key.x, key.y, key.z),
            Chunk::Dense {
                values,
                mask: BitMask::from_words(mask),
            },
        );
    }

    /// Counts present voxels by summing chunk masks.
    pub fn present_voxel_count(&self) -> usize {
        self.chunks.values().map(Chunk::present_count).sum()
    }

    /// Returns a copied value when the voxel is present. Returns `None` when the
    /// coordinate is out of bounds or the voxel is unset.
    pub fn get(&self, pos: UVec3) -> Option<T> {
        if !self.is_in_bounds(pos) {
            return None;
        }

        let key = ChunkKey::from_pos(pos);
        let idx = local_index_vec(local_pos(pos));
        self.chunks.get(&key).and_then(|chunk| chunk.get_index(idx))
    }

    /// True only when the coordinate is in bounds and explicitly present.
    pub fn contains(&self, pos: UVec3) -> bool {
        if !self.is_in_bounds(pos) {
            return false;
        }

        let key = ChunkKey::from_pos(pos);
        let idx = local_index_vec(local_pos(pos));
        self.chunks
            .get(&key)
            .is_some_and(|chunk| chunk.contains_index(idx))
    }

    /// Sets one voxel. Returns false when `pos` is out of bounds.
    pub fn set(&mut self, pos: UVec3, value: T) -> bool {
        if !self.is_in_bounds(pos) {
            return false;
        }

        let key = ChunkKey::from_pos(pos);
        let idx = local_index_vec(local_pos(pos));

        match self.chunks.entry(key) {
            Entry::Occupied(mut occupied) => occupied.get_mut().set_index(idx, value),
            Entry::Vacant(vacant) => {
                let mut mask = BitMask::empty();
                mask.set(idx);
                vacant.insert(Chunk::Uniform { value, mask });
            }
        }

        true
    }

    /// Clears one voxel. Returns true only if a present voxel was removed.
    pub fn clear(&mut self, pos: UVec3) -> bool {
        if !self.is_in_bounds(pos) {
            return false;
        }

        let key = ChunkKey::from_pos(pos);
        let idx = local_index_vec(local_pos(pos));

        let (removed, remove_chunk) = match self.chunks.get_mut(&key) {
            Some(chunk) => {
                let removed = chunk.clear_index(idx);
                (removed, chunk.is_empty())
            }
            None => return false,
        };

        if remove_chunk {
            self.chunks.remove(&key);
        }

        removed
    }

    /// Fills every voxel in `aabb` with `value`, ignoring out-of-bounds portions.
    ///
    /// The AABB is half-open: `[min, max)`.
    pub fn fill_aabb(&mut self, aabb: Aabb3u, value: T) {
        let Some(aabb) = aabb.intersect(self.bounds_aabb()) else {
            return;
        };

        let (min_key, max_key) = chunk_key_range_for_aabb(aabb);

        for cz in min_key.z..=max_key.z {
            for cy in min_key.y..=max_key.y {
                for cx in min_key.x..=max_key.x {
                    let key = ChunkKey::new(cx, cy, cz);
                    let Some((local_min, local_max)) = chunk_local_intersection(key, aabb) else {
                        continue;
                    };

                    match self.chunks.entry(key) {
                        Entry::Occupied(mut occupied) => {
                            occupied
                                .get_mut()
                                .fill_local_aabb(local_min, local_max, value);
                        }
                        Entry::Vacant(vacant) => {
                            let mut mask = BitMask::empty();
                            mask.set_aabb(local_min, local_max);
                            vacant.insert(Chunk::Uniform { value, mask });
                        }
                    }
                }
            }
        }
    }

    /// Writes a dense x-fastest value slice into `aabb`, ignoring out-of-bounds portions.
    ///
    /// `values` must match the full, unclipped AABB volume. This lets callers pass
    /// a logical source block and rely on clipping only for destination writes.
    ///
    /// Slice layout:
    ///
    /// ```text
    /// source_index = local.x + local.y * extent.x + local.z * extent.x * extent.y
    /// ```
    pub fn write_aabb_values(&mut self, aabb: Aabb3u, values: &[T]) -> Result<(), GridError> {
        let expected = aabb
            .volume_usize()
            .ok_or(GridError::AabbVolumeDoesNotFitUsize)?;

        if values.len() != expected {
            return Err(GridError::ValuesLengthMismatch {
                expected,
                actual: values.len(),
            });
        }

        let Some(clipped) = aabb.intersect(self.bounds_aabb()) else {
            return Ok(());
        };

        let source_extent = aabb.extent();
        let (min_key, max_key) = chunk_key_range_for_aabb(clipped);

        for cz in min_key.z..=max_key.z {
            for cy in min_key.y..=max_key.y {
                for cx in min_key.x..=max_key.x {
                    let key = ChunkKey::new(cx, cy, cz);
                    let Some((local_min, local_max)) = chunk_local_intersection(key, clipped)
                    else {
                        continue;
                    };
                    let chunk_min = key.world_min();
                    let world_min = add_uvec3(chunk_min, local_min);
                    let source_base = sub_uvec3(world_min, aabb.min);

                    match self.chunks.entry(key) {
                        Entry::Occupied(mut occupied) => {
                            occupied.get_mut().write_values_local(
                                local_min,
                                local_max,
                                source_base,
                                source_extent,
                                values,
                            );
                        }
                        Entry::Vacant(vacant) => {
                            let chunk = Chunk::from_values_local(
                                local_min,
                                local_max,
                                source_base,
                                source_extent,
                                values,
                            );
                            vacant.insert(chunk);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Clears all present voxels inside `aabb`, ignoring out-of-bounds portions.
    pub fn clear_aabb(&mut self, aabb: Aabb3u) {
        let Some(aabb) = aabb.intersect(self.bounds_aabb()) else {
            return;
        };

        let (min_key, max_key) = chunk_key_range_for_aabb(aabb);
        let mut remove_keys = Vec::new();

        for cz in min_key.z..=max_key.z {
            for cy in min_key.y..=max_key.y {
                for cx in min_key.x..=max_key.x {
                    let key = ChunkKey::new(cx, cy, cz);
                    let Some((local_min, local_max)) = chunk_local_intersection(key, aabb) else {
                        continue;
                    };

                    if let Some(chunk) = self.chunks.get_mut(&key) {
                        chunk.clear_local_aabb(local_min, local_max);
                        if chunk.is_empty() {
                            remove_keys.push(key);
                        }
                    }
                }
            }
        }

        for key in remove_keys {
            self.chunks.remove(&key);
        }
    }

    /// True when at least one present voxel exists inside `aabb`.
    pub fn any_present_in_aabb(&self, aabb: Aabb3u) -> bool {
        let Some(aabb) = aabb.intersect(self.bounds_aabb()) else {
            return false;
        };

        let (min_key, max_key) = chunk_key_range_for_aabb(aabb);

        for cz in min_key.z..=max_key.z {
            for cy in min_key.y..=max_key.y {
                for cx in min_key.x..=max_key.x {
                    let key = ChunkKey::new(cx, cy, cz);
                    let Some(chunk) = self.chunks.get(&key) else {
                        continue;
                    };
                    let Some((local_min, local_max)) = chunk_local_intersection(key, aabb) else {
                        continue;
                    };
                    if chunk.any_present_local(local_min, local_max) {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Calls `f(world_pos, value)` for every present voxel inside `aabb`.
    pub fn for_each_present_in_aabb<F>(&self, aabb: Aabb3u, mut f: F)
    where
        F: FnMut(UVec3, T),
    {
        let Some(aabb) = aabb.intersect(self.bounds_aabb()) else {
            return;
        };

        let (min_key, max_key) = chunk_key_range_for_aabb(aabb);

        for cz in min_key.z..=max_key.z {
            for cy in min_key.y..=max_key.y {
                for cx in min_key.x..=max_key.x {
                    let key = ChunkKey::new(cx, cy, cz);
                    let Some(chunk) = self.chunks.get(&key) else {
                        continue;
                    };
                    let Some((local_min, local_max)) = chunk_local_intersection(key, aabb) else {
                        continue;
                    };
                    let chunk_min = key.world_min();

                    chunk.for_each_present_local(local_min, local_max, |idx, value| {
                        let local = local_from_index(idx);
                        let world = add_uvec3(chunk_min, local);
                        f(world, value);
                    });
                }
            }
        }
    }

    /// Merges `src` into `self`.
    ///
    /// Every present source voxel overwrites the destination. Missing source voxels
    /// leave the destination unchanged. Source voxels outside destination bounds
    /// are ignored.
    pub fn union_replace_from(&mut self, src: &Self) {
        if self.bounds.x == 0 || self.bounds.y == 0 || self.bounds.z == 0 {
            return;
        }

        let dst_bounds = self.bounds_aabb();

        for (&key, src_chunk) in &src.chunks {
            let Some((local_min, local_max)) = chunk_local_intersection(key, dst_bounds) else {
                continue;
            };

            if !src_chunk.any_present_local(local_min, local_max) {
                continue;
            }

            match self.chunks.entry(key) {
                Entry::Occupied(mut occupied) => {
                    let dst_chunk = occupied.get_mut();

                    if let Some(value) =
                        src_chunk.uniform_value_if_all_present(local_min, local_max)
                    {
                        dst_chunk.fill_local_aabb(local_min, local_max, value);
                    } else {
                        src_chunk.for_each_present_local(local_min, local_max, |idx, value| {
                            dst_chunk.set_index(idx, value);
                        });
                    }
                }
                Entry::Vacant(vacant) => {
                    if let Some(value) =
                        src_chunk.uniform_value_if_all_present(local_min, local_max)
                    {
                        let mut mask = BitMask::empty();
                        mask.set_aabb(local_min, local_max);
                        vacant.insert(Chunk::Uniform { value, mask });
                    } else if let Some(chunk) = src_chunk.copy_present_local(local_min, local_max) {
                        vacant.insert(chunk);
                    }
                }
            }
        }
    }

    /// Removes destination voxels whenever any corresponding fine-mask voxel is present.
    ///
    /// `scale` maps one destination voxel to a direction-wise scaled AABB in
    /// `fine_mask` coordinates:
    ///
    /// ```text
    /// dst voxel p -> fine AABB
    ///     [p * scale + translation, (p + 1) * scale + translation)
    /// ```
    ///
    /// `translation` is signed so grids with different logical index origins can
    /// be related without first rebasing either grid. Portions of the transformed
    /// AABB outside `fine_mask` are clipped.
    ///
    /// Returns the number of destination voxels removed. A scale of zero is a no-op.

    pub fn mask_out_by_presence_scaled<M>(
        &mut self,
        fine_mask: &SparseGrid3<M>,
        scale: UVec3,
        translation: I64Vec3,
    ) -> usize
    where
        T: Send,
        M: Copy + PartialEq + Sync,
    {
        if scale.cmpeq(UVec3::ZERO).any() || fine_mask.is_empty() || self.is_empty() {
            return 0;
        }

        let dst_bounds = self.bounds_aabb();
        let fine_bounds = fine_mask.bounds;

        use rayon::iter::IntoParallelRefMutIterator;
        use rayon::iter::ParallelIterator;

        let removed: usize = self
            .chunks
            .par_iter_mut()
            .map(|(&key, chunk)| {
                let Some((local_min, local_max)) = chunk_local_intersection(key, dst_bounds) else {
                    return 0;
                };

                let chunk_min = key.world_min();

                chunk.clear_where_present(local_min, local_max, |idx| {
                    let local = local_from_index(idx);
                    let coarse_pos = add_uvec3(chunk_min, local);

                    let Some(mask_aabb) =
                        scaled_voxel_aabb_in_fine_grid(coarse_pos, scale, translation, fine_bounds)
                    else {
                        return false;
                    };

                    fine_mask.any_present_in_aabb(mask_aabb)
                })
            })
            .sum();

        // Can't modify the HashMap structure while par_iter_mut() is active,
        // so remove emptied chunks afterward.
        self.chunks.retain(|_, chunk| !chunk.is_empty());

        removed
    }

    /// Removes empty chunks and compacts dense chunks that are uniform over all
    /// present voxels.
    pub fn prune(&mut self) {
        self.chunks.retain(|_, chunk| {
            if chunk.is_empty() {
                return false;
            }
            chunk.try_compact();
            !chunk.is_empty()
        });
    }

    /// Immutable cached accessor for repeated local reads.
    pub fn accessor(&self) -> GridAccessor<'_, T> {
        GridAccessor::new(self)
    }

    /// Mutable convenience accessor. This keeps the same public API shape as the
    /// immutable accessor, but avoids unsafe self-referential mutable caching.
    pub fn accessor_mut(&mut self) -> GridAccessorMut<'_, T> {
        GridAccessorMut::new(self)
    }
}

/// Cached read accessor. When queries stay inside the same chunk, this avoids
/// repeated hash-map lookups.
pub struct GridAccessor<'a, T>
where
    T: Copy + PartialEq,
{
    grid: &'a SparseGrid3<T>,
    cached_key: Option<ChunkKey>,
    cached_chunk: Option<&'a Chunk<T>>,
}

impl<'a, T> GridAccessor<'a, T>
where
    T: Copy + PartialEq,
{
    #[inline]
    fn new(grid: &'a SparseGrid3<T>) -> Self {
        Self {
            grid,
            cached_key: None,
            cached_chunk: None,
        }
    }

    #[inline]
    /// Return a copied value, reusing the cached chunk when possible.
    pub fn get(&mut self, pos: UVec3) -> Option<T> {
        if !self.grid.is_in_bounds(pos) {
            return None;
        }

        let key = ChunkKey::from_pos(pos);
        if self.cached_key != Some(key) {
            self.cached_key = Some(key);
            self.cached_chunk = self.grid.chunks.get(&key);
        }

        let idx = local_index_vec(local_pos(pos));
        self.cached_chunk.and_then(|chunk| chunk.get_index(idx))
    }

    #[inline]
    /// True when the coordinate is present, reusing the cached chunk when possible.
    pub fn contains(&mut self, pos: UVec3) -> bool {
        self.get(pos).is_some()
    }
}

/// Mutable grid accessor.
///
/// This deliberately delegates mutation to `SparseGrid3` methods to keep the
/// module fully safe Rust. If profiling later proves this path hot, this can be
/// replaced with an unsafe raw-pointer cached mutable chunk accessor.
pub struct GridAccessorMut<'a, T>
where
    T: Copy + PartialEq,
{
    grid: &'a mut SparseGrid3<T>,
}

impl<'a, T> GridAccessorMut<'a, T>
where
    T: Copy + PartialEq,
{
    #[inline]
    fn new(grid: &'a mut SparseGrid3<T>) -> Self {
        Self { grid }
    }

    #[inline]
    /// Return a copied value from the underlying grid.
    pub fn get(&self, pos: UVec3) -> Option<T> {
        self.grid.get(pos)
    }

    #[inline]
    /// True when the coordinate is present in the underlying grid.
    pub fn contains(&self, pos: UVec3) -> bool {
        self.grid.contains(pos)
    }

    #[inline]
    /// Set one voxel in the underlying grid.
    pub fn set(&mut self, pos: UVec3, value: T) -> bool {
        self.grid.set(pos, value)
    }

    #[inline]
    /// Clear one voxel in the underlying grid.
    pub fn clear(&mut self, pos: UVec3) -> bool {
        self.grid.clear(pos)
    }
}
