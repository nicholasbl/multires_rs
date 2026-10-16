use glam::UVec3;

use super::{
    CHUNK_VOLUME, MASK_WORD_BITS, MASK_WORDS,
    layout::{
        copy_source_region_into_dense, fill_dense_aabb, is_full_chunk_local_aabb, local_extent,
        local_index, source_index, uniform_value_in_source_region, valid_local_aabb,
    },
};

#[derive(Debug, Clone)]
pub(super) enum Chunk<T>
where
    T: Copy + PartialEq,
{
    /// One value for every present voxel, plus a presence mask.
    Uniform { value: T, mask: BitMask },

    /// Dense value array, plus a presence mask.
    Dense { values: Box<[T]>, mask: BitMask },
}

impl<T> Chunk<T>
where
    T: Copy + PartialEq,
{
    #[inline]
    pub(super) fn is_empty(&self) -> bool {
        match self {
            Self::Uniform { mask, .. } | Self::Dense { mask, .. } => mask.is_empty(),
        }
    }

    #[inline]
    pub(super) fn present_count(&self) -> usize {
        match self {
            Self::Uniform { mask, .. } | Self::Dense { mask, .. } => mask.count_ones(),
        }
    }

    #[inline]
    pub(super) fn contains_index(&self, idx: usize) -> bool {
        match self {
            Self::Uniform { mask, .. } | Self::Dense { mask, .. } => mask.get(idx),
        }
    }

    #[inline]
    pub(super) fn get_index(&self, idx: usize) -> Option<T> {
        match self {
            Self::Uniform { value, mask } => mask.get(idx).then_some(*value),
            Self::Dense { values, mask } => mask.get(idx).then_some(values[idx]),
        }
    }

    pub(super) fn set_index(&mut self, idx: usize, new_value: T) {
        match self {
            Self::Uniform { value, mask } => {
                if mask.is_empty() || *value == new_value {
                    *value = new_value;
                    mask.set(idx);
                    return;
                }

                let old_value = *value;
                let mut dense = vec![old_value; CHUNK_VOLUME].into_boxed_slice();
                dense[idx] = new_value;
                mask.set(idx);
                let mask = mask.clone();
                *self = Self::Dense {
                    values: dense,
                    mask,
                };
            }
            Self::Dense { values, mask } => {
                values[idx] = new_value;
                mask.set(idx);
            }
        }
    }

    pub(super) fn clear_index(&mut self, idx: usize) -> bool {
        match self {
            Self::Uniform { mask, .. } | Self::Dense { mask, .. } => mask.clear(idx),
        }
    }

    pub(super) fn fill_local_aabb(&mut self, local_min: UVec3, local_max: UVec3, new_value: T) {
        debug_assert!(valid_local_aabb(local_min, local_max));

        if is_full_chunk_local_aabb(local_min, local_max) {
            *self = Self::Uniform {
                value: new_value,
                mask: BitMask::full(),
            };
            return;
        }

        match self {
            Self::Uniform { value, mask } => {
                if mask.is_empty() || *value == new_value {
                    *value = new_value;
                    mask.set_aabb(local_min, local_max);
                    return;
                }

                let old_value = *value;
                let mut dense = vec![old_value; CHUNK_VOLUME].into_boxed_slice();
                fill_dense_aabb(&mut dense, local_min, local_max, new_value);
                mask.set_aabb(local_min, local_max);
                let mask = mask.clone();
                *self = Self::Dense {
                    values: dense,
                    mask,
                };
            }
            Self::Dense { values, mask } => {
                fill_dense_aabb(values, local_min, local_max, new_value);
                mask.set_aabb(local_min, local_max);
            }
        }
    }

    pub(super) fn clear_local_aabb(&mut self, local_min: UVec3, local_max: UVec3) {
        debug_assert!(valid_local_aabb(local_min, local_max));
        match self {
            Self::Uniform { mask, .. } | Self::Dense { mask, .. } => {
                mask.clear_aabb(local_min, local_max);
            }
        }
    }

    pub(super) fn any_present_local(&self, local_min: UVec3, local_max: UVec3) -> bool {
        debug_assert!(valid_local_aabb(local_min, local_max));
        match self {
            Self::Uniform { mask, .. } | Self::Dense { mask, .. } => {
                mask.any_in_aabb(local_min, local_max)
            }
        }
    }

    pub(super) fn uniform_value_if_all_present(
        &self,
        local_min: UVec3,
        local_max: UVec3,
    ) -> Option<T> {
        match self {
            Self::Uniform { value, mask } => {
                mask.all_in_aabb(local_min, local_max).then_some(*value)
            }
            Self::Dense { .. } => None,
        }
    }

    pub(super) fn for_each_present_local<F>(&self, local_min: UVec3, local_max: UVec3, mut f: F)
    where
        F: FnMut(usize, T),
    {
        debug_assert!(valid_local_aabb(local_min, local_max));

        match self {
            Self::Uniform { value, mask } => {
                mask.for_each_set_index_in_aabb(local_min, local_max, |idx| f(idx, *value));
            }
            Self::Dense { values, mask } => {
                mask.for_each_set_index_in_aabb(local_min, local_max, |idx| f(idx, values[idx]));
            }
        }
    }

    pub(super) fn clear_where_present<F>(
        &mut self,
        local_min: UVec3,
        local_max: UVec3,
        mut pred: F,
    ) -> usize
    where
        F: FnMut(usize) -> bool,
    {
        debug_assert!(valid_local_aabb(local_min, local_max));

        let mut to_clear = Vec::new();
        match self {
            Self::Uniform { mask, .. } | Self::Dense { mask, .. } => {
                mask.for_each_set_index_in_aabb(local_min, local_max, |idx| {
                    if pred(idx) {
                        to_clear.push(idx);
                    }
                });

                for idx in &to_clear {
                    mask.clear(*idx);
                }
            }
        }

        to_clear.len()
    }

    pub(super) fn from_values_local(
        local_min: UVec3,
        local_max: UVec3,
        source_base: UVec3,
        source_extent: UVec3,
        source: &[T],
    ) -> Self {
        debug_assert!(valid_local_aabb(local_min, local_max));

        if let Some(value) = uniform_value_in_source_region(
            source,
            source_base,
            source_extent,
            local_extent(local_min, local_max),
        ) {
            let mut mask = BitMask::empty();
            mask.set_aabb(local_min, local_max);
            return Self::Uniform { value, mask };
        }

        let first = source[source_index(source_base, source_extent)];
        let mut values = vec![first; CHUNK_VOLUME].into_boxed_slice();
        copy_source_region_into_dense(
            &mut values,
            local_min,
            local_max,
            source_base,
            source_extent,
            source,
        );

        let mut mask = BitMask::empty();
        mask.set_aabb(local_min, local_max);
        Self::Dense { values, mask }
    }

    pub(super) fn write_values_local(
        &mut self,
        local_min: UVec3,
        local_max: UVec3,
        source_base: UVec3,
        source_extent: UVec3,
        source: &[T],
    ) {
        debug_assert!(valid_local_aabb(local_min, local_max));

        let region_extent = local_extent(local_min, local_max);
        let uniform =
            uniform_value_in_source_region(source, source_base, source_extent, region_extent);

        match self {
            Self::Uniform { value, mask } => {
                if let Some(new_value) = uniform {
                    if mask.is_empty() || *value == new_value {
                        *value = new_value;
                        mask.set_aabb(local_min, local_max);
                        return;
                    }

                    if is_full_chunk_local_aabb(local_min, local_max) {
                        *self = Self::Uniform {
                            value: new_value,
                            mask: BitMask::full(),
                        };
                        return;
                    }
                }

                let old_value = *value;
                let mut dense = vec![old_value; CHUNK_VOLUME].into_boxed_slice();
                copy_source_region_into_dense(
                    &mut dense,
                    local_min,
                    local_max,
                    source_base,
                    source_extent,
                    source,
                );
                mask.set_aabb(local_min, local_max);
                let mask = mask.clone();
                *self = Self::Dense {
                    values: dense,
                    mask,
                };
            }
            Self::Dense { values, mask } => {
                if let Some(new_value) = uniform {
                    fill_dense_aabb(values, local_min, local_max, new_value);
                } else {
                    copy_source_region_into_dense(
                        values,
                        local_min,
                        local_max,
                        source_base,
                        source_extent,
                        source,
                    );
                }
                mask.set_aabb(local_min, local_max);
            }
        }
    }

    pub(super) fn copy_present_local(&self, local_min: UVec3, local_max: UVec3) -> Option<Self> {
        let mut out: Option<Self> = None;

        self.for_each_present_local(local_min, local_max, |idx, value| {
            if let Some(chunk) = out.as_mut() {
                chunk.set_index(idx, value);
            } else {
                let mut mask = BitMask::empty();
                mask.set(idx);
                out = Some(Self::Uniform { value, mask });
            }
        });

        if let Some(chunk) = &mut out {
            chunk.try_compact();
        }

        out
    }

    pub(super) fn try_compact(&mut self) {
        let replacement = match self {
            Self::Dense { values, mask } => {
                let Some(first_idx) = mask.first_set_index() else {
                    return;
                };
                let first_value = values[first_idx];

                let mut all_same = true;
                mask.for_each_set_index(|idx| {
                    if values[idx] != first_value {
                        all_same = false;
                    }
                });

                all_same.then(|| (first_value, mask.clone()))
            }
            Self::Uniform { .. } => None,
        };

        if let Some((value, mask)) = replacement {
            *self = Self::Uniform { value, mask };
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct BitMask {
    words: [u64; MASK_WORDS],
}

impl BitMask {
    #[inline]
    pub(super) fn words(&self) -> [u64; MASK_WORDS] {
        self.words
    }

    #[inline]
    pub(super) fn from_words(words: [u64; MASK_WORDS]) -> Self {
        Self { words }
    }

    #[inline]
    pub(super) fn empty() -> Self {
        Self {
            words: [0; MASK_WORDS],
        }
    }

    #[inline]
    pub(super) fn full() -> Self {
        Self {
            words: [u64::MAX; MASK_WORDS],
        }
    }

    #[inline]
    pub(super) fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    #[inline]
    pub(super) fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    #[inline]
    pub(super) fn get(&self, idx: usize) -> bool {
        debug_assert!(idx < CHUNK_VOLUME);
        let word = idx / MASK_WORD_BITS;
        let bit = idx % MASK_WORD_BITS;
        (self.words[word] & (1u64 << bit)) != 0
    }

    #[inline]
    pub(super) fn set(&mut self, idx: usize) {
        debug_assert!(idx < CHUNK_VOLUME);
        let word = idx / MASK_WORD_BITS;
        let bit = idx % MASK_WORD_BITS;
        self.words[word] |= 1u64 << bit;
    }

    /// Clears `idx`. Returns true if it was previously set.
    #[inline]
    pub(super) fn clear(&mut self, idx: usize) -> bool {
        debug_assert!(idx < CHUNK_VOLUME);
        let word = idx / MASK_WORD_BITS;
        let bit = idx % MASK_WORD_BITS;
        let mask = 1u64 << bit;
        let was_set = (self.words[word] & mask) != 0;
        self.words[word] &= !mask;
        was_set
    }

    pub(super) fn set_aabb(&mut self, local_min: UVec3, local_max: UVec3) {
        debug_assert!(valid_local_aabb(local_min, local_max));

        if is_full_chunk_local_aabb(local_min, local_max) {
            self.words.fill(u64::MAX);
            return;
        }

        for z in local_min.z..local_max.z {
            for y in local_min.y..local_max.y {
                for x in local_min.x..local_max.x {
                    self.set(local_index(x, y, z));
                }
            }
        }
    }

    pub(super) fn clear_aabb(&mut self, local_min: UVec3, local_max: UVec3) {
        debug_assert!(valid_local_aabb(local_min, local_max));

        if is_full_chunk_local_aabb(local_min, local_max) {
            self.words.fill(0);
            return;
        }

        for z in local_min.z..local_max.z {
            for y in local_min.y..local_max.y {
                for x in local_min.x..local_max.x {
                    self.clear(local_index(x, y, z));
                }
            }
        }
    }

    pub(super) fn any_in_aabb(&self, local_min: UVec3, local_max: UVec3) -> bool {
        debug_assert!(valid_local_aabb(local_min, local_max));

        if is_full_chunk_local_aabb(local_min, local_max) {
            return !self.is_empty();
        }

        for z in local_min.z..local_max.z {
            for y in local_min.y..local_max.y {
                for x in local_min.x..local_max.x {
                    if self.get(local_index(x, y, z)) {
                        return true;
                    }
                }
            }
        }

        false
    }

    pub(super) fn all_in_aabb(&self, local_min: UVec3, local_max: UVec3) -> bool {
        debug_assert!(valid_local_aabb(local_min, local_max));

        for z in local_min.z..local_max.z {
            for y in local_min.y..local_max.y {
                for x in local_min.x..local_max.x {
                    if !self.get(local_index(x, y, z)) {
                        return false;
                    }
                }
            }
        }

        true
    }

    pub(super) fn first_set_index(&self) -> Option<usize> {
        for (word_i, &word) in self.words.iter().enumerate() {
            if word != 0 {
                return Some(word_i * MASK_WORD_BITS + word.trailing_zeros() as usize);
            }
        }
        None
    }

    pub(super) fn for_each_set_index<F>(&self, mut f: F)
    where
        F: FnMut(usize),
    {
        for (word_i, &word) in self.words.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                let idx = word_i * MASK_WORD_BITS + bit;
                f(idx);
                bits &= bits - 1;
            }
        }
    }

    pub(super) fn for_each_set_index_in_aabb<F>(&self, local_min: UVec3, local_max: UVec3, mut f: F)
    where
        F: FnMut(usize),
    {
        debug_assert!(valid_local_aabb(local_min, local_max));

        if is_full_chunk_local_aabb(local_min, local_max) {
            self.for_each_set_index(f);
            return;
        }

        for z in local_min.z..local_max.z {
            for y in local_min.y..local_max.y {
                for x in local_min.x..local_max.x {
                    let idx = local_index(x, y, z);
                    if self.get(idx) {
                        f(idx);
                    }
                }
            }
        }
    }
}
