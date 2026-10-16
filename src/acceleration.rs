//! Rebuildable point-grid acceleration used by isosurface extraction.

use glam::UVec3;

use crate::{
    ChunkValueRange, MsgFieldAcceleration, MsgScalarGrid,
    sparse::{Aabb3u, CHUNK_SIZE, MASK_WORDS, SparseGrid3, SparseGridChunkView},
};

pub(crate) fn build_active_cubes(samples: &MsgScalarGrid) -> SparseGrid3<()> {
    match samples {
        MsgScalarGrid::F32(grid) => build_active_cubes_for_grid(grid),
        MsgScalarGrid::F64(grid) => build_active_cubes_for_grid(grid),
    }
}

fn build_active_cubes_for_grid<T>(samples: &SparseGrid3<T>) -> SparseGrid3<()>
where
    T: Copy + PartialEq + Sync,
{
    let sample_bounds = samples.bounds();
    let cube_bounds = UVec3::new(
        sample_bounds.x.saturating_sub(1),
        sample_bounds.y.saturating_sub(1),
        sample_bounds.z.saturating_sub(1),
    );
    let sample_regions = samples.chunk_aabbs();
    let mut candidates = SparseGrid3::with_chunk_capacity(cube_bounds, sample_regions.len());
    for region in sample_regions {
        candidates.fill_aabb(
            Aabb3u::new(
                UVec3::new(
                    region.min.x.saturating_sub(1),
                    region.min.y.saturating_sub(1),
                    region.min.z.saturating_sub(1),
                ),
                region.max,
            ),
            (),
        );
    }

    let active_chunks = candidates.par_map_chunks(|chunk| {
        let (key, mask) = match chunk {
            SparseGridChunkView::Uniform { key, mask, .. }
            | SparseGridChunkView::Dense { key, mask, .. } => (key, mask),
        };
        let chunk_min = key * CHUNK_SIZE;
        let mut active_mask = [0_u64; MASK_WORDS];
        let mut accessor = samples.accessor();
        for (word_index, &word) in mask.iter().enumerate() {
            let mut remaining = word;
            while remaining != 0 {
                let bit = remaining.trailing_zeros() as usize;
                let index = word_index * u64::BITS as usize + bit;
                let edge = CHUNK_SIZE as usize;
                let anchor = chunk_min
                    + UVec3::new(
                        (index % edge) as u32,
                        ((index / edge) % edge) as u32,
                        (index / (edge * edge)) as u32,
                    );
                let mut complete = true;
                for dz in 0..=1 {
                    for dy in 0..=1 {
                        for dx in 0..=1 {
                            complete &= accessor.contains(anchor + UVec3::new(dx, dy, dz));
                        }
                    }
                }
                if complete {
                    active_mask[word_index] |= 1_u64 << bit;
                }
                remaining &= remaining - 1;
            }
        }
        active_mask
            .iter()
            .any(|&word| word != 0)
            .then_some((key, active_mask))
    });
    let mut active_chunks = active_chunks.into_iter().flatten().collect::<Vec<_>>();
    active_chunks.sort_by_key(|(key, _)| (key.z, key.y, key.x));
    let mut active = SparseGrid3::with_chunk_capacity(cube_bounds, active_chunks.len());
    for (key, mask) in active_chunks {
        active.insert_uniform_chunk(key, (), mask);
    }
    active
}

pub(crate) fn build_value_ranges(
    samples: &MsgScalarGrid,
    active_cubes: &SparseGrid3<()>,
) -> Vec<ChunkValueRange> {
    active_cubes
        .chunk_aabbs()
        .into_iter()
        .filter_map(|cube_aabb| {
            let bounds = samples.bounds();
            let sample_max = UVec3::new(
                cube_aabb.max.x.saturating_add(1).min(bounds.x),
                cube_aabb.max.y.saturating_add(1).min(bounds.y),
                cube_aabb.max.z.saturating_add(1).min(bounds.z),
            );
            let mut min = f64::INFINITY;
            let mut max = f64::NEG_INFINITY;
            samples.for_each_value_in_aabb(Aabb3u::new(cube_aabb.min, sample_max), |_, value| {
                min = min.min(value);
                max = max.max(value);
            });
            (min <= max).then_some(ChunkValueRange {
                chunk_origin: cube_aabb.min,
                min,
                max,
            })
        })
        .collect()
}

pub(crate) fn finish_acceleration(
    samples: &MsgScalarGrid,
    active_cubes: SparseGrid3<()>,
) -> MsgFieldAcceleration {
    let value_ranges = build_value_ranges(samples, &active_cubes);
    MsgFieldAcceleration {
        value_ranges,
        active_cubes,
    }
}
