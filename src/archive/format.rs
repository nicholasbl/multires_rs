use anyhow::{Context, Result, ensure};
use rkyv::{Archive, Deserialize, Serialize};

use crate::sparse::{CHUNK_SIZE, MASK_WORDS, SparseGrid3, SparseGridChunkView};

#[derive(Archive, Deserialize, Serialize)]
pub(super) struct MsgArchiveHeader {
    pub(super) chunk_bits: u32,
    pub(super) simulation_time: f64,
    pub(super) grid_to_world: [f64; 16],
    pub(super) sample_offset: [f64; 3],
    pub(super) levels: Vec<LevelRecord>,
    pub(super) fields: Vec<FieldRecord>,
}

#[derive(Archive, Deserialize, Serialize)]
pub(super) struct LevelRecord {
    pub(super) index_origin: [i32; 3],
    pub(super) point_extent: [u32; 3],
    pub(super) scale: [f64; 3],
    /// Zero on level zero; otherwise a direction-wise positive ratio.
    pub(super) refinement_ratio: [u32; 3],
}

#[derive(Archive, Deserialize, Serialize)]
pub(super) struct FieldRecord {
    pub(super) name: String,
    pub(super) scalar_type: u8,
    pub(super) error_tolerance_kind: u8,
    pub(super) error_absolute: f64,
    pub(super) error_relative_percent: f64,
    pub(super) levels: Vec<FieldLevelRecord>,
}

#[derive(Archive, Deserialize, Serialize)]
pub(super) struct FieldLevelRecord {
    pub(super) data: BlockRecord,
    pub(super) data_masks: BlockRecord,
    pub(super) data_payloads: BlockRecord,
    pub(super) value_ranges: Vec<ChunkRangeRecord>,
    pub(super) active_cubes: BlockRecord,
    pub(super) active_cube_masks: BlockRecord,
}

#[derive(Archive, Deserialize, Serialize, Clone, Copy)]
pub(super) struct BlockRecord {
    pub(super) offset: u64,
    pub(super) stored_length: u64,
    pub(super) decoded_length: u64,
    pub(super) compression: CompressionRecord,
}

#[derive(Archive, Deserialize, Serialize, Clone, Copy)]
#[repr(u8)]
pub(super) enum CompressionRecord {
    None,
}

#[derive(Archive, Deserialize, Serialize)]
pub(super) struct ChunkRangeRecord {
    pub(super) chunk_origin: [u32; 3],
    pub(super) value_min: f64,
    pub(super) value_max: f64,
    pub(super) max_abs_error: f64,
}

/// A scalar grid whose chunks carry independently versioned codec payloads.
///
/// Codec ids are integers rather than an archived enum so a reader can inspect
/// a file containing a future codec and reject only when decoding is requested.
#[derive(Archive, Deserialize, Serialize)]
pub(super) struct ScalarGridRecord {
    pub(super) bounds: [u32; 3],
    pub(super) decoded_type: u8,
    pub(super) chunks: Vec<ScalarChunkRecord>,
}

#[derive(Archive, Deserialize, Serialize)]
pub(super) struct ScalarChunkRecord {
    pub(super) key: [u32; 3],
    pub(super) mask_kind: u8,
    pub(super) mask_offset: u64,
    pub(super) codec_id: u32,
    pub(super) codec_version: u16,
    pub(super) codec_flags: u16,
    pub(super) max_abs_error: f64,
    pub(super) metadata: Vec<u8>,
    pub(super) payload_offset: u64,
    pub(super) payload_length: u64,
}

#[derive(Archive, Deserialize, Serialize)]
pub(super) struct MaskGridRecord {
    bounds: [u32; 3],
    chunks: Vec<MaskChunkRecord>,
}

#[derive(Archive, Deserialize, Serialize)]
struct MaskChunkRecord {
    key: [u32; 3],
    mask_kind: u8,
    mask_offset: u64,
}

pub(super) const MASK_DENSE: u8 = 0;
pub(super) const MASK_FULL: u8 = 1;

pub(super) struct EncodedMaskGrid {
    pub(super) directory: MaskGridRecord,
    pub(super) masks: Vec<u8>,
}

impl MaskGridRecord {
    pub(super) fn from_grid(grid: &SparseGrid3<()>) -> EncodedMaskGrid {
        let mut chunks = Vec::with_capacity(grid.chunk_count());
        let mut masks = Vec::new();
        grid.for_each_chunk(|chunk| {
            let (key, mask) = match chunk {
                SparseGridChunkView::Uniform { key, mask, .. }
                | SparseGridChunkView::Dense { key, mask, .. } => (key, mask),
            };
            let (mask_kind, mask_offset) = if mask.is_full() {
                (MASK_FULL, 0)
            } else {
                let offset = masks.len() as u64;
                for word in mask.as_words() {
                    masks.extend_from_slice(&word.to_le_bytes());
                }
                (MASK_DENSE, offset)
            };
            chunks.push(MaskChunkRecord {
                key: uvec3_to_array(key),
                mask_kind,
                mask_offset,
            });
        });
        EncodedMaskGrid {
            directory: Self {
                bounds: uvec3_to_array(grid.bounds()),
                chunks,
            },
            masks,
        }
    }
}

impl ArchivedMaskGridRecord {
    pub(super) fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub(super) fn validate(&self, masks: &[u8], expected_bounds: glam::UVec3) -> Result<()> {
        let bounds = self.bounds.map(|value| value.to_native());
        ensure!(
            bounds == expected_bounds.to_array(),
            "active-cube grid bounds do not match the field level"
        );
        let mut previous_key = None;
        let mut expected_mask_offset = 0_u64;
        for chunk in self.chunks.iter() {
            let key = chunk.key.map(|value| value.to_native());
            ensure!(
                previous_key.is_none_or(|previous| previous < key),
                "active-cube chunk keys are not unique and sorted"
            );
            let full_chunk = validate_chunk_key(key, bounds)?;
            if chunk.mask_kind == MASK_FULL {
                ensure!(
                    chunk.mask_offset.to_native() == 0,
                    "full mask has a nonzero offset"
                );
            } else {
                ensure!(
                    chunk.mask_offset.to_native() == expected_mask_offset,
                    "active-cube mask block is not canonical"
                );
                expected_mask_offset += (MASK_WORDS * 8) as u64;
            }
            validate_archived_mask(
                chunk.mask_kind,
                chunk.mask_offset.to_native(),
                masks,
                key,
                bounds,
                full_chunk,
            )?;
            previous_key = Some(key);
        }
        ensure!(
            expected_mask_offset as usize == masks.len(),
            "active-cube mask block has trailing data"
        );
        Ok(())
    }

    pub(super) fn to_grid(
        &self,
        masks: &[u8],
        expected_bounds: glam::UVec3,
    ) -> Result<SparseGrid3<()>> {
        self.validate(masks, expected_bounds)?;
        let bounds = self.bounds.map(|value| value.to_native());
        let mut grid = SparseGrid3::with_chunk_capacity(array_to_uvec3(bounds), self.chunks.len());
        for chunk in self.chunks.iter() {
            let key = chunk.key.map(|value| value.to_native());
            let mask = archived_mask_words(chunk.mask_kind, chunk.mask_offset.to_native(), masks)?;
            grid.insert_uniform_chunk(array_to_uvec3(key), (), mask);
        }
        Ok(grid)
    }

    pub(super) fn contains_chunk_key(&self, key: [u32; 3]) -> bool {
        self.chunks
            .binary_search_by(|chunk| chunk.key.map(|component| component.to_native()).cmp(&key))
            .is_ok()
    }

    pub(super) fn contains(&self, masks: &[u8], point: glam::UVec3) -> bool {
        let key = (point / glam::UVec3::splat(CHUNK_SIZE)).to_array();
        let Ok(index) = self
            .chunks
            .binary_search_by(|chunk| chunk.key.map(|component| component.to_native()).cmp(&key))
        else {
            return false;
        };
        let local = point & glam::UVec3::splat(CHUNK_SIZE - 1);
        let bit = local.x as usize
            + local.y as usize * CHUNK_SIZE as usize
            + local.z as usize * CHUNK_SIZE as usize * CHUNK_SIZE as usize;
        let chunk = &self.chunks[index];
        if chunk.mask_kind == MASK_FULL {
            return true;
        }
        read_mask_word(masks, chunk.mask_offset.to_native(), bit / 64)
            .is_some_and(|word| word & (1_u64 << (bit % 64)) != 0)
    }

    pub(super) fn for_each_present_in_aabb(
        &self,
        masks: &[u8],
        aabb: crate::sparse::Aabb3u,
        mut visit: impl FnMut(glam::UVec3),
    ) {
        if aabb.min.cmpeq(aabb.max).any() {
            return;
        }
        let first = aabb.min / glam::UVec3::splat(CHUNK_SIZE);
        let last = (aabb.max - glam::UVec3::ONE) / glam::UVec3::splat(CHUNK_SIZE);
        for z in first.z..=last.z {
            for y in first.y..=last.y {
                for x in first.x..=last.x {
                    let key_array = [x, y, z];
                    let Ok(chunk_index) = self.chunks.binary_search_by(|chunk| {
                        chunk.key.map(|value| value.to_native()).cmp(&key_array)
                    }) else {
                        continue;
                    };
                    let origin = glam::UVec3::new(x, y, z) * glam::UVec3::splat(CHUNK_SIZE);
                    let chunk = &self.chunks[chunk_index];
                    for word_index in 0..MASK_WORDS {
                        let mut remaining = if chunk.mask_kind == MASK_FULL {
                            u64::MAX
                        } else {
                            read_mask_word(masks, chunk.mask_offset.to_native(), word_index)
                                .expect("validated active-cube mask")
                        };
                        while remaining != 0 {
                            let bit = remaining.trailing_zeros() as usize;
                            let index = word_index * 64 + bit;
                            let edge = CHUNK_SIZE as usize;
                            let point = origin
                                + glam::UVec3::new(
                                    (index % edge) as u32,
                                    ((index / edge) % edge) as u32,
                                    (index / (edge * edge)) as u32,
                                );
                            if point.cmpge(aabb.min).all() && point.cmplt(aabb.max).all() {
                                visit(point);
                            }
                            remaining &= remaining - 1;
                        }
                    }
                }
            }
        }
    }
}

pub(super) fn archived_mask_words(
    kind: u8,
    offset: u64,
    masks: &[u8],
) -> Result<[u64; MASK_WORDS]> {
    match kind {
        MASK_FULL => Ok([u64::MAX; MASK_WORDS]),
        MASK_DENSE => {
            let offset = usize::try_from(offset).context("mask offset exceeds usize")?;
            let length = MASK_WORDS * 8;
            let bytes = masks
                .get(offset..offset.checked_add(length).context("mask range overflow")?)
                .context("mask range is outside its block")?;
            Ok(std::array::from_fn(|index| {
                u64::from_le_bytes(bytes[index * 8..index * 8 + 8].try_into().unwrap())
            }))
        }
        kind => anyhow::bail!("unsupported chunk mask kind {kind}"),
    }
}

pub(super) fn archived_mask_contains(
    kind: u8,
    offset: u64,
    masks: &[u8],
    index: usize,
) -> Result<bool> {
    ensure!(
        index < MASK_WORDS * 64,
        "chunk-local mask index is out of bounds"
    );
    match kind {
        MASK_FULL => Ok(true),
        MASK_DENSE => Ok(read_mask_word(masks, offset, index / 64)
            .context("mask range is outside its block")?
            & (1_u64 << (index % 64))
            != 0),
        kind => anyhow::bail!("unsupported chunk mask kind {kind}"),
    }
}

fn validate_archived_mask(
    kind: u8,
    offset: u64,
    masks: &[u8],
    key: [u32; 3],
    bounds: [u32; 3],
    full_chunk: bool,
) -> Result<()> {
    match kind {
        MASK_FULL => ensure!(full_chunk, "full mask is invalid for a partial edge chunk"),
        MASK_DENSE => {
            let words = archived_mask_words(kind, offset, masks)?;
            ensure!(
                words.iter().any(|&word| word != 0),
                "empty chunk mask is not canonical"
            );
            ensure!(
                words.iter().any(|&word| word != u64::MAX),
                "dense full chunk mask is not canonical"
            );
            if !full_chunk {
                validate_chunk_mask(key, bounds, &words)?;
            }
        }
        kind => anyhow::bail!("unsupported chunk mask kind {kind}"),
    }
    Ok(())
}

fn read_mask_word(masks: &[u8], offset: u64, index: usize) -> Option<u64> {
    let start = usize::try_from(offset)
        .ok()?
        .checked_add(index.checked_mul(8)?)?;
    Some(u64::from_le_bytes(
        masks.get(start..start + 8)?.try_into().ok()?,
    ))
}

pub(super) fn validate_chunk_mask(
    key: [u32; 3],
    bounds: [u32; 3],
    mask: &[u64; MASK_WORDS],
) -> Result<()> {
    if validate_chunk_key(key, bounds)? {
        return Ok(());
    }
    let chunk_min = key.map(|component| component.saturating_mul(CHUNK_SIZE));
    let local_extent =
        std::array::from_fn::<_, 3, _>(|axis| (bounds[axis] - chunk_min[axis]).min(CHUNK_SIZE));
    for (word_index, &word) in mask.iter().enumerate() {
        let mut remaining = word;
        while remaining != 0 {
            let bit = remaining.trailing_zeros() as usize;
            let index = word_index * u64::BITS as usize + bit;
            let edge = CHUNK_SIZE as usize;
            let local = [
                (index % edge) as u32,
                ((index / edge) % edge) as u32,
                (index / (edge * edge)) as u32,
            ];
            ensure!(
                local
                    .iter()
                    .zip(local_extent)
                    .all(|(&coordinate, extent)| coordinate < extent),
                "chunk key {key:?} has present values outside grid bounds {bounds:?}"
            );
            remaining &= remaining - 1;
        }
    }
    Ok(())
}

pub(super) fn validate_chunk_key(key: [u32; 3], bounds: [u32; 3]) -> Result<bool> {
    let chunk_min = key.map(|component| component.saturating_mul(CHUNK_SIZE));
    ensure!(
        chunk_min
            .iter()
            .zip(bounds)
            .all(|(&minimum, bound)| minimum < bound),
        "chunk key {key:?} is outside grid bounds {bounds:?}"
    );
    let local_extent =
        std::array::from_fn::<_, 3, _>(|axis| (bounds[axis] - chunk_min[axis]).min(CHUNK_SIZE));
    Ok(local_extent == [CHUNK_SIZE; 3])
}

pub(super) fn uvec3_to_array(value: glam::UVec3) -> [u32; 3] {
    [value.x, value.y, value.z]
}

pub(super) fn array_to_uvec3(value: [u32; 3]) -> glam::UVec3 {
    glam::UVec3::from_array(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archived_mask_grid_rejects_unexpected_bounds() -> Result<()> {
        let record = MaskGridRecord {
            bounds: [CHUNK_SIZE; 3],
            chunks: Vec::new(),
        };
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&record)?;
        let archived = rkyv::access::<ArchivedMaskGridRecord, rkyv::rancor::Error>(&bytes)?;

        let error = archived
            .to_grid(&[], glam::UVec3::splat(CHUNK_SIZE - 1))
            .unwrap_err();
        assert!(error.to_string().contains("bounds do not match"));
        Ok(())
    }
}
