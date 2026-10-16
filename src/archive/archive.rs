use rkyv::{Archive, Deserialize, Serialize};

use crate::sparse::{MASK_WORDS, SparseGrid3, SparseGridChunkView};

#[derive(Archive, Deserialize, Serialize)]
pub(super) struct MsgArchiveHeader {
    pub(super) chunk_bits: u32,
    pub(super) simulation_time: f64,
    pub(super) grid_to_world: [f64; 16],
    pub(super) sample_offset: [f64; 3],
    pub(super) levels: Vec<ArchiveLevel>,
    pub(super) fields: Vec<ArchiveField>,
}

#[derive(Archive, Deserialize, Serialize)]
pub(super) struct ArchiveLevel {
    pub(super) index_origin: [i32; 3],
    pub(super) point_extent: [u32; 3],
    pub(super) scale: [f64; 3],
    /// Zero on level zero; otherwise a direction-wise positive ratio.
    pub(super) refinement_ratio: [u32; 3],
}

#[derive(Archive, Deserialize, Serialize)]
pub(super) struct ArchiveField {
    pub(super) name: String,
    pub(super) scalar_type: u8,
    pub(super) error_tolerance_kind: u8,
    pub(super) error_absolute: f64,
    pub(super) error_relative_percent: f64,
    pub(super) levels: Vec<ArchiveFieldLevel>,
}

#[derive(Archive, Deserialize, Serialize)]
pub(super) struct ArchiveFieldLevel {
    pub(super) data: ArchiveBlock,
    pub(super) value_ranges: Vec<ArchiveChunkRange>,
    pub(super) active_cubes: Option<ArchiveBlock>,
}

#[derive(Archive, Deserialize, Serialize, Clone, Copy)]
pub(super) struct ArchiveBlock {
    pub(super) offset: u64,
    pub(super) stored_length: u64,
    pub(super) decoded_length: u64,
    pub(super) compression: ArchiveCompression,
}

#[derive(Archive, Deserialize, Serialize, Clone, Copy)]
#[repr(u8)]
pub(super) enum ArchiveCompression {
    None,
}

#[derive(Archive, Deserialize, Serialize)]
pub(super) struct ArchiveChunkRange {
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
pub(super) struct ArchiveScalarGrid {
    pub(super) bounds: [u32; 3],
    pub(super) decoded_type: u8,
    pub(super) chunks: Vec<ArchiveScalarChunk>,
}

#[derive(Archive, Deserialize, Serialize)]
pub(super) struct ArchiveScalarChunk {
    pub(super) key: [u32; 3],
    pub(super) mask: Box<[u64; MASK_WORDS]>,
    pub(super) codec_id: u32,
    pub(super) codec_version: u16,
    pub(super) codec_flags: u16,
    pub(super) max_abs_error: f64,
    pub(super) metadata: Vec<u8>,
    pub(super) payload_length: u64,
    pub(super) payload: Vec<u64>,
}

#[derive(Archive, Deserialize, Serialize)]
pub(super) struct ArchiveMaskGrid {
    bounds: [u32; 3],
    chunks: Vec<ArchiveMaskChunk>,
}

#[derive(Archive, Deserialize, Serialize)]
struct ArchiveMaskChunk {
    key: [u32; 3],
    mask: Box<[u64; MASK_WORDS]>,
}

impl ArchiveMaskGrid {
    pub(super) fn from_grid(grid: &SparseGrid3<()>) -> Self {
        let mut chunks = Vec::with_capacity(grid.chunk_count());
        grid.for_each_chunk(|chunk| match chunk {
            SparseGridChunkView::Uniform { key, mask, .. }
            | SparseGridChunkView::Dense { key, mask, .. } => chunks.push(ArchiveMaskChunk {
                key: uvec3_to_array(key),
                mask: Box::new(mask),
            }),
        });
        chunks.sort_by_key(|chunk| chunk.key);
        Self {
            bounds: uvec3_to_array(grid.bounds()),
            chunks,
        }
    }

    pub(super) fn into_grid(self) -> SparseGrid3<()> {
        let mut grid =
            SparseGrid3::with_chunk_capacity(array_to_uvec3(self.bounds), self.chunks.len());
        for chunk in self.chunks {
            grid.insert_uniform_chunk(array_to_uvec3(chunk.key), (), *chunk.mask);
        }
        grid
    }
}

pub(super) fn uvec3_to_array(value: glam::UVec3) -> [u32; 3] {
    [value.x, value.y, value.z]
}

pub(super) fn array_to_uvec3(value: [u32; 3]) -> glam::UVec3 {
    glam::UVec3::from_array(value)
}
