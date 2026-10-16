//! Chunked sparse 3D grid for very large bounded voxel domains.
//!
//! Design goals:
//! - Sparse by chunk, not sparse by point.
//! - Dense, cache-friendly storage inside populated chunks.
//! - Per-voxel presence is tracked separately from stored values.
//! - Chunks can be represented as a single uniform value plus an occupancy mask,
//!   or as a dense value array plus an occupancy mask.
//! - Coordinates are `glam::UVec3` and AABBs are half-open: `[min, max)`.
//!
//! Bulk slice layout is x-fastest, then y, then z:
//!
//! ```text
//! index = x + y * extent.x + z * extent.x * extent.y
//! ```

mod aabb;
mod chunk;
mod grid;
mod layout;

pub use aabb::Aabb3u;
pub(crate) use grid::SparseGridChunkView;
pub use grid::{GridAccessor, GridAccessorMut, GridError, SparseGrid3};

/// log2 of the chunk edge length.
pub const CHUNK_BITS: u32 = 5;

/// Number of cells along one chunk axis.
pub const CHUNK_SIZE: u32 = 1 << CHUNK_BITS; // 32

/// Mask used to convert a world coordinate into a local chunk coordinate.
pub const CHUNK_MASK: u32 = CHUNK_SIZE - 1;

/// Number of voxels in a chunk.
pub const CHUNK_VOLUME: usize =
    (CHUNK_SIZE as usize) * (CHUNK_SIZE as usize) * (CHUNK_SIZE as usize); // 32768

pub(super) const MASK_WORD_BITS: usize = 64;
pub(crate) const MASK_WORDS: usize = CHUNK_VOLUME / MASK_WORD_BITS; // 512

#[cfg(test)]
mod tests;
