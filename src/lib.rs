//! Multiresolution sparse-grid (MSG) data structures and algorithms.
//!
//! MSG datasets contain finite, scalar, point-associated fields. Each field
//! owns its sparse topology; level geometry is shared by the dataset.

mod acceleration;
pub mod archive;
mod dataset;
pub mod decimate;
mod geometry;
pub mod isosurface;
mod mesh;
mod point_grid;
pub mod relax;
mod scalar;
pub mod slice;
pub mod sparse;

pub use archive::{
    MSG_FORMAT_MAGIC, MSG_FORMAT_VERSION, MappedMsgArchive, MsgArchive, MsgArchiveChunkRange,
    MsgArchiveField, MsgArchiveFieldInfo, MsgArchiveFieldLevel, MsgArchiveLevelInfo,
    MsgChunkBuffer, MsgChunkCodec, MsgChunkCodecSelection, MsgErrorTolerance, MsgFieldCodec,
    MsgWriteOptions, read_msg, read_msg_fields, transcode_msg, view_msg, write_msg,
};
pub use dataset::{
    ChunkValueRange, MsgDataset, MsgField, MsgFieldAcceleration, MsgFieldLevel, MsgLevel,
    MsgValidationError,
};
pub use decimate::{
    DecimateErrorMode, DecimateOptions, DecimateParallelOptions, DecimatePipelineOptions,
    DecimatePipelineResult, DecimatePipelineTimings, DecimateResult, DecimateTarget,
    DecimateTimings, decimate_mesh, decimate_mesh_pipeline,
};
pub use geometry::{ArchivedMat4, MsgGeometry, cell_center_to_point_geometry};
pub use isosurface::{
    IsosurfaceOptions, IsosurfaceTimings, PerLevelTimings, Sample, Surface, isosurface,
    isosurface_archive,
};
pub use mesh::{Mesh3D, RemoveDegenerateTrianglesOptions, remove_degenerate_triangles};
pub use relax::{VertexRelaxOptions, VertexRelaxResult, VertexRelaxTimings, relax_mesh_vertices};
pub use scalar::{MsgScalarGrid, MsgScalarType};
pub use slice::{SliceAxis, SliceOptions, SlicePlane, slice};
