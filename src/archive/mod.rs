//! Versioned `.msg` archive reading, writing, and memory-mapped inspection.

mod chunk_codec;
mod format;

use std::{fs::File, io::Write, path::Path};

use anyhow::{Context, Result, bail, ensure};
use glam::{DMat4, UVec3};
use memmap2::Mmap;
use rkyv::rancor::Error;

use crate::{
    ChunkValueRange, MsgDataset, MsgField, MsgFieldAcceleration, MsgFieldLevel, MsgGeometry,
    MsgLevel, MsgScalarGrid, MsgScalarType,
    sparse::{CHUNK_BITS, CHUNK_SIZE, CHUNK_VOLUME, MASK_WORDS},
};
use format::*;

pub const MSG_FORMAT_MAGIC: u32 = u32::from_le_bytes(*b"MSG\0");
pub const MSG_FORMAT_VERSION: u32 = 1;
const PREFIX_LENGTH: u64 = 8;
const FOOTER_LENGTH: u64 = 8;

/// A concrete codec available for scalar chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsgChunkCodec {
    UniformF64,
    DenseF32,
    DenseF64,
    QuantizedU8,
    QuantizedU16,
    QuantizedU32,
    AffineF64,
    AffineResidualU8,
    AffineResidualU16,
    AffineResidualU32,
}

/// How the writer chooses the codec for non-uniform chunks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MsgChunkCodecSelection {
    #[default]
    AutoLossless,
    AutoSmallest,
    Forced(MsgChunkCodec),
}

/// Maximum pointwise physical error accepted from a lossy codec.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MsgErrorTolerance {
    pub absolute: f64,
    pub relative_percent: f64,
}

/// Per-field archive encoding policy. Unlisted fields use `auto:lossless`.
#[derive(Debug, Clone, PartialEq)]
pub struct MsgFieldCodec {
    pub field: String,
    pub selection: MsgChunkCodecSelection,
    pub error_tolerance: Option<MsgErrorTolerance>,
}

#[derive(Debug, Clone, Default)]
pub struct MsgWriteOptions {
    pub codecs: Vec<MsgFieldCodec>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ResolvedFieldCodec {
    pub(crate) selection: MsgChunkCodecSelection,
    pub(crate) error_tolerance: Option<MsgErrorTolerance>,
    pub(crate) physical_offset: f64,
    pub(crate) physical_scale: f64,
}

/// Borrowed, validated metadata view over `.msg` bytes.
pub struct MsgArchive<'a> {
    bytes: &'a [u8],
    data_end: usize,
    archive: &'a ArchivedMsgArchiveHeader,
}

/// Owner for a read-only memory mapping. Call [`MappedMsgArchive::view`] to
/// obtain a validated borrowing archive view.
pub struct MappedMsgArchive {
    mmap: Mmap,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MsgArchiveLevelInfo {
    pub index_origin: [i32; 3],
    pub point_extent: [u32; 3],
    pub scale: [f64; 3],
    pub refinement_ratio_from_coarse: Option<[u32; 3]>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MsgArchiveFieldInfo<'a> {
    pub name: &'a str,
    pub scalar_type: MsgScalarType,
    pub error_tolerance: Option<MsgErrorTolerance>,
}

/// Borrowed view of one scalar field in an MSG archive.
pub struct MsgArchiveField<'a> {
    bytes: &'a [u8],
    data_end: usize,
    archive: &'a ArchivedMsgArchiveHeader,
    field: &'a ArchivedFieldRecord,
}

/// Borrowed, lazily decoded view of one archived field level.
#[derive(Clone, Copy)]
pub struct MsgArchiveFieldLevel<'a> {
    scalar_type: MsgScalarType,
    bounds: UVec3,
    scalar: &'a ArchivedScalarGridRecord,
    scalar_masks: &'a [u8],
    scalar_payloads: &'a [u8],
    active_cubes: &'a ArchivedMaskGridRecord,
    active_cube_masks: &'a [u8],
    field_level: &'a ArchivedFieldLevelRecord,
}

/// Conservative value range for one archived active-cube chunk.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MsgArchiveChunkRange {
    pub chunk_origin: UVec3,
    pub min: f64,
    pub max: f64,
}

#[derive(Debug)]
enum MsgChunkValues {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

/// Reusable storage for decoding one archived scalar chunk without allocating
/// on every fill.
#[derive(Debug)]
pub struct MsgChunkBuffer {
    key: UVec3,
    extent: UVec3,
    mask: Box<[u64; MASK_WORDS]>,
    values: MsgChunkValues,
}

pub(crate) struct MsgArchiveScalarAccessor<'a> {
    level: MsgArchiveFieldLevel<'a>,
    cached_key: Option<[u32; 3]>,
    cached_chunk: Option<&'a ArchivedScalarChunkRecord>,
}

impl MappedMsgArchive {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: The mapping is read-only and is owned for the lifetime of Self.
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("memory-mapping {}", path.display()))?;
        view_msg(&mmap).context("validating mapped MSG archive")?;
        Ok(Self { mmap })
    }

    pub fn view(&self) -> Result<MsgArchive<'_>> {
        view_msg(&self.mmap)
    }

    pub fn bytes(&self) -> &[u8] {
        &self.mmap
    }
}

impl<'a> MsgArchive<'a> {
    pub fn simulation_time(&self) -> f64 {
        self.archive.simulation_time.to_native()
    }

    pub fn level_count(&self) -> usize {
        self.archive.levels.len()
    }

    pub fn field_names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.archive.fields.iter().map(|field| field.name.as_str())
    }

    pub fn levels(&self) -> impl ExactSizeIterator<Item = MsgArchiveLevelInfo> + '_ {
        self.archive
            .levels
            .iter()
            .enumerate()
            .map(|(index, level)| MsgArchiveLevelInfo {
                index_origin: level.index_origin.map(|value| value.to_native()),
                point_extent: level.point_extent.map(|value| value.to_native()),
                scale: level.scale.map(|value| value.to_native()),
                refinement_ratio_from_coarse: (index != 0)
                    .then(|| level.refinement_ratio.map(|value| value.to_native())),
            })
    }

    pub fn fields(&self) -> impl ExactSizeIterator<Item = MsgArchiveFieldInfo<'_>> + '_ {
        self.archive.fields.iter().map(|field| MsgArchiveFieldInfo {
            name: field.name.as_str(),
            scalar_type: scalar_type_from_tag(field.scalar_type)
                .expect("validated MSG scalar type"),
            error_tolerance: match field.error_tolerance_kind {
                0 => None,
                1 => Some(MsgErrorTolerance {
                    absolute: field.error_absolute.to_native(),
                    relative_percent: field.error_relative_percent.to_native(),
                }),
                _ => unreachable!("validated MSG error tolerance kind"),
            },
        })
    }

    pub fn field(&self, name: &str) -> Option<MsgArchiveField<'a>> {
        self.archive
            .fields
            .iter()
            .find(|field| field.name == name)
            .map(|field| MsgArchiveField {
                bytes: self.bytes,
                data_end: self.data_end,
                archive: self.archive,
                field,
            })
    }

    pub fn grid_to_world_columns(&self) -> [f64; 16] {
        self.archive
            .grid_to_world
            .map(|component| component.to_native())
    }

    pub fn geometry(&self) -> MsgGeometry {
        MsgGeometry::new(
            DMat4::from_cols_array(&self.grid_to_world_columns()),
            glam::DVec3::from_array(
                self.archive
                    .sample_offset
                    .map(|component| component.to_native()),
            ),
        )
        .expect("validated MSG geometry")
    }

    pub fn decode(&self) -> Result<MsgDataset> {
        decode_msg(self, None)
    }

    pub fn decode_fields(&self, field_names: &[&str]) -> Result<MsgDataset> {
        decode_msg(self, Some(field_names))
    }
}

impl<'a> MsgArchiveField<'a> {
    pub fn name(&self) -> &str {
        self.field.name.as_str()
    }

    pub fn scalar_type(&self) -> MsgScalarType {
        scalar_type_from_tag(self.field.scalar_type).expect("validated MSG scalar type")
    }

    pub fn level_count(&self) -> usize {
        self.field.levels.len()
    }

    pub fn level(&self, index: usize) -> Result<MsgArchiveFieldLevel<'a>> {
        let archived_level = self
            .field
            .levels
            .get(index)
            .with_context(|| format!("MSG field level {index} is out of bounds"))?;
        let level = &self.archive.levels[index];
        let bounds = UVec3::from_array(level.point_extent.map(|value| value.to_native()));
        let scalar = rkyv::access::<ArchivedScalarGridRecord, Error>(block_slice(
            self.bytes,
            self.data_end,
            &archived_level.data,
        )?)
        .context("validating MSG scalar block")?;
        let scalar_masks = block_slice(self.bytes, self.data_end, &archived_level.data_masks)?;
        let scalar_payloads =
            block_slice(self.bytes, self.data_end, &archived_level.data_payloads)?;
        let scalar_type = self.scalar_type();
        let scalar_view = chunk_codec::ArchivedScalarGridView {
            directory: scalar,
            masks: scalar_masks,
            payloads: scalar_payloads,
        };
        chunk_codec::validate_archived_grid(scalar_view, bounds, scalar_type_tag(scalar_type))?;
        let active_cubes = rkyv::access::<ArchivedMaskGridRecord, Error>(block_slice(
            self.bytes,
            self.data_end,
            &archived_level.active_cubes,
        )?)
        .context("validating MSG active-cube block")?;
        let active_cube_masks =
            block_slice(self.bytes, self.data_end, &archived_level.active_cube_masks)?;
        let cube_bounds = bounds.saturating_sub(UVec3::ONE);
        active_cubes.validate(active_cube_masks, cube_bounds)?;
        validate_archived_ranges(active_cubes, &archived_level.value_ranges, cube_bounds)?;
        Ok(MsgArchiveFieldLevel {
            scalar_type,
            bounds,
            scalar,
            scalar_masks,
            scalar_payloads,
            active_cubes,
            active_cube_masks,
            field_level: archived_level,
        })
    }

    pub(crate) fn level_if_value_may_be_present(
        &self,
        index: usize,
        value: f64,
    ) -> Result<Option<MsgArchiveFieldLevel<'a>>> {
        let archived_level = self
            .field
            .levels
            .get(index)
            .with_context(|| format!("MSG field level {index} is out of bounds"))?;
        let level = &self.archive.levels[index];
        let bounds = UVec3::from_array(level.point_extent.map(|value| value.to_native()));
        let cube_bounds = bounds.saturating_sub(UVec3::ONE);
        let active_cubes = rkyv::access::<ArchivedMaskGridRecord, Error>(block_slice(
            self.bytes,
            self.data_end,
            &archived_level.active_cubes,
        )?)
        .context("validating MSG active-cube block")?;
        let active_cube_masks =
            block_slice(self.bytes, self.data_end, &archived_level.active_cube_masks)?;
        active_cubes.validate(active_cube_masks, cube_bounds)?;
        validate_archived_ranges(active_cubes, &archived_level.value_ranges, cube_bounds)?;
        let may_be_present = archived_level.value_ranges.iter().any(|range| {
            let error = range.max_abs_error.to_native();
            let min = (range.value_min.to_native() - error).max(-f64::MAX);
            let max = (range.value_max.to_native() + error).min(f64::MAX);
            min <= value && value <= max
        });
        if !may_be_present {
            return Ok(None);
        }
        self.level(index).map(Some)
    }
}

impl<'a> MsgArchiveFieldLevel<'a> {
    fn scalar_view(&self) -> chunk_codec::ArchivedScalarGridView<'a> {
        chunk_codec::ArchivedScalarGridView {
            directory: self.scalar,
            masks: self.scalar_masks,
            payloads: self.scalar_payloads,
        }
    }

    pub fn scalar_type(&self) -> MsgScalarType {
        self.scalar_type
    }

    pub fn bounds(&self) -> UVec3 {
        self.bounds
    }

    pub fn chunk_keys(&self) -> impl ExactSizeIterator<Item = UVec3> + '_ {
        self.scalar
            .chunks
            .iter()
            .map(|chunk| UVec3::from_array(chunk.key.map(|value| value.to_native())))
    }

    pub fn value_ranges(&self) -> impl ExactSizeIterator<Item = MsgArchiveChunkRange> + '_ {
        self.field_level.value_ranges.iter().map(|range| {
            let error = range.max_abs_error.to_native();
            MsgArchiveChunkRange {
                chunk_origin: UVec3::from_array(range.chunk_origin.map(|value| value.to_native())),
                min: (range.value_min.to_native() - error).max(-f64::MAX),
                max: (range.value_max.to_native() + error).min(f64::MAX),
            }
        })
    }

    pub fn is_active_cube(&self, anchor: UVec3) -> bool {
        anchor.cmplt(self.bounds.saturating_sub(UVec3::ONE)).all()
            && self.active_cubes.contains(self.active_cube_masks, anchor)
    }

    pub(crate) fn for_each_active_cube_in_aabb(
        &self,
        aabb: crate::sparse::Aabb3u,
        visit: impl FnMut(UVec3),
    ) {
        self.active_cubes
            .for_each_present_in_aabb(self.active_cube_masks, aabb, visit);
    }

    pub fn get(&self, point: UVec3) -> Result<Option<f64>> {
        if !point.cmplt(self.bounds).all() {
            return Ok(None);
        }
        let key = (point / UVec3::splat(CHUNK_SIZE)).to_array();
        let Some(chunk) = chunk_codec::find_archived_chunk(self.scalar, key) else {
            return Ok(None);
        };
        let index = chunk_local_index(point);
        if !chunk_codec::archived_chunk_present_in(self.scalar_view(), chunk, index)? {
            return Ok(None);
        }
        chunk_codec::decode_archived_value(self.scalar_view(), chunk, index).map(Some)
    }

    pub(crate) fn accessor(&self) -> MsgArchiveScalarAccessor<'a> {
        MsgArchiveScalarAccessor {
            level: *self,
            cached_key: None,
            cached_chunk: None,
        }
    }

    /// Decodes one chunk into reusable storage. `key` is in chunk coordinates.
    /// Returns false when the archive contains no chunk at that key.
    pub fn fill_chunk(&self, key: UVec3, buffer: &mut MsgChunkBuffer) -> Result<bool> {
        ensure!(
            buffer.scalar_type() == self.scalar_type,
            "chunk buffer scalar type does not match the archived field"
        );
        let Some(chunk) = chunk_codec::find_archived_chunk(self.scalar, key.to_array()) else {
            buffer.clear(key);
            return Ok(false);
        };
        let origin = key * UVec3::splat(CHUNK_SIZE);
        buffer.key = key;
        buffer.extent = (self.bounds - origin).min(UVec3::splat(CHUNK_SIZE));
        *buffer.mask = format::archived_mask_words(
            chunk.mask_kind,
            chunk.mask_offset.to_native(),
            self.scalar_masks,
        )?;
        match &mut buffer.values {
            MsgChunkValues::F32(values) => {
                chunk_codec::decode_archived_chunk_f32(self.scalar_view(), chunk, values)?;
            }
            MsgChunkValues::F64(values) => {
                chunk_codec::decode_archived_chunk_f64(self.scalar_view(), chunk, values)?;
            }
        }
        Ok(true)
    }
}

impl MsgArchiveScalarAccessor<'_> {
    pub(crate) fn get(&mut self, point: UVec3) -> Result<Option<f64>> {
        if !point.cmplt(self.level.bounds).all() {
            return Ok(None);
        }
        let key = (point / UVec3::splat(CHUNK_SIZE)).to_array();
        if self.cached_key != Some(key) {
            self.cached_key = Some(key);
            self.cached_chunk = chunk_codec::find_archived_chunk(self.level.scalar, key);
        }
        let Some(chunk) = self.cached_chunk else {
            return Ok(None);
        };
        let index = chunk_local_index(point);
        if !chunk_codec::archived_chunk_present_in(self.level.scalar_view(), chunk, index)? {
            return Ok(None);
        }
        chunk_codec::decode_archived_value(self.level.scalar_view(), chunk, index).map(Some)
    }
}

impl MsgChunkBuffer {
    pub fn new(scalar_type: MsgScalarType) -> Self {
        let values = match scalar_type {
            MsgScalarType::F32 => MsgChunkValues::F32(vec![0.0; CHUNK_VOLUME]),
            MsgScalarType::F64 => MsgChunkValues::F64(vec![0.0; CHUNK_VOLUME]),
        };
        Self {
            key: UVec3::ZERO,
            extent: UVec3::ZERO,
            mask: Box::new([0; MASK_WORDS]),
            values,
        }
    }

    pub fn scalar_type(&self) -> MsgScalarType {
        match self.values {
            MsgChunkValues::F32(_) => MsgScalarType::F32,
            MsgChunkValues::F64(_) => MsgScalarType::F64,
        }
    }

    pub fn key(&self) -> UVec3 {
        self.key
    }

    pub fn origin(&self) -> UVec3 {
        self.key * UVec3::splat(CHUNK_SIZE)
    }

    pub fn extent(&self) -> UVec3 {
        self.extent
    }

    pub fn contains_local(&self, position: UVec3) -> bool {
        if !position.cmplt(self.extent).all() {
            return false;
        }
        let index = chunk_local_index(position);
        self.mask[index / 64] & (1_u64 << (index % 64)) != 0
    }

    pub fn get_local(&self, position: UVec3) -> Option<f64> {
        self.contains_local(position).then(|| {
            let index = chunk_local_index(position);
            match &self.values {
                MsgChunkValues::F32(values) => f64::from(values[index]),
                MsgChunkValues::F64(values) => values[index],
            }
        })
    }

    fn clear(&mut self, key: UVec3) {
        self.key = key;
        self.extent = UVec3::ZERO;
        self.mask.fill(0);
    }
}

fn chunk_local_index(position: UVec3) -> usize {
    let local = position & UVec3::splat(CHUNK_SIZE - 1);
    local.x as usize
        + local.y as usize * CHUNK_SIZE as usize
        + local.z as usize * CHUNK_SIZE as usize * CHUNK_SIZE as usize
}

/// Write one owned MSG timestep. Blocks are emitted coarse-to-fine and aligned
/// to eight bytes; the rkyv metadata header follows all blocks and its offset is
/// stored in the final little-endian `u64`.
pub fn write_msg(
    dataset: &MsgDataset,
    options: &MsgWriteOptions,
    dest: &mut impl Write,
) -> Result<()> {
    dataset.validate().context("validating MSG dataset")?;
    let policies = resolve_policies(dataset, options)?;
    let acceleration = dataset
        .derived_acceleration()
        .context("deriving MSG acceleration")?;
    dest.write_all(&MSG_FORMAT_MAGIC.to_le_bytes())?;
    dest.write_all(&MSG_FORMAT_VERSION.to_le_bytes())?;
    let mut offset = PREFIX_LENGTH;
    let mut archived_field_levels = dataset
        .fields()
        .iter()
        .map(|_| Vec::with_capacity(dataset.levels().len()))
        .collect::<Vec<_>>();

    for level_index in 0..dataset.levels().len() {
        for (field_index, field) in dataset.fields().iter().enumerate() {
            let field_level = &field.levels()[level_index];
            let field_acceleration = &acceleration[field_index][level_index];
            let encoded = match field_level.values() {
                MsgScalarGrid::F32(grid) => chunk_codec::encode_f32(grid, policies[field_index]),
                MsgScalarGrid::F64(grid) => chunk_codec::encode_f64(grid, policies[field_index]),
            }
            .with_context(|| format!("encoding field {:?} level {level_index}", field.name()))?;
            let ranges = field_acceleration
                .value_ranges
                .iter()
                .map(|range| ChunkRangeRecord {
                    chunk_origin: range.chunk_origin.to_array(),
                    value_min: range.min,
                    value_max: range.max,
                    max_abs_error: range_max_abs_error(
                        &encoded.directory,
                        range.chunk_origin,
                        field_level.values().bounds(),
                    ),
                })
                .collect::<Vec<_>>();
            let encoded_bytes = rkyv::to_bytes::<Error>(&encoded.directory)
                .context("serializing MSG scalar directory")?;
            let data = write_block(dest, &mut offset, &encoded_bytes)?;
            let data_masks = write_block(dest, &mut offset, &encoded.masks)?;
            let data_payloads = write_block(dest, &mut offset, &encoded.payloads)?;
            let mask = MaskGridRecord::from_grid(&field_acceleration.active_cubes);
            let mask_bytes = rkyv::to_bytes::<Error>(&mask.directory)
                .context("serializing active-cube directory")?;
            let active_cubes = write_block(dest, &mut offset, &mask_bytes)?;
            let active_cube_masks = write_block(dest, &mut offset, &mask.masks)?;
            archived_field_levels[field_index].push(FieldLevelRecord {
                data,
                data_masks,
                data_payloads,
                value_ranges: ranges,
                active_cubes,
                active_cube_masks,
            });
        }
    }

    let levels = dataset
        .levels()
        .iter()
        .enumerate()
        .map(|(index, level)| LevelRecord {
            index_origin: level.index_origin.to_array(),
            point_extent: level.point_extent.to_array(),
            scale: level.scale.to_array(),
            refinement_ratio: if index == 0 {
                [0; 3]
            } else {
                level
                    .refinement_ratio_from_coarse
                    .expect("validated finer level ratio")
                    .to_array()
            },
        })
        .collect();
    let fields = dataset
        .fields()
        .iter()
        .zip(policies)
        .zip(archived_field_levels)
        .map(|((field, policy), levels)| {
            let (kind, absolute, relative_percent) = match policy.error_tolerance {
                None => (0, 0.0, 0.0),
                Some(error) => (1, error.absolute, error.relative_percent),
            };
            FieldRecord {
                name: field.name().to_owned(),
                scalar_type: scalar_type_tag(field.scalar_type()),
                error_tolerance_kind: kind,
                error_absolute: absolute,
                error_relative_percent: relative_percent,
                levels,
            }
        })
        .collect();
    let archive = MsgArchiveHeader {
        chunk_bits: CHUNK_BITS,
        simulation_time: dataset.time(),
        grid_to_world: dataset.geometry().grid_to_world().to_cols_array(),
        sample_offset: dataset.geometry().sample_offset().to_array(),
        levels,
        fields,
    };
    let header = rkyv::to_bytes::<Error>(&archive).context("serializing MSG header")?;
    align_writer(dest, &mut offset)?;
    let header_offset = offset;
    dest.write_all(&header)?;
    dest.write_all(&header_offset.to_le_bytes())?;
    Ok(())
}

fn range_max_abs_error(encoded: &ScalarGridRecord, cube_origin: UVec3, point_bounds: UVec3) -> f64 {
    let first = cube_origin / UVec3::splat(CHUNK_SIZE);
    let point_max = cube_origin
        .saturating_add(UVec3::splat(CHUNK_SIZE + 1))
        .min(point_bounds);
    let last = (point_max - UVec3::ONE) / UVec3::splat(CHUNK_SIZE);
    let mut maximum = 0.0_f64;
    for z in first.z..=last.z {
        for y in first.y..=last.y {
            for x in first.x..=last.x {
                let key = [x, y, z];
                if let Ok(index) = encoded.chunks.binary_search_by_key(&key, |chunk| chunk.key) {
                    maximum = maximum.max(encoded.chunks[index].max_abs_error);
                }
            }
        }
    }
    maximum
}

pub fn view_msg(bytes: &[u8]) -> Result<MsgArchive<'_>> {
    let (data_end, header) = archive_header(bytes)?;
    let archive = rkyv::access::<ArchivedMsgArchiveHeader, Error>(header)
        .context("validating MSG metadata")?;
    validate_archive(archive, bytes, data_end)?;
    Ok(MsgArchive {
        bytes,
        data_end,
        archive,
    })
}

pub fn read_msg(bytes: &[u8]) -> Result<MsgDataset> {
    view_msg(bytes)?.decode()
}

pub fn read_msg_fields(bytes: &[u8], field_names: &[&str]) -> Result<MsgDataset> {
    view_msg(bytes)?.decode_fields(field_names)
}

/// Decode and rewrite an MSG archive with new field codec policies.
///
/// Transcoding a lossy source operates on its reconstructed physical values and
/// cannot recover precision discarded by the source archive.
pub fn transcode_msg(bytes: &[u8], options: &MsgWriteOptions, dest: &mut impl Write) -> Result<()> {
    let dataset = read_msg(bytes).context("decoding source MSG archive")?;
    write_msg(&dataset, options, dest)
}

fn decode_msg(view: &MsgArchive<'_>, requested: Option<&[&str]>) -> Result<MsgDataset> {
    let bytes = view.bytes;
    let data_end = view.data_end;
    let archive = view.archive;
    let geometry = MsgGeometry::new(
        DMat4::from_cols_array(&archive.grid_to_world.map(|value| value.to_native())),
        glam::DVec3::from_array(archive.sample_offset.map(|value| value.to_native())),
    )?;
    let levels = archive
        .levels
        .iter()
        .enumerate()
        .map(|(index, level)| MsgLevel {
            index_origin: glam::IVec3::from_array(
                level.index_origin.map(|value| value.to_native()),
            ),
            point_extent: UVec3::from_array(level.point_extent.map(|value| value.to_native())),
            scale: glam::DVec3::from_array(level.scale.map(|value| value.to_native())),
            refinement_ratio_from_coarse: (index != 0)
                .then(|| UVec3::from_array(level.refinement_ratio.map(|value| value.to_native()))),
        })
        .collect::<Vec<_>>();
    let mut dataset =
        MsgDataset::from_validated_parts(archive.simulation_time.to_native(), geometry, levels);
    let slots = selected_field_slots(archive, requested)?;
    for slot in slots {
        let field = &archive.fields[slot];
        let scalar_type = scalar_type_from_tag(field.scalar_type)?;
        let mut field_levels = Vec::with_capacity(field.levels.len());
        for (level_index, archived_level) in field.levels.iter().enumerate() {
            let expected_bounds = dataset.levels()[level_index].point_extent;
            let block = block_slice(bytes, data_end, &archived_level.data)?;
            let scalar = rkyv::access::<ArchivedScalarGridRecord, Error>(block)
                .context("validating MSG scalar block")?;
            let scalar_view = chunk_codec::ArchivedScalarGridView {
                directory: scalar,
                masks: block_slice(bytes, data_end, &archived_level.data_masks)?,
                payloads: block_slice(bytes, data_end, &archived_level.data_payloads)?,
            };
            let values = match scalar_type {
                MsgScalarType::F32 => MsgScalarGrid::F32(
                    chunk_codec::decode_f32(scalar_view, expected_bounds)
                        .context("decoding f32 MSG field")?,
                ),
                MsgScalarType::F64 => MsgScalarGrid::F64(
                    chunk_codec::decode_f64(scalar_view, expected_bounds)
                        .context("decoding f64 MSG field")?,
                ),
            };
            let mut field_level = MsgFieldLevel::new(values);
            let archived_mask = rkyv::access::<ArchivedMaskGridRecord, Error>(block_slice(
                bytes,
                data_end,
                &archived_level.active_cubes,
            )?)
            .context("validating MSG active-cube block")?;
            let cube_bounds = expected_bounds.saturating_sub(UVec3::ONE);
            let archived_mask_bytes =
                block_slice(bytes, data_end, &archived_level.active_cube_masks)?;
            let mask = archived_mask.to_grid(archived_mask_bytes, cube_bounds)?;
            let mut previous_origin = None;
            let ranges = archived_level
                .value_ranges
                .iter()
                .map(|range| {
                    let chunk_origin =
                        UVec3::from_array(range.chunk_origin.map(|value| value.to_native()));
                    ensure!(
                        (chunk_origin & UVec3::splat(CHUNK_SIZE - 1))
                            .cmpeq(UVec3::ZERO)
                            .all()
                            && chunk_origin.cmplt(cube_bounds).all(),
                        "acceleration range origin is not an in-bounds chunk origin"
                    );
                    let order = (chunk_origin.z, chunk_origin.y, chunk_origin.x);
                    ensure!(
                        previous_origin.is_none_or(|previous| previous < order),
                        "acceleration range origins are not unique and sorted"
                    );
                    previous_origin = Some(order);
                    let chunk_key = (chunk_origin / UVec3::splat(CHUNK_SIZE)).to_array();
                    ensure!(
                        archived_mask.contains_chunk_key(chunk_key),
                        "acceleration range has no corresponding active-cube chunk"
                    );
                    Ok(ChunkValueRange {
                        chunk_origin,
                        min: (range.value_min.to_native() - range.max_abs_error.to_native())
                            .max(-f64::MAX),
                        max: (range.value_max.to_native() + range.max_abs_error.to_native())
                            .min(f64::MAX),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            ensure!(
                ranges.len() == mask.chunk_count(),
                "acceleration range count does not match active-cube chunks"
            );
            field_level.set_acceleration(MsgFieldAcceleration {
                value_ranges: ranges,
                active_cubes: mask,
            });
            field_levels.push(field_level);
        }
        dataset.push_validated_field(MsgField::new(
            field.name.as_str(),
            scalar_type,
            field_levels,
        ));
    }
    Ok(dataset)
}

fn selected_field_slots(
    archive: &ArchivedMsgArchiveHeader,
    requested: Option<&[&str]>,
) -> Result<Vec<usize>> {
    let Some(requested) = requested else {
        return Ok((0..archive.fields.len()).collect());
    };
    let mut slots = Vec::with_capacity(requested.len());
    for (index, &name) in requested.iter().enumerate() {
        ensure!(
            !requested[..index].contains(&name),
            "MSG field {name:?} was selected more than once"
        );
        slots.push(
            archive
                .fields
                .iter()
                .position(|field| field.name == name)
                .with_context(|| format!("MSG field {name:?} is not stored"))?,
        );
    }
    Ok(slots)
}

fn validate_archived_ranges(
    active_cubes: &ArchivedMaskGridRecord,
    ranges: &[ArchivedChunkRangeRecord],
    cube_bounds: UVec3,
) -> Result<()> {
    ensure!(
        ranges.len() == active_cubes.chunk_count(),
        "acceleration range count does not match active-cube chunks"
    );
    let mut previous_origin = None;
    for range in ranges {
        let chunk_origin = UVec3::from_array(range.chunk_origin.map(|value| value.to_native()));
        ensure!(
            (chunk_origin & UVec3::splat(CHUNK_SIZE - 1))
                .cmpeq(UVec3::ZERO)
                .all()
                && chunk_origin.cmplt(cube_bounds).all(),
            "acceleration range origin is not an in-bounds chunk origin"
        );
        let order = (chunk_origin.z, chunk_origin.y, chunk_origin.x);
        ensure!(
            previous_origin.is_none_or(|previous| previous < order),
            "acceleration range origins are not unique and sorted"
        );
        previous_origin = Some(order);
        let chunk_key = (chunk_origin / UVec3::splat(CHUNK_SIZE)).to_array();
        ensure!(
            active_cubes.contains_chunk_key(chunk_key),
            "acceleration range has no corresponding active-cube chunk"
        );
    }
    Ok(())
}

fn resolve_policies(
    dataset: &MsgDataset,
    options: &MsgWriteOptions,
) -> Result<Vec<ResolvedFieldCodec>> {
    let mut policies = dataset
        .fields()
        .iter()
        .map(|_| ResolvedFieldCodec {
            selection: MsgChunkCodecSelection::AutoLossless,
            error_tolerance: None,
            physical_offset: 0.0,
            physical_scale: 1.0,
        })
        .collect::<Vec<_>>();
    let mut assigned = vec![false; policies.len()];
    for requested in &options.codecs {
        let slot = dataset
            .fields()
            .iter()
            .position(|field| field.name() == requested.field)
            .with_context(|| format!("codec field {:?} is not present", requested.field))?;
        ensure!(
            !assigned[slot],
            "field {:?} has multiple codec policies",
            requested.field
        );
        validate_codec_policy(dataset.fields()[slot].scalar_type(), requested)?;
        assigned[slot] = true;
        policies[slot] = ResolvedFieldCodec {
            selection: requested.selection,
            error_tolerance: requested.error_tolerance,
            physical_offset: 0.0,
            physical_scale: 1.0,
        };
    }
    Ok(policies)
}

fn validate_codec_policy(scalar_type: MsgScalarType, policy: &MsgFieldCodec) -> Result<()> {
    if let Some(error) = policy.error_tolerance {
        ensure!(
            error.absolute.is_finite() && error.absolute >= 0.0,
            "invalid absolute error"
        );
        ensure!(
            error.relative_percent.is_finite() && (0.0..=100.0).contains(&error.relative_percent),
            "invalid relative error percentage"
        );
    }
    let requires_error = match policy.selection {
        MsgChunkCodecSelection::AutoLossless => false,
        MsgChunkCodecSelection::AutoSmallest => true,
        MsgChunkCodecSelection::Forced(MsgChunkCodec::DenseF32)
            if scalar_type == MsgScalarType::F32 =>
        {
            false
        }
        MsgChunkCodecSelection::Forced(MsgChunkCodec::DenseF64)
            if scalar_type == MsgScalarType::F64 =>
        {
            false
        }
        MsgChunkCodecSelection::Forced(MsgChunkCodec::DenseF64) => {
            bail!("dense:f64 is incompatible with an f32 logical field")
        }
        MsgChunkCodecSelection::Forced(_) => true,
    };
    ensure!(
        policy.error_tolerance.is_some() == requires_error,
        if requires_error {
            "lossy codec selection requires an error tolerance"
        } else {
            "lossless codec selection does not accept an error tolerance"
        }
    );
    Ok(())
}

fn scalar_type_tag(scalar_type: MsgScalarType) -> u8 {
    match scalar_type {
        MsgScalarType::F32 => chunk_codec::TYPE_F32,
        MsgScalarType::F64 => chunk_codec::TYPE_F64,
    }
}

fn scalar_type_from_tag(tag: u8) -> Result<MsgScalarType> {
    match tag {
        chunk_codec::TYPE_F32 => Ok(MsgScalarType::F32),
        chunk_codec::TYPE_F64 => Ok(MsgScalarType::F64),
        _ => bail!("unsupported MSG scalar type {tag}"),
    }
}

fn write_block(dest: &mut impl Write, offset: &mut u64, bytes: &[u8]) -> Result<BlockRecord> {
    align_writer(dest, offset)?;
    let block_offset = *offset;
    dest.write_all(bytes)?;
    *offset = offset
        .checked_add(u64::try_from(bytes.len())?)
        .context("MSG file length overflow")?;
    Ok(BlockRecord {
        offset: block_offset,
        stored_length: u64::try_from(bytes.len())?,
        decoded_length: u64::try_from(bytes.len())?,
        compression: CompressionRecord::None,
    })
}

fn align_writer(dest: &mut impl Write, offset: &mut u64) -> Result<()> {
    let padding = (8 - (*offset % 8)) % 8;
    if padding != 0 {
        dest.write_all(&[0; 7][..padding as usize])?;
        *offset += padding;
    }
    Ok(())
}

fn archive_header(bytes: &[u8]) -> Result<(usize, &[u8])> {
    ensure!(
        bytes.len() >= (PREFIX_LENGTH + FOOTER_LENGTH) as usize,
        "MSG file is truncated"
    );
    ensure!(
        u32::from_le_bytes(bytes[0..4].try_into().unwrap()) == MSG_FORMAT_MAGIC,
        "invalid MSG archive magic"
    );
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    ensure!(
        version == MSG_FORMAT_VERSION,
        "unsupported MSG archive version {version}"
    );
    let footer_start = bytes.len() - FOOTER_LENGTH as usize;
    let offset = u64::from_le_bytes(bytes[footer_start..].try_into().unwrap());
    ensure!(offset.is_multiple_of(8), "MSG header is not 8-byte aligned");
    let offset = usize::try_from(offset).context("MSG header offset exceeds usize")?;
    ensure!(
        offset >= PREFIX_LENGTH as usize && offset < footer_start,
        "MSG header offset is out of bounds"
    );
    Ok((offset, &bytes[offset..footer_start]))
}

fn block_slice<'a>(
    bytes: &'a [u8],
    data_end: usize,
    block: &ArchivedBlockRecord,
) -> Result<&'a [u8]> {
    ensure!(
        matches!(block.compression, ArchivedCompressionRecord::None),
        "unsupported compression"
    );
    let stored_length = block.stored_length.to_native();
    ensure!(
        stored_length == block.decoded_length.to_native(),
        "invalid uncompressed block length"
    );
    let offset = block.offset.to_native();
    ensure!(offset.is_multiple_of(8), "MSG block is not 8-byte aligned");
    let start = usize::try_from(offset)?;
    let end = start
        .checked_add(usize::try_from(stored_length)?)
        .context("MSG block extent overflow")?;
    ensure!(
        start >= PREFIX_LENGTH as usize && end <= data_end,
        "MSG block is out of bounds"
    );
    Ok(&bytes[start..end])
}

fn validate_archive(
    archive: &ArchivedMsgArchiveHeader,
    bytes: &[u8],
    data_end: usize,
) -> Result<()> {
    ensure!(
        archive.chunk_bits.to_native() == CHUNK_BITS,
        "unsupported MSG chunk size"
    );
    let simulation_time = archive.simulation_time.to_native();
    ensure!(simulation_time.is_finite(), "non-finite simulation time");
    MsgGeometry::new(
        DMat4::from_cols_array(&archive.grid_to_world.map(|value| value.to_native())),
        glam::DVec3::from_array(archive.sample_offset.map(|value| value.to_native())),
    )?;
    ensure!(!archive.levels.is_empty(), "MSG archive has no levels");
    let mut previous_scale: Option<glam::DVec3> = None;
    for (index, level) in archive.levels.iter().enumerate() {
        let extent = UVec3::from_array(level.point_extent.map(|value| value.to_native()));
        ensure!(
            !extent.cmpeq(UVec3::ZERO).any(),
            "level {index} has an empty point extent"
        );
        let scale = glam::DVec3::from_array(level.scale.map(|value| value.to_native()));
        ensure!(
            scale.is_finite() && !scale.cmple(glam::DVec3::ZERO).any(),
            "level {index} scale must be finite and strictly positive"
        );
        let ratio = UVec3::from_array(level.refinement_ratio.map(|value| value.to_native()));
        if let Some(coarse_scale) = previous_scale {
            ensure!(
                !ratio.cmpeq(UVec3::ZERO).any(),
                "level {index} refinement ratio must be nonzero"
            );
            let expected = coarse_scale / ratio.as_dvec3();
            let relative = (scale - expected).abs() / expected;
            ensure!(
                !relative
                    .cmpgt(glam::DVec3::splat(64.0 * f64::EPSILON))
                    .any(),
                "level {index} scale {scale:?} disagrees with refinement ratio {ratio:?}; expected {expected:?}"
            );
        } else {
            ensure!(
                ratio == UVec3::ZERO,
                "level zero refinement ratio must be zero"
            );
        }
        previous_scale = Some(scale);
    }
    for (index, field) in archive.fields.iter().enumerate() {
        ensure!(!field.name.is_empty(), "MSG field name is empty");
        ensure!(
            !archive.fields[..index]
                .iter()
                .any(|other| other.name == field.name),
            "duplicate MSG field name {:?}",
            field.name
        );
        scalar_type_from_tag(field.scalar_type)?;
        match field.error_tolerance_kind {
            0 => ensure!(
                field.error_absolute.to_native() == 0.0
                    && field.error_relative_percent.to_native() == 0.0,
                "lossless field has nonzero error metadata"
            ),
            1 => {
                let absolute = field.error_absolute.to_native();
                let relative_percent = field.error_relative_percent.to_native();
                ensure!(
                    absolute.is_finite() && absolute >= 0.0,
                    "invalid archived absolute error"
                );
                ensure!(
                    relative_percent.is_finite() && (0.0..=100.0).contains(&relative_percent),
                    "invalid archived relative error"
                );
            }
            kind => bail!("unsupported MSG error tolerance kind {kind}"),
        }
        ensure!(
            field.levels.len() == archive.levels.len(),
            "MSG field level count mismatch"
        );
        for level in field.levels.iter() {
            block_slice(bytes, data_end, &level.data)?;
            block_slice(bytes, data_end, &level.data_masks)?;
            block_slice(bytes, data_end, &level.data_payloads)?;
            block_slice(bytes, data_end, &level.active_cubes)?;
            block_slice(bytes, data_end, &level.active_cube_masks)?;
            for range in level.value_ranges.iter() {
                let value_min = range.value_min.to_native();
                let value_max = range.value_max.to_native();
                let max_abs_error = range.max_abs_error.to_native();
                ensure!(
                    value_min.is_finite()
                        && value_max.is_finite()
                        && value_min <= value_max
                        && max_abs_error.is_finite()
                        && max_abs_error >= 0.0,
                    "invalid MSG chunk value range"
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use glam::{DVec3, IVec3};

    use super::*;

    fn sample_dataset() -> MsgDataset {
        let mut dataset = MsgDataset::new(
            2.5,
            MsgGeometry::point_centered(DMat4::from_translation(DVec3::new(1.0, 2.0, 3.0)))
                .unwrap(),
            vec![MsgLevel {
                index_origin: IVec3::new(-2, 4, 0),
                point_extent: UVec3::splat(4),
                scale: DVec3::splat(0.25),
                refinement_ratio_from_coarse: None,
            }],
        )
        .unwrap();
        let mut values = MsgScalarGrid::new(UVec3::splat(4), MsgScalarType::F64);
        values.set(UVec3::new(1, 2, 3), 42.25);
        dataset
            .add_field(MsgField::new(
                "density",
                MsgScalarType::F64,
                vec![MsgFieldLevel::new(values)],
            ))
            .unwrap();
        dataset
    }

    #[test]
    fn archive_round_trip_and_selective_decode() -> Result<()> {
        let dataset = sample_dataset();
        let mut bytes = Vec::new();
        write_msg(&dataset, &MsgWriteOptions::default(), &mut bytes)?;
        assert_eq!(&bytes[..4], b"MSG\0");
        let view = view_msg(&bytes)?;
        assert_eq!(view.simulation_time(), 2.5);
        assert_eq!(view.field_names().collect::<Vec<_>>(), vec!["density"]);
        let decoded = view.decode_fields(&["density"])?;
        assert!(decoded.fields()[0].levels()[0].acceleration().is_some());
        assert_eq!(
            decoded.fields()[0].levels()[0]
                .values()
                .get(UVec3::new(1, 2, 3)),
            Some(42.25)
        );
        Ok(())
    }

    #[test]
    fn range_error_includes_neighboring_point_chunks() {
        let chunk = |key, max_abs_error| ScalarChunkRecord {
            key,
            mask_kind: MASK_DENSE,
            mask_offset: 0,
            codec_id: 0,
            codec_version: 0,
            codec_flags: 0,
            max_abs_error,
            metadata: Vec::new(),
            payload_offset: 0,
            payload_length: 0,
        };
        let encoded = ScalarGridRecord {
            bounds: [CHUNK_SIZE * 2, CHUNK_SIZE, CHUNK_SIZE],
            decoded_type: 0,
            chunks: vec![chunk([0, 0, 0], 0.01), chunk([1, 0, 0], 0.25)],
        };

        assert_eq!(
            range_max_abs_error(&encoded, UVec3::ZERO, UVec3::from_array(encoded.bounds)),
            0.25
        );
        assert_eq!(
            range_max_abs_error(
                &encoded,
                UVec3::new(CHUNK_SIZE, 0, 0),
                UVec3::from_array(encoded.bounds),
            ),
            0.25
        );
    }

    #[test]
    fn lazy_point_and_reusable_chunk_match_eager_decode_for_every_codec() -> Result<()> {
        let cases = [
            (
                MsgScalarType::F64,
                MsgChunkCodec::UniformF64,
                Some(0.0),
                true,
            ),
            (MsgScalarType::F32, MsgChunkCodec::DenseF32, None, false),
            (MsgScalarType::F64, MsgChunkCodec::DenseF64, None, false),
            (
                MsgScalarType::F64,
                MsgChunkCodec::QuantizedU8,
                Some(0.2),
                false,
            ),
            (
                MsgScalarType::F64,
                MsgChunkCodec::QuantizedU16,
                Some(0.001),
                false,
            ),
            (
                MsgScalarType::F64,
                MsgChunkCodec::QuantizedU32,
                Some(1.0e-7),
                false,
            ),
            (
                MsgScalarType::F64,
                MsgChunkCodec::AffineF64,
                Some(0.2),
                false,
            ),
            (
                MsgScalarType::F64,
                MsgChunkCodec::AffineResidualU8,
                Some(0.001),
                false,
            ),
            (
                MsgScalarType::F64,
                MsgChunkCodec::AffineResidualU16,
                Some(1.0e-5),
                false,
            ),
            (
                MsgScalarType::F64,
                MsgChunkCodec::AffineResidualU32,
                Some(1.0e-9),
                false,
            ),
        ];
        let bounds = UVec3::splat(4);
        for (scalar_type, codec, tolerance, uniform) in cases {
            let mut dataset = MsgDataset::new(
                0.0,
                MsgGeometry::point_centered(DMat4::IDENTITY)?,
                vec![MsgLevel {
                    index_origin: IVec3::ZERO,
                    point_extent: bounds,
                    scale: DVec3::ONE,
                    refinement_ratio_from_coarse: None,
                }],
            )?;
            let mut values = MsgScalarGrid::new(bounds, scalar_type);
            for z in 0..bounds.z {
                for y in 0..bounds.y {
                    for x in 0..bounds.x {
                        let value = if uniform {
                            3.25
                        } else {
                            10.0 + 1.25 * f64::from(x) - 0.5 * f64::from(y)
                                + 0.125 * f64::from(z)
                                + f64::from((x + 7 * y + 13 * z) % 5) / 100.0
                        };
                        values.set(UVec3::new(x, y, z), value);
                    }
                }
            }
            dataset.add_field(MsgField::new(
                "density",
                scalar_type,
                vec![MsgFieldLevel::new(values)],
            ))?;
            let mut bytes = Vec::new();
            write_msg(
                &dataset,
                &MsgWriteOptions {
                    codecs: vec![MsgFieldCodec {
                        field: "density".into(),
                        selection: MsgChunkCodecSelection::Forced(codec),
                        error_tolerance: tolerance.map(|absolute| MsgErrorTolerance {
                            absolute,
                            relative_percent: 0.0,
                        }),
                    }],
                },
                &mut bytes,
            )?;

            let archive = view_msg(&bytes)?;
            let level = archive.field("density").unwrap().level(0)?;
            let eager = archive.decode_fields(&["density"])?;
            let eager_values = eager.field("density").unwrap().levels()[0].values();
            let mut chunk = MsgChunkBuffer::new(scalar_type);
            assert!(level.fill_chunk(UVec3::ZERO, &mut chunk)?);
            for point in [UVec3::ZERO, UVec3::new(1, 2, 3), UVec3::splat(3)] {
                let expected = eager_values.get(point);
                assert_eq!(level.get(point)?, expected, "codec {codec:?}");
                assert_eq!(chunk.get_local(point), expected, "codec {codec:?}");
            }
            assert!(!level.is_active_cube(UVec3::splat(3)));
            assert!(level.is_active_cube(UVec3::splat(2)));
            assert_eq!(level.value_ranges().len(), 1);
        }
        Ok(())
    }
}
