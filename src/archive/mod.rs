//! Versioned `.msg` archive reading, writing, and memory-mapped inspection.

mod archive;
mod chunk_codec;

use std::{fs::File, io::Write, path::Path};

use anyhow::{Context, Result, bail, ensure};
use glam::{DMat4, UVec3};
use memmap2::Mmap;
use rkyv::rancor::Error;

use crate::{
    ChunkValueRange, MsgDataset, MsgField, MsgFieldAcceleration, MsgFieldLevel, MsgGeometry,
    MsgLevel, MsgScalarGrid, MsgScalarType,
    sparse::{CHUNK_BITS, CHUNK_SIZE},
};
use archive::*;

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

impl MsgArchive<'_> {
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

    pub fn grid_to_world_columns(&self) -> [f64; 16] {
        self.archive
            .grid_to_world
            .map(|component| component.to_native())
    }

    pub fn decode(&self) -> Result<MsgDataset> {
        decode_msg(self.bytes, None)
    }

    pub fn decode_fields(&self, field_names: &[&str]) -> Result<MsgDataset> {
        decode_msg(self.bytes, Some(field_names))
    }
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
            let encoded = match field_level.values() {
                MsgScalarGrid::F32(grid) => chunk_codec::encode_f32(grid, policies[field_index]),
                MsgScalarGrid::F64(grid) => chunk_codec::encode_f64(grid, policies[field_index]),
            }
            .with_context(|| format!("encoding field {:?} level {level_index}", field.name()))?;
            let ranges = field_level
                .acceleration()
                .map(|acceleration| {
                    acceleration
                        .value_ranges
                        .iter()
                        .map(|range| {
                            let chunk_key = range.chunk_origin / UVec3::splat(CHUNK_SIZE);
                            let max_abs_error = encoded
                                .chunks
                                .iter()
                                .find(|chunk| chunk.key == chunk_key.to_array())
                                .map_or(0.0, |chunk| chunk.max_abs_error);
                            ArchiveChunkRange {
                                chunk_origin: range.chunk_origin.to_array(),
                                value_min: range.min,
                                value_max: range.max,
                                max_abs_error,
                            }
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let encoded_bytes =
                rkyv::to_bytes::<Error>(&encoded).context("serializing MSG scalar block")?;
            let data = write_block(dest, &mut offset, &encoded_bytes)?;
            let active_cubes = if let Some(acceleration) = field_level.acceleration() {
                let mask = ArchiveMaskGrid::from_grid(&acceleration.active_cubes);
                let mask_bytes =
                    rkyv::to_bytes::<Error>(&mask).context("serializing active-cube block")?;
                Some(write_block(dest, &mut offset, &mask_bytes)?)
            } else {
                None
            };
            archived_field_levels[field_index].push(ArchiveFieldLevel {
                data,
                value_ranges: ranges,
                active_cubes,
            });
        }
    }

    let levels = dataset
        .levels()
        .iter()
        .enumerate()
        .map(|(index, level)| ArchiveLevel {
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
            ArchiveField {
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

pub fn view_msg(bytes: &[u8]) -> Result<MsgArchive<'_>> {
    let header = archive_header_bytes(bytes)?;
    let archive = rkyv::access::<ArchivedMsgArchiveHeader, Error>(header)
        .context("validating MSG metadata")?;
    let native = rkyv::deserialize::<MsgArchiveHeader, Error>(archive)
        .context("deserializing MSG metadata for validation")?;
    validate_archive(&native, bytes)?;
    Ok(MsgArchive { bytes, archive })
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

fn decode_msg(bytes: &[u8], requested: Option<&[&str]>) -> Result<MsgDataset> {
    let view = view_msg(bytes)?;
    let archive = rkyv::deserialize::<MsgArchiveHeader, Error>(view.archive)
        .context("deserializing MSG metadata")?;
    let geometry = MsgGeometry::new(
        DMat4::from_cols_array(&archive.grid_to_world),
        glam::DVec3::from_array(archive.sample_offset),
    )?;
    let levels = archive
        .levels
        .iter()
        .enumerate()
        .map(|(index, level)| MsgLevel {
            index_origin: glam::IVec3::from_array(level.index_origin),
            point_extent: UVec3::from_array(level.point_extent),
            scale: glam::DVec3::from_array(level.scale),
            refinement_ratio_from_coarse: (index != 0)
                .then(|| UVec3::from_array(level.refinement_ratio)),
        })
        .collect::<Vec<_>>();
    let mut dataset = MsgDataset::new(archive.simulation_time, geometry, levels)?;
    let slots = selected_field_slots(&archive, requested)?;
    for slot in slots {
        let field = &archive.fields[slot];
        let scalar_type = scalar_type_from_tag(field.scalar_type)?;
        let mut field_levels = Vec::with_capacity(field.levels.len());
        for archived_level in &field.levels {
            let block = block_slice(bytes, &archived_level.data)?;
            let scalar = rkyv::from_bytes::<ArchiveScalarGrid, Error>(block)
                .context("reading MSG scalar block")?;
            let values = match scalar_type {
                MsgScalarType::F32 => MsgScalarGrid::F32(
                    chunk_codec::decode_f32(scalar).context("decoding f32 MSG field")?,
                ),
                MsgScalarType::F64 => MsgScalarGrid::F64(
                    chunk_codec::decode_f64(scalar).context("decoding f64 MSG field")?,
                ),
            };
            let mut field_level = MsgFieldLevel::new(values);
            if let Some(active_block) = archived_level.active_cubes {
                let mask =
                    rkyv::from_bytes::<ArchiveMaskGrid, Error>(block_slice(bytes, &active_block)?)
                        .context("reading MSG active-cube block")?
                        .into_grid();
                let ranges = archived_level
                    .value_ranges
                    .iter()
                    .map(|range| ChunkValueRange {
                        chunk_origin: UVec3::from_array(range.chunk_origin),
                        min: (range.value_min - range.max_abs_error).max(-f64::MAX),
                        max: (range.value_max + range.max_abs_error).min(f64::MAX),
                    })
                    .collect();
                field_level.set_acceleration(MsgFieldAcceleration {
                    value_ranges: ranges,
                    active_cubes: mask,
                });
            }
            field_levels.push(field_level);
        }
        dataset.add_field(MsgField::new(field.name.clone(), scalar_type, field_levels))?;
    }
    Ok(dataset)
}

fn selected_field_slots(
    archive: &MsgArchiveHeader,
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

fn write_block(dest: &mut impl Write, offset: &mut u64, bytes: &[u8]) -> Result<ArchiveBlock> {
    align_writer(dest, offset)?;
    let block_offset = *offset;
    dest.write_all(bytes)?;
    *offset = offset
        .checked_add(u64::try_from(bytes.len())?)
        .context("MSG file length overflow")?;
    Ok(ArchiveBlock {
        offset: block_offset,
        stored_length: u64::try_from(bytes.len())?,
        decoded_length: u64::try_from(bytes.len())?,
        compression: ArchiveCompression::None,
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

fn archive_header_bytes(bytes: &[u8]) -> Result<&[u8]> {
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
    Ok(&bytes[offset..footer_start])
}

fn block_slice<'a>(bytes: &'a [u8], block: &ArchiveBlock) -> Result<&'a [u8]> {
    ensure!(
        matches!(block.compression, ArchiveCompression::None),
        "unsupported compression"
    );
    ensure!(
        block.stored_length == block.decoded_length,
        "invalid uncompressed block length"
    );
    ensure!(
        block.offset.is_multiple_of(8),
        "MSG block is not 8-byte aligned"
    );
    let start = usize::try_from(block.offset)?;
    let end = start
        .checked_add(usize::try_from(block.stored_length)?)
        .context("MSG block extent overflow")?;
    let header_start = bytes.len() - FOOTER_LENGTH as usize - archive_header_bytes(bytes)?.len();
    ensure!(
        start >= PREFIX_LENGTH as usize && end <= header_start,
        "MSG block is out of bounds"
    );
    Ok(&bytes[start..end])
}

fn validate_archive(archive: &MsgArchiveHeader, bytes: &[u8]) -> Result<()> {
    ensure!(
        archive.chunk_bits == CHUNK_BITS,
        "unsupported MSG chunk size"
    );
    ensure!(
        archive.simulation_time.is_finite(),
        "non-finite simulation time"
    );
    let geometry = MsgGeometry::new(
        DMat4::from_cols_array(&archive.grid_to_world),
        glam::DVec3::from_array(archive.sample_offset),
    )?;
    ensure!(!archive.levels.is_empty(), "MSG archive has no levels");
    let levels = archive
        .levels
        .iter()
        .enumerate()
        .map(|(index, level)| MsgLevel {
            index_origin: glam::IVec3::from_array(level.index_origin),
            point_extent: UVec3::from_array(level.point_extent),
            scale: glam::DVec3::from_array(level.scale),
            refinement_ratio_from_coarse: (index != 0)
                .then(|| UVec3::from_array(level.refinement_ratio)),
        })
        .collect();
    MsgDataset::new(archive.simulation_time, geometry, levels)
        .context("validating archived level geometry")?;
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
                field.error_absolute == 0.0 && field.error_relative_percent == 0.0,
                "lossless field has nonzero error metadata"
            ),
            1 => {
                ensure!(
                    field.error_absolute.is_finite() && field.error_absolute >= 0.0,
                    "invalid archived absolute error"
                );
                ensure!(
                    field.error_relative_percent.is_finite()
                        && (0.0..=100.0).contains(&field.error_relative_percent),
                    "invalid archived relative error"
                );
            }
            kind => bail!("unsupported MSG error tolerance kind {kind}"),
        }
        ensure!(
            field.levels.len() == archive.levels.len(),
            "MSG field level count mismatch"
        );
        for level in &field.levels {
            block_slice(bytes, &level.data)?;
            if let Some(block) = level.active_cubes {
                block_slice(bytes, &block)?;
            }
            for range in &level.value_ranges {
                ensure!(
                    range.value_min.is_finite()
                        && range.value_max.is_finite()
                        && range.value_min <= range.value_max
                        && range.max_abs_error.is_finite()
                        && range.max_abs_error >= 0.0,
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
        dataset.rebuild_acceleration().unwrap();
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
}
