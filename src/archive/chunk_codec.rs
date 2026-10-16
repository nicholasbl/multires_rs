use anyhow::{Context, Result, bail, ensure};

use super::archive::{ArchiveScalarChunk, ArchiveScalarGrid, array_to_uvec3, uvec3_to_array};
use super::{MsgChunkCodec, MsgChunkCodecSelection, ResolvedFieldCodec};
use crate::sparse::{CHUNK_BITS, CHUNK_VOLUME, MASK_WORDS, SparseGrid3, SparseGridChunkView};

const CHUNK_SIZE: u32 = 1 << CHUNK_BITS;

pub(super) const TYPE_F32: u8 = 0;
pub(super) const TYPE_F64: u8 = 1;

// On-disk identifiers: never renumber or reuse these. New codecs get new ids;
// incompatible changes to an existing codec increment its codec version.
pub(super) const CODEC_UNIFORM: u32 = 1;
pub(super) const CODEC_DENSE_F32: u32 = 2;
pub(super) const CODEC_DENSE_F64: u32 = 3;
// Codec id 4 belonged to the pre-MSG normalized-u32 representation and is
// intentionally never reused.
pub(super) const CODEC_QUANTIZED_U8: u32 = 5;
pub(super) const CODEC_QUANTIZED_U16: u32 = 6;
pub(super) const CODEC_QUANTIZED_U32: u32 = 7;
pub(super) const CODEC_AFFINE_F64: u32 = 8;
pub(super) const CODEC_AFFINE_RESIDUAL_U8: u32 = 9;
pub(super) const CODEC_AFFINE_RESIDUAL_U16: u32 = 10;
pub(super) const CODEC_AFFINE_RESIDUAL_U32: u32 = 11;
const CODEC_VERSION: u16 = 1;

trait Scalar: Copy + PartialEq + Default + Sync {
    const TYPE: u8;
    const WIDTH: usize;

    fn to_domain(self) -> f64;
    fn from_domain(value: f64) -> Self;
    fn append_raw(self, bytes: &mut Vec<u8>);
    fn read_raw(bytes: &[u8]) -> Self;
}

impl Scalar for f32 {
    const TYPE: u8 = TYPE_F32;
    const WIDTH: usize = 4;

    fn to_domain(self) -> f64 {
        f64::from(self)
    }

    fn from_domain(value: f64) -> Self {
        value as f32
    }

    fn append_raw(self, bytes: &mut Vec<u8>) {
        bytes.extend_from_slice(&self.to_le_bytes());
    }

    fn read_raw(bytes: &[u8]) -> Self {
        Self::from_le_bytes(bytes.try_into().expect("validated f32 width"))
    }
}

impl Scalar for f64 {
    const TYPE: u8 = TYPE_F64;
    const WIDTH: usize = 8;

    fn to_domain(self) -> f64 {
        self
    }

    fn from_domain(value: f64) -> Self {
        value
    }

    fn append_raw(self, bytes: &mut Vec<u8>) {
        bytes.extend_from_slice(&self.to_le_bytes());
    }

    fn read_raw(bytes: &[u8]) -> Self {
        Self::from_le_bytes(bytes.try_into().expect("validated f64 width"))
    }
}

struct EncodedCandidate {
    codec: MsgChunkCodec,
    max_abs_error: f64,
    metadata: Vec<u8>,
    payload: Vec<u8>,
}

struct Candidate<T> {
    codec: MsgChunkCodec,
    max_abs_error: f64,
    plan: CandidatePlan<T>,
}

#[derive(Clone, Copy)]
enum CandidatePlan<T> {
    DenseNative,
    DenseF32,
    Uniform(T),
    Quantized {
        coefficients: Option<[f64; 4]>,
        min: f64,
        step: f64,
        bits: u8,
    },
    Affine([f64; 4]),
}

#[derive(Clone, Copy)]
struct ErrorMetric {
    absolute: f64,
    relative_fraction: f64,
    physical_offset: f64,
    physical_scale: f64,
}

impl ErrorMetric {
    fn accepts(self, original: f64, reconstructed: f64) -> bool {
        let original = self.physical_offset + self.physical_scale * original;
        let reconstructed = self.physical_offset + self.physical_scale * reconstructed;
        let error = (original - reconstructed).abs();
        let allowed = self.absolute + self.relative_fraction * original.abs();
        error <= allowed
    }
}

impl<T: Scalar> Candidate<T> {
    fn stored_len(&self) -> usize {
        match self.plan {
            CandidatePlan::DenseNative => CHUNK_VOLUME * T::WIDTH,
            CandidatePlan::DenseF32 => CHUNK_VOLUME * 4,
            CandidatePlan::Uniform(_) => 8,
            CandidatePlan::Quantized {
                coefficients, bits, ..
            } => {
                let metadata = if coefficients.is_some() { 48 } else { 16 };
                metadata + CHUNK_VOLUME * usize::from(bits / 8)
            }
            CandidatePlan::Affine(_) => 32,
        }
    }
}

pub(super) fn encode_f32(
    grid: &SparseGrid3<f32>,
    policy: ResolvedFieldCodec,
) -> Result<ArchiveScalarGrid> {
    encode_grid(grid, policy)
}

pub(super) fn encode_f64(
    grid: &SparseGrid3<f64>,
    policy: ResolvedFieldCodec,
) -> Result<ArchiveScalarGrid> {
    encode_grid(grid, policy)
}

fn encode_grid<T: Scalar>(
    grid: &SparseGrid3<T>,
    policy: ResolvedFieldCodec,
) -> Result<ArchiveScalarGrid> {
    if let Some(error) = policy.error_tolerance {
        ensure!(
            error.absolute.is_finite() && error.absolute >= 0.0,
            "absolute chunk error must be finite and non-negative"
        );
        ensure!(
            error.relative_percent.is_finite() && (0.0..=100.0).contains(&error.relative_percent),
            "relative chunk error must be a finite percentage from 0 to 100"
        );
    }
    ensure!(
        policy.physical_offset.is_finite()
            && policy.physical_scale.is_finite()
            && policy.physical_scale > 0.0,
        "invalid codec physical-domain transform"
    );

    let encoded = grid.par_map_chunks(|chunk| match chunk {
        SparseGridChunkView::Uniform { key, value, mask } => Ok(ArchiveScalarChunk {
            key: uvec3_to_array(key),
            mask: Box::new(mask),
            codec_id: CODEC_UNIFORM,
            codec_version: CODEC_VERSION,
            codec_flags: 0,
            max_abs_error: 0.0,
            metadata: encode_uniform(value),
            payload_length: 0,
            payload: Vec::new(),
        }),
        SparseGridChunkView::Dense { key, values, mask } => encode_dense(key, values, mask, policy),
    });
    let mut chunks = encoded.into_iter().collect::<Result<Vec<_>>>()?;
    chunks.sort_by_key(|chunk| chunk.key);
    Ok(ArchiveScalarGrid {
        bounds: uvec3_to_array(grid.bounds()),
        decoded_type: T::TYPE,
        chunks,
    })
}

fn encode_dense<T: Scalar>(
    key: glam::UVec3,
    values: &[T],
    mask: [u64; MASK_WORDS],
    policy: ResolvedFieldCodec,
) -> Result<ArchiveScalarChunk> {
    ensure!(values.len() == CHUNK_VOLUME, "invalid dense chunk length");
    let selected = select_candidate(values, &mask, policy)
        .with_context(|| format!("encoding chunk ({}, {}, {})", key.x, key.y, key.z))?;
    let best = materialize_candidate(selected, values, &mask);

    let (payload_length, payload) = pack_payload(&best.payload)?;
    Ok(ArchiveScalarChunk {
        key: uvec3_to_array(key),
        mask: Box::new(mask),
        codec_id: codec_id(best.codec),
        codec_version: CODEC_VERSION,
        codec_flags: 0,
        max_abs_error: best.max_abs_error,
        metadata: best.metadata,
        payload_length,
        payload,
    })
}

fn materialize_candidate<T: Scalar>(
    candidate: Candidate<T>,
    values: &[T],
    mask: &[u64; MASK_WORDS],
) -> EncodedCandidate {
    let Candidate {
        codec,
        max_abs_error,
        plan,
    } = candidate;
    let (metadata, payload) = match plan {
        CandidatePlan::DenseNative => {
            let mut payload = Vec::with_capacity(values.len() * T::WIDTH);
            for &value in values {
                value.append_raw(&mut payload);
            }
            (Vec::new(), payload)
        }
        CandidatePlan::DenseF32 => {
            let mut payload = Vec::with_capacity(CHUNK_VOLUME * 4);
            for &value in values {
                (value.to_domain() as f32).append_raw(&mut payload);
            }
            (Vec::new(), payload)
        }
        CandidatePlan::Uniform(value) => (encode_uniform(value), Vec::new()),
        CandidatePlan::Quantized {
            coefficients,
            min,
            step,
            bits,
        } => {
            let mut metadata = Vec::with_capacity(if coefficients.is_some() { 48 } else { 16 });
            if let Some(coefficients) = coefficients {
                append_f64s(&mut metadata, &coefficients);
            }
            append_f64s(&mut metadata, &[min, step]);
            let mut payload = Vec::with_capacity(CHUNK_VOLUME * usize::from(bits / 8));
            for (index, &value) in values.iter().enumerate() {
                let code = if present(mask, index) {
                    quantized_code(value.to_domain(), index, coefficients, min, step, bits)
                } else {
                    0
                };
                append_code(&mut payload, bits, code);
            }
            (metadata, payload)
        }
        CandidatePlan::Affine(coefficients) => {
            let mut metadata = Vec::with_capacity(32);
            append_f64s(&mut metadata, &coefficients);
            (metadata, Vec::new())
        }
    };
    EncodedCandidate {
        codec,
        max_abs_error,
        metadata,
        payload,
    }
}

fn quantized_code(
    value: f64,
    index: usize,
    coefficients: Option<[f64; 4]>,
    min: f64,
    step: f64,
    bits: u8,
) -> u64 {
    if step == 0.0 {
        return 0;
    }
    let code_max = match bits {
        8 => u8::MAX as u64,
        16 => u16::MAX as u64,
        32 => u32::MAX as u64,
        _ => unreachable!(),
    };
    (((value - coefficients.map_or(0.0, |c| predict(c, index)) - min) / step)
        .round()
        .clamp(0.0, code_max as f64)) as u64
}

fn select_candidate<T: Scalar>(
    values: &[T],
    mask: &[u64; MASK_WORDS],
    policy: ResolvedFieldCodec,
) -> Result<Candidate<T>> {
    match policy.selection {
        MsgChunkCodecSelection::AutoLossless => {
            ensure!(
                policy.error_tolerance.is_none(),
                "auto:lossless does not accept an error tolerance"
            );
            Ok(dense_native_candidate(values))
        }
        MsgChunkCodecSelection::AutoSmallest => {
            let limit = required_error(policy)?;
            let mut best = dense_native_candidate(values);
            if let Some(candidate) = dense_f32_candidate(values, mask, limit) {
                choose_smaller(&mut best, candidate);
            }
            if let Some(candidate) = approximate_uniform(values, mask, limit) {
                choose_smaller(&mut best, candidate);
            }
            if let Some(candidate) = quantized_candidate(values, mask, limit, None, &[8, 16, 32]) {
                choose_smaller(&mut best, candidate);
            }
            if let Some(coefficients) = fit_affine(values, mask) {
                if let Some(candidate) = affine_candidate(values, mask, limit, coefficients) {
                    choose_smaller(&mut best, candidate);
                }
                if let Some(candidate) =
                    quantized_candidate(values, mask, limit, Some(coefficients), &[8, 16, 32])
                {
                    choose_smaller(&mut best, candidate);
                }
            }
            Ok(best)
        }
        MsgChunkCodecSelection::Forced(codec) => {
            let candidate = match codec {
                MsgChunkCodec::DenseF32 if T::TYPE == TYPE_F32 => {
                    ensure!(
                        policy.error_tolerance.is_none(),
                        "dense:f32 is exact for scalar_type f32 and does not accept an error tolerance"
                    );
                    Some(dense_native_candidate(values))
                }
                MsgChunkCodec::DenseF32 => {
                    dense_f32_candidate(values, mask, required_error(policy)?)
                }
                MsgChunkCodec::DenseF64 if T::TYPE == TYPE_F64 => {
                    ensure!(
                        policy.error_tolerance.is_none(),
                        "dense:f64 is exact and does not accept an error tolerance"
                    );
                    Some(dense_native_candidate(values))
                }
                MsgChunkCodec::DenseF64 => {
                    bail!(
                        "forced codec {} is incompatible with this scalar_type",
                        codec_name(codec)
                    )
                }
                MsgChunkCodec::UniformF64 => {
                    approximate_uniform(values, mask, required_error(policy)?)
                }
                MsgChunkCodec::QuantizedU8 => {
                    quantized_candidate(values, mask, required_error(policy)?, None, &[8])
                }
                MsgChunkCodec::QuantizedU16 => {
                    quantized_candidate(values, mask, required_error(policy)?, None, &[16])
                }
                MsgChunkCodec::QuantizedU32 => {
                    quantized_candidate(values, mask, required_error(policy)?, None, &[32])
                }
                MsgChunkCodec::AffineF64 => {
                    let limit = required_error(policy)?;
                    fit_affine(values, mask).and_then(|coefficients| {
                        affine_candidate(values, mask, limit, coefficients)
                    })
                }
                MsgChunkCodec::AffineResidualU8
                | MsgChunkCodec::AffineResidualU16
                | MsgChunkCodec::AffineResidualU32 => {
                    let limit = required_error(policy)?;
                    let bits = match codec {
                        MsgChunkCodec::AffineResidualU8 => 8,
                        MsgChunkCodec::AffineResidualU16 => 16,
                        MsgChunkCodec::AffineResidualU32 => 32,
                        _ => unreachable!(),
                    };
                    fit_affine(values, mask).and_then(|coefficients| {
                        quantized_candidate(values, mask, limit, Some(coefficients), &[bits])
                    })
                }
            };
            candidate.with_context(|| {
                format!(
                    "forced codec {} cannot satisfy the error tolerance",
                    codec_name(codec)
                )
            })
        }
    }
}

fn required_error(policy: ResolvedFieldCodec) -> Result<ErrorMetric> {
    let tolerance = policy
        .error_tolerance
        .context("lossy codec selection requires an error tolerance")?;
    Ok(ErrorMetric {
        absolute: tolerance.absolute,
        relative_fraction: tolerance.relative_percent / 100.0,
        physical_offset: policy.physical_offset,
        physical_scale: policy.physical_scale,
    })
}

fn codec_id(codec: MsgChunkCodec) -> u32 {
    match codec {
        MsgChunkCodec::UniformF64 => CODEC_UNIFORM,
        MsgChunkCodec::DenseF32 => CODEC_DENSE_F32,
        MsgChunkCodec::DenseF64 => CODEC_DENSE_F64,
        MsgChunkCodec::QuantizedU8 => CODEC_QUANTIZED_U8,
        MsgChunkCodec::QuantizedU16 => CODEC_QUANTIZED_U16,
        MsgChunkCodec::QuantizedU32 => CODEC_QUANTIZED_U32,
        MsgChunkCodec::AffineF64 => CODEC_AFFINE_F64,
        MsgChunkCodec::AffineResidualU8 => CODEC_AFFINE_RESIDUAL_U8,
        MsgChunkCodec::AffineResidualU16 => CODEC_AFFINE_RESIDUAL_U16,
        MsgChunkCodec::AffineResidualU32 => CODEC_AFFINE_RESIDUAL_U32,
    }
}

pub(super) fn codec_name(codec: MsgChunkCodec) -> &'static str {
    match codec {
        MsgChunkCodec::UniformF64 => "uniform:f64",
        MsgChunkCodec::DenseF32 => "dense:f32",
        MsgChunkCodec::DenseF64 => "dense:f64",
        MsgChunkCodec::QuantizedU8 => "quantized:u8",
        MsgChunkCodec::QuantizedU16 => "quantized:u16",
        MsgChunkCodec::QuantizedU32 => "quantized:u32",
        MsgChunkCodec::AffineF64 => "affine:f64",
        MsgChunkCodec::AffineResidualU8 => "affine-residual:u8",
        MsgChunkCodec::AffineResidualU16 => "affine-residual:u16",
        MsgChunkCodec::AffineResidualU32 => "affine-residual:u32",
    }
}

fn dense_f32_candidate<T: Scalar>(
    values: &[T],
    mask: &[u64; MASK_WORDS],
    metric: ErrorMetric,
) -> Option<Candidate<T>> {
    if T::TYPE == TYPE_F32 {
        return None;
    }
    let mut error = 0.0_f64;
    for (index, &value) in values.iter().enumerate() {
        let stored = value.to_domain() as f32;
        if present(mask, index) {
            let reconstructed = T::from_domain(f64::from(stored)).to_domain();
            if !reconstructed.is_finite() {
                return None;
            }
            let original = value.to_domain();
            if !metric.accepts(original, reconstructed) {
                return None;
            }
            error = error.max((original - reconstructed).abs());
        }
    }
    Some(Candidate {
        codec: MsgChunkCodec::DenseF32,
        max_abs_error: error,
        plan: CandidatePlan::DenseF32,
    })
}

fn choose_smaller<T: Scalar>(best: &mut Candidate<T>, candidate: Candidate<T>) {
    if candidate.stored_len() < best.stored_len() {
        *best = candidate;
    }
}

fn dense_native_candidate<T: Scalar>(_values: &[T]) -> Candidate<T> {
    Candidate {
        codec: match T::TYPE {
            TYPE_F32 => MsgChunkCodec::DenseF32,
            TYPE_F64 => MsgChunkCodec::DenseF64,
            _ => unreachable!(),
        },
        max_abs_error: 0.0,
        plan: CandidatePlan::DenseNative,
    }
}

fn approximate_uniform<T: Scalar>(
    values: &[T],
    mask: &[u64; MASK_WORDS],
    metric: ErrorMetric,
) -> Option<Candidate<T>> {
    let (min, max) = present_range(values, mask)?;
    let center = min + (max - min) * 0.5;
    let decoded = T::from_domain(center);
    let actual = decoded.to_domain();
    let error = present_max_error(values, mask, metric, |_| actual)?;
    Some(Candidate {
        codec: MsgChunkCodec::UniformF64,
        max_abs_error: error,
        plan: CandidatePlan::Uniform(decoded),
    })
}

fn quantized_candidate<T: Scalar>(
    values: &[T],
    mask: &[u64; MASK_WORDS],
    metric: ErrorMetric,
    affine: Option<[f64; 4]>,
    bit_widths: &[u8],
) -> Option<Candidate<T>> {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for (index, &value) in values.iter().enumerate() {
        if present(mask, index) {
            let residual = value.to_domain() - affine.map_or(0.0, |c| predict(c, index));
            min = min.min(residual);
            max = max.max(residual);
        }
    }
    if min > max || !(max - min).is_finite() {
        return None;
    }

    for &bits in bit_widths {
        let code_max = match bits {
            8 => u8::MAX as u64,
            16 => u16::MAX as u64,
            32 => u32::MAX as u64,
            _ => unreachable!(),
        };
        let step = if max == min {
            0.0
        } else {
            (max - min) / code_max as f64
        };
        if !step.is_finite() {
            continue;
        }

        let mut error = 0.0_f64;
        for (index, &value) in values.iter().enumerate() {
            if present(mask, index) {
                let code = quantized_code(value.to_domain(), index, affine, min, step, bits);
                let reconstructed =
                    affine.map_or(0.0, |c| predict(c, index)) + min + code as f64 * step;
                let actual = T::from_domain(reconstructed).to_domain();
                let original = value.to_domain();
                if !metric.accepts(original, actual) {
                    error = f64::INFINITY;
                } else {
                    error = error.max((original - actual).abs());
                }
            }
        }
        if error.is_finite() {
            return Some(Candidate {
                codec: match (affine.is_some(), bits) {
                    (false, 8) => MsgChunkCodec::QuantizedU8,
                    (false, 16) => MsgChunkCodec::QuantizedU16,
                    (false, 32) => MsgChunkCodec::QuantizedU32,
                    (true, 8) => MsgChunkCodec::AffineResidualU8,
                    (true, 16) => MsgChunkCodec::AffineResidualU16,
                    (true, 32) => MsgChunkCodec::AffineResidualU32,
                    _ => unreachable!(),
                },
                max_abs_error: error,
                plan: CandidatePlan::Quantized {
                    coefficients: affine,
                    min,
                    step,
                    bits,
                },
            });
        }
    }
    None
}

fn affine_candidate<T: Scalar>(
    values: &[T],
    mask: &[u64; MASK_WORDS],
    metric: ErrorMetric,
    coefficients: [f64; 4],
) -> Option<Candidate<T>> {
    let error = present_max_error(values, mask, metric, |index| {
        T::from_domain(predict(coefficients, index)).to_domain()
    })?;
    Some(Candidate {
        codec: MsgChunkCodec::AffineF64,
        max_abs_error: error,
        plan: CandidatePlan::Affine(coefficients),
    })
}

fn fit_affine<T: Scalar>(values: &[T], mask: &[u64; MASK_WORDS]) -> Option<[f64; 4]> {
    let mut normal = [[0.0_f64; 5]; 4];
    for (index, &value) in values.iter().enumerate() {
        if !present(mask, index) {
            continue;
        }
        let [x, y, z] = coordinates(index);
        let feature = [1.0, x, y, z];
        for row in 0..4 {
            for column in 0..4 {
                normal[row][column] += feature[row] * feature[column];
            }
            normal[row][4] += feature[row] * value.to_domain();
        }
    }

    for pivot in 0..4 {
        let best = (pivot..4)
            .max_by(|&a, &b| normal[a][pivot].abs().total_cmp(&normal[b][pivot].abs()))?;
        if normal[best][pivot].abs() <= f64::EPSILON {
            return None;
        }
        normal.swap(pivot, best);
        let divisor = normal[pivot][pivot];
        for value in &mut normal[pivot][pivot..] {
            *value /= divisor;
        }
        let pivot_values = normal[pivot];
        for (row_index, row) in normal.iter_mut().enumerate() {
            if row_index == pivot {
                continue;
            }
            let factor = row[pivot];
            for (value, &pivot_value) in row[pivot..].iter_mut().zip(&pivot_values[pivot..]) {
                *value -= factor * pivot_value;
            }
        }
    }
    let coefficients = std::array::from_fn(|row| normal[row][4]);
    coefficients
        .iter()
        .all(|value| value.is_finite())
        .then_some(coefficients)
}

pub(super) fn decode_f32(archive: ArchiveScalarGrid) -> Result<SparseGrid3<f32>> {
    decode_grid(archive)
}

pub(super) fn decode_f64(archive: ArchiveScalarGrid) -> Result<SparseGrid3<f64>> {
    decode_grid(archive)
}

fn decode_grid<T: Scalar>(archive: ArchiveScalarGrid) -> Result<SparseGrid3<T>> {
    ensure!(
        archive.decoded_type == T::TYPE,
        "MSG scalar chunk type mismatch"
    );
    let mut grid =
        SparseGrid3::with_chunk_capacity(array_to_uvec3(archive.bounds), archive.chunks.len());
    for chunk in archive.chunks {
        ensure!(
            chunk.codec_version == CODEC_VERSION,
            "unsupported chunk codec {} version {}",
            chunk.codec_id,
            chunk.codec_version
        );
        ensure!(chunk.codec_flags == 0, "unsupported chunk codec flags");
        ensure!(
            chunk.max_abs_error.is_finite() && chunk.max_abs_error >= 0.0,
            "invalid chunk error bound"
        );
        let key = array_to_uvec3(chunk.key);
        let mask = *chunk.mask;
        let payload = unpack_payload(&chunk.payload, chunk.payload_length)?;
        match chunk.codec_id {
            CODEC_UNIFORM => {
                ensure!(payload.is_empty(), "uniform chunk has a payload");
                let value = read_f64_array::<1>(&chunk.metadata)?[0];
                grid.insert_uniform_chunk(key, T::from_domain(value), mask);
            }
            id @ CODEC_DENSE_F64 => {
                ensure!(
                    id == CODEC_DENSE_F64 && T::TYPE == TYPE_F64,
                    "dense chunk codec is incompatible with decoded scalar type"
                );
                ensure!(chunk.metadata.is_empty(), "dense chunk has metadata");
                ensure!(
                    payload.len() == CHUNK_VOLUME * T::WIDTH,
                    "invalid native dense chunk length"
                );
                let values = payload
                    .chunks_exact(T::WIDTH)
                    .map(T::read_raw)
                    .collect::<Vec<_>>();
                grid.insert_dense_chunk(key, values.into_boxed_slice(), mask);
            }
            CODEC_DENSE_F32 => {
                ensure!(chunk.metadata.is_empty(), "dense f32 chunk has metadata");
                ensure!(
                    payload.len() == CHUNK_VOLUME * 4,
                    "invalid dense f32 chunk length"
                );
                let values = payload
                    .chunks_exact(4)
                    .map(|bytes| {
                        T::from_domain(f64::from(f32::from_le_bytes(bytes.try_into().unwrap())))
                    })
                    .collect::<Vec<_>>();
                grid.insert_dense_chunk(key, values.into_boxed_slice(), mask);
            }
            id @ (CODEC_QUANTIZED_U8 | CODEC_QUANTIZED_U16 | CODEC_QUANTIZED_U32) => {
                let bits = codec_bits(id);
                let offset = decode_quantized_metadata(&chunk.metadata, false)?;
                let values = decode_codes::<T>(&payload, bits, |_, code| {
                    offset[0] + code as f64 * offset[1]
                })?;
                grid.insert_dense_chunk(key, values.into_boxed_slice(), mask);
            }
            CODEC_AFFINE_F64 => {
                ensure!(payload.is_empty(), "affine chunk has a payload");
                let coefficients = read_f64_array::<4>(&chunk.metadata)?;
                let values = (0..CHUNK_VOLUME)
                    .map(|index| T::from_domain(predict(coefficients, index)))
                    .collect::<Vec<_>>();
                grid.insert_dense_chunk(key, values.into_boxed_slice(), mask);
            }
            id @ (CODEC_AFFINE_RESIDUAL_U8
            | CODEC_AFFINE_RESIDUAL_U16
            | CODEC_AFFINE_RESIDUAL_U32) => {
                let bits = codec_bits(id);
                let values = decode_quantized_metadata(&chunk.metadata, true)?;
                let coefficients = [values[0], values[1], values[2], values[3]];
                let residual_min = values[4];
                let residual_step = values[5];
                let decoded = decode_codes::<T>(&payload, bits, |index, code| {
                    predict(coefficients, index) + residual_min + code as f64 * residual_step
                })?;
                grid.insert_dense_chunk(key, decoded.into_boxed_slice(), mask);
            }
            id => bail!("unsupported MSG chunk codec {id}"),
        }
    }
    Ok(grid)
}

fn decode_quantized_metadata(metadata: &[u8], affine: bool) -> Result<Vec<f64>> {
    let count = if affine { 6 } else { 2 };
    ensure!(
        metadata.len() == count * 8,
        "invalid quantized chunk metadata"
    );
    let values = metadata
        .chunks_exact(8)
        .map(|bytes| f64::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "non-finite quantized chunk metadata"
    );
    ensure!(values[count - 1] >= 0.0, "negative quantization step");
    Ok(values)
}

fn codec_bits(codec_id: u32) -> u8 {
    match codec_id {
        CODEC_QUANTIZED_U8 | CODEC_AFFINE_RESIDUAL_U8 => 8,
        CODEC_QUANTIZED_U16 | CODEC_AFFINE_RESIDUAL_U16 => 16,
        CODEC_QUANTIZED_U32 | CODEC_AFFINE_RESIDUAL_U32 => 32,
        _ => unreachable!(),
    }
}

fn decode_codes<T: Scalar>(
    payload: &[u8],
    bits: u8,
    mut reconstruct: impl FnMut(usize, u64) -> f64,
) -> Result<Vec<T>> {
    let width = usize::from(bits / 8);
    ensure!(
        payload.len() == CHUNK_VOLUME * width,
        "invalid quantized chunk length"
    );
    Ok(payload
        .chunks_exact(width)
        .enumerate()
        .map(|(index, bytes)| {
            let code = match bits {
                8 => u64::from(bytes[0]),
                16 => u64::from(u16::from_le_bytes(bytes.try_into().unwrap())),
                32 => u64::from(u32::from_le_bytes(bytes.try_into().unwrap())),
                _ => unreachable!(),
            };
            T::from_domain(reconstruct(index, code))
        })
        .collect())
}

fn present(mask: &[u64; MASK_WORDS], index: usize) -> bool {
    mask[index / 64] & (1_u64 << (index % 64)) != 0
}

fn present_range<T: Scalar>(values: &[T], mask: &[u64; MASK_WORDS]) -> Option<(f64, f64)> {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for (index, &value) in values.iter().enumerate() {
        if present(mask, index) {
            min = min.min(value.to_domain());
            max = max.max(value.to_domain());
        }
    }
    (min <= max).then_some((min, max))
}

fn present_max_error<T: Scalar>(
    values: &[T],
    mask: &[u64; MASK_WORDS],
    metric: ErrorMetric,
    mut prediction: impl FnMut(usize) -> f64,
) -> Option<f64> {
    let mut error = 0.0_f64;
    for (index, &value) in values.iter().enumerate() {
        if !present(mask, index) {
            continue;
        }
        let original = value.to_domain();
        let reconstructed = prediction(index);
        if !metric.accepts(original, reconstructed) {
            return None;
        }
        error = error.max((original - reconstructed).abs());
    }
    Some(error)
}

fn coordinates(index: usize) -> [f64; 3] {
    let edge = CHUNK_SIZE as usize;
    let x = index % edge;
    let y = (index / edge) % edge;
    let z = index / (edge * edge);
    [x as f64, y as f64, z as f64]
}

fn predict(coefficients: [f64; 4], index: usize) -> f64 {
    let [x, y, z] = coordinates(index);
    coefficients[0] + coefficients[1] * x + coefficients[2] * y + coefficients[3] * z
}

fn encode_uniform<T: Scalar>(value: T) -> Vec<u8> {
    value.to_domain().to_le_bytes().to_vec()
}

fn append_f64s(bytes: &mut Vec<u8>, values: &[f64]) {
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

fn read_f64_array<const N: usize>(bytes: &[u8]) -> Result<[f64; N]> {
    ensure!(
        bytes.len() == N * 8,
        "invalid floating-point codec metadata"
    );
    let values = std::array::from_fn(|index| {
        f64::from_le_bytes(bytes[index * 8..(index + 1) * 8].try_into().unwrap())
    });
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "non-finite codec metadata"
    );
    Ok(values)
}

fn append_code(bytes: &mut Vec<u8>, bits: u8, code: u64) {
    match bits {
        8 => bytes.push(code as u8),
        16 => bytes.extend_from_slice(&(code as u16).to_le_bytes()),
        32 => bytes.extend_from_slice(&(code as u32).to_le_bytes()),
        _ => unreachable!(),
    }
}

fn pack_payload(bytes: &[u8]) -> Result<(u64, Vec<u64>)> {
    let length = u64::try_from(bytes.len()).context("chunk payload length exceeds u64")?;
    let mut words = Vec::with_capacity(bytes.len().div_ceil(8));
    for chunk in bytes.chunks(8) {
        let mut word = [0_u8; 8];
        word[..chunk.len()].copy_from_slice(chunk);
        words.push(u64::from_le_bytes(word));
    }
    Ok((length, words))
}

fn unpack_payload(words: &[u64], length: u64) -> Result<Vec<u8>> {
    let length = usize::try_from(length).context("chunk payload length exceeds usize")?;
    ensure!(
        words.len() == length.div_ceil(8),
        "invalid aligned chunk payload length"
    );
    let mut bytes = Vec::with_capacity(words.len() * 8);
    for &word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes.truncate(length);
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::archive::MsgErrorTolerance;
    use glam::UVec3;

    fn auto_policy(max_abs_error: f64) -> ResolvedFieldCodec {
        ResolvedFieldCodec {
            selection: MsgChunkCodecSelection::AutoSmallest,
            error_tolerance: Some(MsgErrorTolerance {
                absolute: max_abs_error,
                relative_percent: 0.0,
            }),
            physical_offset: 0.0,
            physical_scale: 1.0,
        }
    }

    fn forced_policy(codec: MsgChunkCodec, max_abs_error: Option<f64>) -> ResolvedFieldCodec {
        ResolvedFieldCodec {
            selection: MsgChunkCodecSelection::Forced(codec),
            error_tolerance: max_abs_error.map(|absolute| MsgErrorTolerance {
                absolute,
                relative_percent: 0.0,
            }),
            physical_offset: 0.0,
            physical_scale: 1.0,
        }
    }

    #[test]
    fn affine_chunk_round_trips_with_requested_error() -> Result<()> {
        let bounds = UVec3::splat(CHUNK_SIZE);
        let mut grid = SparseGrid3::new(bounds);
        let values = (0..CHUNK_VOLUME)
            .map(|index| {
                let [x, y, z] = coordinates(index);
                10.0 + 2.0 * x - 0.5 * y + 0.25 * z
            })
            .collect::<Vec<_>>();
        grid.write_aabb_values(grid.bounds_aabb(), &values)
            .map_err(|error| anyhow::anyhow!("writing test grid: {error:?}"))?;
        let archive = encode_f64(&grid, auto_policy(1.0e-10))?;
        assert_eq!(archive.chunks[0].codec_id, CODEC_AFFINE_F64);
        let decoded = decode_f64(archive)?;
        for (expected, actual) in values.into_iter().zip((0..CHUNK_VOLUME).map(|index| {
            let [x, y, z] = coordinates(index);
            decoded
                .get(UVec3::new(x as u32, y as u32, z as u32))
                .unwrap()
        })) {
            assert!((expected - actual).abs() <= 1.0e-10);
        }
        Ok(())
    }

    #[test]
    fn quantized_chunk_respects_error_bound() -> Result<()> {
        let bounds = UVec3::splat(CHUNK_SIZE);
        let mut grid = SparseGrid3::new(bounds);
        let values = (0..CHUNK_VOLUME)
            .map(|index| ((index * 7919) % 10000) as f32 / 10000.0)
            .collect::<Vec<_>>();
        grid.write_aabb_values(grid.bounds_aabb(), &values)
            .map_err(|error| anyhow::anyhow!("writing test grid: {error:?}"))?;
        let archive = encode_f32(&grid, auto_policy(0.002))?;
        assert_eq!(archive.chunks[0].codec_id, CODEC_QUANTIZED_U8);
        let decoded = decode_f32(archive)?;
        for (index, &expected) in values.iter().enumerate() {
            let [x, y, z] = coordinates(index);
            let actual = decoded
                .get(UVec3::new(x as u32, y as u32, z as u32))
                .unwrap();
            assert!((expected - actual).abs() <= 0.002);
        }
        Ok(())
    }

    #[test]
    fn forced_quantized_codec_does_not_substitute_a_smaller_codec() -> Result<()> {
        let bounds = UVec3::splat(CHUNK_SIZE);
        let mut grid = SparseGrid3::new(bounds);
        let values = (0..CHUNK_VOLUME)
            .map(|index| ((index * 7919) % 10000) as f32 / 10000.0)
            .collect::<Vec<_>>();
        grid.write_aabb_values(grid.bounds_aabb(), &values)
            .map_err(|error| anyhow::anyhow!("writing test grid: {error:?}"))?;

        let archive = encode_f32(
            &grid,
            forced_policy(MsgChunkCodec::QuantizedU16, Some(0.002)),
        )?;
        assert_eq!(archive.chunks[0].codec_id, CODEC_QUANTIZED_U16);
        Ok(())
    }

    #[test]
    fn forced_codec_fails_instead_of_falling_back() -> Result<()> {
        let bounds = UVec3::splat(CHUNK_SIZE);
        let mut grid = SparseGrid3::new(bounds);
        let values = (0..CHUNK_VOLUME)
            .map(|index| ((index * 7919) % 10000) as f32 / 10000.0)
            .collect::<Vec<_>>();
        grid.write_aabb_values(grid.bounds_aabb(), &values)
            .map_err(|error| anyhow::anyhow!("writing test grid: {error:?}"))?;

        let result = encode_f32(
            &grid,
            forced_policy(MsgChunkCodec::QuantizedU8, Some(1.0e-6)),
        );
        let error = match result {
            Ok(_) => panic!("forced quantized:u8 unexpectedly satisfied the error bound"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("encoding chunk"));
        Ok(())
    }

    #[test]
    fn structural_uniform_chunk_stays_uniform_under_forced_dense_policy() -> Result<()> {
        let bounds = UVec3::splat(CHUNK_SIZE);
        let mut grid = SparseGrid3::new(bounds);
        assert!(grid.set(UVec3::ZERO, 42.0_f32));

        let archive = encode_f32(&grid, forced_policy(MsgChunkCodec::DenseF32, None))?;
        assert_eq!(archive.chunks[0].codec_id, CODEC_UNIFORM);
        Ok(())
    }

    #[test]
    fn f64_chunk_can_select_dense_f32() -> Result<()> {
        let bounds = UVec3::splat(CHUNK_SIZE);
        let mut grid = SparseGrid3::new(bounds);
        let values = (0..CHUNK_VOLUME)
            .map(|index| (((index * 7919) % 10000) as f32 / 10000.0) as f64)
            .collect::<Vec<_>>();
        grid.write_aabb_values(grid.bounds_aabb(), &values)
            .map_err(|error| anyhow::anyhow!("writing test grid: {error:?}"))?;
        let archive = encode_f64(&grid, auto_policy(0.0))?;
        assert_eq!(archive.chunks[0].codec_id, CODEC_DENSE_F32);
        let decoded = decode_f64(archive)?;
        for (index, &expected) in values.iter().enumerate() {
            let [x, y, z] = coordinates(index);
            assert_eq!(
                decoded
                    .get(UVec3::new(x as u32, y as u32, z as u32))
                    .unwrap(),
                expected
            );
        }
        Ok(())
    }

    #[test]
    fn parallel_encoding_sorts_chunk_directory() -> Result<()> {
        let bounds = UVec3::new(CHUNK_SIZE * 3, CHUNK_SIZE, CHUNK_SIZE);
        let mut grid = SparseGrid3::new(bounds);
        let values = (0..bounds.x as usize * bounds.y as usize * bounds.z as usize)
            .map(|index| index as f32)
            .collect::<Vec<_>>();
        grid.write_aabb_values(grid.bounds_aabb(), &values)
            .map_err(|error| anyhow::anyhow!("writing test grid: {error:?}"))?;

        let archive = encode_f32(
            &grid,
            ResolvedFieldCodec {
                selection: MsgChunkCodecSelection::AutoLossless,
                error_tolerance: None,
                physical_offset: 0.0,
                physical_scale: 1.0,
            },
        )?;
        assert_eq!(
            archive
                .chunks
                .iter()
                .map(|chunk| chunk.key)
                .collect::<Vec<_>>(),
            vec![[0, 0, 0], [1, 0, 0], [2, 0, 0]]
        );
        Ok(())
    }
}
