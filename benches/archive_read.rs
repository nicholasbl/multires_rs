use std::{
    alloc::{GlobalAlloc, Layout, System},
    fs::File,
    hint::black_box,
    io::Write,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use glam::{DMat4, DVec3, IVec3, UVec3};
use multires_rs::{
    IsosurfaceOptions, MappedMsgArchive, MsgChunkBuffer, MsgChunkCodec, MsgChunkCodecSelection,
    MsgDataset, MsgErrorTolerance, MsgField, MsgFieldCodec, MsgFieldLevel, MsgGeometry, MsgLevel,
    MsgScalarGrid, MsgScalarType, MsgWriteOptions, Surface, isosurface, isosurface_archive,
    read_msg, read_msg_fields,
    sparse::{Aabb3u, CHUNK_SIZE},
    view_msg, write_msg,
};

struct CountingAllocator;

static ALLOCATION_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        // SAFETY: This allocator delegates the allocation unchanged to System.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` and `layout` came from the delegated System allocation.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        // SAFETY: The reallocation request is delegated unchanged to System.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

#[derive(Clone, Copy)]
struct CodecCase {
    label: &'static str,
    scalar_type: MsgScalarType,
    codec: MsgChunkCodec,
    tolerance: Option<f64>,
}

struct Fixture {
    bytes: Vec<u8>,
    decoded_bytes: u64,
    field_names: Vec<String>,
}

impl Fixture {
    fn selected_names(&self) -> Vec<&str> {
        self.field_names.iter().map(String::as_str).collect()
    }
}

struct TemporaryArchive {
    path: PathBuf,
}

impl TemporaryArchive {
    fn new(bytes: &[u8]) -> Self {
        let path = std::env::temp_dir().join(format!(
            "multires-rs-archive-bench-{}.msg",
            std::process::id()
        ));
        let mut file = File::create(&path).expect("creating temporary benchmark archive");
        file.write_all(bytes)
            .expect("writing temporary benchmark archive");
        file.sync_all()
            .expect("syncing temporary benchmark archive");
        Self { path }
    }
}

impl Drop for TemporaryArchive {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn archive_benchmarks(criterion: &mut Criterion) {
    benchmark_validation(criterion);
    benchmark_decode_scaling(criterion);
    benchmark_codecs(criterion);
    benchmark_selection(criterion);
    benchmark_lazy_access(criterion);
    benchmark_lazy_isosurface(criterion);
    benchmark_mapped_workflow(criterion);
}

fn benchmark_validation(criterion: &mut Criterion) {
    let fixture = build_fixture(1, 4, dense_f32());
    report_allocations("view_msg/4_fields_1_level", || {
        black_box(view_msg(&fixture.bytes).unwrap());
    });

    let mut group = criterion.benchmark_group("archive_validation");
    group.bench_function("4_fields_1_level", |bencher| {
        bencher.iter(|| black_box(view_msg(black_box(&fixture.bytes)).unwrap()));
    });
    group.finish();
}

fn benchmark_decode_scaling(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("archive_decode_scaling_dense_f32");
    for chunks_per_axis in [1, 4, 8] {
        let fixture = build_fixture(chunks_per_axis, 1, dense_f32());
        let view = view_msg(&fixture.bytes).unwrap();
        let names = fixture.selected_names();
        group.throughput(Throughput::Bytes(fixture.decoded_bytes));
        group.bench_function(
            BenchmarkId::from_parameter(format!("{}_chunks", chunks_per_axis.pow(3))),
            |bencher| {
                bencher.iter(|| {
                    let dataset = view.decode_fields(black_box(&names)).unwrap();
                    black_box(&dataset);
                });
            },
        );
    }
    group.finish();
}

fn benchmark_codecs(criterion: &mut Criterion) {
    let cases = [
        dense_f64(),
        CodecCase {
            label: "quantized_u8",
            scalar_type: MsgScalarType::F64,
            codec: MsgChunkCodec::QuantizedU8,
            tolerance: Some(0.2),
        },
        CodecCase {
            label: "quantized_u16",
            scalar_type: MsgScalarType::F64,
            codec: MsgChunkCodec::QuantizedU16,
            tolerance: Some(0.001),
        },
        CodecCase {
            label: "quantized_u32",
            scalar_type: MsgScalarType::F64,
            codec: MsgChunkCodec::QuantizedU32,
            tolerance: Some(1.0e-7),
        },
        CodecCase {
            label: "affine_f64",
            scalar_type: MsgScalarType::F64,
            codec: MsgChunkCodec::AffineF64,
            tolerance: Some(0.2),
        },
        CodecCase {
            label: "affine_residual_u8",
            scalar_type: MsgScalarType::F64,
            codec: MsgChunkCodec::AffineResidualU8,
            tolerance: Some(0.001),
        },
        CodecCase {
            label: "affine_residual_u16",
            scalar_type: MsgScalarType::F64,
            codec: MsgChunkCodec::AffineResidualU16,
            tolerance: Some(1.0e-5),
        },
        CodecCase {
            label: "affine_residual_u32",
            scalar_type: MsgScalarType::F64,
            codec: MsgChunkCodec::AffineResidualU32,
            tolerance: Some(1.0e-9),
        },
    ];

    let mut group = criterion.benchmark_group("archive_decode_codec_64_chunks");
    for codec in cases {
        let fixture = build_fixture(4, 1, codec);
        let view = view_msg(&fixture.bytes).unwrap();
        let names = fixture.selected_names();
        group.throughput(Throughput::Bytes(fixture.decoded_bytes));
        group.bench_function(codec.label, |bencher| {
            bencher.iter(|| {
                let dataset = view.decode_fields(black_box(&names)).unwrap();
                black_box(&dataset);
            });
        });
    }
    group.finish();
}

fn benchmark_selection(criterion: &mut Criterion) {
    let fixture = build_fixture(4, 4, dense_f32());
    let view = view_msg(&fixture.bytes).unwrap();
    let one_field = [fixture.field_names[0].as_str()];
    let all_fields = fixture.selected_names();

    let mut group = criterion.benchmark_group("archive_field_selection_64_chunks");
    group.throughput(Throughput::Bytes(fixture.decoded_bytes / 4));
    group.bench_function("1_of_4", |bencher| {
        bencher.iter(|| {
            let dataset = view.decode_fields(black_box(&one_field)).unwrap();
            black_box(&dataset);
        });
    });
    group.throughput(Throughput::Bytes(fixture.decoded_bytes));
    group.bench_function("4_of_4", |bencher| {
        bencher.iter(|| {
            let dataset = view.decode_fields(black_box(&all_fields)).unwrap();
            black_box(&dataset);
        });
    });
    group.bench_function("read_msg_4_fields", |bencher| {
        bencher.iter(|| {
            let dataset = read_msg(black_box(&fixture.bytes)).unwrap();
            black_box(&dataset);
        });
    });
    group.finish();
}

fn benchmark_mapped_workflow(criterion: &mut Criterion) {
    let fixture = build_fixture(4, 1, dense_f32());
    let temporary = TemporaryArchive::new(&fixture.bytes);
    let names = fixture.selected_names();
    let mapped = MappedMsgArchive::open(&temporary.path).unwrap();
    let view = mapped.view().unwrap();

    report_allocations("decode_fields/dense_f32_64_chunks", || {
        black_box(view.decode_fields(&names).unwrap());
    });
    report_allocations("read_msg_fields/dense_f32_64_chunks", || {
        black_box(read_msg_fields(&fixture.bytes, &names).unwrap());
    });

    let mut group = criterion.benchmark_group("archive_mapped_workflow_64_chunks");
    group.bench_function("open_and_validate", |bencher| {
        bencher.iter(|| {
            let mapped = MappedMsgArchive::open(black_box(&temporary.path)).unwrap();
            black_box(mapped.bytes().len());
        });
    });
    group.throughput(Throughput::Bytes(fixture.decoded_bytes));
    group.bench_function("open_validate_and_decode", |bencher| {
        bencher.iter(|| {
            let mapped = MappedMsgArchive::open(black_box(&temporary.path)).unwrap();
            let dataset = mapped.view().unwrap().decode_fields(&names).unwrap();
            black_box(&dataset);
        });
    });
    group.finish();
}

fn benchmark_lazy_access(criterion: &mut Criterion) {
    let fixture = build_fixture(4, 1, dense_f32());
    let archive = view_msg(&fixture.bytes).unwrap();
    let level = archive.field("field_0").unwrap().level(0).unwrap();
    let point = UVec3::new(37, 41, 43);
    let key = point / UVec3::splat(CHUNK_SIZE);
    let mut buffer = MsgChunkBuffer::new(MsgScalarType::F32);
    level.fill_chunk(key, &mut buffer).unwrap();

    report_allocations("archive_level/get", || {
        black_box(level.get(point).unwrap());
    });
    report_allocations("archive_level/fill_chunk_reused", || {
        black_box(level.fill_chunk(key, &mut buffer).unwrap());
    });

    let mut point_group = criterion.benchmark_group("archive_lazy_point_dense_f32");
    point_group.bench_function("open_level", |bencher| {
        bencher.iter(|| {
            black_box(archive.field("field_0").unwrap().level(0).unwrap());
        });
    });
    point_group.bench_function("present", |bencher| {
        bencher.iter(|| black_box(level.get(black_box(point)).unwrap()));
    });
    point_group.finish();

    let mut chunk_group = criterion.benchmark_group("archive_lazy_chunk_dense_f32");
    chunk_group.throughput(Throughput::Bytes(
        u64::from(CHUNK_SIZE).pow(3) * std::mem::size_of::<f32>() as u64,
    ));
    chunk_group.bench_function("reused_buffer", |bencher| {
        bencher.iter(|| {
            black_box(level.fill_chunk(black_box(key), &mut buffer).unwrap());
        });
    });
    chunk_group.finish();
}

fn benchmark_lazy_isosurface(criterion: &mut Criterion) {
    let fixture = build_fixture(2, 1, dense_f32());
    let archive = view_msg(&fixture.bytes).unwrap();
    let eager = archive.decode_fields(&["field_0"]).unwrap();
    let options = || IsosurfaceOptions {
        surface: Surface {
            field: "field_0".into(),
            value: -1000.0,
        },
        ..IsosurfaceOptions::default()
    };
    let mut group = criterion.benchmark_group("archive_isosurface_rejected_by_ranges");
    group.bench_function("mapped_lazy", |bencher| {
        bencher.iter(|| {
            black_box(isosurface_archive(&archive, options()).unwrap());
        });
    });
    group.bench_function("owned_already_decoded", |bencher| {
        bencher.iter(|| {
            black_box(isosurface(&eager, options()).unwrap());
        });
    });
    group.bench_function("eager_decode_then_extract", |bencher| {
        bencher.iter(|| {
            let dataset = archive.decode_fields(&["field_0"]).unwrap();
            black_box(isosurface(&dataset, options()).unwrap());
        });
    });
    group.finish();
}

fn build_fixture(chunks_per_axis: u32, field_count: usize, codec: CodecCase) -> Fixture {
    let extent = UVec3::splat(chunks_per_axis * CHUNK_SIZE);
    let point_count = extent.x as usize * extent.y as usize * extent.z as usize;
    let mut dataset = MsgDataset::new(
        1.25,
        MsgGeometry::point_centered(DMat4::IDENTITY).unwrap(),
        vec![MsgLevel {
            index_origin: IVec3::ZERO,
            point_extent: extent,
            scale: DVec3::ONE,
            refinement_ratio_from_coarse: None,
        }],
    )
    .unwrap();

    for field_index in 0..field_count {
        let mut grid = MsgScalarGrid::new(extent, codec.scalar_type);
        match codec.scalar_type {
            MsgScalarType::F32 => {
                let values = generated_values(point_count, extent, field_index)
                    .map(|value| value as f32)
                    .collect::<Vec<_>>();
                grid.write_f32_values(Aabb3u::full_grid(extent), &values)
                    .unwrap();
            }
            MsgScalarType::F64 => {
                let values = generated_values(point_count, extent, field_index).collect::<Vec<_>>();
                grid.write_f64_values(Aabb3u::full_grid(extent), &values)
                    .unwrap();
            }
        }
        dataset
            .add_field(MsgField::new(
                format!("field_{field_index}"),
                codec.scalar_type,
                vec![MsgFieldLevel::new(grid)],
            ))
            .unwrap();
    }
    let field_names = (0..field_count)
        .map(|index| format!("field_{index}"))
        .collect::<Vec<_>>();
    let codecs = field_names
        .iter()
        .map(|field| MsgFieldCodec {
            field: field.clone(),
            selection: MsgChunkCodecSelection::Forced(codec.codec),
            error_tolerance: codec.tolerance.map(|absolute| MsgErrorTolerance {
                absolute,
                relative_percent: 0.0,
            }),
        })
        .collect();
    let mut bytes = Vec::new();
    write_msg(&dataset, &MsgWriteOptions { codecs }, &mut bytes).unwrap();
    let scalar_width = match codec.scalar_type {
        MsgScalarType::F32 => 4,
        MsgScalarType::F64 => 8,
    };
    Fixture {
        bytes,
        decoded_bytes: (point_count * field_count * scalar_width) as u64,
        field_names,
    }
}

fn generated_values(
    point_count: usize,
    extent: UVec3,
    field_index: usize,
) -> impl Iterator<Item = f64> {
    (0..point_count).map(move |index| {
        let x = index % extent.x as usize;
        let y = (index / extent.x as usize) % extent.y as usize;
        let z = index / (extent.x as usize * extent.y as usize);
        let noise = ((index.wrapping_mul(7919) + field_index * 101) % 1000) as f64 / 5000.0;
        10.0 + 1.25 * x as f64 - 0.5 * y as f64 + 0.125 * z as f64 + noise
    })
}

fn dense_f32() -> CodecCase {
    CodecCase {
        label: "dense_f32",
        scalar_type: MsgScalarType::F32,
        codec: MsgChunkCodec::DenseF32,
        tolerance: None,
    }
}

fn dense_f64() -> CodecCase {
    CodecCase {
        label: "dense_f64",
        scalar_type: MsgScalarType::F64,
        codec: MsgChunkCodec::DenseF64,
        tolerance: None,
    }
}

fn report_allocations(label: &str, mut operation: impl FnMut()) {
    operation();
    let count_before = ALLOCATION_COUNT.load(Ordering::Relaxed);
    let bytes_before = ALLOCATED_BYTES.load(Ordering::Relaxed);
    operation();
    let count = ALLOCATION_COUNT.load(Ordering::Relaxed) - count_before;
    let bytes = ALLOCATED_BYTES.load(Ordering::Relaxed) - bytes_before;
    eprintln!("allocation census {label}: {count} allocations, {bytes} bytes");
}

criterion_group!(benches, archive_benchmarks);
criterion_main!(benches);
