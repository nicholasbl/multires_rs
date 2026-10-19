# multires_rs

`multires_rs` is an implementation of multiresolution sparse grids
(MSG (I was hungry when I wrote this)), including the
`.msg` archive format, and algorithms that operate on MSG data. Source-format importers such as
`amrex_rs` and `msg_vti` depend on this crate.

An MSG file currently contains one timestep. Levels are ordered coarse to fine,
all fields are finite scalar point data, and every field has its own unique sparse
topology. Missing points mean “not present,” not zero. If you wish to store a “background”
value, you must do so explicitly.

The API has two complementary entry points:

- `MsgDataset`
  is an owned, mutable representation for creating or editing data and running
  algorithms.
- `MappedMsgArchive`
  owns a read-only memory map. Its borrowed `MsgArchive` view validates the
  file, exposes metadata without decoding scalar blocks, and can selectively
  decode fields into an owned `MsgDataset`.

## Mutable, in-memory datasets

Build a `MsgDataset` from shared level geometry, then add independently sparse
fields. A field must have one `MsgFieldLevel` per dataset level, and each scalar
grid's bounds and precision must match its level and field declaration.

```rust
use glam::{DMat4, DVec3, IVec3, UVec3};
use multires_rs::{
    MsgDataset, MsgField, MsgFieldLevel, MsgGeometry, MsgLevel, MsgScalarGrid,
    MsgScalarType, MsgWriteOptions, write_msg,
};

fn main() -> anyhow::Result<()> {
    let extent = UVec3::new(64, 64, 64);
    let mut dataset = MsgDataset::new(
        1.25,
        MsgGeometry::point_centered(DMat4::IDENTITY)?,
        vec![MsgLevel {
            index_origin: IVec3::ZERO,
            point_extent: extent,
            scale: DVec3::splat(0.5),
            refinement_ratio_from_coarse: None,
        }],
    )?;

    let mut density = MsgScalarGrid::new(extent, MsgScalarType::F32);
    density.set(UVec3::new(10, 20, 30), 1.5);
    dataset.add_field(MsgField::new(
        "density",
        MsgScalarType::F32,
        vec![MsgFieldLevel::new(density)],
    ))?;

    // Optional for in-memory isosurface extraction. Archive writing always
    // derives fresh acceleration from the current samples.
    dataset.rebuild_acceleration()?;

    let mut file = std::fs::File::create("example.msg")?;
    write_msg(&dataset, &MsgWriteOptions::default(), &mut file)?;
    Ok(())
}
```

Use `field_mut()` followed by `levels_mut()` and `values_mut()` to edit an
existing field. Calling `values_mut()` invalidates that level's derived
acceleration, even if no value is ultimately changed; call
`rebuild_acceleration()` after the edits are complete. `set` returns `false`
for an out-of-bounds point, and `clear` returns whether a present point was
removed. Missing points remain absent rather than reading as zero.

See [`examples/mutable_dataset.rs`](examples/mutable_dataset.rs) for a complete
two-level example. Run it with:

```console
cargo run --example mutable_dataset -- example.msg
```

## Read-only, disk-backed archives

`MappedMsgArchive::open` maps and validates an archive without eagerly
decoding its field data. The `MsgArchive` returned by `view()` borrows the map,
so it cannot outlive its `MappedMsgArchive` owner. Metadata accessors remain
cheap. Field-level `get()` and `fill_chunk()` reconstruct data on demand;
`decode()` and `decode_fields()` are the explicit boundary where complete
scalar fields become an owned, mutable `MsgDataset`.

```rust
use glam::UVec3;
use multires_rs::{MappedMsgArchive, MsgChunkBuffer};

fn main() -> anyhow::Result<()> {
    let mapped = MappedMsgArchive::open("example.msg")?;
    let archive = mapped.view()?;

    println!("time: {}", archive.simulation_time());
    for field in archive.fields() {
        println!("{}: {:?}", field.name, field.scalar_type);
    }

    let level = archive.field("density").unwrap().level(0)?;
    println!("sample: {:?}", level.get(UVec3::new(10, 20, 30))?);

    // Allocate once, then reuse this buffer for subsequent chunk fills.
    let mut chunk = MsgChunkBuffer::new(level.scalar_type());
    level.fill_chunk(UVec3::ZERO, &mut chunk)?;

    // Decode only the data needed by this operation. The result is owned and
    // may be mutated independently of the mapping.
    let mut density = archive.decode_fields(&["density"])?;
    density
        .field_mut("density")
        .unwrap()
        .levels_mut()[0]
        .values_mut()
        .set(UVec3::new(10, 20, 30), 2.0);
    Ok(())
}
```

For bytes already in memory, use `view_msg(&bytes)` for the same borrowed view,
`read_msg(&bytes)` to decode everything, or `read_msg_fields(&bytes, names)` to
decode a selection directly. A selective decode preserves dataset geometry and
levels but contains only the requested fields, in request order. Unknown or
duplicate requested names are errors.

See [`examples/read_only_archive.rs`](examples/read_only_archive.rs). Run it
with an optional field name to demonstrate selective decoding:

```console
cargo run --example read_only_archive -- example.msg density
```

## Geometry

The dataset stores one affine, invertible `f64` 4×4 grid-to-world matrix. Each
level stores an integer index origin, point extent, direction-wise scale, and
direction-wise integer refinement ratio. Converted cell-center data uses a
dataset-wide `[0.5, 0.5, 0.5]` sample offset; native point data normally uses
zero.

Matrices are stored column-major as 16 little-endian `f64` values. Each column
is concatenated in `x, y, z, w` order and positions use column-vector
multiplication:

```text
world = grid_to_world * [x, y, z, 1]
```

## Archive layout

Version 1 uses:

```text
u32 magic "MSG\0"       little-endian
u32 format version      little-endian
8-byte-aligned blocks   coarse levels before fine levels
rkyv metadata header
u64 header offset       little-endian footer
```

Scalar directories, irregular occupancy masks, scalar payloads, acceleration
directories, and acceleration masks are independently addressable. A full
physical 32³ chunk stores its occupancy as a directory tag; only irregular
masks consume a 4 KiB raw-mask record. This keeps mapped level lookup from
opening scalar payload pages. `MsgArchive` provides validated borrowed metadata
and selective field decoding; `MappedMsgArchive` owns a read-only memory
mapping.

## Chunk codecs

- `uniform:f64` stores one value for all present points in a chunk.
- `dense:f32` and `dense:f64` store every chunk slot directly.
- `quantized:u8`, `quantized:u16`, and `quantized:u32` uniformly quantize a
  chunk-local value interval while the API continues to expose physical values.
- `affine:f64` fits a 3D affine function to a chunk and stores its four `f64`
  coefficients.
- `affine-residual:u8`, `affine-residual:u16`, and
  `affine-residual:u32` fit an affine function and quantize the remaining
  residuals.
- `auto:lossless` uses structural uniform chunks and the logical field's dense
  codec.
- `auto:smallest` chooses the smallest candidate satisfying the requested
  mixed absolute and pointwise-relative error bound.

Codec identifiers and versions are integers so future readers can inspect an
archive containing an unknown codec without treating the container enum as
closed.

## Tools

- `msg-probe`: inspect metadata and compare archive size with a dense payload.
- `msg-transcode`: rewrite fields with different codec policies.
- `msg-isosurface`: extract an MC33 surface into Wavefront OBJ.

The library additionally exposes point-based slice extraction, mesh relaxation,
and parallel meshoptimizer-backed decimation. For the latter, a C++ compiler
may be required.

## Benchmarks

```console
cargo bench --bench archive_read
```

Pass a Criterion filter to run one case, for example:

```console
cargo bench --bench archive_read -- archive_decode_scaling_dense_f32/64_chunks
```

Use `--test` for a fast smoke run that executes every benchmark once:

```console
cargo bench --bench archive_read -- --test
```
