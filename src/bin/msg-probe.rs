use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use multires_rs::{MappedMsgArchive, MsgScalarType};

#[derive(Debug, Parser)]
#[command(about = "Inspect an MSG archive without decoding its scalar fields")]
struct Args {
    archive: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let size = fs::metadata(&args.archive)
        .with_context(|| format!("reading metadata for {}", args.archive.display()))?
        .len();
    let mapped = MappedMsgArchive::open(&args.archive)?;
    let archive = mapped.view()?;
    let levels = archive.levels().collect::<Vec<_>>();
    let fields = archive.fields().collect::<Vec<_>>();
    let dense_bytes = levels.iter().fold(0_u128, |total, level| {
        let points = level
            .point_extent
            .iter()
            .map(|&value| u128::from(value))
            .product::<u128>();
        total
            + fields
                .iter()
                .map(|field| {
                    points
                        * match field.scalar_type {
                            MsgScalarType::F32 => 4,
                            MsgScalarType::F64 => 8,
                        }
                })
                .sum::<u128>()
    });

    println!("archive: {}", args.archive.display());
    println!("size: {} bytes", size);
    println!("simulation time: {}", archive.simulation_time());
    println!("levels: {}", levels.len());
    println!("fields: {}", fields.len());
    println!("dense scalar payload estimate: {dense_bytes} bytes");
    if size != 0 {
        println!(
            "archive / dense payload: {:.4}% (dense/archive: {:.2}x)",
            size as f64 / dense_bytes.max(1) as f64 * 100.0,
            dense_bytes as f64 / size as f64
        );
    }
    println!(
        "grid-to-world columns: {:?}",
        archive.grid_to_world_columns()
    );
    for (index, level) in levels.iter().enumerate() {
        println!(
            "  level {index}: origin={:?}, points={:?}, scale={:?}, refinement={:?}",
            level.index_origin, level.point_extent, level.scale, level.refinement_ratio_from_coarse
        );
    }
    for field in fields {
        match field.error_tolerance {
            Some(error) => println!(
                "  field {:?}: {:?}, error absolute={:.7e}, relative-percent={:.7}%",
                field.name, field.scalar_type, error.absolute, error.relative_percent
            ),
            None => println!(
                "  field {:?}: {:?}, lossless",
                field.name, field.scalar_type
            ),
        }
    }
    Ok(())
}
