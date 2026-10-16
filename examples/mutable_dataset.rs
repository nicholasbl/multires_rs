use std::{env, fs::File, path::PathBuf};

use anyhow::{Context, Result};
use glam::{DMat4, DVec3, IVec3, UVec3};
use multires_rs::{
    MsgDataset, MsgField, MsgFieldLevel, MsgGeometry, MsgLevel, MsgScalarGrid, MsgScalarType,
    MsgWriteOptions, write_msg,
};

fn main() -> Result<()> {
    let output = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("example.msg"));

    let levels = vec![
        MsgLevel {
            index_origin: IVec3::ZERO,
            point_extent: UVec3::splat(8),
            scale: DVec3::splat(2.0),
            refinement_ratio_from_coarse: None,
        },
        MsgLevel {
            index_origin: IVec3::ZERO,
            point_extent: UVec3::splat(16),
            scale: DVec3::ONE,
            refinement_ratio_from_coarse: Some(UVec3::splat(2)),
        },
    ];
    let mut dataset = MsgDataset::new(1.25, MsgGeometry::point_centered(DMat4::IDENTITY)?, levels)?;

    let mut coarse = MsgScalarGrid::new(UVec3::splat(8), MsgScalarType::F32);
    coarse.set(UVec3::new(2, 3, 4), 1.5);
    let mut fine = MsgScalarGrid::new(UVec3::splat(16), MsgScalarType::F32);
    fine.set(UVec3::new(4, 6, 8), 1.75);
    fine.set(UVec3::new(5, 6, 8), 1.8);

    dataset.add_field(MsgField::new(
        "density",
        MsgScalarType::F32,
        vec![MsgFieldLevel::new(coarse), MsgFieldLevel::new(fine)],
    ))?;

    // Build optional field-local acceleration after all mutations are done.
    dataset.rebuild_acceleration()?;

    let mut file =
        File::create(&output).with_context(|| format!("creating {}", output.display()))?;
    write_msg(&dataset, &MsgWriteOptions::default(), &mut file)?;
    println!("wrote {}", output.display());
    Ok(())
}
