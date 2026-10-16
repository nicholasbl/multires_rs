use std::{env, path::PathBuf};

use anyhow::{Context, Result};
use multires_rs::MappedMsgArchive;

fn main() -> Result<()> {
    let path = env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .context("usage: read_only_archive <archive.msg> [field]")?;
    let requested_field = env::args().nth(2);

    let mapped = MappedMsgArchive::open(&path)?;
    let archive = mapped.view()?;

    println!("time: {}", archive.simulation_time());
    println!("levels: {}", archive.level_count());
    for field in archive.fields() {
        println!("field: {} ({:?})", field.name, field.scalar_type);
    }

    let dataset = match requested_field.as_deref() {
        Some(name) => archive.decode_fields(&[name])?,
        None => archive.decode()?,
    };
    println!("decoded {} field(s)", dataset.fields().len());
    Ok(())
}
