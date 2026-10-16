use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use multires_rs::{
    MappedMsgArchive, MsgChunkCodec, MsgChunkCodecSelection, MsgErrorTolerance, MsgFieldCodec,
    MsgWriteOptions, transcode_msg,
};
use serde::Deserialize;

#[derive(Debug, Parser)]
#[command(
    about = "Rewrite an MSG archive with different field chunk codecs",
    after_help = "With --config, positional paths override both paths in the TOML file."
)]
struct Cli {
    #[arg(long, value_name = "FILE")]
    config: Option<PathBuf>,
    input: Option<PathBuf>,
    output: Option<PathBuf>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    codec: Option<String>,
    error: Option<ErrorConfig>,
    #[serde(default)]
    fields: Vec<FieldConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FieldConfig {
    name: String,
    codec: String,
    error: Option<ErrorConfig>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct ErrorConfig {
    absolute: Option<f64>,
    relative_percent: Option<f64>,
}

fn main() -> Result<()> {
    let (input, output, config) = resolve(Cli::parse())?;
    reject_same_file(&input, &output)?;
    let mapped = MappedMsgArchive::open(&input)?;
    let view = mapped.view()?;
    let codecs = codec_policies(&view, &config)?;
    let input_size = mapped.bytes().len() as u64;
    let mut writer = BufWriter::new(
        File::create(&output).with_context(|| format!("creating {}", output.display()))?,
    );
    transcode_msg(mapped.bytes(), &MsgWriteOptions { codecs }, &mut writer)?;
    writer.flush()?;
    let output_size = fs::metadata(&output)?.len();
    eprintln!(
        "size: {input_size} -> {output_size} bytes ({:.2}% of source, {:.2}x source/output)",
        output_size as f64 / input_size.max(1) as f64 * 100.0,
        input_size as f64 / output_size.max(1) as f64
    );
    Ok(())
}

fn resolve(cli: Cli) -> Result<(PathBuf, PathBuf, Config)> {
    if let Some(path) = cli.config {
        let source = fs::read_to_string(&path)
            .with_context(|| format!("reading config {}", path.display()))?;
        let mut config: Config = toml::from_str(&source)
            .with_context(|| format!("parsing config {}", path.display()))?;
        let paths = match (cli.input, cli.output) {
            (Some(input), Some(output)) => (input, output),
            (None, None) => (
                config.input.take().context("config has no input")?,
                config.output.take().context("config has no output")?,
            ),
            _ => bail!("config path overrides require both INPUT and OUTPUT"),
        };
        return Ok((paths.0, paths.1, config));
    }
    Ok((
        cli.input.context("missing INPUT archive")?,
        cli.output.context("missing OUTPUT archive")?,
        Config::default(),
    ))
}

fn codec_policies(
    view: &multires_rs::MsgArchive<'_>,
    config: &Config,
) -> Result<Vec<MsgFieldCodec>> {
    let default = match (&config.codec, config.error) {
        (None, None) => None,
        (None, Some(_)) => bail!("a top-level error requires a top-level codec"),
        (Some(codec), error) => Some((parse_codec(codec)?, resolve_error(error)?)),
    };
    let mut policies = view
        .field_names()
        .filter_map(|name| {
            default.map(|(selection, error_tolerance)| MsgFieldCodec {
                field: name.to_owned(),
                selection,
                error_tolerance,
            })
        })
        .collect::<Vec<_>>();
    for (index, field) in config.fields.iter().enumerate() {
        ensure!(
            !config.fields[..index]
                .iter()
                .any(|earlier| earlier.name == field.name),
            "field {:?} is configured more than once",
            field.name
        );
        ensure!(
            view.field_names().any(|name| name == field.name),
            "archive has no field {:?}",
            field.name
        );
        let policy = MsgFieldCodec {
            field: field.name.clone(),
            selection: parse_codec(&field.codec)?,
            error_tolerance: resolve_error(field.error)?,
        };
        if let Some(existing) = policies.iter_mut().find(|item| item.field == field.name) {
            *existing = policy;
        } else {
            policies.push(policy);
        }
    }
    Ok(policies)
}

fn resolve_error(error: Option<ErrorConfig>) -> Result<Option<MsgErrorTolerance>> {
    error
        .map(|error| {
            ensure!(
                error.absolute.is_some() || error.relative_percent.is_some(),
                "error must specify absolute, relative_percent, or both"
            );
            Ok(MsgErrorTolerance {
                absolute: error.absolute.unwrap_or(0.0),
                relative_percent: error.relative_percent.unwrap_or(0.0),
            })
        })
        .transpose()
}

fn parse_codec(value: &str) -> Result<MsgChunkCodecSelection> {
    Ok(match value {
        "auto:lossless" => MsgChunkCodecSelection::AutoLossless,
        "auto:smallest" => MsgChunkCodecSelection::AutoSmallest,
        "uniform:f64" => MsgChunkCodecSelection::Forced(MsgChunkCodec::UniformF64),
        "dense:f32" => MsgChunkCodecSelection::Forced(MsgChunkCodec::DenseF32),
        "dense:f64" => MsgChunkCodecSelection::Forced(MsgChunkCodec::DenseF64),
        "quantized:u8" => MsgChunkCodecSelection::Forced(MsgChunkCodec::QuantizedU8),
        "quantized:u16" => MsgChunkCodecSelection::Forced(MsgChunkCodec::QuantizedU16),
        "quantized:u32" => MsgChunkCodecSelection::Forced(MsgChunkCodec::QuantizedU32),
        "affine:f64" => MsgChunkCodecSelection::Forced(MsgChunkCodec::AffineF64),
        "affine-residual:u8" => MsgChunkCodecSelection::Forced(MsgChunkCodec::AffineResidualU8),
        "affine-residual:u16" => MsgChunkCodecSelection::Forced(MsgChunkCodec::AffineResidualU16),
        "affine-residual:u32" => MsgChunkCodecSelection::Forced(MsgChunkCodec::AffineResidualU32),
        _ => bail!("unsupported MSG codec {value:?}; expected family:value"),
    })
}

fn reject_same_file(input: &Path, output: &Path) -> Result<()> {
    if output.exists() {
        ensure!(
            fs::canonicalize(input)? != fs::canonicalize(output)?,
            "input and output must be different files"
        );
    }
    Ok(())
}
