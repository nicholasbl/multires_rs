use anyhow::{Context, Result, ensure};
use glam::{U16Vec2, UVec3};

use crate::scalar::MsgScalarAccessor;

use super::SampleRange;

/// Interpolate auxiliary quantities along the same edge and with the same
/// parameter used to place an isosurface vertex.
pub(super) fn sampled_values_on_edge(
    sampled_quantities: &mut [MsgScalarAccessor<'_>],
    start: UVec3,
    end: UVec3,
    t: f64,
    ranges: &[SampleRange],
) -> Result<U16Vec2> {
    encode_sampled_values(sampled_quantities, ranges, |grid| {
        let start_value = grid
            .get(start)
            .with_context(|| format!("sample value is absent at edge start {start:?}"))?;
        let end_value = grid
            .get(end)
            .with_context(|| format!("sample value is absent at edge end {end:?}"))?;
        Ok((end_value - start_value).mul_add(t, start_value))
    })
}

/// Blend auxiliary quantities using the weights of an MC33 interior vertex.
pub(super) fn sampled_values_weighted(
    sampled_quantities: &mut [MsgScalarAccessor<'_>],
    positions: &[UVec3; 8],
    weights: &[f64; 8],
    ranges: &[SampleRange],
) -> Result<U16Vec2> {
    let weight_sum = weights.iter().sum::<f64>();
    ensure!(weight_sum > 0.0, "MC33 interior vertex has zero weight");
    encode_sampled_values(sampled_quantities, ranges, |grid| {
        let mut value = 0.0;
        for (&position, &weight) in positions.iter().zip(weights) {
            let sample = grid.get(position).with_context(|| {
                format!("sample value is absent at MC33 interior vertex corner {position:?}")
            })?;
            value += sample * weight;
        }
        Ok(value / weight_sum)
    })
}

fn encode_sampled_values<F>(
    sampled_quantities: &mut [MsgScalarAccessor<'_>],
    ranges: &[SampleRange],
    mut value: F,
) -> Result<U16Vec2>
where
    F: FnMut(&mut MsgScalarAccessor<'_>) -> Result<f64>,
{
    ensure!(
        sampled_quantities.len() == ranges.len(),
        "sample grid and range counts differ"
    );
    ensure!(ranges.len() <= 2, "at most two UV axes are supported");

    let mut encoded = [0_u16; 2];
    for (axis, (grid, range)) in sampled_quantities.iter_mut().zip(ranges).enumerate() {
        encoded[axis] = encode_unorm16(value(grid)?, *range);
    }
    Ok(U16Vec2::new(encoded[0], encoded[1]))
}

fn encode_unorm16(value: f64, range: SampleRange) -> u16 {
    let normalized = ((value - range.min) / (range.max - range.min)).clamp(0.0, 1.0);
    (normalized * f64::from(u16::MAX)).round() as u16
}
