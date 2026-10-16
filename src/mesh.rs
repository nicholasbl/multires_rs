//! Triangle meshes produced and consumed by MSG algorithms.

use anyhow::{Context, Result, ensure};

/// Triangle mesh with up to two per-vertex scalar sampling channels.
#[derive(Debug, Default)]
pub struct Mesh3D {
    pub positions: Vec<[f32; 3]>,
    pub uv: Vec<[f32; 2]>,
    pub indices: Vec<[u32; 3]>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RemoveDegenerateTrianglesOptions {
    /// Minimum triangle area to keep. Repeated-index faces are always removed.
    pub area_epsilon: f32,
}

/// Remove repeated-index and near-zero-area triangles in place.
pub fn remove_degenerate_triangles(
    mesh: &mut Mesh3D,
    options: RemoveDegenerateTrianglesOptions,
) -> Result<usize> {
    ensure!(
        options.area_epsilon.is_finite() && options.area_epsilon >= 0.0,
        "area_epsilon must be finite and non-negative"
    );
    let threshold = options.area_epsilon * options.area_epsilon;
    let mut kept = Vec::with_capacity(mesh.indices.len());
    for (face_index, &face) in mesh.indices.iter().enumerate() {
        if face[0] == face[1] || face[0] == face[2] || face[1] == face[2] {
            continue;
        }
        let a = position(mesh, face_index, face[0])?;
        let b = position(mesh, face_index, face[1])?;
        let c = position(mesh, face_index, face[2])?;
        if triangle_area_squared(a, b, c) > threshold {
            kept.push(face);
        }
    }
    let removed = mesh.indices.len() - kept.len();
    mesh.indices = kept;
    Ok(removed)
}

fn position(mesh: &Mesh3D, face: usize, vertex: u32) -> Result<[f32; 3]> {
    let vertex = usize::try_from(vertex).context("mesh index does not fit in usize")?;
    let value = *mesh
        .positions
        .get(vertex)
        .with_context(|| format!("mesh face {face} index {vertex} is out of bounds"))?;
    ensure!(
        value.iter().all(|value| value.is_finite()),
        "non-finite mesh position"
    );
    Ok(value)
}

fn triangle_area_squared(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let cross = [
        ab[1].mul_add(ac[2], -ab[2] * ac[1]),
        ab[2].mul_add(ac[0], -ab[0] * ac[2]),
        ab[0].mul_add(ac[1], -ab[1] * ac[0]),
    ];
    0.25 * cross[0].mul_add(cross[0], cross[1].mul_add(cross[1], cross[2] * cross[2]))
}
