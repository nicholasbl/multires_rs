//! Topology-correct Marching Cubes 33 extraction.
//!
//! The case resolver and lookup tables follow Lewiner et al.'s MC33
//! implementation as distributed by scikit-image 0.25.2 under BSD-3-Clause.

mod tables;

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, bail};
use glam::{DVec3, U16Vec2, UVec3, Vec3};

use crate::scalar::MsgScalarAccessor;
use crate::sparse::Aabb3u;

use super::sampling::{sampled_values_on_edge, sampled_values_weighted};
use super::{Mesh3D, SampleRange};
use crate::point_grid::PointGridLevel;
use tables::*;

const EPSILON: f64 = f32::EPSILON as f64;

// MC33 uses the conventional cyclic corner ordering on each z plane.
const CUBE_CORNERS: [UVec3; 8] = [
    UVec3::new(0, 0, 0),
    UVec3::new(1, 0, 0),
    UVec3::new(1, 1, 0),
    UVec3::new(0, 1, 0),
    UVec3::new(0, 0, 1),
    UVec3::new(1, 0, 1),
    UVec3::new(1, 1, 1),
    UVec3::new(0, 1, 1),
];

const EDGE_CORNERS: [[usize; 2]; 12] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum VertexKey {
    Edge(UVec3, UVec3),
    Interior(UVec3),
}

#[derive(Clone, Copy)]
struct Tiling {
    table: Lut,
    config: usize,
    subconfig: Option<usize>,
    triangle_count: usize,
}

impl Tiling {
    fn direct(table: Lut, config: usize, triangle_count: usize) -> Self {
        Self {
            table,
            config,
            subconfig: None,
            triangle_count,
        }
    }

    fn nested(table: Lut, config: usize, subconfig: usize, triangle_count: usize) -> Self {
        Self {
            table,
            config,
            subconfig: Some(subconfig),
            triangle_count,
        }
    }

    fn edge(self, index: usize) -> u8 {
        let edge = match self.subconfig {
            Some(subconfig) => self.table.get3(self.config, subconfig, index),
            None => self.table.get2(self.config, index),
        };
        u8::try_from(edge).expect("MC33 tiling contains a negative edge index")
    }
}

struct Mc33Mesher<'a> {
    level: &'a PointGridLevel<'a>,
    samples: MsgScalarAccessor<'a>,
    sampled_quantities: Vec<MsgScalarAccessor<'a>>,
    ranges: &'a [SampleRange],
    isovalue: f64,
    mesh: &'a mut Mesh3D,
    vertex_keys: &'a mut Vec<VertexKey>,
    vertex_ids: HashMap<VertexKey, u32>,
    face_ids: HashSet<[u32; 3]>,
}

impl Mc33Mesher<'_> {
    fn march_cube(&mut self, anchor: UVec3) -> Result<()> {
        let corners = CUBE_CORNERS.map(|offset| anchor + offset);
        let mut values = [0.0_f64; 8];
        let mut case_index = 0_usize;
        for (index, (&corner, value)) in corners.iter().zip(&mut values).enumerate() {
            *value = self
                .samples
                .get(corner)
                .with_context(|| format!("surface sample is absent at {corner:?}"))?
                - self.isovalue;
            if *value > 0.0 {
                case_index |= 1 << index;
            }
        }

        let Some(tiling) = resolve_tiling(case_index, &values)? else {
            return Ok(());
        };
        let quantity_gradient = self
            .level
            .geometry
            .gradient_to_world(cube_quantity_gradient(&values, self.level.scale))
            .as_vec3();
        for triangle in 0..tiling.triangle_count {
            let a = self.vertex(anchor, &corners, &values, tiling.edge(triangle * 3))?;
            let b = self.vertex(anchor, &corners, &values, tiling.edge(triangle * 3 + 1))?;
            let c = self.vertex(anchor, &corners, &values, tiling.edge(triangle * 3 + 2))?;
            self.emit_face(a, b, c, quantity_gradient);
        }
        Ok(())
    }

    fn vertex(
        &mut self,
        anchor: UVec3,
        corners: &[UVec3; 8],
        values: &[f64; 8],
        edge: u8,
    ) -> Result<u32> {
        if edge == 12 {
            return self.interior_vertex(anchor, corners, values);
        }
        let [first_index, second_index] = EDGE_CORNERS
            .get(usize::from(edge))
            .copied()
            .context("MC33 tiling edge is out of range")?;
        let first = corners[first_index];
        let second = corners[second_index];
        let first_value = values[first_index] + self.isovalue;
        let second_value = values[second_index] + self.isovalue;
        let (start, start_value, end, end_value) = if grid_point_le(first, second) {
            (first, first_value, second, second_value)
        } else {
            (second, second_value, first, first_value)
        };
        let key = VertexKey::Edge(start, end);
        if let Some(&id) = self.vertex_ids.get(&key) {
            return Ok(id);
        }

        let denominator = end_value - start_value;
        let t = if denominator == 0.0 {
            0.5
        } else {
            ((self.isovalue - start_value) / denominator).clamp(0.0, 1.0)
        };
        let position = self.grid_position(start.as_dvec3().lerp(end.as_dvec3(), t));
        let sampled_values =
            sampled_values_on_edge(&mut self.sampled_quantities, start, end, t, self.ranges)?;
        self.insert_vertex(key, position, sampled_values)
    }

    fn interior_vertex(
        &mut self,
        anchor: UVec3,
        corners: &[UVec3; 8],
        values: &[f64; 8],
    ) -> Result<u32> {
        let key = VertexKey::Interior(anchor);
        if let Some(&id) = self.vertex_ids.get(&key) {
            return Ok(id);
        }

        let weights = values.map(|value| 1.0 / (EPSILON + value.abs()));
        let weight_sum = weights.iter().sum::<f64>();
        let mut grid_position = DVec3::ZERO;
        for (&corner, &weight) in corners.iter().zip(&weights) {
            grid_position += corner.as_dvec3() * weight;
        }
        grid_position /= weight_sum;
        let sampled_values =
            sampled_values_weighted(&mut self.sampled_quantities, corners, &weights, self.ranges)?;
        self.insert_vertex(key, self.grid_position(grid_position), sampled_values)
    }

    fn insert_vertex(
        &mut self,
        key: VertexKey,
        position: Vec3,
        sampled_values: U16Vec2,
    ) -> Result<u32> {
        let id =
            u32::try_from(self.mesh.positions.len()).context("mesh vertex count exceeds u32")?;
        self.mesh.positions.push(position.to_array());
        self.mesh.uv.push(sampled_values_to_uv(sampled_values));
        self.vertex_keys.push(key);
        self.vertex_ids.insert(key, id);
        Ok(id)
    }

    fn grid_position(&self, position: DVec3) -> glam::Vec3 {
        self.level
            .geometry
            .point_position_to_world(self.level.index_origin, self.level.scale, position)
            .as_vec3()
    }

    fn emit_face(&mut self, a: u32, mut b: u32, mut c: u32, quantity_gradient: Vec3) {
        if a == b || b == c || c == a {
            return;
        }
        let pa = Vec3::from_array(self.mesh.positions[a as usize]);
        let pb = Vec3::from_array(self.mesh.positions[b as usize]);
        let pc = Vec3::from_array(self.mesh.positions[c as usize]);
        let normal = (pb - pa).cross(pc - pa);
        if normal.dot(quantity_gradient) > 0.0 {
            std::mem::swap(&mut b, &mut c);
        }
        if (pb - pa).cross(pc - pa).length_squared() == 0.0 {
            return;
        }
        let mut key = [a, b, c];
        key.sort_unstable();
        if self.face_ids.insert(key) {
            self.mesh.indices.push([a, b, c]);
        }
    }
}

fn sampled_values_to_uv(sampled_values: U16Vec2) -> [f32; 2] {
    let scale = 1.0 / f32::from(u16::MAX);
    [
        f32::from(sampled_values.x) * scale,
        f32::from(sampled_values.y) * scale,
    ]
}

fn cube_quantity_gradient(values: &[f64; 8], cell_size: DVec3) -> DVec3 {
    let gx = ((values[1] + values[2] + values[5] + values[6])
        - (values[0] + values[3] + values[4] + values[7]))
        / (4.0 * cell_size.x);
    let gy = ((values[2] + values[3] + values[6] + values[7])
        - (values[0] + values[1] + values[4] + values[5]))
        / (4.0 * cell_size.y);
    let gz = ((values[4] + values[5] + values[6] + values[7])
        - (values[0] + values[1] + values[2] + values[3]))
        / (4.0 * cell_size.z);
    DVec3::new(gx, gy, gz)
}

pub(super) fn mesh_level_aabb(
    level: &PointGridLevel<'_>,
    active_aabb: Aabb3u,
    ranges: &[SampleRange],
    isovalue: f64,
    mesh: &mut Mesh3D,
    vertex_keys: &mut Vec<VertexKey>,
) -> Result<()> {
    let mut mesher = Mc33Mesher {
        level,
        samples: level.samples.accessor(),
        sampled_quantities: level
            .sampled_quantities
            .iter()
            .map(|grid| grid.accessor())
            .collect(),
        ranges,
        isovalue,
        mesh,
        vertex_keys,
        vertex_ids: HashMap::new(),
        face_ids: HashSet::new(),
    };
    let mut error = None;
    level
        .active_cubes
        .for_each_present_in_aabb(active_aabb, |anchor, ()| {
            if error.is_none() {
                error = mesher.march_cube(anchor).err();
            }
        });
    match error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

fn resolve_tiling(case_index: usize, values: &[f64; 8]) -> Result<Option<Tiling>> {
    let case = CASES.get2(case_index, 0);
    if case <= 0 {
        return Ok(None);
    }
    let case = usize::try_from(case).expect("positive MC33 case");
    let config = usize::try_from(CASES.get2(case_index, 1)).expect("MC33 config is nonnegative");
    let face = |table: Lut, column: usize| test_face(values, table.get2(config, column));
    let internal =
        |subconfig: usize, sign: i8| test_internal(values, case, config, subconfig, sign);

    let tiling = match case {
        1 => Tiling::direct(TILING1, config, 1),
        2 => Tiling::direct(TILING2, config, 2),
        3 if test_face(values, TEST3.get1(config)) => Tiling::direct(TILING3_2, config, 4),
        3 => Tiling::direct(TILING3_1, config, 2),
        4 if internal(0, TEST4.get1(config)) => Tiling::direct(TILING4_1, config, 2),
        4 => Tiling::direct(TILING4_2, config, 6),
        5 => Tiling::direct(TILING5, config, 3),
        6 if face(TEST6, 0) => Tiling::direct(TILING6_2, config, 5),
        6 if internal(0, TEST6.get2(config, 1)) => Tiling::direct(TILING6_1_1, config, 3),
        6 => Tiling::direct(TILING6_1_2, config, 9),
        7 => {
            let subconfig = usize::from(face(TEST7, 0))
                | (usize::from(face(TEST7, 1)) << 1)
                | (usize::from(face(TEST7, 2)) << 2);
            match subconfig {
                0 => Tiling::direct(TILING7_1, config, 3),
                1..=2 => Tiling::nested(TILING7_2, config, subconfig - 1, 5),
                3 => Tiling::nested(TILING7_3, config, 0, 9),
                4 => Tiling::nested(TILING7_2, config, 2, 5),
                5..=6 => Tiling::nested(TILING7_3, config, subconfig - 4, 9),
                7 if internal(7, TEST7.get2(config, 3)) => Tiling::direct(TILING7_4_2, config, 9),
                7 => Tiling::direct(TILING7_4_1, config, 5),
                _ => unreachable!(),
            }
        }
        8 => Tiling::direct(TILING8, config, 2),
        9 => Tiling::direct(TILING9, config, 4),
        10 if face(TEST10, 0) && face(TEST10, 1) => Tiling::direct(TILING10_1_1_, config, 4),
        10 if face(TEST10, 0) => Tiling::direct(TILING10_2, config, 8),
        10 if face(TEST10, 1) => Tiling::direct(TILING10_2_, config, 8),
        10 if internal(0, TEST10.get2(config, 2)) => Tiling::direct(TILING10_1_1, config, 4),
        10 => Tiling::direct(TILING10_1_2, config, 8),
        11 => Tiling::direct(TILING11, config, 4),
        12 if face(TEST12, 0) && face(TEST12, 1) => Tiling::direct(TILING12_1_1_, config, 4),
        12 if face(TEST12, 0) => Tiling::direct(TILING12_2, config, 8),
        12 if face(TEST12, 1) => Tiling::direct(TILING12_2_, config, 8),
        12 if internal(0, TEST12.get2(config, 2)) => Tiling::direct(TILING12_1_1, config, 4),
        12 => Tiling::direct(TILING12_1_2, config, 8),
        13 => resolve_case13(values, config)?,
        14 => Tiling::direct(TILING14, config, 4),
        _ => bail!("invalid MC33 case {case}"),
    };
    Ok(Some(tiling))
}

fn resolve_case13(values: &[f64; 8], config: usize) -> Result<Tiling> {
    let mut face_bits = 0_usize;
    for column in 0..6 {
        if test_face(values, TEST13.get2(config, column)) {
            face_bits |= 1 << column;
        }
    }
    let subconfig = SUBCONFIG13.get1(face_bits);
    if subconfig < 0 {
        bail!("invalid MC33 case 13 face configuration {face_bits}");
    }
    let subconfig = usize::try_from(subconfig).expect("nonnegative case 13 subconfig");
    Ok(match subconfig {
        0 => Tiling::direct(TILING13_1, config, 4),
        1..=6 => Tiling::nested(TILING13_2, config, subconfig - 1, 6),
        7..=18 => Tiling::nested(TILING13_3, config, subconfig - 7, 10),
        19..=22 => Tiling::nested(TILING13_4, config, subconfig - 19, 12),
        23..=26 => {
            let nested = subconfig - 23;
            let sign = TEST13.get2(config, 6);
            if test_internal(values, 13, config, nested, sign) {
                Tiling::nested(TILING13_5_1, config, nested, 6)
            } else {
                Tiling::nested(TILING13_5_2, config, nested, 10)
            }
        }
        27..=38 => Tiling::nested(TILING13_3_, config, subconfig - 27, 10),
        39..=44 => Tiling::nested(TILING13_2_, config, subconfig - 39, 6),
        45 => Tiling::direct(TILING13_1_, config, 4),
        _ => bail!("invalid MC33 case 13 subconfiguration {subconfig}"),
    })
}

fn test_face(values: &[f64; 8], face: i8) -> bool {
    let [a, b, c, d] = match face.unsigned_abs() {
        1 => [0, 4, 5, 1],
        2 => [1, 5, 6, 2],
        3 => [2, 6, 7, 3],
        4 => [3, 7, 4, 0],
        5 => [0, 3, 2, 1],
        6 => [4, 7, 6, 5],
        invalid => panic!("invalid MC33 face {invalid}"),
    };
    let determinant = values[a] * values[c] - values[b] * values[d];
    if determinant.abs() < EPSILON {
        face >= 0
    } else {
        f64::from(face) * values[a] * determinant >= 0.0
    }
}

fn test_internal(
    values: &[f64; 8],
    case: usize,
    config: usize,
    subconfig: usize,
    sign: i8,
) -> bool {
    let (a_t, b_t, c_t, d_t) = if case == 4 || case == 10 {
        let a = (values[4] - values[0]) * (values[6] - values[2])
            - (values[7] - values[3]) * (values[5] - values[1]);
        let b = values[2] * (values[4] - values[0]) + values[0] * (values[6] - values[2])
            - values[1] * (values[7] - values[3])
            - values[3] * (values[5] - values[1]);
        let t = -b / (2.0 * a + EPSILON);
        if !(0.0..=1.0).contains(&t) {
            return sign > 0;
        }
        (
            values[0] + (values[4] - values[0]) * t,
            values[3] + (values[7] - values[3]) * t,
            values[2] + (values[6] - values[2]) * t,
            values[1] + (values[5] - values[1]) * t,
        )
    } else {
        let edge = match case {
            6 => TEST6.get2(config, 2),
            7 => TEST7.get2(config, 4),
            12 => TEST12.get2(config, 3),
            13 => TILING13_5_1.get3(config, subconfig, 0),
            _ => panic!("invalid ambiguous MC33 case {case}"),
        };
        let (start, end, b0, b1, c0, c1, d0, d1) = match edge {
            0 => (0, 1, 3, 2, 7, 6, 4, 5),
            1 => (1, 2, 0, 3, 4, 7, 5, 6),
            2 => (2, 3, 1, 0, 5, 4, 6, 7),
            3 => (3, 0, 2, 1, 6, 5, 7, 4),
            4 => (4, 5, 7, 6, 3, 2, 0, 1),
            5 => (5, 6, 4, 7, 0, 3, 1, 2),
            6 => (6, 7, 5, 4, 1, 0, 2, 3),
            7 => (7, 4, 6, 5, 2, 1, 3, 0),
            8 => (0, 4, 3, 7, 2, 6, 1, 5),
            9 => (1, 5, 0, 4, 3, 7, 2, 6),
            10 => (2, 6, 1, 5, 0, 4, 3, 7),
            11 => (3, 7, 2, 6, 1, 5, 0, 4),
            invalid => panic!("invalid MC33 reference edge {invalid}"),
        };
        let t = values[start] / (values[start] - values[end] + EPSILON);
        (
            0.0,
            values[b0] + (values[b1] - values[b0]) * t,
            values[c0] + (values[c1] - values[c0]) * t,
            values[d0] + (values[d1] - values[d0]) * t,
        )
    };

    let test = usize::from(a_t >= 0.0)
        | (usize::from(b_t >= 0.0) << 1)
        | (usize::from(c_t >= 0.0) << 2)
        | (usize::from(d_t >= 0.0) << 3);
    match test {
        0..=4 | 6 | 8 | 9 | 12 => sign > 0,
        5 if a_t * c_t - b_t * d_t < EPSILON => sign > 0,
        10 if a_t * c_t - b_t * d_t >= EPSILON => sign > 0,
        5 | 10 => sign < 0,
        7 | 11 | 13..=15 => sign < 0,
        _ => sign < 0,
    }
}

fn grid_point_le(a: UVec3, b: UVec3) -> bool {
    (a.x, a.y, a.z) <= (b.x, b.y, b.z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_binary_case_resolves_to_valid_edges() {
        for case_index in 0..256 {
            let values = std::array::from_fn(|corner| {
                if case_index & (1 << corner) != 0 {
                    1.0
                } else {
                    -1.0
                }
            });
            if let Some(tiling) = resolve_tiling(case_index, &values).unwrap() {
                for index in 0..tiling.triangle_count * 3 {
                    assert!(tiling.edge(index) <= 12, "case {case_index}");
                }
            }
        }
    }
}
