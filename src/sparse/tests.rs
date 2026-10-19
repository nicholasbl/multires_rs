use glam::{I64Vec3, UVec3};

use super::{Aabb3u, SparseGrid3};

#[test]
fn fill_get_clear_single_chunk() {
    let mut grid = SparseGrid3::<u8>::new(UVec3::new(64, 64, 64));
    let aabb = Aabb3u::new(UVec3::new(2, 3, 4), UVec3::new(8, 9, 10));
    grid.fill_aabb(aabb, 7);

    assert_eq!(grid.get(UVec3::new(2, 3, 4)), Some(7));
    assert_eq!(grid.get(UVec3::new(7, 8, 9)), Some(7));
    assert_eq!(grid.get(UVec3::new(8, 8, 9)), None);
    assert!(grid.clear(UVec3::new(2, 3, 4)));
    assert_eq!(grid.get(UVec3::new(2, 3, 4)), None);
}

#[test]
fn fill_crosses_chunks() {
    let mut grid = SparseGrid3::<u16>::new(UVec3::new(80, 80, 80));
    grid.fill_aabb(
        Aabb3u::new(UVec3::new(30, 30, 30), UVec3::new(35, 35, 35)),
        123,
    );

    assert_eq!(grid.get(UVec3::new(30, 30, 30)), Some(123));
    assert_eq!(grid.get(UVec3::new(34, 34, 34)), Some(123));
    assert_eq!(grid.get(UVec3::new(35, 34, 34)), None);
    assert!(grid.chunk_count() > 1);
}

#[test]
fn chunk_aabbs_are_sorted_and_clipped_to_bounds() {
    let mut grid = SparseGrid3::<u8>::new(UVec3::new(40, 40, 40));
    grid.set(UVec3::new(35, 1, 1), 1);
    grid.set(UVec3::new(1, 1, 1), 2);

    let aabbs = grid.chunk_aabbs();
    assert_eq!(
        aabbs,
        vec![
            Aabb3u::new(UVec3::ZERO, UVec3::splat(32)),
            Aabb3u::new(UVec3::new(32, 0, 0), UVec3::new(40, 32, 32)),
        ]
    );
}

#[test]
fn write_values_x_fastest() {
    let mut grid = SparseGrid3::<u32>::new(UVec3::new(16, 16, 16));
    let aabb = Aabb3u::new(UVec3::new(1, 2, 3), UVec3::new(4, 4, 5));
    let extent = aabb.extent();
    let mut values = Vec::new();

    for z in 0..extent.z {
        for y in 0..extent.y {
            for x in 0..extent.x {
                values.push(x + y * 10 + z * 100);
            }
        }
    }

    grid.write_aabb_values(aabb, &values).unwrap();
    assert_eq!(grid.get(UVec3::new(1, 2, 3)), Some(0));
    assert_eq!(grid.get(UVec3::new(2, 2, 3)), Some(1));
    assert_eq!(grid.get(UVec3::new(1, 3, 3)), Some(10));
    assert_eq!(grid.get(UVec3::new(1, 2, 4)), Some(100));
}

#[test]
fn out_of_bounds_writes_are_ignored_but_slice_size_is_unclipped() {
    let mut grid = SparseGrid3::<u8>::new(UVec3::new(4, 4, 4));
    let aabb = Aabb3u::new(UVec3::new(2, 2, 2), UVec3::new(6, 6, 6));
    let values = vec![9u8; 4 * 4 * 4];
    grid.write_aabb_values(aabb, &values).unwrap();

    assert_eq!(grid.get(UVec3::new(3, 3, 3)), Some(9));
    assert_eq!(grid.get(UVec3::new(4, 3, 3)), None);
}

#[test]
fn union_replaces_present_only() {
    let mut dst = SparseGrid3::<u8>::new(UVec3::new(64, 64, 64));
    let mut src = SparseGrid3::<u8>::new(UVec3::new(64, 64, 64));

    dst.fill_aabb(Aabb3u::new(UVec3::new(0, 0, 0), UVec3::new(4, 4, 4)), 1);
    src.set(UVec3::new(1, 1, 1), 9);
    src.set(UVec3::new(10, 10, 10), 8);

    dst.union_replace_from(&src);

    assert_eq!(dst.get(UVec3::new(0, 0, 0)), Some(1));
    assert_eq!(dst.get(UVec3::new(1, 1, 1)), Some(9));
    assert_eq!(dst.get(UVec3::new(10, 10, 10)), Some(8));
}

#[test]
fn scaled_mask_removes_coarse_if_any_fine_present() {
    let mut coarse = SparseGrid3::<u8>::new(UVec3::new(8, 8, 8));
    let mut fine = SparseGrid3::<u8>::new(UVec3::new(32, 32, 32));

    coarse.fill_aabb(Aabb3u::new(UVec3::new(0, 0, 0), UVec3::new(8, 8, 8)), 1);
    fine.set(UVec3::new(9, 0, 0), 1); // maps to coarse x = 2 when scale = 4

    let removed = coarse.mask_out_by_presence_scaled(&fine, UVec3::splat(4), I64Vec3::ZERO);
    assert_eq!(removed, 1);
    assert_eq!(coarse.get(UVec3::new(2, 0, 0)), None);
    assert_eq!(coarse.get(UVec3::new(1, 0, 0)), Some(1));
}

#[test]
fn scaled_mask_applies_signed_translation_and_clips() {
    let mut coarse = SparseGrid3::<u8>::new(UVec3::new(4, 1, 1));
    let mut fine = SparseGrid3::<u8>::new(UVec3::new(8, 1, 1));

    coarse.fill_aabb(coarse.bounds_aabb(), 1);
    fine.set(UVec3::new(0, 0, 0), 1);

    // coarse x=1 maps to fine [-1, 1), whose in-bounds portion contains x=0.
    let removed =
        coarse.mask_out_by_presence_scaled(&fine, UVec3::splat(2), I64Vec3::new(-3, 0, 0));
    assert_eq!(removed, 1);
    assert_eq!(coarse.get(UVec3::new(1, 0, 0)), None);
    assert_eq!(coarse.get(UVec3::new(0, 0, 0)), Some(1));
    assert_eq!(coarse.get(UVec3::new(2, 0, 0)), Some(1));
}

#[test]
fn prune_removes_empty_chunks() {
    let mut grid = SparseGrid3::<u8>::new(UVec3::new(64, 64, 64));
    grid.fill_aabb(Aabb3u::new(UVec3::new(0, 0, 0), UVec3::new(32, 32, 32)), 1);
    assert_eq!(grid.chunk_count(), 1);
    grid.clear_aabb(Aabb3u::new(UVec3::new(0, 0, 0), UVec3::new(32, 32, 32)));
    grid.prune();
    assert_eq!(grid.chunk_count(), 0);
}

#[test]
fn bitmask_keeps_uniform_occupancy_inline() {
    use super::chunk::{BitMask, BitMaskView};

    assert!(std::mem::size_of::<BitMask>() <= 2 * std::mem::size_of::<usize>());
    assert!(matches!(
        BitMask::empty().view(),
        BitMaskView::Uniform(false)
    ));
    assert!(matches!(BitMask::full().view(), BitMaskView::Uniform(true)));

    let mut irregular = BitMask::empty();
    irregular.set(7);
    assert!(matches!(irregular.view(), BitMaskView::Dense(_)));
    assert!(irregular.get(7));
}
