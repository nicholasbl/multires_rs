use std::{
    fs::{File, remove_file},
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
};

use glam::{DMat4, DVec3, IVec3, UVec3};
use multires_rs::{
    MappedMsgArchive, MsgDataset, MsgField, MsgFieldLevel, MsgGeometry, MsgLevel, MsgScalarGrid,
    MsgScalarType, MsgWriteOptions, read_msg, write_msg,
};

fn sample_dataset() -> MsgDataset {
    let extent = UVec3::splat(8);
    let mut dataset = MsgDataset::new(
        3.5,
        MsgGeometry::point_centered(DMat4::from_translation(DVec3::new(1.0, 2.0, 3.0))).unwrap(),
        vec![MsgLevel {
            index_origin: IVec3::new(-1, 2, 0),
            point_extent: extent,
            scale: DVec3::splat(0.5),
            refinement_ratio_from_coarse: None,
        }],
    )
    .unwrap();

    let mut density = MsgScalarGrid::new(extent, MsgScalarType::F64);
    assert!(density.set(UVec3::new(1, 2, 3), 4.25));
    dataset
        .add_field(MsgField::new(
            "density",
            MsgScalarType::F64,
            vec![MsgFieldLevel::new(density)],
        ))
        .unwrap();

    let mut temperature = MsgScalarGrid::new(extent, MsgScalarType::F32);
    assert!(temperature.set(UVec3::new(4, 5, 6), 273.25));
    dataset
        .add_field(MsgField::new(
            "temperature",
            MsgScalarType::F32,
            vec![MsgFieldLevel::new(temperature)],
        ))
        .unwrap();
    dataset
}

#[test]
fn mutable_workflow_invalidates_acceleration_and_round_trips() {
    let mut dataset = sample_dataset();
    dataset.rebuild_acceleration().unwrap();
    assert!(
        dataset.field("density").unwrap().levels()[0]
            .acceleration()
            .is_some()
    );

    // Mutable access invalidates derived data even if no value is changed.
    let _ = dataset.field_mut("density").unwrap().levels_mut()[0].values_mut();
    assert!(
        dataset.field("density").unwrap().levels()[0]
            .acceleration()
            .is_none()
    );
    dataset.rebuild_acceleration().unwrap();

    let values = dataset.field_mut("density").unwrap().levels_mut()[0].values_mut();
    assert!(values.set(UVec3::new(7, 7, 7), 9.5));
    assert!(!values.set(UVec3::new(8, 7, 7), 10.0));
    assert!(
        dataset.field("density").unwrap().levels()[0]
            .acceleration()
            .is_none()
    );

    dataset.rebuild_acceleration().unwrap();
    let mut bytes = Vec::new();
    write_msg(&dataset, &MsgWriteOptions::default(), &mut bytes).unwrap();
    let decoded = read_msg(&bytes).unwrap();
    assert_eq!(
        decoded.field("density").unwrap().levels()[0]
            .values()
            .get(UVec3::new(7, 7, 7)),
        Some(9.5)
    );
}

#[test]
fn mapped_workflow_reads_metadata_and_selectively_decodes() {
    let mut dataset = sample_dataset();
    dataset.rebuild_acceleration().unwrap();
    let temporary = TemporaryMsg::new();
    {
        let mut file = File::create(&temporary.path).unwrap();
        write_msg(&dataset, &MsgWriteOptions::default(), &mut file).unwrap();
    }

    let mapped = MappedMsgArchive::open(&temporary.path).unwrap();
    let archive = mapped.view().unwrap();
    assert_eq!(archive.simulation_time(), 3.5);
    assert_eq!(archive.level_count(), 1);
    assert_eq!(
        archive.field_names().collect::<Vec<_>>(),
        vec!["density", "temperature"]
    );

    let decoded = archive.decode_fields(&["temperature"]).unwrap();
    assert_eq!(decoded.fields().len(), 1);
    assert!(decoded.field("density").is_none());
    assert_eq!(
        decoded.field("temperature").unwrap().levels()[0]
            .values()
            .get(UVec3::new(4, 5, 6)),
        Some(273.25)
    );

    let reordered = archive.decode_fields(&["temperature", "density"]).unwrap();
    assert_eq!(reordered.geometry(), dataset.geometry());
    assert_eq!(reordered.levels(), dataset.levels());
    assert_eq!(
        reordered
            .fields()
            .iter()
            .map(|field| field.name())
            .collect::<Vec<_>>(),
        vec!["temperature", "density"]
    );
    assert!(archive.decode_fields(&["missing"]).is_err());
    assert!(
        archive
            .decode_fields(&["temperature", "temperature"])
            .is_err()
    );
}

static NEXT_TEMPORARY_ID: AtomicU64 = AtomicU64::new(0);

struct TemporaryMsg {
    path: PathBuf,
}

impl TemporaryMsg {
    fn new() -> Self {
        let id = NEXT_TEMPORARY_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "multires_rs-api-workflow-{}-{id}.msg",
            std::process::id()
        ));
        Self { path }
    }
}

impl Drop for TemporaryMsg {
    fn drop(&mut self) {
        let _ = remove_file(&self.path);
    }
}
