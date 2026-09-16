from __future__ import annotations


GOLDEN_MINIMAL_CONFIG_HASH = (
    "4372326ca81102e0055441517c095673647586c8095c5d4c53838877b90345e0"
)


def config_mapping() -> dict[str, object]:
    return {
        "config_schema_version": "1.0.0",
        "input": {"kind": "generic_binary"},
        "temporal": {
            "input_cadence": "monthly",
            "monthly_composite": "supplied",
            "composite_owner": "caller",
        },
    }


def test_minimal_scientific_config_has_stable_golden_hash() -> None:
    from hydrofragments.config import HydroConfig

    config = HydroConfig.from_mapping(config_mapping())

    assert config.config_hash == GOLDEN_MINIMAL_CONFIG_HASH


def test_execution_and_human_fields_do_not_change_config_hash() -> None:
    from hydrofragments.config import HydroConfig

    windows = config_mapping() | {
        "run_label": "Windows run",
        "compute": {
            "accelerator": "cuda",
            "scheduler": "distributed",
            "workers": 8,
            "scheduler_address": "tcp://scheduler:8786",
            "checkpoint_path": r"D:\work\checkpoints",
        },
        "output": {"output_dir": r"D:\results\run-a", "formats": ["parquet"]},
    }
    posix = config_mapping() | {
        "run_label": "Linux run",
        "compute": {
            "accelerator": "none",
            "scheduler": "local",
            "workers": 1,
            "checkpoint_path": "/work/checkpoints",
        },
        "output": {"output_dir": "/results/run-a", "formats": ["csv"]},
    }

    left = HydroConfig.from_mapping(windows)
    right = HydroConfig.from_mapping(posix)

    assert left.scientific_config() == right.scientific_config()
    assert left.config_hash == right.config_hash == GOLDEN_MINIMAL_CONFIG_HASH
    assert left.execution_config() != right.execution_config()
    assert left.execution_hash != right.execution_hash


def test_scientific_threshold_changes_config_hash() -> None:
    from hydrofragments.config import HydroConfig

    baseline = HydroConfig.from_mapping(config_mapping())
    changed = HydroConfig.from_mapping(
        config_mapping()
        | {"validity": {"min_valid_fraction_month": 0.75}}
    )

    assert changed.config_hash != baseline.config_hash


def test_metric_profile_order_is_canonicalized_as_a_set() -> None:
    from hydrofragments.config import HydroConfig

    left = HydroConfig.from_mapping(
        config_mapping() | {"metric_profiles": ["secondary", "contracts_core"]}
    )
    right = HydroConfig.from_mapping(
        config_mapping() | {"metric_profiles": ["contracts_core", "secondary"]}
    )

    assert left.config_hash == right.config_hash


def test_dynamics_threshold_changes_alter_scientific_hash() -> None:
    from hydrofragments.config import HydroConfig

    baseline = HydroConfig.from_mapping(config_mapping())
    changed = HydroConfig.from_mapping(
        config_mapping()
        | {
            "config_schema_version": "1.1.0",
            "dynamics": {"reconnection_lpi_threshold_pct": 60.0},
        }
    )

    assert changed.config_hash != baseline.config_hash


def test_equivalent_config_schema_versions_normalize_to_same_scientific_hash() -> None:
    from hydrofragments.config import HydroConfig

    legacy = HydroConfig.from_mapping(config_mapping())
    explicit = HydroConfig.from_mapping(
        config_mapping()
        | {
            "config_schema_version": "1.1.0",
            "output": {"spatial_products": [], "raster_formats": ["geotiff"]},
        }
    )

    assert legacy.scientific_config() == explicit.scientific_config()
    assert legacy.config_hash == explicit.config_hash


def test_output_product_changes_alter_execution_hash_only() -> None:
    from hydrofragments.config import HydroConfig

    baseline = HydroConfig.from_mapping(config_mapping())
    with_exports = HydroConfig.from_mapping(
        config_mapping()
        | {
            "config_schema_version": "1.1.0",
            "output": {
                "output_dir": "/tmp/out",
                "spatial_products": ["monthly_pools", "zones"],
                "raster_formats": ["geotiff", "netcdf"],
            },
        }
    )

    assert with_exports.scientific_config() == baseline.scientific_config()
    assert with_exports.config_hash == baseline.config_hash
    assert with_exports.execution_hash != baseline.execution_hash
