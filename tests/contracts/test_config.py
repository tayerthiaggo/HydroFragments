from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest


def minimal_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "config_schema_version": "1.0.0",
        "input": {"kind": "generic_binary"},
        "temporal": {
            "input_cadence": "monthly",
            "monthly_composite": "supplied",
            "composite_owner": "caller",
        },
    }
    config.update(overrides)
    return config


def test_unknown_top_level_config_key_is_rejected() -> None:
    from hydrofragments.config import ConfigError, HydroConfig

    with pytest.raises(ConfigError, match=r"unknown config key.*mystery"):
        HydroConfig.from_mapping(minimal_config(mystery=True))


def test_unknown_nested_config_key_is_rejected_with_its_path() -> None:
    from hydrofragments.config import ConfigError, HydroConfig

    with pytest.raises(ConfigError, match=r"unknown config key.*validity\.mystery"):
        HydroConfig.from_mapping(
            minimal_config(validity={"mystery": "not-scientific"})
        )


@pytest.mark.parametrize(
    "missing_key",
    ["water_threshold", "threshold_method", "probability_source"],
)
def test_probability_input_requires_threshold_provenance(missing_key: str) -> None:
    from hydrofragments.config import ConfigError, HydroConfig

    probability = {
        "kind": "generic_probability",
        "water_threshold": 0.5,
        "threshold_method": "fixed",
        "probability_source": "model-v3",
    }
    probability.pop(missing_key)

    with pytest.raises(ConfigError, match=missing_key):
        HydroConfig.from_mapping(minimal_config(input=probability))


@pytest.mark.parametrize(
    "state",
    [
        {"enabled": True, "connectivity_metric": "LPI"},
        {"enabled": True, "connectivity_threshold": 50.0},
    ],
)
def test_enabled_state_requires_metric_and_threshold(
    state: dict[str, object],
) -> None:
    from hydrofragments.config import ConfigError, HydroConfig

    with pytest.raises(ConfigError, match="state"):
        HydroConfig.from_mapping(minimal_config(state=state))


def test_cuda_strict_requires_cuda_accelerator() -> None:
    from hydrofragments.config import ConfigError, HydroConfig

    with pytest.raises(ConfigError, match="cuda_strict"):
        HydroConfig.from_mapping(
            minimal_config(compute={"accelerator": "auto", "cuda_strict": True})
        )


def test_compute_workers_defaults_to_one() -> None:
    from hydrofragments.config import HydroConfig

    config = HydroConfig.from_mapping(minimal_config())
    assert config.compute.workers == 1


@pytest.mark.parametrize("workers", [0, -1, -5])
def test_compute_workers_below_one_is_rejected(workers: int) -> None:
    from hydrofragments.config import ConfigError, HydroConfig

    with pytest.raises(ConfigError, match="compute.workers"):
        HydroConfig.from_mapping(minimal_config(compute={"workers": workers}))


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
def test_compute_workers_at_least_one_is_accepted(workers: int) -> None:
    from hydrofragments.config import HydroConfig

    config = HydroConfig.from_mapping(minimal_config(compute={"workers": workers}))
    assert config.compute.workers == workers


def test_compute_accelerator_defaults_to_auto_detection() -> None:
    from hydrofragments.config import HydroConfig

    config = HydroConfig.from_mapping(minimal_config())

    assert config.compute.accelerator == "auto"


@pytest.mark.parametrize("floor", [0, -1, float("nan")])
def test_width_resolution_floor_must_be_positive_and_finite(floor: float) -> None:
    from hydrofragments.config import ConfigError, HydroConfig

    with pytest.raises(ConfigError, match="width_resolution_floor_pixels"):
        HydroConfig.from_mapping(
            minimal_config(patches={"width_resolution_floor_pixels": floor})
        )


def test_approved_validity_contract_is_the_resolved_default() -> None:
    from hydrofragments.config import HydroConfig

    config = HydroConfig.from_mapping(minimal_config())

    assert config.validity.policy == "p_native_season_stratified_v1"
    assert config.validity.min_valid_obs == 20
    assert config.validity.min_valid_fraction_month == 0.70
    assert config.validity.low_support_behavior == "suppress_value"


def test_extent_contraction_defaults_are_locked() -> None:
    from hydrofragments.config import HydroConfig

    config = HydroConfig.from_mapping(minimal_config())

    assert config.dynamics.contraction_method == "linear"
    assert config.dynamics.minimum_points == 3


def test_config_is_immutable() -> None:
    from hydrofragments.config import HydroConfig

    config = HydroConfig.from_mapping(minimal_config())

    with pytest.raises(FrozenInstanceError):
        config.run_label = "changed"  # type: ignore[misc]


def test_config_schema_1_0_0_disables_all_spatial_products() -> None:
    from hydrofragments.config import HydroConfig

    config = HydroConfig.from_mapping(minimal_config())

    assert config.config_schema_version == "1.0.0"
    assert config.output.spatial_products == ()
    assert config.output.raster_formats == ("geotiff",)


@pytest.mark.parametrize(
    "spatial_products",
    [
        ["monthly_pools"],
        ["zones", "persistence_rasters"],
        ["temporal_rasters", "refuge_stability_rasters", "reach_profiles"],
    ],
)
def test_non_empty_spatial_products_require_output_dir(
    spatial_products: list[str],
) -> None:
    from hydrofragments.config import ConfigError, HydroConfig

    with pytest.raises(ConfigError, match="output.output_dir"):
        HydroConfig.from_mapping(
            minimal_config(
                config_schema_version="1.1.0",
                output={
                    "spatial_products": spatial_products,
                    "output_dir": None,
                },
            )
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("spatial_products", ["not_a_product"]),
        ("raster_formats", ["shapefile"]),
        ("formats", ["xml"]),
    ],
)
def test_config_schema_1_1_0_rejects_unknown_output_literals(
    field: str, value: list[str]
) -> None:
    from hydrofragments.config import ConfigError, HydroConfig

    with pytest.raises(ConfigError):
        HydroConfig.from_mapping(
            minimal_config(
                config_schema_version="1.1.0",
                output={field: value, "output_dir": "/tmp/out"},
            )
        )


@pytest.mark.parametrize(
    "product",
    [
        "monthly_pools",
        "zones",
        "persistence_rasters",
        "temporal_rasters",
        "refuge_stability_rasters",
        "reach_profiles",
    ],
)
def test_config_schema_1_1_0_accepts_spatial_product_literals(
    product: str,
) -> None:
    from hydrofragments.config import HydroConfig

    config = HydroConfig.from_mapping(
        minimal_config(
            config_schema_version="1.1.0",
            output={
                "spatial_products": [product],
                "output_dir": "/tmp/out",
            },
        )
    )

    assert config.output.spatial_products == (product,)


def test_include_vectors_alias_maps_to_monthly_pools_once() -> None:
    from hydrofragments.config import HydroConfig

    config = HydroConfig.from_mapping(
        minimal_config(
            config_schema_version="1.1.0",
            output={
                "include_vectors": True,
                "output_dir": "/tmp/out",
            },
        )
    )

    assert config.output.spatial_products == ("monthly_pools",)


def test_include_vectors_alias_rejects_contradictory_spatial_products() -> None:
    from hydrofragments.config import ConfigError, HydroConfig

    with pytest.raises(ConfigError, match="include_vectors"):
        HydroConfig.from_mapping(
            minimal_config(
                config_schema_version="1.1.0",
                output={
                    "include_vectors": True,
                    "spatial_products": ["zones"],
                    "output_dir": "/tmp/out",
                },
            )
        )


def test_netcdf_format_is_accepted_at_parse_time() -> None:
    from hydrofragments.config import HydroConfig

    config = HydroConfig.from_mapping(
        minimal_config(
            config_schema_version="1.1.0",
            output={
                "raster_formats": ["netcdf", "geotiff"],
                "output_dir": "/tmp/out",
            },
        )
    )

    assert config.output.raster_formats == ("geotiff", "netcdf")


def test_netcdf_preflight_requires_optional_runtime_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hydrofragments.config import ConfigError, HydroConfig

    config = HydroConfig.from_mapping(
        minimal_config(
            config_schema_version="1.1.0",
            output={
                "raster_formats": ["netcdf"],
                "output_dir": "/tmp/out",
            },
        )
    )
    monkeypatch.setattr(
        "hydrofragments.config._netcdf_writer_available",
        lambda: False,
    )

    with pytest.raises(ConfigError, match="netcdf"):
        config.validate_output_preflight()


@pytest.mark.parametrize(
    "threshold",
    [float("nan"), float("inf"), -1.0, 101.0],
)
def test_dynamics_percentage_thresholds_reject_invalid_values(
    threshold: float,
) -> None:
    from hydrofragments.config import ConfigError, HydroConfig

    with pytest.raises(ConfigError, match="dynamics"):
        HydroConfig.from_mapping(
            minimal_config(
                config_schema_version="1.1.0",
                dynamics={"reconnection_lpi_threshold_pct": threshold},
            )
        )


def test_dynamics_threshold_defaults_are_locked() -> None:
    from hydrofragments.config import HydroConfig

    config = HydroConfig.from_mapping(minimal_config(config_schema_version="1.1.0"))

    assert config.dynamics.reconnection_lpi_threshold_pct == 50.0
    assert config.dynamics.reconnection_lpsec_threshold_pct == 50.0


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"add": ["not_a_metric"]}, "unknown metric"),
        ({"add": ["lpi", "lpi"]}, "duplicate"),
        ({"remove": ["lpi", "lpi"]}, "duplicate"),
        ({"add": ["lpi"], "remove": ["lpi"]}, "contradict"),
    ],
)
def test_metric_override_validation_failures(
    overrides: dict[str, list[str]], match: str
) -> None:
    from hydrofragments.config import ConfigError, HydroConfig

    with pytest.raises(ConfigError, match=match):
        HydroConfig.from_mapping(minimal_config(metric_overrides=overrides))


def test_riverscape_config_has_documented_defaults() -> None:
    from hydrofragments.config import HydroConfig

    config = HydroConfig.from_mapping(minimal_config())

    assert config.riverscape.mode == "auto"
    assert config.riverscape.dem_product == "ga_srtm_dem1sv1_0"
    assert config.riverscape.dem_band == "dem_s"
    assert config.riverscape.fc_product == "ga_ls_fc_pc_cyear_3"
    assert config.riverscape.bare_band == "bs_pc_50"
    assert config.riverscape.green_band == "pv_pc_50"
    assert 0.0 < config.riverscape.f_seed < config.riverscape.f_chan_high < 1.0
    assert config.riverscape.corridor_min_m < config.riverscape.corridor_max_m
    assert config.riverscape.min_channel_confidence >= 1


def test_riverscape_config_rejects_bad_mode() -> None:
    from hydrofragments.config import ConfigError, HydroConfig

    with pytest.raises(ConfigError, match="riverscape.mode"):
        HydroConfig.from_mapping(minimal_config(riverscape={"mode": "sometimes"}))


def test_riverscape_config_rejects_f_seed_above_f_chan_high() -> None:
    from hydrofragments.config import ConfigError, HydroConfig

    with pytest.raises(ConfigError, match="f_seed"):
        HydroConfig.from_mapping(
            minimal_config(riverscape={"f_seed": 0.5, "f_chan_high": 0.1})
        )


def test_riverscape_config_rejects_inverted_corridor_bounds() -> None:
    from hydrofragments.config import ConfigError, HydroConfig

    with pytest.raises(ConfigError, match="corridor_min_m"):
        HydroConfig.from_mapping(
            minimal_config(riverscape={"corridor_min_m": 600, "corridor_max_m": 90})
        )


def test_unknown_riverscape_key_is_rejected() -> None:
    from hydrofragments.config import ConfigError, HydroConfig

    with pytest.raises(ConfigError, match=r"unknown config key.*riverscape\.mystery"):
        HydroConfig.from_mapping(minimal_config(riverscape={"mystery": True}))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rem_k", -1),
        ("envelope_min_bin_pixels", -1),
        ("narrow_width_px", -1),
        ("trough_radius_m", -150.0),
        ("profile_bin_m", -300.0),
        ("rem_max_distance_m", -5000.0),
        ("h_chan_m", -2.0),
        ("width_growth_factor", -3.0),
        ("trough_radius_m", float("nan")),
        ("profile_bin_m", float("inf")),
        ("rem_max_distance_m", float("nan")),
        # 0 boundary: zero is rejected, not treated as a valid minimum.
        ("rem_k", 0),
        ("trough_radius_m", 0.0),
        # Fields newly guarded in this round.
        ("trough_depth_m", -0.5),
        ("trough_depth_m", float("nan")),
        ("envelope_h_max_m", -15.0),
        ("envelope_h_max_m", float("nan")),
        ("bridge_max_length_m", -2000.0),
        ("bridge_max_length_m", float("inf")),
        ("bridge_rem_max_m", -5.0),
        ("bridge_rem_max_m", float("-inf")),
        ("slope_max_deg", -1.0),
        ("slope_max_deg", 0.0),
        ("slope_max_deg", 90.0),
        ("bridge_max_cost_per_m", -2.0),
        ("bridge_max_cost_per_m", 0.0),
        # width_growth_factor's bound was tightened from > 0 to >= 1.0: a
        # factor between 0 and 1 would cap the corridor narrower than the
        # seed itself, which is degenerate.
        ("width_growth_factor", 0.5),
    ],
)
def test_riverscape_config_rejects_non_positive_or_non_finite_numerics(
    field: str, value: float
) -> None:
    from hydrofragments.config import ConfigError, HydroConfig

    with pytest.raises(ConfigError, match=f"riverscape.{field}"):
        HydroConfig.from_mapping(minimal_config(riverscape={field: value}))
