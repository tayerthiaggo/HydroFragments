"""Immutable, validated HydroFragments configuration contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import importlib
import json
import math
from typing import Any, Literal, Mapping

ACCEPTED_CONFIG_SCHEMA_VERSIONS = frozenset({"1.0.0", "1.1.0"})
SCIENTIFIC_HASH_SCHEMA_VERSION = "1.2.0"
HASH_ALGORITHM_VERSION = "sha256-json-v1"

SpatialProduct = Literal[
    "monthly_pools",
    "zones",
    "persistence_rasters",
    "temporal_rasters",
    "refuge_stability_rasters",
    "reach_profiles",
]
RasterFormat = Literal["geotiff", "netcdf"]

SPATIAL_PRODUCTS = frozenset(
    {
        "monthly_pools",
        "zones",
        "persistence_rasters",
        "temporal_rasters",
        "refuge_stability_rasters",
        "reach_profiles",
    }
)
RASTER_FORMATS = frozenset({"geotiff", "netcdf"})
OUTPUT_FORMATS = frozenset({"parquet", "csv"})


class ConfigError(ValueError):
    """Raised when a configuration mapping violates the public contract."""


class LowSupportBehavior(str, Enum):
    """Approved handling for values with insufficient valid observations."""

    SUPPRESS_VALUE = "suppress_value"
    EMIT_FLAGGED_VALUE = "emit_flagged_value"
    METRIC_SPECIFIC = "metric_specific"


@dataclass(frozen=True)
class InputConfig:
    kind: str
    variable_map: tuple[tuple[str, str], ...] = ()
    water_threshold: float | None = None
    threshold_method: str | None = None
    probability_source: str | None = None


@dataclass(frozen=True)
class ValidityConfig:
    policy: str = "p_native_season_stratified_v1"
    min_valid_obs: int = 20
    min_valid_fraction_month: float = 0.70
    low_support_behavior: str = LowSupportBehavior.SUPPRESS_VALUE.value


@dataclass(frozen=True)
class WindowingConfig:
    mode: str = "none"
    length_m: float | None = None


@dataclass(frozen=True)
class SpatialConfig:
    target_crs: str = "EPSG:3577"
    area_method: str = "projected"
    windowing: WindowingConfig = field(default_factory=WindowingConfig)


@dataclass(frozen=True)
class PatchesConfig:
    min_patch_pixels: int = 3
    connectivity_rule: int = 8
    width_resolution_floor_pixels: float | None = None


@dataclass(frozen=True)
class PersistenceConfig:
    refuge_threshold: float = 0.90


@dataclass(frozen=True)
class ZonesConfig:
    t_persist: float = 0.50
    t_season: float = 0.10


@dataclass(frozen=True)
class RiverscapeConfig:
    mode: str = "auto"
    dem_product: str = "ga_srtm_dem1sv1_0"
    dem_band: str = "dem_s"
    fc_product: str = "ga_ls_fc_pc_cyear_3"
    bare_band: str = "bs_pc_50"
    green_band: str = "pv_pc_50"
    npv_band: str = "npv_pc_50"
    waterbodies_source: str | None = None
    f_seed: float = 0.05
    f_chan_high: float = 0.10
    # Basin-scale rerun (Plan 2, Task 3; see
    # docs/superpowers/specs/2026-09-14-riverscape-phase0-findings.md,
    # "Basin-scale rerun" section): bare_threshold_floor_pct is a
    # degraded-path *fallback* used only when per-run calibration (spec
    # §4.2 step 4) cannot run -- not the primary threshold and not a
    # min-clamp on the calibrated value. The evidence bit's semantic (spec
    # §4.2, bit B: "bare percentile >= bare_threshold") is "barer than
    # local background", so this floor must be anchored to a background
    # baseline, not to a seed-water figure taken in isolation.
    #
    # The narrow-AOI seed-water figure (31.5) is NOT a safe anchor: the
    # findings doc's own basin-wide rerun shows the narrow AOI's reaches
    # "sit almost exactly at the top 1% of basin reaches by upstream
    # drainage area, confirming it sampled only the largest (near-outlet)
    # reaches, not a representative cross-section." Worse, the sign of
    # the seed-vs-background contrast flips between the two AOIs: in the
    # narrow AOI, seed-water bs_pc_50 (31.5) is HIGHER than AOI-wide
    # background (20.5) -- seed is barer than background, as intended --
    # but basin-wide, seed-water bs_pc_50 (7.5) is LOWER than AOI-wide
    # background (24.5) -- seed is LESS bare than background, the
    # opposite relationship. These two figures cannot be averaged or
    # range-picked between; they disagree in sign, not just magnitude.
    # Resolving that disagreement needs per-reach calibration, which this
    # fallback floor does not attempt -- it is left open for Plan 3 (see
    # the findings doc's "Implications for Plan 3" bullet on
    # bare_threshold calibration).
    #
    # Anchor: the basin-wide AOI-wide background median, 24.5 (the more
    # representative baseline per the findings doc's own assessment).
    # 30.0 is set just above that median, on the barer-than-typical side
    # of the background baseline, without relying on either AOI's
    # seed-water figure.
    bare_threshold_floor_pct: float = 30.0
    bare_year_fraction: float = 0.6
    trough_radius_m: float = 150.0
    trough_depth_m: float = 0.5
    h_chan_m: float = 2.0
    corridor_min_m: float = 90.0
    # Basin-scale rerun (Plan 2, Task 3): the basin-wide line->EO-skeleton
    # offset p95 is 924.18 m (up from the narrow AOI's already-saturating
    # 630.71 m), so the Plan 1 placeholder ceiling of 600.0 m is too tight
    # at basin scale ("The 600 m ceiling remains too tight for this
    # catchment's measured AHGF/EO offset at basin scale." -- findings
    # doc). 1200.0 m clears the measured p95 with roughly 30% headroom;
    # it is deliberately not set to exactly 924.18 since p95 is a soft
    # distributional boundary and this field is a hard ceiling, not a
    # p95 pin.
    corridor_max_m: float = 1200.0
    alignment_quantile: float = 0.95
    width_growth_factor: float = 3.0
    profile_bin_m: float = 300.0
    profile_percentile: float = 10.0
    rem_k: int = 8
    rem_max_distance_m: float = 5000.0
    envelope_quantile: float = 0.95
    envelope_h_max_m: float = 15.0
    envelope_min_bin_pixels: int = 200
    slope_max_deg: float = 2.0
    min_channel_confidence: int = 2
    include_line_fallback_in_channel: bool = False
    bridge_enabled: bool = True
    bridge_max_length_m: float = 2000.0
    bridge_max_cost_per_m: float = 1.0
    bridge_rem_max_m: float = 5.0
    riparian_green_pct: float = 40.0
    narrow_width_px: int = 2


@dataclass(frozen=True)
class TemporalConfig:
    input_cadence: str
    monthly_composite: str
    composite_owner: str


@dataclass(frozen=True)
class DynamicsConfig:
    composite_sensitivity_tolerance_pp: float = 10.0
    contraction_method: str = "linear"
    minimum_points: int = 3
    reconnection_lpi_threshold_pct: float = 50.0
    reconnection_lpsec_threshold_pct: float = 50.0


@dataclass(frozen=True)
class HydroYearConfig:
    algorithm: str | None = None
    parameters: tuple[tuple[str, Any], ...] = ()


_HYDROSEASON_DEFAULT_PARAMETERS: dict[str, Any] = {
    "wet_start_month": 11,
    "wet_end_month": 4,
    "dry_start_month": 7,
    "dry_end_month": 12,
    "min_wet_months": 2,
    "min_dry_months": 2,
    "low_confidence_ratio": 0.25,
    "medium_confidence_ratio": 0.5,
}


@dataclass(frozen=True)
class ChannelConfig:
    source: str | None = None
    node_source: str | None = None


@dataclass(frozen=True)
class StateConfig:
    enabled: bool = False
    connectivity_metric: str | None = None
    connectivity_threshold: float | None = None


@dataclass(frozen=True)
class ConnectivityConfig:
    edge_rule: str | None = None


@dataclass(frozen=True)
class ComputeConfig:
    accelerator: str = "auto"
    cuda_strict: bool = False
    target_chunk_bytes: int | None = None
    worker_memory_fraction: float | None = None
    checkpoint: str = "zarr"
    scheduler: str = "local"
    workers: int = 1
    scheduler_address: str | None = None
    checkpoint_path: str | None = None


@dataclass(frozen=True)
class OutputConfig:
    formats: tuple[str, ...] = ("parquet",)
    include_patch_table: bool = False
    include_vectors: bool = False
    output_dir: str | None = None
    spatial_products: tuple[str, ...] = ()
    raster_formats: tuple[str, ...] = ("geotiff",)


@dataclass(frozen=True)
class MetricOverrides:
    add: tuple[str, ...] = ()
    remove: tuple[str, ...] = ()
    reasons: tuple[tuple[str, str], ...] = ()


_TOP_LEVEL_KEYS = {
    "config_schema_version",
    "run_label",
    "metric_profiles",
    "metric_overrides",
    "input",
    "validity",
    "spatial",
    "patches",
    "persistence",
    "zones",
    "riverscape",
    "temporal",
    "dynamics",
    "hydroyear",
    "channel",
    "state",
    "connectivity",
    "compute",
    "output",
}


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{path} keys must be strings")
    return value


def _strict_section(
    source: Mapping[str, Any], path: str, allowed: set[str]
) -> Mapping[str, Any]:
    unknown = sorted(set(source) - allowed)
    if unknown:
        raise ConfigError(f"unknown config key: {path}.{unknown[0]}")
    return source


def _section(
    source: Mapping[str, Any], name: str, allowed: set[str]
) -> Mapping[str, Any]:
    return _strict_section(_mapping(source.get(name, {}), name), name, allowed)


def _required(source: Mapping[str, Any], key: str, path: str) -> Any:
    value = source.get(key)
    if value is None or value == "":
        raise ConfigError(f"{path}.{key} is required")
    return value


def _fraction(value: object, path: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{path} must be a number between 0 and 1") from error
    if not 0.0 <= result <= 1.0:
        raise ConfigError(f"{path} must be between 0 and 1")
    return result


def _percentage(value: object, path: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{path} must be a number between 0 and 100") from error
    if not math.isfinite(result):
        raise ConfigError(f"{path} must be finite")
    if not 0.0 <= result <= 100.0:
        raise ConfigError(f"{path} must be between 0 and 100")
    return result


def _literal_sequence(
    value: object,
    *,
    path: str,
    allowed: frozenset[str],
) -> tuple[str, ...]:
    if isinstance(value, str):
        raise ConfigError(f"{path} must be a sequence, not a string")
    try:
        items = [str(item) for item in value]
    except TypeError as error:
        raise ConfigError(f"{path} must be a sequence") from error
    unknown = sorted({item for item in items if item not in allowed})
    if unknown:
        raise ConfigError(f"{path} has unsupported value: {unknown[0]}")
    return tuple(sorted(set(items)))


def _validate_override_ids(
    *,
    add: tuple[str, ...],
    remove: tuple[str, ...],
    add_raw: object,
    remove_raw: object,
) -> None:
    from hydrofragments.metrics.registry import METRIC_REGISTRY

    for label, raw, resolved in (
        ("add", add_raw, add),
        ("remove", remove_raw, remove),
    ):
        if raw is None:
            continue
        if isinstance(raw, str):
            raise ConfigError(f"metric_overrides.{label} must be a sequence, not a string")
        try:
            items = [str(item) for item in raw]
        except TypeError as error:
            raise ConfigError(f"metric_overrides.{label} must be a sequence") from error
        if len(items) != len(set(items)):
            raise ConfigError(f"metric_overrides.{label} contains duplicate metric ids")
    overlap = sorted(set(add).intersection(remove))
    if overlap:
        raise ConfigError(
            "metric_overrides contains contradictory add/remove entries: "
            + overlap[0]
        )
    for metric_id in add + remove:
        if metric_id not in METRIC_REGISTRY:
            raise ConfigError(f"unknown metric override id: {metric_id}")


def _netcdf_writer_available() -> bool:
    for module_name in ("netCDF4", "h5netcdf"):
        try:
            importlib.import_module(module_name)
        except ImportError:
            continue
        return True
    return False


def _canonical_json(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ConfigError("configuration must contain JSON-compatible values") from error


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class HydroConfig:
    """Resolved scientific and execution configuration."""

    config_schema_version: str
    input: InputConfig
    temporal: TemporalConfig
    run_label: str | None = None
    metric_profiles: tuple[str, ...] = ("all_available",)
    metric_overrides: MetricOverrides = field(default_factory=MetricOverrides)
    validity: ValidityConfig = field(default_factory=ValidityConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    patches: PatchesConfig = field(default_factory=PatchesConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    zones: ZonesConfig = field(default_factory=ZonesConfig)
    riverscape: RiverscapeConfig = field(default_factory=RiverscapeConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    hydroyear: HydroYearConfig = field(default_factory=HydroYearConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    state: StateConfig = field(default_factory=StateConfig)
    connectivity: ConnectivityConfig = field(default_factory=ConnectivityConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "HydroConfig":
        source = _strict_section(_mapping(raw, "config"), "config", _TOP_LEVEL_KEYS)

        config_schema_version = str(
            _required(source, "config_schema_version", "config")
        )
        if config_schema_version not in ACCEPTED_CONFIG_SCHEMA_VERSIONS:
            raise ConfigError(
                "config.config_schema_version has unsupported value: "
                f"{config_schema_version}"
            )

        input_raw = _section(
            source,
            "input",
            {
                "kind",
                "variable_map",
                "water_threshold",
                "threshold_method",
                "probability_source",
            },
        )
        kind = str(_required(input_raw, "kind", "input"))
        if kind not in {
            "watermask_tsfill",
            "generic_binary",
            "generic_probability",
        }:
            raise ConfigError(f"input.kind has unsupported value: {kind}")
        if kind == "generic_probability":
            for key in (
                "water_threshold",
                "threshold_method",
                "probability_source",
            ):
                _required(input_raw, key, "input")
        variable_map_raw = _mapping(input_raw.get("variable_map", {}), "input.variable_map")
        input_config = InputConfig(
            kind=kind,
            variable_map=tuple(
                sorted((str(key), str(value)) for key, value in variable_map_raw.items())
            ),
            water_threshold=(
                None
                if input_raw.get("water_threshold") is None
                else _fraction(input_raw["water_threshold"], "input.water_threshold")
            ),
            threshold_method=input_raw.get("threshold_method"),
            probability_source=input_raw.get("probability_source"),
        )

        validity_raw = _section(
            source,
            "validity",
            {
                "policy",
                "min_valid_obs",
                "min_valid_fraction_month",
                "low_support_behavior",
            },
        )
        min_valid_obs = int(validity_raw.get("min_valid_obs", 20))
        if min_valid_obs < 1:
            raise ConfigError("validity.min_valid_obs must be at least 1")
        low_support_behavior = str(
            validity_raw.get(
                "low_support_behavior", LowSupportBehavior.SUPPRESS_VALUE.value
            )
        )
        if low_support_behavior not in {item.value for item in LowSupportBehavior}:
            raise ConfigError(
                "validity.low_support_behavior has unsupported value: "
                f"{low_support_behavior}"
            )
        validity = ValidityConfig(
            policy=str(
                validity_raw.get("policy", "p_native_season_stratified_v1")
            ),
            min_valid_obs=min_valid_obs,
            min_valid_fraction_month=_fraction(
                validity_raw.get("min_valid_fraction_month", 0.70),
                "validity.min_valid_fraction_month",
            ),
            low_support_behavior=low_support_behavior,
        )

        windowing_raw = _strict_section(
            _mapping(
                _section(
                    source,
                    "spatial",
                    {"target_crs", "area_method", "windowing"},
                ).get("windowing", {}),
                "spatial.windowing",
            ),
            "spatial.windowing",
            {"mode", "length_m"},
        )
        spatial_raw = _section(
            source, "spatial", {"target_crs", "area_method", "windowing"}
        )
        window_mode = str(windowing_raw.get("mode", "none"))
        if window_mode not in {"none", "channel_length", "regular_grid"}:
            raise ConfigError(f"spatial.windowing.mode has unsupported value: {window_mode}")
        window_length = windowing_raw.get("length_m")
        if window_mode == "channel_length" and window_length is None:
            window_length = 5000.0
        spatial = SpatialConfig(
            target_crs=str(spatial_raw.get("target_crs", "EPSG:3577")),
            area_method=str(spatial_raw.get("area_method", "projected")),
            windowing=WindowingConfig(
                mode=window_mode,
                length_m=None if window_length is None else float(window_length),
            ),
        )
        if spatial.area_method not in {"projected", "per_pixel"}:
            raise ConfigError(
                f"spatial.area_method has unsupported value: {spatial.area_method}"
            )

        patches_raw = _section(
            source,
            "patches",
            {
                "min_patch_pixels",
                "connectivity_rule",
                "width_resolution_floor_pixels",
            },
        )
        patches = PatchesConfig(
            min_patch_pixels=int(patches_raw.get("min_patch_pixels", 3)),
            connectivity_rule=int(patches_raw.get("connectivity_rule", 8)),
            width_resolution_floor_pixels=(
                None
                if patches_raw.get("width_resolution_floor_pixels") is None
                else float(patches_raw["width_resolution_floor_pixels"])
            ),
        )
        if patches.min_patch_pixels < 1:
            raise ConfigError("patches.min_patch_pixels must be at least 1")
        if patches.connectivity_rule not in {4, 8}:
            raise ConfigError("patches.connectivity_rule must be 4 or 8")
        if patches.width_resolution_floor_pixels is not None and (
            not math.isfinite(patches.width_resolution_floor_pixels)
            or patches.width_resolution_floor_pixels <= 0
        ):
            raise ConfigError(
                "patches.width_resolution_floor_pixels must be positive and finite"
            )

        persistence_raw = _section(
            source, "persistence", {"refuge_threshold"}
        )
        persistence = PersistenceConfig(
            refuge_threshold=_fraction(
                persistence_raw.get("refuge_threshold", 0.90),
                "persistence.refuge_threshold",
            )
        )

        zones_raw = _section(source, "zones", {"t_persist", "t_season"})
        zones = ZonesConfig(
            t_persist=_fraction(zones_raw.get("t_persist", 0.50), "zones.t_persist"),
            t_season=_fraction(zones_raw.get("t_season", 0.10), "zones.t_season"),
        )
        if zones.t_season >= zones.t_persist:
            raise ConfigError("zones.t_season must be less than zones.t_persist")

        riverscape_raw = _section(
            source,
            "riverscape",
            {
                "mode", "dem_product", "dem_band", "fc_product", "bare_band",
                "green_band", "npv_band", "waterbodies_source", "f_seed",
                "f_chan_high", "bare_threshold_floor_pct", "bare_year_fraction",
                "trough_radius_m", "trough_depth_m", "h_chan_m",
                "corridor_min_m", "corridor_max_m", "alignment_quantile",
                "width_growth_factor", "profile_bin_m", "profile_percentile",
                "rem_k", "rem_max_distance_m", "envelope_quantile",
                "envelope_h_max_m", "envelope_min_bin_pixels", "slope_max_deg",
                "min_channel_confidence", "include_line_fallback_in_channel",
                "bridge_enabled", "bridge_max_length_m", "bridge_max_cost_per_m",
                "bridge_rem_max_m", "riparian_green_pct", "narrow_width_px",
            },
        )
        riverscape_defaults = RiverscapeConfig()
        riverscape_mode = str(riverscape_raw.get("mode", riverscape_defaults.mode))
        if riverscape_mode not in {"off", "auto", "required"}:
            raise ConfigError(
                f"riverscape.mode has unsupported value: {riverscape_mode}"
            )
        riverscape_f_seed = _fraction(
            riverscape_raw.get("f_seed", riverscape_defaults.f_seed),
            "riverscape.f_seed",
        )
        riverscape_f_chan_high = _fraction(
            riverscape_raw.get("f_chan_high", riverscape_defaults.f_chan_high),
            "riverscape.f_chan_high",
        )
        if riverscape_f_seed > riverscape_f_chan_high:
            raise ConfigError("riverscape.f_seed must not exceed riverscape.f_chan_high")
        riverscape_corridor_min = float(
            riverscape_raw.get("corridor_min_m", riverscape_defaults.corridor_min_m)
        )
        riverscape_corridor_max = float(
            riverscape_raw.get("corridor_max_m", riverscape_defaults.corridor_max_m)
        )
        if not (
            math.isfinite(riverscape_corridor_min)
            and math.isfinite(riverscape_corridor_max)
            and 0.0 < riverscape_corridor_min < riverscape_corridor_max
        ):
            raise ConfigError(
                "riverscape.corridor_min_m must be positive and less than "
                "riverscape.corridor_max_m"
            )
        riverscape = RiverscapeConfig(
            mode=riverscape_mode,
            dem_product=str(
                riverscape_raw.get("dem_product", riverscape_defaults.dem_product)
            ),
            dem_band=str(riverscape_raw.get("dem_band", riverscape_defaults.dem_band)),
            fc_product=str(
                riverscape_raw.get("fc_product", riverscape_defaults.fc_product)
            ),
            bare_band=str(
                riverscape_raw.get("bare_band", riverscape_defaults.bare_band)
            ),
            green_band=str(
                riverscape_raw.get("green_band", riverscape_defaults.green_band)
            ),
            npv_band=str(riverscape_raw.get("npv_band", riverscape_defaults.npv_band)),
            waterbodies_source=riverscape_raw.get("waterbodies_source"),
            f_seed=riverscape_f_seed,
            f_chan_high=riverscape_f_chan_high,
            bare_threshold_floor_pct=_percentage(
                riverscape_raw.get(
                    "bare_threshold_floor_pct",
                    riverscape_defaults.bare_threshold_floor_pct,
                ),
                "riverscape.bare_threshold_floor_pct",
            ),
            bare_year_fraction=_fraction(
                riverscape_raw.get(
                    "bare_year_fraction", riverscape_defaults.bare_year_fraction
                ),
                "riverscape.bare_year_fraction",
            ),
            trough_radius_m=float(
                riverscape_raw.get("trough_radius_m", riverscape_defaults.trough_radius_m)
            ),
            trough_depth_m=float(
                riverscape_raw.get("trough_depth_m", riverscape_defaults.trough_depth_m)
            ),
            h_chan_m=float(riverscape_raw.get("h_chan_m", riverscape_defaults.h_chan_m)),
            corridor_min_m=riverscape_corridor_min,
            corridor_max_m=riverscape_corridor_max,
            alignment_quantile=_fraction(
                riverscape_raw.get(
                    "alignment_quantile", riverscape_defaults.alignment_quantile
                ),
                "riverscape.alignment_quantile",
            ),
            width_growth_factor=float(
                riverscape_raw.get(
                    "width_growth_factor", riverscape_defaults.width_growth_factor
                )
            ),
            profile_bin_m=float(
                riverscape_raw.get("profile_bin_m", riverscape_defaults.profile_bin_m)
            ),
            profile_percentile=_percentage(
                riverscape_raw.get(
                    "profile_percentile", riverscape_defaults.profile_percentile
                ),
                "riverscape.profile_percentile",
            ),
            rem_k=int(riverscape_raw.get("rem_k", riverscape_defaults.rem_k)),
            rem_max_distance_m=float(
                riverscape_raw.get(
                    "rem_max_distance_m", riverscape_defaults.rem_max_distance_m
                )
            ),
            envelope_quantile=_fraction(
                riverscape_raw.get(
                    "envelope_quantile", riverscape_defaults.envelope_quantile
                ),
                "riverscape.envelope_quantile",
            ),
            envelope_h_max_m=float(
                riverscape_raw.get(
                    "envelope_h_max_m", riverscape_defaults.envelope_h_max_m
                )
            ),
            envelope_min_bin_pixels=int(
                riverscape_raw.get(
                    "envelope_min_bin_pixels",
                    riverscape_defaults.envelope_min_bin_pixels,
                )
            ),
            slope_max_deg=float(
                riverscape_raw.get("slope_max_deg", riverscape_defaults.slope_max_deg)
            ),
            min_channel_confidence=int(
                riverscape_raw.get(
                    "min_channel_confidence",
                    riverscape_defaults.min_channel_confidence,
                )
            ),
            include_line_fallback_in_channel=bool(
                riverscape_raw.get(
                    "include_line_fallback_in_channel",
                    riverscape_defaults.include_line_fallback_in_channel,
                )
            ),
            bridge_enabled=bool(
                riverscape_raw.get(
                    "bridge_enabled", riverscape_defaults.bridge_enabled
                )
            ),
            bridge_max_length_m=float(
                riverscape_raw.get(
                    "bridge_max_length_m", riverscape_defaults.bridge_max_length_m
                )
            ),
            bridge_max_cost_per_m=float(
                riverscape_raw.get(
                    "bridge_max_cost_per_m", riverscape_defaults.bridge_max_cost_per_m
                )
            ),
            bridge_rem_max_m=float(
                riverscape_raw.get(
                    "bridge_rem_max_m", riverscape_defaults.bridge_rem_max_m
                )
            ),
            riparian_green_pct=_percentage(
                riverscape_raw.get(
                    "riparian_green_pct", riverscape_defaults.riparian_green_pct
                ),
                "riverscape.riparian_green_pct",
            ),
            narrow_width_px=int(
                riverscape_raw.get(
                    "narrow_width_px", riverscape_defaults.narrow_width_px
                )
            ),
        )
        if riverscape.min_channel_confidence < 1:
            raise ConfigError("riverscape.min_channel_confidence must be at least 1")
        if riverscape.rem_k < 1:
            raise ConfigError(
                f"riverscape.rem_k must be a positive integer, got {riverscape.rem_k}"
            )
        if riverscape.envelope_min_bin_pixels < 1:
            raise ConfigError(
                "riverscape.envelope_min_bin_pixels must be a positive integer, "
                f"got {riverscape.envelope_min_bin_pixels}"
            )
        if riverscape.narrow_width_px < 1:
            raise ConfigError(
                "riverscape.narrow_width_px must be a positive integer, "
                f"got {riverscape.narrow_width_px}"
            )
        if not (
            math.isfinite(riverscape.trough_radius_m)
            and riverscape.trough_radius_m > 0
        ):
            raise ConfigError("riverscape.trough_radius_m must be finite and positive")
        if not (
            math.isfinite(riverscape.profile_bin_m) and riverscape.profile_bin_m > 0
        ):
            raise ConfigError("riverscape.profile_bin_m must be finite and positive")
        if not (
            math.isfinite(riverscape.rem_max_distance_m)
            and riverscape.rem_max_distance_m > 0
        ):
            raise ConfigError(
                "riverscape.rem_max_distance_m must be finite and positive"
            )
        if not (math.isfinite(riverscape.h_chan_m) and riverscape.h_chan_m > 0):
            raise ConfigError("riverscape.h_chan_m must be finite and positive")
        if not (
            math.isfinite(riverscape.width_growth_factor)
            and riverscape.width_growth_factor > 0
        ):
            raise ConfigError(
                "riverscape.width_growth_factor must be finite and positive"
            )

        temporal_raw = _section(
            source,
            "temporal",
            {"input_cadence", "monthly_composite", "composite_owner"},
        )
        temporal = TemporalConfig(
            input_cadence=str(_required(temporal_raw, "input_cadence", "temporal")),
            monthly_composite=str(
                _required(temporal_raw, "monthly_composite", "temporal")
            ),
            composite_owner=str(
                _required(temporal_raw, "composite_owner", "temporal")
            ),
        )
        if temporal.monthly_composite not in {
            "max_water",
            "median",
            "mode",
            "end_of_month_nearest",
            "supplied",
        }:
            raise ConfigError(
                "temporal.monthly_composite has unsupported value: "
                f"{temporal.monthly_composite}"
            )
        if temporal.composite_owner not in {"hydrofragments", "upstream", "caller"}:
            raise ConfigError(
                "temporal.composite_owner has unsupported value: "
                f"{temporal.composite_owner}"
            )

        dynamics_raw = _section(
            source,
            "dynamics",
            {
                "composite_sensitivity_tolerance_pp",
                "contraction_method",
                "minimum_points",
                "reconnection_lpi_threshold_pct",
                "reconnection_lpsec_threshold_pct",
            },
        )
        dynamics = DynamicsConfig(
            composite_sensitivity_tolerance_pp=float(
                dynamics_raw.get("composite_sensitivity_tolerance_pp", 10.0)
            ),
            contraction_method=str(
                dynamics_raw.get("contraction_method") or "linear"
            ),
            minimum_points=int(
                dynamics_raw.get("minimum_points")
                if dynamics_raw.get("minimum_points") is not None
                else 3
            ),
            reconnection_lpi_threshold_pct=_percentage(
                dynamics_raw.get("reconnection_lpi_threshold_pct", 50.0),
                "dynamics.reconnection_lpi_threshold_pct",
            ),
            reconnection_lpsec_threshold_pct=_percentage(
                dynamics_raw.get("reconnection_lpsec_threshold_pct", 50.0),
                "dynamics.reconnection_lpsec_threshold_pct",
            ),
        )
        if dynamics.contraction_method not in {"linear", "theil_sen"}:
            raise ConfigError(
                "dynamics.contraction_method must be 'linear' or 'theil_sen'"
            )
        if dynamics.minimum_points < 3:
            raise ConfigError("dynamics.minimum_points must be at least 3")

        hydroyear_raw = _section(source, "hydroyear", {"algorithm", "parameters"})
        hydroyear_parameters = _mapping(
            hydroyear_raw.get("parameters", {}), "hydroyear.parameters"
        )
        unknown_hydroseason = sorted(
            set(hydroyear_parameters) - set(_HYDROSEASON_DEFAULT_PARAMETERS)
        )
        if unknown_hydroseason:
            raise ConfigError(
                "unknown hydroseason config key: hydroyear.parameters."
                + unknown_hydroseason[0]
            )
        resolved_hydroyear_parameters = {
            **_HYDROSEASON_DEFAULT_PARAMETERS,
            **hydroyear_parameters,
        }
        hydroyear = HydroYearConfig(
            algorithm=(
                str(hydroyear_raw.get("algorithm"))
                if hydroyear_raw.get("algorithm")
                else "hydroseason.detect_hydrological_years"
            ),
            parameters=tuple(sorted(resolved_hydroyear_parameters.items())),
        )

        channel_raw = _section(source, "channel", {"source", "node_source"})
        channel = ChannelConfig(
            source=channel_raw.get("source"),
            node_source=channel_raw.get("node_source"),
        )

        state_raw = _section(
            source,
            "state",
            {"enabled", "connectivity_metric", "connectivity_threshold"},
        )
        state = StateConfig(
            enabled=bool(state_raw.get("enabled", False)),
            connectivity_metric=state_raw.get("connectivity_metric"),
            connectivity_threshold=(
                None
                if state_raw.get("connectivity_threshold") is None
                else float(state_raw["connectivity_threshold"])
            ),
        )
        if state.enabled and (
            state.connectivity_metric is None or state.connectivity_threshold is None
        ):
            raise ConfigError(
                "state.enabled requires state.connectivity_metric and "
                "state.connectivity_threshold"
            )
        if state.connectivity_metric is not None and state.connectivity_metric not in {
            "RC",
            "LPSEC",
            "LPI",
            "DCI",
        }:
            raise ConfigError(
                "state.connectivity_metric must be one of RC, LPSEC, LPI, DCI"
            )

        connectivity_raw = _section(source, "connectivity", {"edge_rule"})
        connectivity = ConnectivityConfig(edge_rule=connectivity_raw.get("edge_rule"))

        compute_raw = _section(
            source,
            "compute",
            {
                "accelerator",
                "cuda_strict",
                "target_chunk_bytes",
                "worker_memory_fraction",
                "checkpoint",
                "scheduler",
                "workers",
                "scheduler_address",
                "checkpoint_path",
            },
        )
        compute = ComputeConfig(
            accelerator=str(compute_raw.get("accelerator", "auto")),
            cuda_strict=bool(compute_raw.get("cuda_strict", False)),
            target_chunk_bytes=(
                None
                if compute_raw.get("target_chunk_bytes") is None
                else int(compute_raw["target_chunk_bytes"])
            ),
            worker_memory_fraction=(
                None
                if compute_raw.get("worker_memory_fraction") is None
                else _fraction(
                    compute_raw["worker_memory_fraction"],
                    "compute.worker_memory_fraction",
                )
            ),
            checkpoint=str(compute_raw.get("checkpoint", "zarr")),
            scheduler=str(compute_raw.get("scheduler", "local")),
            workers=int(compute_raw.get("workers", 1)),
            scheduler_address=compute_raw.get("scheduler_address"),
            checkpoint_path=compute_raw.get("checkpoint_path"),
        )
        if compute.accelerator not in {"none", "auto", "cuda"}:
            raise ConfigError(
                f"compute.accelerator has unsupported value: {compute.accelerator}"
            )
        if compute.cuda_strict and compute.accelerator != "cuda":
            raise ConfigError("compute.cuda_strict requires accelerator='cuda'")
        if compute.workers < 1:
            raise ConfigError("compute.workers must be at least 1")

        output_raw = _section(
            source,
            "output",
            {
                "formats",
                "include_patch_table",
                "include_vectors",
                "output_dir",
                "spatial_products",
                "raster_formats",
            },
        )
        output_formats = _literal_sequence(
            output_raw.get("formats", ("parquet",)),
            path="output.formats",
            allowed=OUTPUT_FORMATS,
        )
        include_vectors_raw = output_raw.get("include_vectors", False)
        include_vectors_flag = bool(include_vectors_raw)
        spatial_products_raw = output_raw.get("spatial_products")
        if spatial_products_raw is None:
            spatial_products: tuple[str, ...] = (
                ("monthly_pools",) if include_vectors_flag else ()
            )
        else:
            spatial_products = _literal_sequence(
                spatial_products_raw,
                path="output.spatial_products",
                allowed=SPATIAL_PRODUCTS,
            )
        if include_vectors_flag and "monthly_pools" not in spatial_products:
            raise ConfigError(
                "output.include_vectors conflicts with output.spatial_products"
            )
        raster_formats = _literal_sequence(
            output_raw.get("raster_formats", ("geotiff",)),
            path="output.raster_formats",
            allowed=RASTER_FORMATS,
        )
        output_dir = output_raw.get("output_dir")
        if spatial_products and not output_dir:
            raise ConfigError(
                "output.output_dir is required when output.spatial_products is non-empty"
            )
        output = OutputConfig(
            formats=output_formats,
            include_patch_table=bool(output_raw.get("include_patch_table", False)),
            include_vectors=include_vectors_flag or "monthly_pools" in spatial_products,
            output_dir=output_dir,
            spatial_products=spatial_products,
            raster_formats=raster_formats or ("geotiff",),
        )

        profiles_raw = source.get("metric_profiles", ("all_available",))
        if isinstance(profiles_raw, str):
            raise ConfigError("metric_profiles must be a sequence, not a string")
        metric_profiles = tuple(sorted(set(str(item) for item in profiles_raw)))
        if not metric_profiles:
            raise ConfigError("metric_profiles must not be empty")

        overrides_raw = _section(
            source, "metric_overrides", {"add", "remove", "reasons"}
        )
        reasons_raw = _mapping(
            overrides_raw.get("reasons", {}), "metric_overrides.reasons"
        )
        add_raw = overrides_raw.get("add", ())
        remove_raw = overrides_raw.get("remove", ())
        metric_overrides = MetricOverrides(
            add=tuple(sorted(set(add_raw))),
            remove=tuple(sorted(set(remove_raw))),
            reasons=tuple(
                sorted((str(key), str(value)) for key, value in reasons_raw.items())
            ),
        )
        _validate_override_ids(
            add=metric_overrides.add,
            remove=metric_overrides.remove,
            add_raw=add_raw,
            remove_raw=remove_raw,
        )

        return cls(
            config_schema_version=config_schema_version,
            input=input_config,
            temporal=temporal,
            run_label=source.get("run_label"),
            metric_profiles=metric_profiles,
            metric_overrides=metric_overrides,
            validity=validity,
            spatial=spatial,
            patches=patches,
            persistence=persistence,
            zones=zones,
            riverscape=riverscape,
            dynamics=dynamics,
            hydroyear=hydroyear,
            channel=channel,
            state=state,
            connectivity=connectivity,
            compute=compute,
            output=output,
        )

    def scientific_config(self) -> dict[str, Any]:
        """Return the canonical scientifically meaningful configuration."""

        return {
            "channel": {
                "node_source": self.channel.node_source,
                "source": self.channel.source,
            },
            "connectivity": {"edge_rule": self.connectivity.edge_rule},
            "dynamics": {
                "composite_sensitivity_tolerance_pp": (
                    self.dynamics.composite_sensitivity_tolerance_pp
                ),
                "contraction_method": self.dynamics.contraction_method,
                "minimum_points": self.dynamics.minimum_points,
                "reconnection_lpi_threshold_pct": (
                    self.dynamics.reconnection_lpi_threshold_pct
                ),
                "reconnection_lpsec_threshold_pct": (
                    self.dynamics.reconnection_lpsec_threshold_pct
                ),
            },
            "hash_algorithm_version": HASH_ALGORITHM_VERSION,
            "hydroyear": {
                "algorithm": self.hydroyear.algorithm,
                "parameters": dict(self.hydroyear.parameters),
            },
            "input": {
                "kind": self.input.kind,
                "probability_source": self.input.probability_source,
                "threshold_method": self.input.threshold_method,
                "variable_map": dict(self.input.variable_map),
                "water_threshold": self.input.water_threshold,
            },
            "metric_overrides": {
                "add": list(self.metric_overrides.add),
                "reasons": dict(self.metric_overrides.reasons),
                "remove": list(self.metric_overrides.remove),
            },
            "metric_profiles": list(self.metric_profiles),
            "patches": {
                "connectivity_rule": self.patches.connectivity_rule,
                "min_patch_pixels": self.patches.min_patch_pixels,
                "width_resolution_floor_pixels": (
                    self.patches.width_resolution_floor_pixels
                ),
            },
            "persistence": {"refuge_threshold": self.persistence.refuge_threshold},
            "riverscape": {
                "alignment_quantile": self.riverscape.alignment_quantile,
                "bare_band": self.riverscape.bare_band,
                "bare_threshold_floor_pct": self.riverscape.bare_threshold_floor_pct,
                "bare_year_fraction": self.riverscape.bare_year_fraction,
                "bridge_enabled": self.riverscape.bridge_enabled,
                "bridge_max_cost_per_m": self.riverscape.bridge_max_cost_per_m,
                "bridge_max_length_m": self.riverscape.bridge_max_length_m,
                "bridge_rem_max_m": self.riverscape.bridge_rem_max_m,
                "corridor_max_m": self.riverscape.corridor_max_m,
                "corridor_min_m": self.riverscape.corridor_min_m,
                "dem_band": self.riverscape.dem_band,
                "dem_product": self.riverscape.dem_product,
                "envelope_h_max_m": self.riverscape.envelope_h_max_m,
                "envelope_min_bin_pixels": self.riverscape.envelope_min_bin_pixels,
                "envelope_quantile": self.riverscape.envelope_quantile,
                "f_chan_high": self.riverscape.f_chan_high,
                "f_seed": self.riverscape.f_seed,
                "fc_product": self.riverscape.fc_product,
                "green_band": self.riverscape.green_band,
                "h_chan_m": self.riverscape.h_chan_m,
                "include_line_fallback_in_channel": (
                    self.riverscape.include_line_fallback_in_channel
                ),
                "min_channel_confidence": self.riverscape.min_channel_confidence,
                "mode": self.riverscape.mode,
                "narrow_width_px": self.riverscape.narrow_width_px,
                "npv_band": self.riverscape.npv_band,
                "profile_bin_m": self.riverscape.profile_bin_m,
                "profile_percentile": self.riverscape.profile_percentile,
                "rem_k": self.riverscape.rem_k,
                "rem_max_distance_m": self.riverscape.rem_max_distance_m,
                "riparian_green_pct": self.riverscape.riparian_green_pct,
                "slope_max_deg": self.riverscape.slope_max_deg,
                "trough_depth_m": self.riverscape.trough_depth_m,
                "trough_radius_m": self.riverscape.trough_radius_m,
                "waterbodies_source": self.riverscape.waterbodies_source,
                "width_growth_factor": self.riverscape.width_growth_factor,
            },
            "spatial": {
                "area_method": self.spatial.area_method,
                "target_crs": self.spatial.target_crs,
                "windowing": {
                    "length_m": self.spatial.windowing.length_m,
                    "mode": self.spatial.windowing.mode,
                },
            },
            "state": {
                "connectivity_metric": self.state.connectivity_metric,
                "connectivity_threshold": self.state.connectivity_threshold,
                "enabled": self.state.enabled,
            },
            "temporal": {
                "composite_owner": self.temporal.composite_owner,
                "input_cadence": self.temporal.input_cadence,
                "monthly_composite": self.temporal.monthly_composite,
            },
            "validity": {
                "low_support_behavior": self.validity.low_support_behavior,
                "min_valid_fraction_month": self.validity.min_valid_fraction_month,
                "min_valid_obs": self.validity.min_valid_obs,
                "policy": self.validity.policy,
            },
            "zones": {
                "t_persist": self.zones.t_persist,
                "t_season": self.zones.t_season,
            },
            "scientific_hash_schema_version": SCIENTIFIC_HASH_SCHEMA_VERSION,
        }

    def validate_output_preflight(self) -> None:
        """Validate optional output writers and spatial export prerequisites."""

        if "netcdf" in self.output.raster_formats and not _netcdf_writer_available():
            raise ConfigError(
                "output.raster_formats includes 'netcdf' but no NetCDF writer is "
                "installed; install the optional 'netcdf4' or 'h5netcdf' package"
            )

    def execution_config(self) -> dict[str, Any]:
        """Return execution and output settings excluded from ``config_hash``."""

        return {
            "compute": {
                "accelerator": self.compute.accelerator,
                "checkpoint": self.compute.checkpoint,
                "checkpoint_path": self.compute.checkpoint_path,
                "cuda_strict": self.compute.cuda_strict,
                "scheduler": self.compute.scheduler,
                "scheduler_address": self.compute.scheduler_address,
                "target_chunk_bytes": self.compute.target_chunk_bytes,
                "worker_memory_fraction": self.compute.worker_memory_fraction,
                "workers": self.compute.workers,
            },
            "output": {
                "formats": list(self.output.formats),
                "include_patch_table": self.output.include_patch_table,
                "include_vectors": self.output.include_vectors,
                "output_dir": self.output.output_dir,
                "raster_formats": list(self.output.raster_formats),
                "spatial_products": list(self.output.spatial_products),
            },
        }

    @property
    def config_hash(self) -> str:
        return _sha256_json(self.scientific_config())

    @property
    def execution_hash(self) -> str:
        return _sha256_json(self.execution_config())


__all__ = [
    "ACCEPTED_CONFIG_SCHEMA_VERSIONS",
    "ConfigError",
    "HASH_ALGORITHM_VERSION",
    "HydroConfig",
    "LowSupportBehavior",
    "RiverscapeConfig",
    "SCIENTIFIC_HASH_SCHEMA_VERSION",
]
