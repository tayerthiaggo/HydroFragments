"""``build_landform`` orchestration: stub loaders, real Phase 3-5 kernels.

Loaders are injected through ``build_landform``'s own keyword parameters
(spec 6a section 3.2) rather than monkeypatched, so no test here touches
STAC/WFS. The kernels themselves are NOT stubbed: this module's contract is
call order, argument forwarding, and packaging, and the only honest way to
prove the order is right is to let the real kernels consume each other's
output. Every expected value is derived by hand in the plan's "Test fixture
arithmetic" section.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from affine import Affine

gpd = pytest.importorskip("geopandas")
xr = pytest.importorskip("xarray")
pytest.importorskip("rasterio")
pytest.importorskip("skimage")
from shapely.geometry import LineString

from hydrofragments.config import RiverscapeConfig
from hydrofragments.io.riverscape_sources import RiverscapeSourceUnavailable
from hydrofragments.riverscape import pipeline as pipeline_module
from hydrofragments.riverscape.bridging import BridgeResult, Gap, GapSearchResult
from hydrofragments.riverscape.channel import RULE_CONNECTED_WATER
from hydrofragments.riverscape.codes import (
    LANDFORM_IN_CHANNEL,
    LANDFORM_NON_RIVERINE,
    LANDFORM_OFF_CHANNEL_RIVERINE,
    LANDFORM_OUTSIDE,
)
from hydrofragments.riverscape.pipeline import (
    CHANNEL_CONFIDENCE_NODATA,
    CHANNEL_SOURCE_BRIDGED,
    CHANNEL_SOURCE_NONE,
    CHANNEL_SOURCE_OBSERVED,
    _merge_degraded_reasons,
    _package_channel_confidence,
    _package_channel_source,
    build_landform,
    validate_drainage_columns,
    with_grid,
)

SHAPE = (12, 20)
PIXEL_M = 30.0
TRANSFORM = Affine(PIXEL_M, 0.0, 0.0, 0.0, -PIXEL_M, 360.0)
CRS = "EPSG:3577"
YEARS = (2019, 2021)
BARE_BAND = "bs_pc_50"
GREEN_BAND = "pv_pc_50"
NPV_BAND = "npv_pc_50"


def _cfg(**overrides) -> RiverscapeConfig:
    """Tiny-grid config. Every override is justified in the plan's fixture notes."""
    values = {
        "bare_band": BARE_BAND,
        "green_band": GREEN_BAND,
        "npv_band": NPV_BAND,
        "f_seed": 0.05,
        "f_chan_high": 0.10,
        "corridor_min_m": 60.0,
        "corridor_max_m": 300.0,
        "h_chan_m": 1.0,
        "trough_depth_m": 0.5,
        "trough_radius_m": 60.0,
        "profile_bin_m": 120.0,
        "rem_k": 4,
        "rem_max_distance_m": 2000.0,
        "envelope_min_bin_pixels": 1,
        "slope_max_deg": 2.0,
        "min_channel_confidence": 2,
        "bridge_enabled": False,
    }
    values.update(overrides)
    return RiverscapeConfig(**values)


def _domain_and_frequency() -> tuple[np.ndarray, np.ndarray]:
    domain = np.zeros(SHAPE, dtype=bool)
    domain[3:8, 1:19] = True
    frequency = np.full(SHAPE, 1.0, dtype=float)
    frequency[4:7, 1:19] = 60.0
    return domain, frequency


def _drainage() -> "gpd.GeoDataFrame":
    return gpd.GeoDataFrame(
        {
            "HydroID": [1],
            "From_Node": [1],
            "To_Node": [2],
            "NextDownID": [-1],
            "UpstrDArea": [5000.0],
        },
        geometry=[LineString([(75.0, 195.0), (525.0, 195.0)])],
        crs=CRS,
    )


def _reach_context() -> tuple[np.ndarray, dict[int, str], dict[int, float]]:
    return np.ones(SHAPE, dtype=np.int32), {1: "1"}, {1: 5000.0}


def _dem() -> "xr.DataArray":
    dem = np.full(SHAPE, 10.0, dtype=np.float32)
    dem[4:7, :] = 8.0
    return xr.DataArray(dem, dims=("y", "x"))


def _fc() -> dict[str, "xr.DataArray"]:
    bare = np.full((3, *SHAPE), 5.0, dtype=float)
    bare[:, 4:7, :] = 40.0
    return {
        BARE_BAND: xr.DataArray(bare, dims=("time", "y", "x")),
        GREEN_BAND: xr.DataArray(np.full((3, *SHAPE), 10.0), dims=("time", "y", "x")),
        NPV_BAND: xr.DataArray(np.full((3, *SHAPE), 10.0), dims=("time", "y", "x")),
    }


def _waterbodies() -> "gpd.GeoDataFrame":
    return gpd.GeoDataFrame(geometry=[], crs=CRS)


def _loaders(**overrides):
    loaders = {
        "load_dem": lambda geobox, *, product, band: _dem(),
        "load_fc_percentiles": lambda geobox, *, product, bands, years: _fc(),
        "load_waterbodies": lambda bounds, crs, *, source=None: _waterbodies(),
    }
    loaders.update(overrides)
    return loaders


def _build(cfg=None, *, domain=None, frequency=None, **loader_overrides):
    labels, reach_keys, upstr_darea = _reach_context()
    default_domain, default_frequency = _domain_and_frequency()
    return build_landform(
        default_domain if domain is None else domain,
        default_frequency if frequency is None else frequency,
        _drainage(),
        labels,
        reach_keys,
        upstr_darea,
        geobox=SimpleNamespace(),
        transform=TRANSFORM,
        pixel_m=PIXEL_M,
        cfg=cfg or _cfg(),
        years=YEARS,
        **_loaders(**loader_overrides),
    )


# --------------------------------------------------------------------------
# Happy path (spec section 7.1, bullet 1 and bullet 4)
# --------------------------------------------------------------------------


def test_happy_path_emits_in_channel_and_off_channel_riverine() -> None:
    result = _build()

    assert result.landform.dtype == np.uint8
    assert set(np.unique(result.landform).tolist()) == {
        LANDFORM_OUTSIDE,
        LANDFORM_IN_CHANNEL,
        LANDFORM_OFF_CHANNEL_RIVERINE,
    }
    assert not (result.landform == LANDFORM_NON_RIVERINE).any()
    assert result.landform[5, 10] == LANDFORM_IN_CHANNEL
    assert result.landform[3, 10] == LANDFORM_OFF_CHANNEL_RIVERINE
    assert result.landform[0, 0] == LANDFORM_OUTSIDE


def test_happy_path_channel_source_marks_observed_only() -> None:
    result = _build()
    observed = result.landform == LANDFORM_IN_CHANNEL

    assert result.channel_source.dtype == np.uint8
    assert (result.channel_source[observed] == CHANNEL_SOURCE_OBSERVED).all()
    assert (result.channel_source[~observed] == CHANNEL_SOURCE_NONE).all()
    assert not (result.channel_source == CHANNEL_SOURCE_BRIDGED).any()
    assert not result.bridged_mask.any()


def test_happy_path_confidence_is_255_outside_channel() -> None:
    result = _build()
    channel = result.landform == LANDFORM_IN_CHANNEL

    assert result.channel_confidence.dtype == np.uint8
    assert (result.channel_confidence[channel] == 3).all()
    assert (result.channel_confidence[~channel] == CHANNEL_CONFIDENCE_NODATA).all()


def test_happy_path_rule_id_is_zero_outside_channel() -> None:
    result = _build()
    channel = result.landform == LANDFORM_IN_CHANNEL

    assert (result.rule_id[channel] == RULE_CONNECTED_WATER).all()
    assert (result.rule_id[~channel] == 0).all()


def test_happy_path_rem_is_flat_channel_relative() -> None:
    result = _build()

    assert result.rem.dtype == np.float32
    assert result.rem[5, 10] == pytest.approx(0.0, abs=1e-5)
    assert result.rem[3, 10] == pytest.approx(2.0, abs=1e-5)


def test_happy_path_degraded_reasons_are_sorted_unique() -> None:
    result = _build()

    assert result.degraded_reasons == (
        "bare_threshold_fallback",
        "envelope_single_bin",
    )
    assert list(result.degraded_reasons) == sorted(set(result.degraded_reasons))


def test_happy_path_provenance_carries_stable_keys_for_phase_6b() -> None:
    result = _build()
    provenance = result.provenance

    assert provenance["channel_ruleset_version"] == "1.0.0"
    assert provenance["waterbody_ruleset_version"] == "1.0.0"
    assert provenance["riverine_ruleset_version"] == "1.0.0"
    assert provenance["dem_product"] == "ga_srtm_dem1sv1_0"
    assert provenance["fc_bands"] == (BARE_BAND, GREEN_BAND, NPV_BAND)
    assert provenance["years"] == YEARS
    assert provenance["bridge_enabled"] is False
    assert provenance["bare_threshold_pct"] == pytest.approx(30.0)
    assert provenance["envelope"]["n_bins"] == 1
    assert provenance["pixel_counts"]["domain"] == 90
    assert provenance["pixel_counts"]["observed_channel"] == 54
    assert provenance["pixel_counts"]["bridged_channel"] == 0
    assert provenance["pixel_counts"]["in_channel"] == 54
    assert provenance["pixel_counts"]["non_riverine"] == 0
    assert provenance["bridge_counts_by_cause"] == {}
    assert provenance["unbridged_counts_by_reason"] == {}
    assert provenance["reach_counts"] == {
        "total": 1,
        "line_fallback": 0,
        "multithread": 0,
    }


def test_envelope_and_bridge_frame_are_carried_through() -> None:
    result = _build()

    assert result.envelope.n_bins == 1
    assert result.envelope.a == pytest.approx(2.0)
    assert result.envelope.b == pytest.approx(0.0)
    assert result.unbridged == ()
    assert result.channel_bridges.empty
    assert "gap_cause" in result.channel_bridges.columns
    assert result.grid is None


# --------------------------------------------------------------------------
# Bridging (spec section 7.1 bullet 1, MUTANT 4)
# --------------------------------------------------------------------------


def test_bridged_pixels_are_channel_source_2_and_in_channel(monkeypatch) -> None:
    """MUTANT 4: painting bridged pixels as observed (1) must fail here."""
    bridged_rc = (1, 10)

    def fake_find_gaps(*args, **kwargs):
        return GapSearchResult(
            (
                Gap(
                    gap_id="gap-1",
                    reach_ids=("1",),
                    upstream_anchor=(5, 2),
                    downstream_anchor=(5, 17),
                    upstream_width_m=60.0,
                    downstream_width_m=60.0,
                    straight_length_m=450.0,
                ),
            ),
            (),
            (),
        )

    def fake_bridge_gaps(gaps, observed_channel, cost_inputs, **kwargs):
        painted = np.zeros(np.shape(observed_channel), dtype=bool)
        painted[bridged_rc] = True
        frame = gpd.GeoDataFrame(
            {"gap_id": ["gap-1"], "gap_cause": ["vegetated"]},
            geometry=[LineString([(75.0, 315.0), (525.0, 315.0)])],
            crs=CRS,
        )
        return BridgeResult(painted, frame, (), ())

    monkeypatch.setattr(pipeline_module, "find_gaps", fake_find_gaps)
    monkeypatch.setattr(pipeline_module, "bridge_gaps", fake_bridge_gaps)

    result = _build(_cfg(bridge_enabled=True))

    assert result.bridged_mask[bridged_rc]
    assert result.channel_source[bridged_rc] == CHANNEL_SOURCE_BRIDGED
    assert result.channel_source[5, 10] == CHANNEL_SOURCE_OBSERVED
    assert result.landform[bridged_rc] == LANDFORM_IN_CHANNEL
    assert result.provenance["pixel_counts"]["bridged_channel"] == 1
    assert result.provenance["bridge_counts_by_cause"] == {"vegetated": 1}


def test_bridged_mask_never_overlaps_the_observed_domain(monkeypatch) -> None:
    """The LandformResult invariant classify_hydroperiod depends on."""

    def fake_find_gaps(*args, **kwargs):
        return GapSearchResult((), (), ())

    def fake_bridge_gaps(gaps, observed_channel, cost_inputs, **kwargs):
        painted = np.ones(np.shape(observed_channel), dtype=bool)
        return BridgeResult(painted, _waterbodies(), (), ())

    monkeypatch.setattr(pipeline_module, "find_gaps", fake_find_gaps)
    monkeypatch.setattr(pipeline_module, "bridge_gaps", fake_bridge_gaps)

    domain, _frequency = _domain_and_frequency()
    result = _build(_cfg(bridge_enabled=True))

    assert not (result.bridged_mask & domain).any()
    assert not (result.bridged_mask & (result.channel_source == CHANNEL_SOURCE_OBSERVED)).any()


def test_bridge_disabled_never_calls_the_bridging_kernels(monkeypatch) -> None:
    def explode(*args, **kwargs):
        raise AssertionError("bridging must not run when bridge_enabled is False")

    monkeypatch.setattr(pipeline_module, "find_gaps", explode)
    monkeypatch.setattr(pipeline_module, "bridge_gaps", explode)

    result = _build(_cfg(bridge_enabled=False))

    assert not result.bridged_mask.any()
    assert not (result.channel_source == CHANNEL_SOURCE_BRIDGED).any()


# --------------------------------------------------------------------------
# Loader failures (spec section 7.1 bullet 2, MUTANT 5)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "loader",
    ["load_dem", "load_fc_percentiles", "load_waterbodies"],
)
def test_loader_failure_propagates_unchanged(loader: str) -> None:
    """MUTANT 5: catching RiverscapeSourceUnavailable inside build_landform
    must fail here -- mode behaviour is the caller's decision, not this
    function's (spec section 4, closing paragraph)."""

    def raising(*args, **kwargs):
        raise RiverscapeSourceUnavailable(f"{loader} unavailable")

    with pytest.raises(RiverscapeSourceUnavailable, match=f"{loader} unavailable"):
        _build(**{loader: raising})


def test_missing_fc_band_is_a_source_failure() -> None:
    def partial_fc(geobox, *, product, bands, years):
        loaded = _fc()
        del loaded[GREEN_BAND]
        return loaded

    with pytest.raises(RiverscapeSourceUnavailable, match=GREEN_BAND):
        _build(load_fc_percentiles=partial_fc)


# --------------------------------------------------------------------------
# Input validation (spec section 7.1 bullet 3, section 3.2)
# --------------------------------------------------------------------------


def test_shape_mismatch_raises_value_error() -> None:
    domain, _frequency = _domain_and_frequency()
    with pytest.raises(ValueError, match="domain and frequency must share shape"):
        _build(frequency=np.ones((4, 4), dtype=float))

    with pytest.raises(ValueError, match="domain must be a boolean array"):
        _build(domain=np.ones(SHAPE, dtype=np.uint8))

    with pytest.raises(ValueError, match="domain must be a 2-D array"):
        _build(domain=np.ones((2, *SHAPE), dtype=bool))


def test_non_finite_frequency_inside_domain_raises() -> None:
    domain, frequency = _domain_and_frequency()
    frequency[5, 10] = np.nan
    with pytest.raises(ValueError, match="frequency must be finite inside the domain"):
        _build(frequency=frequency)


def test_dem_off_grid_raises_value_error() -> None:
    def wrong_shape_dem(geobox, *, product, band):
        return xr.DataArray(np.zeros((4, 4), dtype=np.float32), dims=("y", "x"))

    with pytest.raises(ValueError, match="does not match the zoning grid"):
        _build(load_dem=wrong_shape_dem)


@pytest.mark.parametrize(
    "column", ["HydroID", "NextDownID", "UpstrDArea", "From_Node", "To_Node"]
)
def test_missing_drainage_column_is_a_programming_error(column: str) -> None:
    """Spec section 3.2: missing columns raise ValueError, never a mode fallback."""
    drainage = _drainage().drop(columns=[column])
    with pytest.raises(ValueError, match=column):
        validate_drainage_columns(drainage)


def test_validate_drainage_columns_accepts_the_full_schema() -> None:
    assert validate_drainage_columns(_drainage()) is None


def test_drainage_crs_must_match_geobox() -> None:
    domain, frequency = _domain_and_frequency()
    labels, reach_keys, upstr_darea = _reach_context()
    from pyproj import Transformer

    to_wgs84 = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    x1, y1 = to_wgs84.transform(75.0, 195.0)
    x2, y2 = to_wgs84.transform(525.0, 195.0)
    drainage_wgs84 = gpd.GeoDataFrame(
        {
            "HydroID": [1],
            "From_Node": [1],
            "To_Node": [2],
            "NextDownID": [-1],
            "UpstrDArea": [5000.0],
        },
        geometry=[LineString([(x1, y1), (x2, y2)])],
        crs="EPSG:4326",
    )

    with pytest.raises(ValueError, match="drainage CRS must match"):
        build_landform(
            domain,
            frequency,
            drainage_wgs84,
            labels,
            reach_keys,
            upstr_darea,
            geobox=SimpleNamespace(crs=CRS),
            transform=TRANSFORM,
            pixel_m=PIXEL_M,
            cfg=_cfg(),
            years=YEARS,
            **_loaders(),
        )


def test_empty_drainage_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one line feature"):
        validate_drainage_columns(_drainage().iloc[0:0])


# --------------------------------------------------------------------------
# Packaging helpers, tested directly
# --------------------------------------------------------------------------


def test_package_channel_source_prefers_observed_over_bridged() -> None:
    observed = np.array([[True, False, False]], dtype=bool)
    bridged = np.array([[True, True, False]], dtype=bool)

    source = _package_channel_source(observed, bridged)

    assert source.dtype == np.uint8
    assert source.tolist() == [
        [CHANNEL_SOURCE_OBSERVED, CHANNEL_SOURCE_BRIDGED, CHANNEL_SOURCE_NONE]
    ]


def test_package_channel_source_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="must share shape"):
        _package_channel_source(np.zeros((2, 2), bool), np.zeros((3, 3), bool))


def test_package_channel_confidence_masks_off_channel_to_255() -> None:
    confidence = np.array([[3, 2, 1]], dtype=np.uint8)
    channel = np.array([[True, True, False]], dtype=bool)

    packaged = _package_channel_confidence(confidence, channel)

    assert packaged.tolist() == [[3, 2, CHANNEL_CONFIDENCE_NODATA]]


def test_merge_degraded_reasons_sorts_and_deduplicates() -> None:
    merged = _merge_degraded_reasons(
        ("envelope_single_bin",),
        ("bare_threshold_fallback", "envelope_single_bin"),
        (),
    )
    assert merged == ("bare_threshold_fallback", "envelope_single_bin")


def test_with_grid_attaches_the_phase_6b_export_grid() -> None:
    result = _build()
    grid = SimpleNamespace(height=SHAPE[0], width=SHAPE[1])

    attached = with_grid(result, grid)

    assert result.grid is None
    assert attached.grid is grid
    assert attached.landform is result.landform
