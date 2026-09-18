"""``zones_from_riverscape``: DEA stats + landform -> riverscape ZoneResult.

``build_landform`` is stubbed (it has its own suite in
``tests/riverscape/test_riverscape_pipeline.py`` and needs network-shaped
loaders); ``wet_domain``, ``classify_hydroperiod`` and ``combine_zones`` all
run for real, because the behaviour under test IS how this adapter wires
those three together -- in particular that the bridged mask reaches
``classify_hydroperiod`` as ``unobserved_mask``.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
gpd = pytest.importorskip("geopandas")
pytest.importorskip("rioxarray")
from affine import Affine
from shapely.geometry import LineString

from hydrofragments.config import HydroConfig
from hydrofragments.riverscape.riverine import FloodplainEnvelope
from hydrofragments.riverscape.pipeline import LandformResult
from hydrofragments.spatial import zones as zones_module
from hydrofragments.spatial.zones import zones_from_riverscape

SHAPE = (3, 4)
PIXEL_M = 30.0
TRANSFORM = Affine(PIXEL_M, 0.0, 0.0, 0.0, -PIXEL_M, 120.0)
CRS = "EPSG:3577"
PRODUCT = "ga_ls_wo_fq_myear_3"

# Domain is exactly the six cells with non-zero frequency: count_wet > 0 and
# count_clear (40) >= the default min_valid_obs (20).
FREQUENCY = np.array(
    [
        [80.0, 30.0, 5.0, 0.0],
        [80.0, 30.0, 5.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)
EXPECTED_DOMAIN = np.array(
    [
        [True, True, True, False],
        [True, True, True, False],
        [False, False, False, False],
    ]
)


def _config() -> HydroConfig:
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "input": {"kind": "watermask_tsfill"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "max_water",
                "composite_owner": "upstream",
            },
            "riverscape": {"mode": "auto"},
        }
    )


def _stats(*, years=None):
    y = 120.0 - np.arange(SHAPE[0]) * PIXEL_M - PIXEL_M / 2.0
    x = np.arange(SHAPE[1]) * PIXEL_M + PIXEL_M / 2.0
    frequency = xr.DataArray(
        FREQUENCY, dims=("y", "x"), coords={"y": y, "x": x}
    ).rio.write_crs(CRS)
    stats = SimpleNamespace(
        frequency=frequency,
        count_wet=np.where(FREQUENCY > 0, 5, 0),
        count_clear=np.full(SHAPE, 40),
        product=PRODUCT,
        version="0.1.0",
        crs=CRS,
        time_span=None,
        provenance={},
    )
    if years is not None:
        stats.years = years
    return stats


def _drainage() -> "gpd.GeoDataFrame":
    return gpd.GeoDataFrame(
        {
            "HydroID": [1],
            "From_Node": [1],
            "To_Node": [2],
            "NextDownID": [-1],
            "UpstrDArea": [5000.0],
        },
        geometry=[LineString([(15.0, 105.0), (105.0, 105.0)])],
        crs=CRS,
    )


def _landform_result(landform, *, bridged=None, degraded_reasons=()):
    landform = np.asarray(landform, dtype=np.uint8)
    zeros = np.zeros(landform.shape, dtype=np.uint8)
    return LandformResult(
        landform=landform,
        channel_source=zeros.copy(),
        channel_confidence=zeros.copy(),
        rule_id=zeros.copy(),
        rem=np.zeros(landform.shape, dtype=np.float32),
        bridged_mask=(
            np.zeros(landform.shape, dtype=bool) if bridged is None else np.asarray(bridged, bool)
        ),
        channel_bridges=None,
        unbridged=(),
        envelope=FloodplainEnvelope(
            a=2.0,
            b=0.0,
            n_bins=1,
            bin_log_a_mid=(3.7,),
            bin_rem_quantile=(2.0,),
            bin_counts=(90,),
            degraded_reasons=(),
        ),
        provenance={"years": (2019, 2021)},
        degraded_reasons=tuple(degraded_reasons),
    )


def _install_build_landform(monkeypatch, result: LandformResult) -> list[dict]:
    calls: list[dict] = []

    def fake_build_landform(
        domain, frequency, drainage, reach_labels, reach_keys, upstr_darea, **kwargs
    ):
        calls.append(
            {
                "domain": np.asarray(domain).copy(),
                "frequency": np.asarray(frequency).copy(),
                "drainage": drainage,
                "reach_labels": reach_labels,
                "reach_keys": reach_keys,
                "upstr_darea": upstr_darea,
                **kwargs,
            }
        )
        return result

    monkeypatch.setattr(zones_module, "build_landform", fake_build_landform)
    return calls


def _zones(monkeypatch, result: LandformResult, **overrides):
    calls = _install_build_landform(monkeypatch, result)
    kwargs = {
        "drainage": _drainage(),
        "config": _config(),
        "reach_labels": np.ones(SHAPE, dtype=np.int32),
        "reach_keys": {1: "1"},
        "upstr_darea": {1: 5000.0},
        "geobox": SimpleNamespace(),
        "transform": TRANSFORM,
        "pixel_m": PIXEL_M,
        "years": (2019, 2021),
    }
    stats = overrides.pop("stats", None) or _stats()
    kwargs.update(overrides)
    return zones_from_riverscape(stats, **kwargs), calls


# --------------------------------------------------------------------------
# Crosstab and legacy mask (spec section 7.2 bullet 1)
# --------------------------------------------------------------------------


def test_crosstab_and_legacy_mask_follow_the_landform_hydroperiod_table(monkeypatch) -> None:
    landform = [
        [1, 2, 2, 0],
        [2, 2, 2, 0],
        [0, 0, 0, 0],
    ]
    result, _calls = _zones(monkeypatch, _landform_result(landform))

    # hydroperiod: 80 > 50 -> persistent(1); 30 in [10, 50] -> seasonal(2);
    # 5 < 10 -> marginal(3). crosstab = landform * 10 + hydroperiod.
    assert result.crosstab.tolist() == [
        [11, 22, 23, 0],
        [21, 22, 23, 0],
        [0, 0, 0, 0],
    ]
    assert result.mask.tolist() == [
        [1, 3, 4, 0],
        [2, 3, 4, 0],
        [0, 0, 0, 0],
    ]
    assert result.emitted_zones == (1, 2, 3, 4)
    assert result.has_zone_1 is True
    assert result.mode == "riverscape"


def test_result_is_stamped_with_the_dea_product_and_a_grid(monkeypatch) -> None:
    landform = [
        [1, 2, 2, 0],
        [2, 2, 2, 0],
        [0, 0, 0, 0],
    ]
    result, _calls = _zones(monkeypatch, _landform_result(landform))

    assert result.source == PRODUCT
    assert result.grid is not None
    assert (result.grid.height, result.grid.width) == SHAPE
    assert result.as_dataarray().shape == SHAPE


# --------------------------------------------------------------------------
# Bridged pixels (spec section 7.2 bullet 1, MUTANT 3)
# --------------------------------------------------------------------------


def test_bridged_pixel_is_hydroperiod_4_and_zone_1(monkeypatch) -> None:
    """MUTANT 3: dropping unobserved_mask from the classify_hydroperiod call
    makes the bridged pixel hydroperiod 0 while landform says 1, and
    combine_zones raises "landform and hydroperiod disagree on the zoned
    extent" -- so this test fails loudly rather than silently losing code 4.
    """
    landform = [
        [1, 2, 2, 0],
        [2, 2, 2, 0],
        [1, 0, 0, 0],
    ]
    bridged = np.zeros(SHAPE, dtype=bool)
    bridged[2, 0] = True

    result, _calls = _zones(
        monkeypatch, _landform_result(landform, bridged=bridged)
    )

    assert int(result.crosstab[2, 0]) == 14
    assert int(result.mask[2, 0]) == 1
    assert result.has_zone_1 is True


# --------------------------------------------------------------------------
# Degraded reasons (spec section 7.2 bullet 2)
# --------------------------------------------------------------------------


def test_degraded_reasons_are_forwarded_onto_the_zone_result(monkeypatch) -> None:
    landform = [
        [1, 2, 2, 0],
        [2, 2, 2, 0],
        [0, 0, 0, 0],
    ]
    result, _calls = _zones(
        monkeypatch,
        _landform_result(
            landform, degraded_reasons=("bare_threshold_fallback", "envelope_single_bin")
        ),
    )

    assert result.degraded_reasons == (
        "bare_threshold_fallback",
        "envelope_single_bin",
    )


# --------------------------------------------------------------------------
# Argument forwarding into build_landform
# --------------------------------------------------------------------------


def test_build_landform_receives_the_wet_domain_and_stats_frequency(monkeypatch) -> None:
    landform = [
        [1, 2, 2, 0],
        [2, 2, 2, 0],
        [0, 0, 0, 0],
    ]
    _result, calls = _zones(monkeypatch, _landform_result(landform))

    assert len(calls) == 1
    call = calls[0]
    assert call["domain"].dtype == np.bool_
    assert call["domain"].tolist() == EXPECTED_DOMAIN.tolist()
    np.testing.assert_allclose(call["frequency"], FREQUENCY)
    assert call["reach_keys"] == {1: "1"}
    assert call["upstr_darea"] == {1: 5000.0}
    assert call["transform"] == TRANSFORM
    assert call["pixel_m"] == PIXEL_M
    assert call["years"] == (2019, 2021)
    assert call["cfg"].mode == "auto"


def test_years_are_resolved_from_stats_when_not_supplied(monkeypatch) -> None:
    landform = [
        [1, 2, 2, 0],
        [2, 2, 2, 0],
        [0, 0, 0, 0],
    ]
    _result, calls = _zones(
        monkeypatch,
        _landform_result(landform),
        stats=_stats(years=(2018, 2019, 2020, 2021)),
        years=None,
    )

    assert calls[0]["years"] == (2018, 2021)


def test_missing_years_without_stats_years_raises(monkeypatch) -> None:
    landform = [
        [1, 2, 2, 0],
        [2, 2, 2, 0],
        [0, 0, 0, 0],
    ]
    _install_build_landform(monkeypatch, _landform_result(landform))

    with pytest.raises(ValueError, match="years"):
        zones_from_riverscape(
            _stats(),
            drainage=_drainage(),
            config=_config(),
            reach_labels=np.ones(SHAPE, dtype=np.int32),
            reach_keys={1: "1"},
            upstr_darea={1: 5000.0},
            geobox=SimpleNamespace(),
            transform=TRANSFORM,
            years=None,
        )
