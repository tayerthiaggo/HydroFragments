from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from hydrofragments.hydroperiod import (
    HYDROPERIOD_MARGINAL,
    HYDROPERIOD_OUTSIDE,
    HYDROPERIOD_PERSISTENT,
    HYDROPERIOD_SEASONAL,
    HYDROPERIOD_UNOBSERVED,
    HydroperiodResult,
    classify_hydroperiod,
)
from hydrofragments.spatial.zones import build_zones

HYDROPERIOD_PACKAGE = Path(__file__).resolve().parents[2] / "hydrofragments" / "hydroperiod"


def _single(value: float, *, t_persist: float = 0.50, t_season: float = 0.10) -> int:
    result = classify_hydroperiod(
        np.array([[value]]),
        np.array([[True]]),
        t_persist=t_persist,
        t_season=t_season,
    )
    return int(result.classes[0, 0])


@pytest.mark.parametrize(
    ("frequency", "expected"),
    [
        (0.5, HYDROPERIOD_MARGINAL),
        (9.9, HYDROPERIOD_MARGINAL),
        (10.0, HYDROPERIOD_SEASONAL),
        (45.0, HYDROPERIOD_SEASONAL),
        (50.0, HYDROPERIOD_SEASONAL),
        (50.1, HYDROPERIOD_PERSISTENT),
        (100.0, HYDROPERIOD_PERSISTENT),
    ],
)
def test_percent_boundaries_match_spec(frequency: float, expected: int) -> None:
    assert _single(frequency) == expected


def test_result_carries_method_and_thresholds() -> None:
    result = classify_hydroperiod(
        np.array([[60.0]]), np.array([[True]]), t_persist=0.6, t_season=0.2
    )
    assert isinstance(result, HydroperiodResult)
    assert result.classes.dtype == np.uint8
    assert result.method == "wofs_multiyear"
    assert (result.t_persist, result.t_season) == (0.6, 0.2)


def test_pixels_outside_domain_are_outside_even_if_frequency_is_high() -> None:
    result = classify_hydroperiod(
        np.array([[90.0, 90.0]]),
        np.array([[True, False]]),
        t_persist=0.5,
        t_season=0.1,
    )
    assert result.classes.tolist() == [[HYDROPERIOD_PERSISTENT, HYDROPERIOD_OUTSIDE]]


def test_unobserved_mask_marks_bridged_pixels_outside_domain() -> None:
    result = classify_hydroperiod(
        np.array([[30.0, np.nan, 0.0]]),
        np.array([[True, False, False]]),
        t_persist=0.5,
        t_season=0.1,
        unobserved_mask=np.array([[False, True, False]]),
    )
    assert result.classes.tolist() == [
        [HYDROPERIOD_SEASONAL, HYDROPERIOD_UNOBSERVED, HYDROPERIOD_OUTSIDE]
    ]


def test_unobserved_mask_must_not_overlap_domain() -> None:
    with pytest.raises(ValueError, match="unobserved_mask must not overlap"):
        classify_hydroperiod(
            np.array([[30.0]]),
            np.array([[True]]),
            t_persist=0.5,
            t_season=0.1,
            unobserved_mask=np.array([[True]]),
        )


def test_non_finite_frequency_inside_domain_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite inside the domain"):
        classify_hydroperiod(
            np.array([[np.nan]]), np.array([[True]]), t_persist=0.5, t_season=0.1
        )


@pytest.mark.parametrize(("t_persist", "t_season"), [(0.5, 0.5), (0.4, 0.5), (1.1, 0.1), (0.5, -0.1)])
def test_invalid_thresholds_are_rejected(t_persist: float, t_season: float) -> None:
    with pytest.raises(ValueError, match="t_season < t_persist"):
        classify_hydroperiod(
            np.array([[30.0]]), np.array([[True]]), t_persist=t_persist, t_season=t_season
        )


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="share shape"):
        classify_hydroperiod(
            np.ones((2, 2)), np.ones((2, 3), dtype=bool), t_persist=0.5, t_season=0.1
        )
    with pytest.raises(ValueError, match="share shape"):
        classify_hydroperiod(
            np.ones((2, 2)),
            np.ones((2, 2), dtype=bool),
            t_persist=0.5,
            t_season=0.1,
            unobserved_mask=np.zeros((1, 2), dtype=bool),
        )


def test_matches_build_zones_classes_on_the_same_domain() -> None:
    rng = np.random.default_rng(7)
    frequency = rng.uniform(0.0, 100.0, size=(50, 50))
    frequency[::7, ::5] = 10.0
    frequency[::9, ::4] = 50.0
    count_wet = rng.integers(0, 3, size=(50, 50))
    count_clear = rng.integers(15, 30, size=(50, 50))
    domain = (count_wet > 0) & np.isfinite(frequency) & (count_clear >= 20)

    zones = build_zones(
        frequency, max_wet_mask=count_wet > 0, valid_count=count_clear, min_valid_obs=20
    ).mask
    classes = classify_hydroperiod(frequency, domain, t_persist=0.5, t_season=0.1).classes

    expected = np.select(
        [zones == 2, zones == 3, zones == 4],
        [HYDROPERIOD_PERSISTENT, HYDROPERIOD_SEASONAL, HYDROPERIOD_MARGINAL],
        default=HYDROPERIOD_OUTSIDE,
    )
    np.testing.assert_array_equal(classes, expected)


def test_hydroperiod_package_never_imports_riverscape() -> None:
    offenders: list[str] = []
    for path in sorted(HYDROPERIOD_PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("hydrofragments.riverscape"):
                offenders.append(f"{path.name}: from {node.module}")
            if isinstance(node, ast.Import):
                offenders.extend(
                    f"{path.name}: import {alias.name}"
                    for alias in node.names
                    if alias.name.startswith("hydrofragments.riverscape")
                )
    assert offenders == []
