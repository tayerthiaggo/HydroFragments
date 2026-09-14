from __future__ import annotations

import numpy as np
import pytest

from hydrofragments.spatial.zones import ZoneResult, build_zones, combine_zones

VALID_PAIRS = [
    # (landform, hydroperiod, legacy_zone, crosstab)
    (0, 0, 0, 0),
    (1, 1, 1, 11),
    (1, 2, 1, 12),
    (1, 3, 1, 13),
    (1, 4, 1, 14),
    (2, 1, 2, 21),
    (2, 2, 3, 22),
    (2, 3, 4, 23),
    (3, 1, 0, 31),
    (3, 2, 0, 32),
    (3, 3, 0, 33),
]


@pytest.mark.parametrize(("landform", "hydroperiod", "zone", "crosstab"), VALID_PAIRS)
def test_mapping_table_matches_spec(landform: int, hydroperiod: int, zone: int, crosstab: int) -> None:
    result = combine_zones(np.array([[landform]]), np.array([[hydroperiod]]))

    assert int(result.mask[0, 0]) == zone
    assert int(result.crosstab[0, 0]) == crosstab


def test_combined_result_contract() -> None:
    landform = np.array([[1, 2], [3, 0]], dtype=np.uint8)
    hydroperiod = np.array([[4, 1], [2, 0]], dtype=np.uint8)

    result = combine_zones(landform, hydroperiod, source="riverscape", degraded_reasons=("envelope_single_bin",))

    assert result.mask.dtype == np.uint8
    assert result.crosstab.dtype == np.uint8
    assert result.mask.tolist() == [[1, 2], [0, 0]]
    assert result.crosstab.tolist() == [[14, 21], [32, 0]]
    assert result.emitted_zones == (1, 2, 3, 4)
    assert result.has_zone_1 is True
    assert result.mode == "riverscape"
    assert result.source == "riverscape"
    assert result.degraded_reasons == ("envelope_single_bin",)


@pytest.mark.parametrize(
    ("landform", "hydroperiod", "message"),
    [
        (0, 1, "disagree on the zoned extent"),
        (1, 0, "disagree on the zoned extent"),
        (2, 0, "disagree on the zoned extent"),
        (2, 4, "unobserved hydroperiod is only valid for in-channel"),
        (3, 4, "unobserved hydroperiod is only valid for in-channel"),
    ],
)
def test_inconsistent_layers_are_rejected(landform: int, hydroperiod: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        combine_zones(np.array([[landform]]), np.array([[hydroperiod]]))


def test_invalid_codes_are_rejected() -> None:
    with pytest.raises(ValueError, match="landform has invalid codes"):
        combine_zones(np.array([[5]]), np.array([[1]]))
    with pytest.raises(ValueError, match="hydroperiod has invalid codes"):
        combine_zones(np.array([[1]]), np.array([[7]]))


def test_shape_mismatch_and_dimensionality_are_rejected() -> None:
    with pytest.raises(ValueError, match="share shape"):
        combine_zones(np.zeros((2, 2)), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="2-D"):
        combine_zones(np.zeros(4), np.zeros(4))


def test_zone_result_defaults_keep_occurrence_contract() -> None:
    result = build_zones(
        np.array([[90.0]]),
        max_wet_mask=np.array([[True]]),
        valid_count=np.array([[20]]),
    )
    assert result.mode == "occurrence"
    assert result.crosstab is None
    assert result.degraded_reasons == ()


def test_zone_result_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        ZoneResult(mask=np.zeros((1, 1), dtype=np.uint8), emitted_zones=(), has_zone_1=False, mode="hybrid")


def test_riverscape_zone_result_requires_matching_crosstab() -> None:
    mask = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="crosstab"):
        ZoneResult(mask=mask, emitted_zones=(1, 2, 3, 4), has_zone_1=True, mode="riverscape")
    with pytest.raises(ValueError, match="crosstab"):
        ZoneResult(
            mask=mask,
            emitted_zones=(1, 2, 3, 4),
            has_zone_1=True,
            mode="riverscape",
            crosstab=np.zeros((1, 2), dtype=np.uint8),
        )
