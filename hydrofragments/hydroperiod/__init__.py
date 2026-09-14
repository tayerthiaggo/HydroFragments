"""Hydroperiod layer: how often observed-wet pixels are wet."""

from hydrofragments.hydroperiod.classify import HydroperiodResult, classify_hydroperiod
from hydrofragments.hydroperiod.codes import (
    HYDROPERIOD_CODES,
    HYDROPERIOD_MARGINAL,
    HYDROPERIOD_NAMES,
    HYDROPERIOD_OUTSIDE,
    HYDROPERIOD_PERSISTENT,
    HYDROPERIOD_SEASONAL,
    HYDROPERIOD_UNOBSERVED,
)

__all__ = [
    "HYDROPERIOD_CODES",
    "HYDROPERIOD_MARGINAL",
    "HYDROPERIOD_NAMES",
    "HYDROPERIOD_OUTSIDE",
    "HYDROPERIOD_PERSISTENT",
    "HYDROPERIOD_SEASONAL",
    "HYDROPERIOD_UNOBSERVED",
    "HydroperiodResult",
    "classify_hydroperiod",
]
