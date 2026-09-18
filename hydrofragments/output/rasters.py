"""Spatial raster products and verified GeoTIFF / NetCDF writers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr
import zarr

import rioxarray  # noqa: F401 — registers the .rio accessor
from rasterio import windows as rio_windows
from rasterio.enums import Compression
from rasterio.transform import Affine

from hydrofragments.config import HydroConfig
from hydrofragments.metrics.persistence import (
    HydroperiodResult,
    OccurrenceResult,
    RecurrenceResult,
)
from hydrofragments.output.checkpoints import (
    CheckpointError,
    SPATIAL_RASTER_ALGORITHM_VERSION,
    SpatialRasterCheckpoint,
    SpatialRasterCheckpointAccumulator,
)
from hydrofragments.output.spatial import SpatialGrid
from hydrofragments.schema import EdgeFlag

RASTER_TILE_SIZE = 256
NETCDF_SPATIAL_FILENAME = "spatial.nc"
REFUGE_OVERLAP_NODATA = np.uint8(255)

_FLOAT32_NODATA = float("nan")
_UINT32_NODATA = np.uint32(4294967295)
_UINT16_NODATA = np.uint16(65535)
_UINT8_NODATA = np.uint8(255)


class RasterExportError(ValueError):
    """Raised when raster export inputs or artifacts are invalid."""


@dataclass(frozen=True)
class RasterProductContract:
    """Frozen Section 2.4 dtype/nodata/units contract for one raster product."""

    filename: str
    dtype: np.dtype
    nodata: float | int | np.floating | np.integer
    units: str
    codebook: str | None = None


RASTER_PRODUCT_CONTRACTS: dict[str, RasterProductContract] = {
    "occurrence": RasterProductContract(
        filename="occurrence.tif",
        dtype=np.dtype(np.float32),
        nodata=_FLOAT32_NODATA,
        units="percent",
        codebook="0-100",
    ),
    "valid_observation_count": RasterProductContract(
        filename="valid_observation_count.tif",
        dtype=np.dtype(np.uint32),
        nodata=_UINT32_NODATA,
        units="months",
    ),
    "refuge_mask": RasterProductContract(
        filename="refuge_mask.tif",
        dtype=np.dtype(np.uint8),
        nodata=_UINT8_NODATA,
        units="boolean",
        codebook="0=false,1=true",
    ),
    "zones": RasterProductContract(
        filename="zones.tif",
        dtype=np.dtype(np.uint8),
        nodata=np.uint8(0),
        units="zone_code",
        codebook="0=outside/no zone,1-4=configured zone codes",
    ),
    "recurrence": RasterProductContract(
        filename="recurrence.tif",
        dtype=np.dtype(np.float32),
        nodata=_FLOAT32_NODATA,
        units="percent",
        codebook="0-100",
    ),
    "recurrence_valid_year_count": RasterProductContract(
        filename="recurrence_valid_year_count.tif",
        dtype=np.dtype(np.uint16),
        nodata=_UINT16_NODATA,
        units="calendar_years",
    ),
    "hydroperiod": RasterProductContract(
        filename="hydroperiod_by_year.tif",
        dtype=np.dtype(np.float32),
        nodata=_FLOAT32_NODATA,
        units="fraction",
        codebook="0-1",
    ),
    "hydroperiod_valid_month_count": RasterProductContract(
        filename="hydroperiod_valid_month_count_by_year.tif",
        dtype=np.dtype(np.uint8),
        nodata=_UINT8_NODATA,
        units="months",
        codebook="0-12",
    ),
    "refuge_overlap": RasterProductContract(
        filename="refuge_overlap_by_hy.tif",
        dtype=np.dtype(np.uint8),
        nodata=REFUGE_OVERLAP_NODATA,
        units="overlap_category",
        codebook="0=dry,1=lost,2=new,3=stable,255=unsupported",
    ),
    "refuge_stability_frequency": RasterProductContract(
        filename="refuge_stability_frequency.tif",
        dtype=np.dtype(np.float32),
        nodata=_FLOAT32_NODATA,
        units="percent",
        codebook="0-100",
    ),
    "refuge_stability_union_pair_count": RasterProductContract(
        filename="refuge_stability_union_pair_count.tif",
        dtype=np.dtype(np.uint16),
        nodata=_UINT16_NODATA,
        units="eligible_hy_pairs",
        codebook="valid HY pairs wet in either year",
    ),
    "riverscape_landform": RasterProductContract(
        filename="landform.tif",
        dtype=np.dtype(np.uint8),
        nodata=np.uint8(0),
        units="landform_code",
        codebook="0=outside,1=in_channel,2=off_channel_riverine,3=non_riverine",
    ),
    "riverscape_hydroperiod": RasterProductContract(
        filename="hydroperiod.tif",
        dtype=np.dtype(np.uint8),
        nodata=np.uint8(0),
        units="hydroperiod_code",
        codebook="0=outside,1=persistent,2=seasonal,3=marginal,4=unobserved",
    ),
    "riverscape_zone_crosstab": RasterProductContract(
        filename="zone_crosstab.tif",
        dtype=np.dtype(np.uint16),
        nodata=np.uint16(0),
        units="crosstab_code",
        codebook="landform*10+hydroperiod; 0=outside",
    ),
    "riverscape_channel_source": RasterProductContract(
        filename="channel_source.tif",
        dtype=np.dtype(np.uint8),
        nodata=_UINT8_NODATA,
        units="channel_source_code",
        codebook="0=not channel,1=observed,2=bridged,255=nodata",
    ),
    "riverscape_channel_confidence": RasterProductContract(
        filename="channel_confidence.tif",
        dtype=np.dtype(np.uint8),
        nodata=_UINT8_NODATA,
        units="confidence_score",
        codebook="0-3 inside channel,255=not a channel pixel",
    ),
    "riverscape_rem": RasterProductContract(
        filename="rem.tif",
        dtype=np.dtype(np.float32),
        nodata=_FLOAT32_NODATA,
        units="metres",
    ),
}


def build_persistence_rasters(
    occurrence: OccurrenceResult, *, config: HydroConfig
) -> xr.Dataset:
    """Assemble occurrence, valid-count, and refuge-mask rasters."""

    threshold_pct = config.persistence.refuge_threshold * 100.0
    min_valid_obs = config.validity.min_valid_obs

    refuge_mask = (occurrence.occurrence >= threshold_pct) & (
        occurrence.valid_count >= min_valid_obs
    )

    rasters = xr.Dataset(
        {
            "occurrence": occurrence.occurrence,
            "valid_count": occurrence.valid_count,
            "refuge_mask": refuge_mask,
        }
    )
    rasters.attrs.update(
        {
            "refuge_threshold": config.persistence.refuge_threshold,
            "min_valid_obs": min_valid_obs,
            "validity_policy": config.validity.policy,
        }
    )
    return rasters


def write_persistence_rasters(
    rasters: xr.Dataset, output_dir: str | Path
) -> dict[str, Path]:
    """Write minimal-core rasters as independent, reopenable Zarr stores."""

    required = ("occurrence", "valid_count", "refuge_mask")
    missing = [name for name in required if name not in rasters.data_vars]
    if missing:
        raise ValueError(f"missing persistence raster: {missing[0]}")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}
    for name in required:
        path = root / name
        rasters[[name]].to_zarr(path, mode="w")
        artifacts[name] = path
    return artifacts


def build_persistence_rasters_from_checkpoint(
    checkpoint: SpatialRasterCheckpoint | SpatialRasterCheckpointAccumulator,
    *,
    config: HydroConfig,
) -> xr.Dataset:
    """Assemble persistence rasters from a completed spatial checkpoint."""

    if isinstance(checkpoint, SpatialRasterCheckpoint):
        checkpoint.validate_complete()
        accumulator = SpatialRasterCheckpointAccumulator._from_existing(
            checkpoint,
            config=config,
            template=None,
        )
    else:
        accumulator = checkpoint
    occurrence = accumulator.finalize_occurrence()
    return build_persistence_rasters(occurrence, config=config)


def build_temporal_rasters_from_checkpoint(
    checkpoint: SpatialRasterCheckpoint | SpatialRasterCheckpointAccumulator,
    *,
    config: HydroConfig,
) -> xr.Dataset:
    """Assemble recurrence and hydroperiod rasters from a completed checkpoint."""

    if isinstance(checkpoint, SpatialRasterCheckpoint):
        checkpoint.validate_complete()
        accumulator = SpatialRasterCheckpointAccumulator._from_existing(
            checkpoint,
            config=config,
            template=None,
        )
    else:
        accumulator = checkpoint
    recurrence = accumulator.finalize_recurrence()
    hydroperiod = accumulator.finalize_hydroperiod()
    return xr.Dataset(
        {
            "recurrence": recurrence.recurrence,
            "recurrence_valid_year_count": recurrence.valid_year_count,
            "hydroperiod": hydroperiod.hydroperiod,
            "hydroperiod_valid_month_count": hydroperiod.valid_observed_months,
        }
    ).assign_attrs(
        {
            "min_valid_obs": config.validity.min_valid_obs,
            "validity_policy": config.validity.policy,
        }
    )


def build_refuge_stability_rasters_from_checkpoint(
    checkpoint: SpatialRasterCheckpoint | SpatialRasterCheckpointAccumulator,
    *,
    config: HydroConfig,
) -> xr.Dataset:
    """Assemble refuge-stability frequency rasters from a completed checkpoint."""

    if isinstance(checkpoint, SpatialRasterCheckpoint):
        checkpoint.validate_complete()
        accumulator = SpatialRasterCheckpointAccumulator._from_existing(
            checkpoint,
            config=config,
            template=None,
        )
    else:
        accumulator = checkpoint
    return accumulator.finalize_refuge_stability_rasters()


def _load_end_dry_states(
    accumulator: SpatialRasterCheckpointAccumulator,
    *,
    root: Path,
    hydrological_years: Sequence[int],
) -> list[object]:
    from hydrofragments.metrics.dynamics import EndDryState

    if accumulator._end_dry_states:
        return list(accumulator._end_dry_states)

    states: list[EndDryState] = []
    for hy in hydrological_years:
        hy_dir = root / "end_dry" / str(hy)
        water_path = hy_dir / "water.zarr"
        valid_path = hy_dir / "valid.zarr"
        if not water_path.exists() or not valid_path.exists():
            continue
        water = np.asarray(zarr.open(water_path, mode="r"), dtype=bool)
        valid_obs = np.asarray(zarr.open(valid_path, mode="r"), dtype=bool)
        date = None
        anchor_missing = False
        for anchor in accumulator.end_dry_lookup.values():
            if int(anchor.get("hy", -1)) != int(hy):
                continue
            end_dry = anchor.get("end_dry_month")
            anchor_missing = end_dry is None or pd.isna(end_dry)
            if not anchor_missing:
                date = pd.Timestamp(end_dry).to_pydatetime()
            break
        states.append(
            EndDryState(
                hy=int(hy),
                date=date if date is not None else pd.Timestamp("1970-01-01").to_pydatetime(),
                water=water,
                valid_obs=valid_obs,
                hy_confidence="unknown",
                anchor_missing=anchor_missing,
            )
        )
    return states


def build_refuge_overlap_from_checkpoint(
    checkpoint: SpatialRasterCheckpoint | SpatialRasterCheckpointAccumulator,
    *,
    config: HydroConfig,
    analysis_mask: np.ndarray | None = None,
) -> xr.DataArray:
    """Build per-HY-pair refuge overlap categories from checkpoint end-dry states."""

    if isinstance(checkpoint, SpatialRasterCheckpoint):
        checkpoint.validate_complete()
        accumulator = SpatialRasterCheckpointAccumulator._from_existing(
            checkpoint,
            config=config,
            template=None,
        )
        root = checkpoint.root
        grid = checkpoint.metadata.grid
        hydrological_years = checkpoint.metadata.hydrological_years
    else:
        accumulator = checkpoint
        root = accumulator.root
        grid = accumulator.grid
        hydrological_years = tuple(sorted({state.hy for state in accumulator._end_dry_states}))

    if not hydrological_years:
        raise RasterExportError("refuge overlap requires captured end-dry states")

    states = _load_end_dry_states(
        accumulator,
        root=root,
        hydrological_years=hydrological_years,
    )
    if len(states) < 2:
        raise RasterExportError("refuge overlap requires at least two end-dry states")

    states = sorted(states, key=lambda item: item.hy)
    pair_labels: list[str] = []
    pair_arrays: list[np.ndarray] = []
    mask = (
        analysis_mask
        if analysis_mask is not None
        else accumulator.analysis_mask
        if accumulator.analysis_mask is not None
        else np.ones((grid.height, grid.width), dtype=bool)
    )

    from hydrofragments.metrics.dynamics import evaluate_refuge_spatial_stability

    for index in range(1, len(states)):
        previous = states[index - 1]
        current = states[index]
        evaluation = evaluate_refuge_spatial_stability(
            current=current,
            previous=previous,
            analysis_mask=mask,
            min_valid_fraction=config.validity.min_valid_fraction_month,
        )
        overlap = np.full((grid.height, grid.width), REFUGE_OVERLAP_NODATA, dtype=np.uint8)
        if (
            evaluation.edge_flag is not None
            and evaluation.edge_flag != EdgeFlag.EMPTY_REFUGE_UNION
        ):
            pair_arrays.append(overlap)
        else:
            common_valid = mask & previous.valid_obs & current.valid_obs
            previous_refuge = previous.water & common_valid
            current_refuge = current.water & common_valid
            union = previous_refuge | current_refuge
            overlap[common_valid & ~union] = np.uint8(0)
            overlap[previous_refuge & current_refuge] = np.uint8(3)
            overlap[previous_refuge & ~current_refuge] = np.uint8(1)
            overlap[~previous_refuge & current_refuge] = np.uint8(2)
            pair_arrays.append(overlap)
        prev_date = pd.Timestamp(previous.date).strftime("%Y-%m-%d")
        cur_date = pd.Timestamp(current.date).strftime("%Y-%m-%d")
        pair_labels.append(
            f"HY{previous.hy}-HY{current.hy} end_dry={prev_date} end_dry={cur_date}"
        )

    stacked = np.stack(pair_arrays, axis=0)
    return xr.DataArray(
        stacked,
        dims=("hy_pair", grid.y_dim, grid.x_dim),
        coords={
            "hy_pair": pair_labels,
            grid.y_dim: grid.y,
            grid.x_dim: grid.x,
        },
        attrs={"band_descriptions": pair_labels},
    )


def build_rasters_from_checkpoint(
    checkpoint: SpatialRasterCheckpoint,
    *,
    config: HydroConfig,
    analysis_mask: np.ndarray | None = None,
) -> xr.Dataset:
    """Build every raster product present in ``checkpoint``."""

    checkpoint.validate_complete()
    products = set(checkpoint.metadata.products)
    datasets: list[xr.Dataset] = []
    if "persistence_rasters" in products:
        datasets.append(
            build_persistence_rasters_from_checkpoint(checkpoint, config=config)
        )
    if "temporal_rasters" in products:
        datasets.append(
            build_temporal_rasters_from_checkpoint(checkpoint, config=config)
        )
    if "refuge_stability_rasters" in products:
        stability = build_refuge_stability_rasters_from_checkpoint(
            checkpoint, config=config
        )
        overlap = build_refuge_overlap_from_checkpoint(
            checkpoint,
            config=config,
            analysis_mask=analysis_mask,
        )
        datasets.append(stability)
        datasets.append(xr.Dataset({"refuge_overlap": overlap}))
    if not datasets:
        return xr.Dataset()
    merged = xr.merge(datasets)
    merged.attrs["scientific_config_hash"] = checkpoint.metadata.scientific_config_hash
    merged.attrs["algorithm_version"] = checkpoint.metadata.algorithm_version
    return merged


def _tile_block_size(height: int, width: int) -> tuple[int, int]:
    def align(axis: int) -> int:
        size = min(RASTER_TILE_SIZE, axis)
        if size < 16:
            return 16
        aligned = (size // 16) * 16
        return max(16, aligned)

    return align(height), align(width)


def _predictor_for_dtype(dtype: np.dtype) -> int:
    if np.issubdtype(dtype, np.floating):
        return 3
    return 2


def _geotiff_profile(
    grid: SpatialGrid,
    *,
    contract: RasterProductContract,
    count: int,
) -> dict[str, object]:
    block_height, block_width = _tile_block_size(grid.height, grid.width)
    nodata = contract.nodata
    if contract.dtype == np.dtype(np.float32) and isinstance(nodata, float) and np.isnan(nodata):
        nodata_value = float("nan")
    else:
        nodata_value = contract.dtype.type(contract.nodata)
    return {
        "driver": "GTiff",
        "height": grid.height,
        "width": grid.width,
        "count": count,
        "dtype": contract.dtype.name,
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": nodata_value,
        "tiled": True,
        "blockxsize": block_width,
        "blockysize": block_height,
        "compress": Compression.deflate,
        "predictor": _predictor_for_dtype(contract.dtype),
        "BIGTIFF": "IF_SAFER",
    }


def _cast_raster_values(
    values: np.ndarray,
    *,
    contract: RasterProductContract,
    source_name: str,
) -> np.ndarray:
    if source_name == "valid_count":
        cast = np.asarray(values, dtype=np.uint32)
        if np.any(cast > _UINT32_NODATA):
            raise RasterExportError("valid observation count exceeds uint32 range")
        return cast
    if source_name == "refuge_mask":
        return np.where(np.asarray(values, dtype=bool), np.uint8(1), np.uint8(0))
    if contract.dtype == np.dtype(np.float32):
        return np.asarray(values, dtype=np.float32)
    return np.asarray(values, dtype=contract.dtype)


def _projected_crs_identifiers(crs) -> set[str]:
    from rasterio.crs import CRS
    import re

    parsed = CRS.from_user_input(crs)
    identifiers: set[str] = set()
    epsg = parsed.to_epsg()
    if epsg is not None:
        identifiers.add(f"EPSG:{epsg}")
    for code in re.findall(r'EPSG","(\d+)"', parsed.to_wkt()):
        identifiers.add(f"EPSG:{code}")
    for pattern in (r'PROJCS\["([^"]+)"', r'LOCAL_CS\["([^"]+)"'):
        match = re.search(pattern, parsed.to_wkt())
        if match:
            identifiers.add(match.group(1))
    return identifiers


def _crs_equal(left, right) -> bool:
    from rasterio.crs import CRS

    left_crs = CRS.from_user_input(left)
    right_crs = CRS.from_user_input(right)
    left_epsg = left_crs.to_epsg()
    right_epsg = right_crs.to_epsg()
    if left_epsg is not None and right_epsg is not None:
        return left_epsg == right_epsg
    try:
        if left_crs.equals(right_crs):
            return True
    except Exception:
        pass
    left_ids = _projected_crs_identifiers(left_crs)
    right_ids = _projected_crs_identifiers(right_crs)
    return bool(left_ids.intersection(right_ids))


def _require_dataarray_crs(data: xr.DataArray, grid: SpatialGrid) -> xr.DataArray:
    if not hasattr(data, "rio") or data.rio.crs is None:
        raise ValueError("spatial output requires a resolvable CRS")
    return data


def _validation_slice(data: xr.DataArray, grid: SpatialGrid) -> xr.DataArray:
    spatial_dims = tuple(dim for dim in data.dims if dim in (grid.y_dim, grid.x_dim))
    extra_dims = [dim for dim in data.dims if dim not in spatial_dims]
    if extra_dims:
        return data.isel({extra_dims[0]: 0})
    return data


def _align_dataarray_to_grid(data: xr.DataArray, grid: SpatialGrid) -> xr.DataArray:
    aligned = data.copy()
    aligned = aligned.assign_coords({grid.y_dim: grid.y, grid.x_dim: grid.x})
    return aligned.rio.write_crs(grid.crs)


def preflight_raster_artifacts(
    destination: Path,
    *,
    filenames: Sequence[str],
) -> None:
    """Reject export when any final raster artifact already exists."""

    for filename in filenames:
        final_path = destination / filename
        if final_path.exists():
            raise RasterExportError(
                f"refusing to overwrite existing artifact: {final_path}"
            )


def _read_band_window(
    source: np.ndarray | zarr.Array,
    *,
    row_slice: slice,
    col_slice: slice,
) -> np.ndarray:
    if isinstance(source, zarr.Array):
        return np.asarray(source[row_slice, col_slice])
    return source[row_slice, col_slice]


def _write_geotiff_windowed(
    *,
    bands: Sequence[np.ndarray | zarr.Array],
    path: Path,
    grid: SpatialGrid,
    contract: RasterProductContract,
    band_descriptions: Sequence[str],
    metadata: Mapping[str, object],
) -> None:
    import rasterio

    profile = _geotiff_profile(grid, contract=contract, count=len(bands))
    block_height, block_width = (
        int(profile["blockysize"]),
        int(profile["blockxsize"]),
    )

    with rasterio.open(path, "w", **profile) as dataset:
        dataset.update_tags(
            UNITS=contract.units,
            CODEBOOK=contract.codebook or "",
            ALGORITHM_VERSION=str(metadata.get("algorithm_version", "")),
            SCIENTIFIC_CONFIG_HASH=str(metadata.get("scientific_config_hash", "")),
        )
        for band_index, description in enumerate(band_descriptions, start=1):
            dataset.set_band_description(band_index, description)

        for band_index, band_source in enumerate(bands, start=1):
            for row_off in range(0, grid.height, block_height):
                height = min(block_height, grid.height - row_off)
                row_slice = slice(row_off, row_off + height)
                for col_off in range(0, grid.width, block_width):
                    width = min(block_width, grid.width - col_off)
                    col_slice = slice(col_off, col_off + width)
                    window = rio_windows.Window(col_off, row_off, width, height)
                    block = _read_band_window(
                        band_source,
                        row_slice=row_slice,
                        col_slice=col_slice,
                    )
                    block = _cast_raster_values(
                        block,
                        contract=contract,
                        source_name=str(metadata.get("source_name", "")),
                    )
                    dataset.write(block, band_index, window=window)


def validate_geotiff(
    path: Path,
    *,
    grid: SpatialGrid,
    contract: RasterProductContract,
    band_descriptions: Sequence[str],
    metadata: Mapping[str, object],
    expected_arrays: Sequence[np.ndarray] | None = None,
) -> None:
    """Reopen a GeoTIFF and validate Section 2.4 contract alignment."""

    import rasterio
    from rasterio.crs import CRS

    with rasterio.open(path) as dataset:
        opened_crs = CRS.from_user_input(dataset.crs) if dataset.crs is not None else None
        if opened_crs is None or not _crs_equal(opened_crs, grid.crs):
            raise RasterExportError("GeoTIFF CRS mismatch")
        if not np.allclose(
            np.array(dataset.transform),
            np.array(grid.transform),
            rtol=0,
            atol=1e-9,
        ):
            raise RasterExportError("GeoTIFF affine transform mismatch")
        if (dataset.height, dataset.width) != (grid.height, grid.width):
            raise RasterExportError("GeoTIFF shape mismatch")
        if dataset.count != len(band_descriptions):
            raise RasterExportError("GeoTIFF band count mismatch")
        if dataset.dtypes[0] != contract.dtype.name:
            raise RasterExportError(
                f"GeoTIFF dtype mismatch: expected {contract.dtype.name}, got {dataset.dtypes[0]}"
            )
        if not dataset.is_tiled:
            raise RasterExportError("GeoTIFF is not tiled")
        if dataset.block_shapes[0] != _tile_block_size(grid.height, grid.width):
            raise RasterExportError("GeoTIFF block size mismatch")
        compression = dataset.compression
        if compression is None:
            raise RasterExportError("GeoTIFF compression mismatch")
        compression_name = (
            compression.name
            if isinstance(compression, Compression)
            else str(compression)
        )
        if compression_name.lower() != Compression.deflate.name.lower():
            raise RasterExportError("GeoTIFF compression mismatch")

        tags = dataset.tags()
        if tags.get("UNITS") != contract.units:
            raise RasterExportError("GeoTIFF units tag mismatch")
        if tags.get("ALGORITHM_VERSION") != str(metadata.get("algorithm_version", "")):
            raise RasterExportError("GeoTIFF algorithm version tag mismatch")
        if tags.get("SCIENTIFIC_CONFIG_HASH") != str(
            metadata.get("scientific_config_hash", "")
        ):
            raise RasterExportError("GeoTIFF scientific config hash tag mismatch")

        for band_index, description in enumerate(band_descriptions, start=1):
            if dataset.descriptions[band_index - 1] != description:
                raise RasterExportError("GeoTIFF band description mismatch")

        if expected_arrays is not None:
            for band_index, expected in enumerate(expected_arrays, start=1):
                actual = dataset.read(band_index)
                actual = _cast_raster_values(
                    actual,
                    contract=contract,
                    source_name=str(metadata.get("source_name", "")),
                )
                expected_cast = _cast_raster_values(
                    expected,
                    contract=contract,
                    source_name=str(metadata.get("source_name", "")),
                )
                if contract.dtype == np.dtype(np.float32):
                    if not np.allclose(
                        actual,
                        expected_cast,
                        rtol=0,
                        atol=1e-5,
                        equal_nan=True,
                    ):
                        raise RasterExportError("GeoTIFF float32 values mismatch")
                else:
                    if not np.array_equal(actual, expected_cast):
                        raise RasterExportError("GeoTIFF categorical values mismatch")


def write_verified_geotiff(
    *,
    bands: Sequence[np.ndarray | zarr.Array],
    destination: Path,
    grid: SpatialGrid,
    contract: RasterProductContract,
    band_descriptions: Sequence[str],
    metadata: Mapping[str, object],
) -> Path:
    """Write a tiled GeoTIFF via a validated temporary file and atomic replace."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RasterExportError(f"refusing to overwrite existing artifact: {destination}")

    tmp_path = destination.with_name(f".{destination.name}.tmp.tif")
    if tmp_path.exists():
        tmp_path.unlink()

    template = xr.DataArray(
        np.zeros((grid.height, grid.width), dtype=float),
        dims=(grid.y_dim, grid.x_dim),
        coords={grid.y_dim: grid.y, grid.x_dim: grid.x},
    ).rio.write_crs(grid.crs)
    grid.validate_dataarray(template)

    _write_geotiff_windowed(
        bands=bands,
        path=tmp_path,
        grid=grid,
        contract=contract,
        band_descriptions=band_descriptions,
        metadata=metadata,
    )

    expected_arrays = [
        np.asarray(band) if not isinstance(band, zarr.Array) else np.asarray(band[:])
        for band in bands
    ]
    validate_geotiff(
        tmp_path,
        grid=grid,
        contract=contract,
        band_descriptions=band_descriptions,
        metadata=metadata,
        expected_arrays=expected_arrays,
    )
    tmp_path.replace(destination)
    return destination


def _band_descriptions_for_dataarray(data: xr.DataArray, *, default: str) -> list[str]:
    if "year" in data.dims:
        return [f"calendar_year={int(value)}" for value in data["year"].values]
    if "hy_pair" in data.dims:
        if "band_descriptions" in data.attrs:
            return list(data.attrs["band_descriptions"])
        return [str(value) for value in data["hy_pair"].values]
    return [default]


def write_geotiff_from_dataarray(
    data: xr.DataArray,
    destination: Path,
    *,
    grid: SpatialGrid,
    contract: RasterProductContract,
    metadata: Mapping[str, object],
    source_name: str,
) -> Path:
    """Write one raster product from a grid-bearing ``DataArray``."""

    grid.validate_dataarray(_require_dataarray_crs(_validation_slice(data, grid), grid))
    spatial_dims = tuple(dim for dim in data.dims if dim in (grid.y_dim, grid.x_dim))
    if len(spatial_dims) != 2:
        raise RasterExportError("expected a spatial DataArray")

    extra_dims = [dim for dim in data.dims if dim not in spatial_dims]
    if extra_dims:
        stack_dim = extra_dims[0]
        band_arrays = [np.asarray(data.isel({stack_dim: index}).values) for index in range(data.sizes[stack_dim])]
        descriptions = _band_descriptions_for_dataarray(data, default=source_name)
    else:
        band_arrays = [np.asarray(data.values)]
        descriptions = [source_name]

    meta = dict(metadata)
    meta["source_name"] = source_name
    return write_verified_geotiff(
        bands=band_arrays,
        destination=destination,
        grid=grid,
        contract=contract,
        band_descriptions=descriptions,
        metadata=meta,
    )


def _checkpoint_metadata_payload(
    checkpoint: SpatialRasterCheckpoint,
) -> dict[str, str]:
    return {
        "algorithm_version": checkpoint.metadata.algorithm_version,
        "scientific_config_hash": checkpoint.metadata.scientific_config_hash,
    }


def write_hydroperiod_geotiffs_from_checkpoint(
    checkpoint: SpatialRasterCheckpoint,
    destination: Path,
    *,
    metadata: Mapping[str, object],
) -> dict[str, Path]:
    """Write hydroperiod stacks one calendar year at a time from checkpoint slices."""

    grid = checkpoint.metadata.grid
    years = checkpoint.metadata.calendar_years
    if not years:
        raise RasterExportError("hydroperiod checkpoint has no completed calendar years")

    hydroperiod_contract = RASTER_PRODUCT_CONTRACTS["hydroperiod"]
    valid_month_contract = RASTER_PRODUCT_CONTRACTS["hydroperiod_valid_month_count"]
    hydroperiod_bands: list[zarr.Array | np.ndarray] = []
    valid_month_bands: list[zarr.Array | np.ndarray] = []
    descriptions: list[str] = []

    for year in years:
        year_dir = checkpoint.root / "hydroperiod" / str(year)
        wet_path = year_dir / "wet.zarr"
        valid_path = year_dir / "valid.zarr"
        valid_months_path = year_dir / "valid_months.zarr"
        if not wet_path.exists() or not valid_path.exists():
            raise RasterExportError(f"missing hydroperiod slice for calendar year {year}")
        wet = zarr.open(wet_path, mode="r")
        valid = zarr.open(valid_path, mode="r")
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(
                np.asarray(valid, dtype=np.float32) > 0,
                np.asarray(wet, dtype=np.float32) / np.asarray(valid, dtype=np.float32),
                np.nan,
            )
        hydroperiod_bands.append(ratio.astype(np.float32))
        if valid_months_path.exists():
            valid_month_bands.append(zarr.open(valid_months_path, mode="r"))
        else:
            valid_month_bands.append(np.asarray(valid, dtype=np.uint8))
        descriptions.append(f"calendar_year={int(year)}")

    hydroperiod_path = destination / hydroperiod_contract.filename
    valid_month_path = destination / valid_month_contract.filename
    write_verified_geotiff(
        bands=hydroperiod_bands,
        destination=hydroperiod_path,
        grid=grid,
        contract=hydroperiod_contract,
        band_descriptions=descriptions,
        metadata={**metadata, "source_name": "hydroperiod"},
    )
    write_verified_geotiff(
        bands=valid_month_bands,
        destination=valid_month_path,
        grid=grid,
        contract=valid_month_contract,
        band_descriptions=descriptions,
        metadata={**metadata, "source_name": "hydroperiod_valid_month_count"},
    )
    return {
        "hydroperiod": hydroperiod_path,
        "hydroperiod_valid_month_count": valid_month_path,
    }


def write_zones_geotiff(
    zone_mask: np.ndarray,
    destination: Path,
    *,
    grid: SpatialGrid,
    metadata: Mapping[str, object],
) -> Path:
    """Write the hydrological zone raster."""

    contract = RASTER_PRODUCT_CONTRACTS["zones"]
    return write_verified_geotiff(
        bands=[zone_mask],
        destination=destination,
        grid=grid,
        contract=contract,
        band_descriptions=["hydrological_zone"],
        metadata={**metadata, "source_name": "zones"},
    )


#: On-disk order and band names for the ``riverscape_evidence`` product
#: (spec 6b section 3.5). Pairs are ``(RASTER_PRODUCT_CONTRACTS key, array
#: key)``; the array key doubles as the GeoTIFF band description.
RIVERSCAPE_EVIDENCE_BANDS: tuple[tuple[str, str], ...] = (
    ("riverscape_landform", "landform"),
    ("riverscape_hydroperiod", "hydroperiod"),
    ("riverscape_zone_crosstab", "zone_crosstab"),
    ("riverscape_channel_source", "channel_source"),
    ("riverscape_channel_confidence", "channel_confidence"),
    ("riverscape_rem", "rem"),
)


def write_riverscape_evidence_geotiffs(
    arrays: Mapping[str, np.ndarray],
    destination: Path | str,
    *,
    grid: SpatialGrid,
    metadata: Mapping[str, object],
) -> dict[str, Path]:
    """Write the six ``riverscape_evidence`` rasters onto one shared grid.

    ``arrays`` is a plain mapping keyed by band name, not a
    ``RiverscapeExportBundle`` -- this module stays free of the export
    bundle type, and ``output.finalize`` owns the translation. Every band
    goes through ``write_verified_geotiff``, so each file is reopened and
    checked against its contract (dtype, CRS, transform, tiling,
    compression, tags, values) before the atomic replace.
    """
    raster_dir = Path(destination)
    raster_dir.mkdir(parents=True, exist_ok=True)
    preflight_raster_artifacts(
        raster_dir,
        filenames=[
            RASTER_PRODUCT_CONTRACTS[key].filename
            for key, _band in RIVERSCAPE_EVIDENCE_BANDS
        ],
    )

    written: dict[str, Path] = {}
    for key, band in RIVERSCAPE_EVIDENCE_BANDS:
        contract = RASTER_PRODUCT_CONTRACTS[key]
        values = np.asarray(arrays[band])
        if values.shape != (grid.height, grid.width):
            raise RasterExportError(
                f"riverscape evidence band {band} does not align with the export grid"
            )
        written[key] = write_verified_geotiff(
            bands=[values.astype(contract.dtype)],
            destination=raster_dir / contract.filename,
            grid=grid,
            contract=contract,
            band_descriptions=[band],
            metadata={**metadata, "source_name": band},
        )
    return written


def _require_h5netcdf() -> None:
    try:
        import h5netcdf  # noqa: F401
    except ImportError:
        raise RasterExportError(
            "NetCDF export requires the optional netcdf extra: "
            "pip install 'hydrofragments[netcdf]'"
        )


def _spatial_slice_for_grid(data_array: xr.DataArray, grid: SpatialGrid) -> xr.DataArray:
    """Return a 2-D (y, x) slice so stacked NetCDF variables can share the grid."""

    indexers = {
        dim: 0
        for dim in data_array.dims
        if dim not in (grid.y_dim, grid.x_dim)
    }
    sliced = data_array.isel(indexers) if indexers else data_array
    if sliced.ndim != 2:
        raise RasterExportError(
            f"NetCDF variable {data_array.name} is not a spatial (y, x) field"
        )
    return sliced


def write_verified_netcdf(
    dataset: xr.Dataset,
    destination: Path,
    *,
    grid: SpatialGrid,
    metadata: Mapping[str, object],
) -> Path:
    """Write one consolidated NetCDF file with validation and atomic replace."""

    _require_h5netcdf()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise RasterExportError(f"refusing to overwrite existing artifact: {destination}")

    tmp_path = destination.with_name(f".{destination.name}.tmp.nc")
    if tmp_path.exists():
        tmp_path.unlink()

    export_ds = dataset.copy()
    crs_wkt = grid.crs.to_wkt()
    export_ds.attrs.update(
        {
            "algorithm_version": metadata.get("algorithm_version", ""),
            "scientific_config_hash": metadata.get("scientific_config_hash", ""),
            "crs": crs_wkt,
            "transform": list(grid.transform),
        }
    )
    encoding: dict[str, dict[str, object]] = {}
    block = _tile_block_size(grid.height, grid.width)
    for name, data_array in export_ds.data_vars.items():
        export_ds[name].attrs["crs"] = crs_wkt
        export_ds[name].attrs["crs_wkt"] = crs_wkt
        spatial_shape = tuple(
            int(data_array.sizes[dim])
            for dim in data_array.dims
            if dim in (grid.y_dim, grid.x_dim)
        )
        if len(spatial_shape) != 2:
            raise RasterExportError(f"NetCDF variable {name} is not spatial")
        chunks = tuple(
            min(block[index], spatial_shape[index]) for index in range(2)
        )
        if data_array.ndim > 2:
            leading = int(data_array.sizes[data_array.dims[0]])
            chunks = (min(leading, 1), *chunks)
        encoding[name] = {"zlib": True, "complevel": 4, "chunksizes": chunks}
    export_ds.to_netcdf(tmp_path, engine="h5netcdf", encoding=encoding)

    reopened = xr.open_dataset(tmp_path)
    try:
        for name, data_array in export_ds.data_vars.items():
            grid.validate_dataarray(_spatial_slice_for_grid(reopened[name], grid))
            if export_ds[name].dtype != reopened[name].dtype:
                raise RasterExportError(f"NetCDF dtype mismatch for {name}")
            expected = export_ds[name].values
            actual = reopened[name].values
            if np.issubdtype(expected.dtype, np.floating):
                if not np.allclose(actual, expected, rtol=0, atol=1e-5, equal_nan=True):
                    raise RasterExportError(f"NetCDF values mismatch for {name}")
            elif not np.array_equal(actual, expected):
                raise RasterExportError(f"NetCDF values mismatch for {name}")
    finally:
        reopened.close()
    tmp_path.replace(destination)
    return destination


def export_rasters_from_checkpoint(
    checkpoint: SpatialRasterCheckpoint,
    destination: Path | str,
    *,
    config: HydroConfig,
    raster_formats: Sequence[str] = ("geotiff",),
    zone_mask: np.ndarray | None = None,
    analysis_mask: np.ndarray | None = None,
) -> dict[str, Path]:
    """Export completed checkpoint products as verified GeoTIFF and/or NetCDF."""

    checkpoint.validate_complete()
    raster_dir = Path(destination)
    raster_dir.mkdir(parents=True, exist_ok=True)
    metadata = _checkpoint_metadata_payload(checkpoint)
    grid = checkpoint.metadata.grid
    products = set(checkpoint.metadata.products)
    formats = tuple(raster_formats)

    filenames: list[str] = []
    if "persistence_rasters" in products:
        filenames.extend(
            [
                RASTER_PRODUCT_CONTRACTS["occurrence"].filename,
                RASTER_PRODUCT_CONTRACTS["valid_observation_count"].filename,
                RASTER_PRODUCT_CONTRACTS["refuge_mask"].filename,
            ]
        )
    if "temporal_rasters" in products:
        filenames.extend(
            [
                RASTER_PRODUCT_CONTRACTS["recurrence"].filename,
                RASTER_PRODUCT_CONTRACTS["recurrence_valid_year_count"].filename,
                RASTER_PRODUCT_CONTRACTS["hydroperiod"].filename,
                RASTER_PRODUCT_CONTRACTS["hydroperiod_valid_month_count"].filename,
            ]
        )
    if "refuge_stability_rasters" in products:
        filenames.extend(
            [
                RASTER_PRODUCT_CONTRACTS["refuge_overlap"].filename,
                RASTER_PRODUCT_CONTRACTS["refuge_stability_frequency"].filename,
                RASTER_PRODUCT_CONTRACTS["refuge_stability_union_pair_count"].filename,
            ]
        )
    if zone_mask is not None:
        filenames.append(RASTER_PRODUCT_CONTRACTS["zones"].filename)
    if "netcdf" in formats:
        filenames.append(NETCDF_SPATIAL_FILENAME)

    preflight_raster_artifacts(raster_dir, filenames=filenames)

    artifacts: dict[str, Path] = {}
    merged_for_netcdf: xr.Dataset | None = None

    if "geotiff" in formats:
        if "persistence_rasters" in products:
            persistence = build_persistence_rasters_from_checkpoint(checkpoint, config=config)
            mapping = {
                "occurrence": persistence["occurrence"],
                "valid_observation_count": persistence["valid_count"],
                "refuge_mask": persistence["refuge_mask"],
            }
            for key, data_array in mapping.items():
                contract = RASTER_PRODUCT_CONTRACTS[key]
                path = raster_dir / contract.filename
                artifacts[key] = write_geotiff_from_dataarray(
                    _align_dataarray_to_grid(data_array, grid),
                    path,
                    grid=grid,
                    contract=contract,
                    metadata=metadata,
                    source_name=key,
                )
            merged_for_netcdf = (
                persistence.rename({"valid_count": "valid_observation_count"})
                if merged_for_netcdf is None
                else xr.merge([merged_for_netcdf, persistence.rename({"valid_count": "valid_observation_count"})])
            )

        if "temporal_rasters" in products:
            temporal = build_temporal_rasters_from_checkpoint(checkpoint, config=config)
            for key in (
                "recurrence",
                "recurrence_valid_year_count",
                "hydroperiod",
                "hydroperiod_valid_month_count",
            ):
                contract = RASTER_PRODUCT_CONTRACTS[key]
                path = raster_dir / contract.filename
                artifacts[key] = write_geotiff_from_dataarray(
                    _align_dataarray_to_grid(temporal[key], grid),
                    path,
                    grid=grid,
                    contract=contract,
                    metadata=metadata,
                    source_name=key,
                )
            merged_for_netcdf = (
                temporal
                if merged_for_netcdf is None
                else xr.merge([merged_for_netcdf, temporal])
            )

        if "refuge_stability_rasters" in products:
            stability = build_refuge_stability_rasters_from_checkpoint(
                checkpoint, config=config
            )
            overlap = build_refuge_overlap_from_checkpoint(
                checkpoint,
                config=config,
                analysis_mask=analysis_mask,
            )
            for key, data_array in {
                "refuge_stability_frequency": stability["refuge_stability_frequency"],
                "refuge_stability_union_pair_count": stability["refuge_stability_union_pair_count"],
                "refuge_overlap": overlap,
            }.items():
                contract = RASTER_PRODUCT_CONTRACTS[key]
                path = raster_dir / contract.filename
                artifacts[key] = write_geotiff_from_dataarray(
                    _align_dataarray_to_grid(data_array, grid),
                    path,
                    grid=grid,
                    contract=contract,
                    metadata=metadata,
                    source_name=key,
                )
            refuge_ds = xr.merge([stability, xr.Dataset({"refuge_overlap": overlap})])
            merged_for_netcdf = (
                refuge_ds
                if merged_for_netcdf is None
                else xr.merge([merged_for_netcdf, refuge_ds])
            )

        if zone_mask is not None:
            contract = RASTER_PRODUCT_CONTRACTS["zones"]
            path = raster_dir / contract.filename
            artifacts["zones"] = write_zones_geotiff(
                zone_mask,
                path,
                grid=grid,
                metadata=metadata,
            )
            zones_da = xr.DataArray(
                zone_mask,
                dims=(grid.y_dim, grid.x_dim),
                coords={grid.y_dim: grid.y, grid.x_dim: grid.x},
            )
            merged_for_netcdf = (
                xr.Dataset({"zones": zones_da})
                if merged_for_netcdf is None
                else xr.merge([merged_for_netcdf, xr.Dataset({"zones": zones_da})])
            )

    if "netcdf" in formats:
        if merged_for_netcdf is None:
            merged_for_netcdf = build_rasters_from_checkpoint(
                checkpoint,
                config=config,
                analysis_mask=analysis_mask,
            )
        netcdf_path = raster_dir / NETCDF_SPATIAL_FILENAME
        artifacts["spatial_nc"] = write_verified_netcdf(
            merged_for_netcdf,
            netcdf_path,
            grid=grid,
            metadata=metadata,
        )

    return artifacts


__all__ = [
    "NETCDF_SPATIAL_FILENAME",
    "RASTER_PRODUCT_CONTRACTS",
    "RASTER_TILE_SIZE",
    "RasterExportError",
    "RasterProductContract",
    "build_persistence_rasters",
    "build_persistence_rasters_from_checkpoint",
    "build_rasters_from_checkpoint",
    "build_refuge_overlap_from_checkpoint",
    "build_refuge_stability_rasters_from_checkpoint",
    "build_temporal_rasters_from_checkpoint",
    "export_rasters_from_checkpoint",
    "preflight_raster_artifacts",
    "validate_geotiff",
    "write_geotiff_from_dataarray",
    "write_persistence_rasters",
    "RIVERSCAPE_EVIDENCE_BANDS",
    "write_verified_geotiff",
    "write_verified_netcdf",
    "write_zones_geotiff",
    "write_riverscape_evidence_geotiffs",
]
