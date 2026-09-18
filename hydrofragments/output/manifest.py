"""Self-contained run configuration and manifest artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import mimetypes
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from hydrofragments.output.riverscape_export import RiverscapeExportBundle
    from hydrofragments.spatial.zones import ZoneResult

import numpy as np

from hydrofragments.config import HydroConfig
from hydrofragments.schema import SCHEMA_VERSION


LEGACY_MANIFEST_SCHEMA_VERSION = "1.0.0"
MANIFEST_SCHEMA_VERSION = "1.1.0"
SUPPORTED_MANIFEST_SCHEMA_VERSIONS = frozenset(
    {LEGACY_MANIFEST_SCHEMA_VERSION, MANIFEST_SCHEMA_VERSION}
)


class ManifestError(ValueError):
    """Raised when run metadata is incomplete or not self-contained."""


@dataclass(frozen=True)
class RunMetadataArtifacts:
    config_path: Path
    manifest_path: Path


_MEDIA_TYPE_OVERRIDES = {
    ".gpkg": "application/geopackage+sqlite3",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".nc": "application/netcdf",
    ".parquet": "application/vnd.apache.parquet",
    ".csv": "text/csv",
    ".json": "application/json",
}


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _hash_json(value: object) -> str:
    return hashlib.sha256(_json_bytes(value).rstrip(b"\n")).hexdigest()


def hash_file(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_directory_tree(root: Path) -> tuple[int, str]:
    """Return total byte size and a deterministic tree digest for one directory."""

    base = Path(root)
    files = sorted(
        path
        for path in base.rglob("*")
        if path.is_file() and path.name != ".bundle_transaction.json"
    )
    total_bytes = 0
    tree = hashlib.sha256()
    for path in files:
        relative = path.relative_to(base).as_posix()
        file_digest = hash_file(path)
        total_bytes += path.stat().st_size
        tree.update(relative.encode("utf-8"))
        tree.update(b"\0")
        tree.update(file_digest.encode("utf-8"))
        tree.update(b"\0")
    return total_bytes, tree.hexdigest()


def guess_media_type(path: Path) -> str:
    """Infer a stable media type for one artifact path."""

    suffix = Path(path).suffix.lower()
    if suffix in _MEDIA_TYPE_OVERRIDES:
        return _MEDIA_TYPE_OVERRIDES[suffix]
    if Path(path).is_dir():
        return "application/vnd.directory"
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        raise ManifestError("created_at must include a timezone")
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _relative_artifacts(
    artifacts: Mapping[str, str | Path],
) -> dict[str, str]:
    resolved: dict[str, str] = {
        "config": "config.json",
        "manifest": "run_manifest.json",
    }
    for name, raw_path in artifacts.items():
        path = Path(raw_path)
        if path.is_absolute() or ".." in path.parts:
            raise ManifestError(
                f"artifact path for {name} must be relative to the run directory"
            )
        resolved[str(name)] = PurePosixPath(*path.parts).as_posix()
    return dict(sorted(resolved.items()))


def _artifact_size_and_digest(root: Path, relative_path: str) -> tuple[int, str]:
    path = root / relative_path
    if path.is_dir():
        return hash_directory_tree(path)
    if not path.is_file():
        raise ManifestError(f"artifact path is not a file or directory: {relative_path}")
    return path.stat().st_size, hash_file(path)


def build_artifact_inventory(
    bundle_root: str | Path,
    registrations: Sequence[Any],
    *,
    config: HydroConfig,
) -> list[dict[str, object]]:
    """Build the manifest 1.1.0 artifact inventory for one staged bundle."""

    root = Path(bundle_root)
    entries: list[dict[str, object]] = []

    def _append_entry(
        *,
        name: str,
        relative_path: str,
        media_type: str | None = None,
        algorithm_version: str | None = None,
        scientific_config_hash: str | None = None,
        execution_config_hash: str | None = None,
        spatial: Mapping[str, object] | None = None,
    ) -> None:
        byte_size, digest = _artifact_size_and_digest(root, relative_path)
        entry: dict[str, object] = {
            "name": name,
            "relative_path": PurePosixPath(*Path(relative_path).parts).as_posix(),
            "media_type": media_type or guess_media_type(root / relative_path),
            "byte_size": byte_size,
            "sha256": digest,
            "scientific_config_hash": scientific_config_hash or config.config_hash,
            "execution_config_hash": execution_config_hash or config.execution_hash,
        }
        if algorithm_version is not None:
            entry["algorithm_version"] = algorithm_version
        if spatial:
            entry["spatial"] = dict(spatial)
        entries.append(entry)

    for registration in registrations:
        _append_entry(
            name=registration.name,
            relative_path=registration.relative_path,
            media_type=registration.media_type,
            algorithm_version=registration.algorithm_version,
            scientific_config_hash=registration.scientific_config_hash,
            execution_config_hash=registration.execution_config_hash,
            spatial=registration.spatial,
        )

    _append_entry(
        name="config",
        relative_path="config.json",
        media_type="application/json",
    )
    entries.sort(key=lambda item: str(item["name"]))
    return entries


def _hash_array(mask: Any) -> str:
    """A stable SHA-256 content digest over a raster mask's raw values.

    Consistent with this module's ``_hash_json``/``_json_bytes`` convention
    (canonical bytes -> sha256 hex digest) rather than importing a different
    hashing utility from elsewhere in the codebase. Uses the array's raw
    bytes plus its shape and dtype (not JSON) since the payload is a numpy
    array, not JSON-serializable structured data -- shape/dtype are included
    so two differently-shaped or differently-typed arrays with coincidentally
    identical byte patterns never collide.
    """
    array = np.asarray(mask)
    hasher = hashlib.sha256()
    hasher.update(np.ascontiguousarray(array).tobytes())
    hasher.update(str(array.shape).encode("utf-8"))
    hasher.update(str(array.dtype).encode("utf-8"))
    return hasher.hexdigest()


def build_dea_provenance(
    config: HydroConfig,
    *,
    product: str,
    version: str | None,
    item_ids: Sequence[str],
    crs: str,
    resolution: float | None,
    time_span: str | None,
    zone_mask: Any,
    planning_footprint: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the ``dea_provenance`` manifest section for a DEA-zoned run."""
    section: dict[str, object] = {
        "product": product,
        "version": version,
        "item_ids": list(item_ids),
        "crs": crs,
        "resolution": resolution,
        "time_span": time_span,
        "zone_thresholds": {
            "t_persist": config.zones.t_persist,
            "t_season": config.zones.t_season,
        },
        "zone_mask_digest": _hash_array(zone_mask),
    }
    if planning_footprint is not None:
        footprint = dict(planning_footprint)
        section["planning_footprint"] = {
            "digest": footprint["digest"],
            "factor": footprint["factor"],
            "safety_cells": footprint["safety_cells"],
            "covered_years": list(footprint["covered_years"]),
            "source_collection": footprint["source_collection"],
            "source_version": footprint["source_version"],
            "source_lineage": footprint["source_lineage"],
        }
    return section


def _finite_float(value: object, *, field: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ManifestError(f"zoning.{field} must be finite")
    return number


def _bridged_length_m(bridges: Any) -> float:
    """Total bridged centreline length in metres; 0 for an empty/absent frame.

    ``LandformResult.channel_bridges`` is either
    ``bridging._empty_bridge_frame``'s typed-empty GeoDataFrame or a frame
    with one row per bridged gap. A stub may leave it ``None``.
    """
    if bridges is None or len(bridges) == 0:
        return 0.0
    return _finite_float(
        np.asarray(bridges["length_m"], dtype=float).sum(),
        field="bridged_length_m",
    )


def build_zoning_section(
    zone_result: "ZoneResult | None",
    bundle: "RiverscapeExportBundle | None",
    *,
    pixel_m: float,
    configured_mode: str,
    workflow_reasons: Sequence[str] = (),
) -> dict[str, object]:
    """Build the manifest's top-level ``zoning`` object (spec 6b section 3.3).

    Three shapes, chosen by what actually exists this run:

    1. ``bundle`` present -> the full riverscape record. Every key under
       ``LandformResult.provenance`` is copied **verbatim** (spec section
       3.3); the ruleset versions and source products are additionally
       regrouped under ``ruleset_versions``/``sources`` exactly as the spec's
       JSON shows. Area and length fields are derived here, not in the
       pipeline, because ``pixel_m`` is an output-layer fact.
    2. ``zone_result`` present without a bundle (occurrence zoning, or an
       ``auto`` run that fell back) -> the thin subset: ``mode``,
       ``domain_pixel_count``, ``domain_digest``, ``degraded_reasons``,
       ``zone_source``. Riverscape-only keys are **omitted**, never emitted
       as empty defaults, so a reader cannot mistake absent science for
       degraded science.
    3. Neither -> ``mode`` (the *configured* riverscape mode, since no
       ``ZoneResult`` exists to report an actual one) plus
       ``degraded_reasons`` from the workflow tuple. No domain digest is
       fabricated.

    ``degraded_reasons`` prefers ``ZoneResult.degraded_reasons`` whenever a
    ``ZoneResult`` exists (spec section 2 row 5); ``workflow_reasons`` are
    already merged onto it by ``workflow._with_reasons``, and are only read
    directly in shape 3.

    ``domain_digest`` reuses :func:`_hash_array` -- the same helper behind
    ``dea_provenance.zone_mask_digest`` -- so the codebase has exactly one
    mask-digest algorithm.
    """
    if zone_result is None:
        return {
            "mode": str(configured_mode),
            "degraded_reasons": [str(reason) for reason in workflow_reasons],
        }

    section: dict[str, object] = {
        "mode": str(zone_result.mode),
        "degraded_reasons": [str(reason) for reason in zone_result.degraded_reasons],
        "domain_digest": _hash_array(zone_result.mask),
    }

    if bundle is None:
        section["domain_pixel_count"] = int(np.count_nonzero(zone_result.mask))
        section["zone_source"] = str(zone_result.source)
        return section

    provenance = dict(bundle.landform.provenance)
    pixel_counts = {
        str(name): int(count)
        for name, count in dict(provenance["pixel_counts"]).items()
    }
    cell_area_m2 = float(pixel_m) ** 2

    section["domain_pixel_count"] = int(pixel_counts["domain"])
    section["ruleset_versions"] = {
        "channel": provenance["channel_ruleset_version"],
        "waterbody": provenance["waterbody_ruleset_version"],
        "riverine": provenance["riverine_ruleset_version"],
    }
    section["sources"] = {
        "dem_product": provenance["dem_product"],
        "dem_band": provenance["dem_band"],
        "fc_product": provenance["fc_product"],
        "fc_bands": [str(band) for band in provenance["fc_bands"]],
        "waterbodies_source": provenance["waterbodies_source"],
        "years": [int(year) for year in provenance["years"]],
        "zone_source": str(zone_result.source),
    }
    section["bridge_enabled"] = bool(provenance["bridge_enabled"])
    section["bare_threshold_pct"] = _finite_float(
        provenance["bare_threshold_pct"], field="bare_threshold_pct"
    )
    envelope = dict(provenance["envelope"])
    section["envelope"] = {
        "a": _finite_float(envelope["a"], field="envelope.a"),
        "b": _finite_float(envelope["b"], field="envelope.b"),
        "n_bins": int(envelope["n_bins"]),
    }
    section["pixel_counts"] = pixel_counts
    section["bridge_counts_by_cause"] = {
        str(cause): int(count)
        for cause, count in dict(provenance["bridge_counts_by_cause"]).items()
    }
    section["unbridged_counts_by_reason"] = {
        str(reason): int(count)
        for reason, count in dict(provenance["unbridged_counts_by_reason"]).items()
    }
    section["reach_counts"] = {
        str(name): int(count)
        for name, count in dict(provenance["reach_counts"]).items()
    }
    section["bridged_length_m"] = _bridged_length_m(bundle.landform.channel_bridges)
    section["bridged_area_m2"] = (
        float(np.count_nonzero(bundle.landform.bridged_mask)) * cell_area_m2
    )
    section["non_riverine_area_m2"] = float(pixel_counts["non_riverine"]) * cell_area_m2
    return section


def build_run_manifest(
    config: HydroConfig,
    *,
    run_id: str,
    package_version: str,
    git_sha: str,
    input_fingerprint: Mapping[str, object],
    planned_backend: str,
    actual_backend_by_stage: Mapping[str, str],
    skipped_metrics: Sequence[Mapping[str, object]] = (),
    warnings: Sequence[str] = (),
    timings_seconds: Mapping[str, float] | None = None,
    comparison_context: Mapping[str, object] | None = None,
    artifacts: Mapping[str, str | Path] | None = None,
    artifact_inventory: Sequence[Mapping[str, object]] | None = None,
    created_at: datetime | None = None,
    dependency_versions: Mapping[str, str] | None = None,
    backend_capabilities: Mapping[str, object] | None = None,
    dea_provenance: Mapping[str, object] | None = None,
    zoning: Mapping[str, object] | None = None,
    manifest_schema_version: str | None = None,
    peak_rss_bytes: int | None = None,
) -> dict[str, object]:
    """Build complete machine-readable provenance for one immutable run."""
    for name, value in {
        "run_id": run_id,
        "package_version": package_version,
        "git_sha": git_sha,
        "planned_backend": planned_backend,
    }.items():
        if not value:
            raise ManifestError(f"{name} must be a non-empty string")
    if not input_fingerprint:
        raise ManifestError("input_fingerprint cannot be empty")

    schema_version = manifest_schema_version
    if schema_version is None:
        schema_version = (
            MANIFEST_SCHEMA_VERSION
            if artifact_inventory is not None
            else LEGACY_MANIFEST_SCHEMA_VERSION
        )
    if schema_version not in SUPPORTED_MANIFEST_SCHEMA_VERSIONS:
        raise ManifestError(f"unsupported manifest_schema_version: {schema_version}")

    versions: dict[str, object] = {
        "package_version": package_version,
        "git_sha": git_sha,
    }
    try:
        import hydroseason

        hydroseason_version = hydroseason.__version__
        versions["hydroseason"] = hydroseason_version
    except ImportError as error:  # pragma: no cover - dependency is mandatory
        raise ManifestError("hydroseason is required for manifest provenance") from error
    supplied_dependencies = dict(dependency_versions or {})
    supplied_hydroseason = supplied_dependencies.pop("hydroseason", None)
    if supplied_hydroseason is not None and supplied_hydroseason != hydroseason_version:
        raise ManifestError(
            "dependency_versions.hydroseason does not match installed hydroseason"
        )
    versions.update(supplied_dependencies)

    comparison = dict(comparison_context or {})
    comparison.update(
        {
            "validity_policy": config.validity.policy,
            "monthly_composite": config.temporal.monthly_composite,
        }
    )

    backend: dict[str, object] = {
        "planned": planned_backend,
        "actual_by_stage": dict(actual_backend_by_stage),
    }
    if backend_capabilities is not None:
        backend["capabilities"] = dict(backend_capabilities)

    manifest: dict[str, object] = {
        "manifest_schema_version": schema_version,
        "output_schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": _utc_text(created_at or datetime.now(timezone.utc)),
        "config_hash": config.config_hash,
        "execution_hash": config.execution_hash,
        "package_version": package_version,
        "git_sha": git_sha,
        "resolved_config": config.scientific_config(),
        "execution_config": config.execution_config(),
        "input_fingerprint": dict(input_fingerprint),
        "versions": versions,
        "backend": backend,
        "skipped_metrics": [dict(item) for item in skipped_metrics],
        "warnings": list(warnings),
        "timings_seconds": dict(timings_seconds or {}),
        "comparison": comparison,
        "artifacts": _relative_artifacts(artifacts or {}),
    }
    if artifact_inventory is not None:
        manifest["artifact_inventory"] = [dict(item) for item in artifact_inventory]
    if peak_rss_bytes is not None:
        manifest["peak_rss_bytes"] = int(peak_rss_bytes)
    if dea_provenance is not None:
        manifest["dea_provenance"] = dict(dea_provenance)
    if zoning is not None:
        manifest["zoning"] = dict(zoning)
    return manifest


def write_run_metadata(
    output_dir: str | Path,
    config: HydroConfig,
    **manifest_arguments: object,
) -> RunMetadataArtifacts:
    """Write canonical scientific config plus its linked run manifest."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.json"
    manifest_path = root / "run_manifest.json"
    manifest = build_run_manifest(
        config,
        manifest_schema_version=LEGACY_MANIFEST_SCHEMA_VERSION,
        **manifest_arguments,  # type: ignore[arg-type]
    )
    config_path.write_bytes(_json_bytes(config.scientific_config()))
    manifest_path.write_bytes(_json_bytes(manifest))
    return RunMetadataArtifacts(config_path, manifest_path)


def read_run_manifest(source: str | Path) -> dict[str, object]:
    """Read a run manifest from a file or bundle directory."""

    path = Path(source)
    if path.is_dir():
        path = path / "run_manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read run manifest: {path}") from error
    if not isinstance(value, dict):
        raise ManifestError("run manifest must be a JSON object")
    return value


def _validate_legacy_artifacts(root: Path, manifest: Mapping[str, object]) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ManifestError("manifest artifacts must be an object")
    relative = _relative_artifacts(
        {str(name): str(path) for name, path in artifacts.items()}
    )
    for name, relative_path in relative.items():
        if not (root / relative_path).exists():
            raise ManifestError(f"missing bundle artifact {name}: {relative_path}")


def _validate_inventory_entry(root: Path, entry: Mapping[str, object]) -> None:
    relative_path = str(entry.get("relative_path", ""))
    if not relative_path:
        raise ManifestError("artifact_inventory entry missing relative_path")
    path = root / relative_path
    if not path.exists():
        raise ManifestError(f"missing inventory artifact: {relative_path}")

    if path.is_dir():
        byte_size, digest = hash_directory_tree(path)
    else:
        byte_size = path.stat().st_size
        digest = hash_file(path)

    expected_size = entry.get("byte_size")
    if expected_size is not None and int(expected_size) != byte_size:
        raise ManifestError(f"byte_size mismatch for {relative_path}")
    expected_digest = entry.get("sha256")
    if expected_digest is not None and str(expected_digest) != digest:
        raise ManifestError(f"sha256 mismatch for {relative_path}")

    spatial = entry.get("spatial")
    if spatial is None or not isinstance(spatial, Mapping):
        return

    if path.suffix.lower() in {".tif", ".tiff"}:
        _validate_geotiff_spatial_metadata(path, spatial)
    elif path.suffix.lower() == ".gpkg":
        _validate_gpkg_spatial_metadata(path, spatial)


def _crs_equal(left: object, right: object) -> bool:
    from rasterio.crs import CRS

    try:
        left_crs = CRS.from_user_input(left)
        right_crs = CRS.from_user_input(right)
    except Exception:
        return str(left) == str(right)
    left_epsg = left_crs.to_epsg()
    right_epsg = right_crs.to_epsg()
    if left_epsg is not None and right_epsg is not None:
        return left_epsg == right_epsg
    try:
        return bool(left_crs.equals(right_crs))
    except Exception:
        return str(left_crs) == str(right_crs)


def _validate_geotiff_spatial_metadata(
    path: Path,
    spatial: Mapping[str, object],
) -> None:
    import rasterio

    with rasterio.open(path) as dataset:
        if spatial.get("crs") and not _crs_equal(dataset.crs, spatial["crs"]):
            raise ManifestError(f"CRS mismatch for {path.name}")
        if spatial.get("shape") and list(dataset.shape) != list(spatial["shape"]):
            raise ManifestError(f"shape mismatch for {path.name}")
        if spatial.get("dtype") and str(dataset.dtypes[0]) != str(spatial["dtype"]):
            raise ManifestError(f"dtype mismatch for {path.name}")
        if spatial.get("band_count") and dataset.count != int(spatial["band_count"]):
            raise ManifestError(f"band_count mismatch for {path.name}")


def _validate_gpkg_spatial_metadata(path: Path, spatial: Mapping[str, object]) -> None:
    import pyogrio

    info = pyogrio.read_info(path)
    layers = spatial.get("layers")
    if layers is not None:
        expected_layers = {str(layer) for layer in layers}
        actual_layers = set(info["layer_names"])
        if expected_layers != actual_layers:
            raise ManifestError(f"layer mismatch for {path.name}")


def validate_result_bundle(output_dir: str | Path) -> dict[str, object]:
    """Validate bundle integrity without reopening any source dataset."""

    root = Path(output_dir)
    manifest = read_run_manifest(root)
    schema_version = manifest.get("manifest_schema_version")
    if schema_version not in SUPPORTED_MANIFEST_SCHEMA_VERSIONS:
        raise ManifestError("unsupported manifest_schema_version")
    if manifest.get("output_schema_version") != SCHEMA_VERSION:
        raise ManifestError("unsupported output_schema_version")

    if schema_version == MANIFEST_SCHEMA_VERSION:
        inventory = manifest.get("artifact_inventory")
        if not isinstance(inventory, list):
            raise ManifestError("manifest artifact_inventory must be a list")
        for entry in inventory:
            if not isinstance(entry, Mapping):
                raise ManifestError("artifact_inventory entries must be objects")
            _validate_inventory_entry(root, entry)
    else:
        _validate_legacy_artifacts(root, manifest)

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ManifestError("manifest artifacts must be an object")
    relative = _relative_artifacts(
        {str(name): str(path) for name, path in artifacts.items()}
    )

    config_path = root / str(relative["config"])
    try:
        config_value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError("config.json is not valid JSON") from error
    if _hash_json(config_value) != manifest.get("config_hash"):
        raise ManifestError("config_hash does not match config.json")
    if config_value != manifest.get("resolved_config"):
        raise ManifestError("resolved_config does not match config.json")
    if not manifest.get("input_fingerprint"):
        raise ManifestError("input_fingerprint is missing")
    return manifest


__all__ = [
    "LEGACY_MANIFEST_SCHEMA_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "ManifestError",
    "RunMetadataArtifacts",
    "SUPPORTED_MANIFEST_SCHEMA_VERSIONS",
    "build_artifact_inventory",
    "build_dea_provenance",
    "build_zoning_section",
    "build_run_manifest",
    "guess_media_type",
    "hash_directory_tree",
    "hash_file",
    "read_run_manifest",
    "validate_result_bundle",
    "write_run_metadata",
]
