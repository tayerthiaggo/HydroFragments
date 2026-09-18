"""Atomic result-bundle staging, validation, and directory-commit publication."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from hydrofragments.config import HydroConfig
from hydrofragments.output.manifest import (
    MANIFEST_SCHEMA_VERSION,
    ManifestError,
    RunMetadataArtifacts,
    _json_bytes,
    _utc_text,
    build_artifact_inventory,
    build_run_manifest,
    validate_result_bundle,
)

TRANSACTION_SCHEMA_VERSION = "1.0.0"
TRANSACTION_RECORD_NAME = ".bundle_transaction.json"
TRANSACTION_STATE_OPEN = "open"
TRANSACTION_STATE_STAGED = "staged"


class BundleError(ValueError):
    """Raised when bundle staging or publication is invalid."""


@dataclass
class ArtifactRegistration:
    """One artifact tracked for manifest inventory and validation."""

    name: str
    relative_path: str
    media_type: str | None = None
    algorithm_version: str | None = None
    scientific_config_hash: str | None = None
    execution_config_hash: str | None = None
    spatial: Mapping[str, object] | None = None


@dataclass
class BundleTransaction:
    """Own one staged output directory until commit or abort."""

    target_dir: Path
    run_id: str
    config: HydroConfig
    staging_dir: Path
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    _artifacts: list[ArtifactRegistration] = field(default_factory=list)
    _committed: bool = False
    _aborted: bool = False

    @property
    def root(self) -> Path:
        """Alias for the writable staging root."""

        return self.staging_dir

    def register_artifact(self, registration: ArtifactRegistration) -> None:
        self._ensure_open()
        path = Path(registration.relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise BundleError(
                f"artifact path for {registration.name} must be relative to the bundle root"
            )
        self._artifacts.append(registration)

    def write_config(self) -> Path:
        """Write ``config.json`` into staging."""

        self._ensure_open()
        config_path = self.staging_dir / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_bytes(_json_bytes(self.config.scientific_config()))
        return config_path

    def finalize(
        self,
        *,
        package_version: str,
        git_sha: str,
        input_fingerprint: Mapping[str, object],
        planned_backend: str,
        actual_backend_by_stage: Mapping[str, str],
        skipped_metrics: Sequence[Mapping[str, object]] = (),
        warnings: Sequence[str] = (),
        timings_seconds: Mapping[str, float] | None = None,
        comparison_context: Mapping[str, object] | None = None,
        dependency_versions: Mapping[str, str] | None = None,
        backend_capabilities: Mapping[str, object] | None = None,
        dea_provenance: Mapping[str, object] | None = None,
        zoning: Mapping[str, object] | None = None,
        peak_rss_bytes: int | None = None,
        extra_artifacts: Mapping[str, str | Path] | None = None,
        validate_hook: Any | None = None,
        manifest_write_hook: Any | None = None,
    ) -> RunMetadataArtifacts:
        """Validate the staged bundle, write the manifest last, and commit."""

        self._ensure_open()
        config_path = self.staging_dir / "config.json"
        if not config_path.exists():
            raise BundleError("config.json must be written before finalize()")

        inventory = build_artifact_inventory(
            self.staging_dir,
            self._artifacts,
            config=self.config,
        )
        artifact_paths = {
            str(entry["name"]): str(entry["relative_path"])
            for entry in inventory
            if str(entry["name"]) != "config"
        }
        if extra_artifacts:
            artifact_paths.update(
                {str(name): str(path) for name, path in extra_artifacts.items()}
            )

        manifest = build_run_manifest(
            self.config,
            run_id=self.run_id,
            package_version=package_version,
            git_sha=git_sha,
            input_fingerprint=input_fingerprint,
            planned_backend=planned_backend,
            actual_backend_by_stage=actual_backend_by_stage,
            skipped_metrics=skipped_metrics,
            warnings=warnings,
            timings_seconds=timings_seconds,
            comparison_context=comparison_context,
            dependency_versions=dependency_versions,
            backend_capabilities=backend_capabilities,
            dea_provenance=dea_provenance,
            zoning=zoning,
            artifacts=artifact_paths,
            artifact_inventory=inventory,
            created_at=self.created_at,
            manifest_schema_version=MANIFEST_SCHEMA_VERSION,
            peak_rss_bytes=peak_rss_bytes,
        )

        if validate_hook is not None:
            validate_hook(self.staging_dir, manifest, inventory)

        manifest_path = self.staging_dir / "run_manifest.json"
        if manifest_write_hook is not None:
            manifest_write_hook(manifest_path, manifest)
        manifest_path.write_bytes(_json_bytes(manifest))

        validate_result_bundle(self.staging_dir)
        self._write_transaction_record(TRANSACTION_STATE_STAGED)
        self._remove_transaction_record()
        commit_staged_bundle(self.staging_dir, self.target_dir)
        self._committed = True
        return RunMetadataArtifacts(
            self.target_dir / "config.json",
            self.target_dir / "run_manifest.json",
        )

    def abort(self) -> None:
        """Remove owned staging state after a failed publication attempt."""

        if self._committed or self._aborted:
            return
        self._aborted = True
        if self.staging_dir.exists():
            shutil.rmtree(self.staging_dir, ignore_errors=True)

    def _ensure_open(self) -> None:
        if self._committed:
            raise BundleError("bundle transaction already committed")
        if self._aborted:
            raise BundleError("bundle transaction already aborted")

    def _write_transaction_record(self, state: str) -> None:
        payload = {
            "transaction_schema_version": TRANSACTION_SCHEMA_VERSION,
            "target_dir": str(self.target_dir.resolve()),
            "run_id": self.run_id,
            "staging_dir": str(self.staging_dir.resolve()),
            "created_at": _utc_text(self.created_at),
            "state": state,
        }
        record_path = self.staging_dir / TRANSACTION_RECORD_NAME
        record_path.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def _remove_transaction_record(self) -> None:
        record_path = self.staging_dir / TRANSACTION_RECORD_NAME
        if record_path.exists():
            record_path.unlink()


def staging_directory_for(target_dir: Path, run_id: str) -> Path:
    """Return the sibling staging directory for one target/run pair."""

    target = Path(target_dir)
    safe_run_id = run_id.replace(os.sep, "_").replace(":", "_")
    return target.parent / f"{target.name}.{safe_run_id}.staging"


def assert_output_dir_available(target_dir: Path) -> None:
    """Reject a non-empty or occupied final output directory."""

    target = Path(target_dir)
    if target.exists():
        if not target.is_dir():
            raise BundleError(f"output_dir exists and is not a directory: {target}")
        contents = list(target.iterdir())
        if contents:
            raise BundleError(
                f"output_dir must be absent or empty before publication: {target}"
            )


def open_bundle_transaction(
    output_dir: str | Path,
    *,
    run_id: str,
    config: HydroConfig,
) -> BundleTransaction:
    """Create a sibling staging directory and ownership record."""

    target = Path(output_dir)
    assert_output_dir_available(target)
    staging_dir = staging_directory_for(target, run_id)
    if staging_dir.exists():
        recover_or_remove_staging(staging_dir, target_dir=target, run_id=run_id)
    if staging_dir.exists():
        raise BundleError(f"staging directory already exists: {staging_dir}")

    target.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=False)
    transaction = BundleTransaction(
        target_dir=target,
        run_id=run_id,
        config=config,
        staging_dir=staging_dir,
    )
    transaction._write_transaction_record(TRANSACTION_STATE_OPEN)
    return transaction


def commit_staged_bundle(staging_dir: Path, target_dir: Path) -> Path:
    """Atomically publish one validated staging directory."""

    staging = Path(staging_dir)
    target = Path(target_dir)
    if not staging.exists():
        raise BundleError(f"staging directory does not exist: {staging}")
    if target.exists():
        if not target.is_dir():
            raise BundleError(f"refusing to overwrite non-directory target: {target}")
        if any(target.iterdir()):
            raise BundleError(f"refusing to overwrite non-empty target: {target}")
        target.rmdir()
    staging.rename(target)
    return target


def read_transaction_record(staging_dir: Path) -> dict[str, object]:
    record_path = Path(staging_dir) / TRANSACTION_RECORD_NAME
    try:
        payload = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BundleError(f"cannot read bundle transaction record: {record_path}") from error
    if not isinstance(payload, dict):
        raise BundleError("bundle transaction record must be a JSON object")
    return payload


def recover_or_remove_staging(
    staging_dir: Path,
    *,
    target_dir: Path,
    run_id: str,
) -> Path | None:
    """Commit a complete staged bundle or remove an owned incomplete transaction."""

    staging = Path(staging_dir)
    if not staging.is_dir():
        return None

    record_path = staging / TRANSACTION_RECORD_NAME
    if not record_path.exists():
        return None

    record = read_transaction_record(staging)
    if record.get("transaction_schema_version") != TRANSACTION_SCHEMA_VERSION:
        return None
    if record.get("run_id") != run_id:
        return None
    if Path(str(record.get("target_dir", ""))).resolve() != Path(target_dir).resolve():
        return None
    if Path(str(record.get("staging_dir", ""))).resolve() != staging.resolve():
        return None

    manifest_path = staging / "run_manifest.json"
    if manifest_path.exists():
        validate_result_bundle(staging)
        record_path.unlink(missing_ok=True)
        return commit_staged_bundle(staging, target_dir)

    shutil.rmtree(staging)
    return None


def discover_owned_staging_directories(
    parent_dir: Path,
    *,
    target_dir: Path,
    run_id: str,
) -> list[Path]:
    """Return sibling staging directories owned by one target/run pair."""

    parent = Path(parent_dir)
    expected = staging_directory_for(target_dir, run_id)
    matches: list[Path] = []
    if expected.exists():
        matches.append(expected)
    pattern = f"{Path(target_dir).name}.{run_id.replace(os.sep, '_').replace(':', '_')}.staging"
    for candidate in parent.glob("*.staging"):
        if candidate == expected:
            continue
        if candidate.name == pattern and (candidate / TRANSACTION_RECORD_NAME).exists():
            record = read_transaction_record(candidate)
            if (
                record.get("run_id") == run_id
                and Path(str(record.get("target_dir", ""))).resolve()
                == Path(target_dir).resolve()
            ):
                matches.append(candidate)
    return matches


__all__ = [
    "ArtifactRegistration",
    "BundleError",
    "BundleTransaction",
    "TRANSACTION_RECORD_NAME",
    "assert_output_dir_available",
    "commit_staged_bundle",
    "discover_owned_staging_directories",
    "open_bundle_transaction",
    "read_transaction_record",
    "recover_or_remove_staging",
    "staging_directory_for",
]
