"""Checkpoint discovery/cleanup, adapter zipping, and Google Drive sync.

The raw Drive *mount* primitive lives in utils.py (pure environment
detection, safe for any module to call); the copy *policy* — what gets
synced where, and when — lives here, since it's tightly coupled to the
artifact-persistence lifecycle (checkpoints, adapter zips) this module
already owns.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Optional

from src import utils

logger = utils.get_logger("checkpoint")

_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    """Return the checkpoint-<N> subdirectory with the highest N, or None."""
    if not output_dir.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        match = _CHECKPOINT_RE.match(child.name)
        if match:
            candidates.append((int(match.group(1)), child))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def resolve_resume_path(output_dir: Path, resume_training: bool) -> Optional[str]:
    """Return a checkpoint path string suitable for Trainer(resume_from_checkpoint=...), or None.

    Returns None (not an error) when resume_training is False, or no prior
    checkpoint exists yet — both are normal for a first run.
    """
    if not resume_training:
        return None
    latest = find_latest_checkpoint(output_dir)
    if latest is None:
        logger.info("resume_training=true but no checkpoint-* found under %s — starting fresh.", output_dir)
        return None
    logger.info("Resuming from checkpoint: %s", latest)
    return str(latest)


def prune_checkpoints(output_dir: Path, keep: set[Path]) -> None:
    """Delete any checkpoint-* directory under output_dir not in `keep`.

    Defensive cleanup alongside HF Trainer's own save_total_limit handling
    (which already protects the checkpoint tracked as "best" from deletion) —
    this catches anything left behind by an interrupted run or a config
    change between runs.
    """
    if not output_dir.is_dir():
        return
    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        if _CHECKPOINT_RE.match(child.name) and child not in keep:
            logger.info("Pruning old checkpoint: %s", child)
            shutil.rmtree(child, ignore_errors=True)


def find_latest_synced_run(drive_base_dir: Path) -> Optional[Path]:
    """Return the most recently Drive-synced run_* directory that has an adapter/ subdir.

    Run directory names (run_YYYYMMDD_HHMMSS, via utils.timestamp_run_id) sort
    correctly as plain strings, so — unlike find_latest_checkpoint's numeric
    suffix parse — a lexicographic sort is sufficient here.
    """
    if not drive_base_dir.is_dir():
        return None
    candidates = [
        child
        for child in drive_base_dir.iterdir()
        if child.is_dir() and child.name.startswith("run_") and (child / "adapter").is_dir()
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


def zip_adapter(adapter_dir: Path, zip_path: Path) -> Path:
    """Zip adapter_dir to zip_path (notebook section 12's shutil.make_archive)."""
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    utils.ensure_dir(zip_path.parent)
    base_name = str(zip_path.with_suffix(""))
    archive_path_str = shutil.make_archive(base_name, "zip", root_dir=str(adapter_dir))
    archive_path = Path(archive_path_str)
    if archive_path != zip_path:
        archive_path.replace(zip_path)
        archive_path = zip_path
    size_mb = archive_path.stat().st_size / (1024 * 1024)
    logger.info("Zipped adapter: %s (%.1f MB)", archive_path, size_mb)
    return archive_path


def sync_run_to_drive(
    run_dir: Path,
    drive_base_dir: Optional[str],
    mount_point: str = "/content/drive",
) -> Optional[Path]:
    """Copy this run's artifacts (adapter/, manifest.json, config.yaml, metrics/, predictions/)
    to Google Drive.

    No-ops gracefully (returns None) if drive_base_dir is unset or Drive
    mounting fails — Drive sync is a convenience, not a hard requirement.
    """
    if not drive_base_dir:
        logger.info("drive.google_drive_directory unset — skipping Drive sync.")
        return None

    mounted = utils.mount_google_drive(mount_point)
    if mounted is None and utils.is_colab():
        logger.warning("Google Drive mount failed — skipping Drive sync.")
        return None
    if mounted is None and not utils.is_colab():
        logger.info("Not running in Colab — skipping Drive sync (would target %s).", drive_base_dir)
        return None

    dest_root = utils.ensure_dir(Path(drive_base_dir) / run_dir.name)

    items_to_sync = ["adapter", "manifest.json", "config.yaml", "metrics", "predictions"]
    for item_name in items_to_sync:
        src_path = run_dir / item_name
        if not src_path.exists():
            continue
        dest_path = dest_root / item_name
        try:
            if src_path.is_dir():
                shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
            else:
                shutil.copy2(src_path, dest_path)
            logger.info("Synced %s -> %s", src_path, dest_path)
        except OSError as exc:
            logger.warning("Failed to sync %s to Drive: %s", src_path, exc)

    return dest_root
