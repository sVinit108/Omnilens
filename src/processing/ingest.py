"""
ingest.py
─────────
Main entrypoint for the data ingestion pipeline (Steps 1–3).

Flow:
  1. Recursively walk the data directory              (os.walk)
  2. Classify each file → DataCategory                (classifier.py)
  3. Group files by category                          (collections.defaultdict)
  4. Dispatch each group to its preprocessing pipeline (pipelines.py)
  5. Collect all ProcessingResult objects             (→ Step 4: vector DB)

Concurrency:
  Each category's pipeline runs in its own thread via ThreadPoolExecutor.
  I/O-bound work (reading files) benefits from threading; swap to
  ProcessPoolExecutor for CPU-bound pipelines (e.g. image processing).
"""

import logging
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from classifier import classify
from models import DataCategory, ProcessingResult
from pipelines import PIPELINE_REGISTRY

logger = logging.getLogger(__name__)


def _walk_and_classify(root: Path) -> dict[DataCategory, list[Path]]:
    """
    Recursively walk root, classify every file, and group paths by category.
    Returns: { DataCategory: [Path, ...] }
    """
    groups: dict[DataCategory, list[Path]] = defaultdict(list)

    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            filepath = Path(dirpath) / filename
            category = classify(filepath)
            groups[category].append(filepath)

    return groups


def _dispatch(
    groups: dict[DataCategory, list[Path]],
    max_workers: int,
) -> list[ProcessingResult]:
    """
    Submit each category's file batch to its pipeline concurrently.
    Collects and returns all ProcessingResult objects.
    """
    all_results: list[ProcessingResult] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit one task per category (not per file — pipelines are batch-aware)
        futures = {
            executor.submit(PIPELINE_REGISTRY[category], files): category
            for category, files in groups.items()
        }

        for future in as_completed(futures):
            category = futures[future]
            try:
                results: list[ProcessingResult] = future.result()
                all_results.extend(results)
            except Exception as exc:
                # Pipeline itself raised unexpectedly — mark all files as failed
                logger.error("Pipeline crashed for category %s: %s", category.value, exc)
                for filepath in groups[category]:
                    all_results.append(
                        ProcessingResult(
                            filepath=filepath,
                            category=category,
                            success=False,
                            error=f"Pipeline crash: {exc}",
                        )
                    )

    return all_results


def run(root_dir: str | Path, max_workers: int = 8) -> list[ProcessingResult]:
    """
    Full ingestion run: walk → classify → group → dispatch → return results.

    Args:
        root_dir:    Root directory containing all data (nested structure supported).
        max_workers: Max concurrent pipeline threads. Tune to your I/O capacity.

    Returns:
        List of ProcessingResult — one per file, success or failure.
        Pass this to Step 4 (vector DB ingestion).
    """
    root = Path(root_dir).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")

    # Step 1 + 2: Walk and classify
    groups = _walk_and_classify(root)

    # Step 3 + 4: Group dispatch (concurrent per-category)
    results = _dispatch(groups, max_workers=max_workers)

    return results


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    root_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    results  = run(root_dir)

    # Summary log
    total    = len(results)
    failed   = sum(1 for r in results if not r.success)
    logger.info("Ingestion complete — %d files processed, %d failed.", total, failed)

    # Serialize results to JSON for downstream pipeline (Step 4)
    output = [
        {
            "filepath": str(r.filepath),
            "category": r.category.value,
            "success":  r.success,
            "output":   r.output,
            "error":    r.error,
        }
        for r in results
    ]

    with open("ingestion_results.json", "w") as f:
        json.dump(output, f, indent=2)
