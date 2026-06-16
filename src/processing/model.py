"""
models.py
─────────
Shared data models used across the ingestion pipeline.
"""

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class DataCategory(str, Enum):
    """
    The 21 data types from the ingestion spec.
    Inherits from str so instances serialize cleanly to JSON.
    """
    DOCUMENTS       = "Documents"
    PLAIN_TEXT      = "Plain Text"
    STRUCTURED_TEXT = "Structured Text"
    TABULAR         = "Tabular Data"
    IMAGES          = "Images"
    VIDEO           = "Video"
    AUDIO           = "Audio"
    ARCHIVES        = "Archives"
    DATABASES       = "Databases"
    CODE            = "Code"
    LOGS            = "Logs"
    ML_MODELS       = "ML Models"
    TIME_SERIES     = "Time-Series Data"
    GEOSPATIAL      = "Geospatial Data"
    GRAPH           = "Graph Data"
    MEDICAL         = "Medical Imaging"
    SCIENTIFIC      = "Scientific Data"
    DATA_3D         = "3D Data"
    POINT_CLOUD     = "Point Cloud Data"
    CAD             = "CAD Data"
    EBOOKS          = "eBooks"
    UNCLASSIFIED    = "Unclassified"


def compute_hash(filepath: Path) -> str:
    """
    SHA-256 hash of raw file bytes.
    Stored alongside every ProcessingResult for lineage tracking.
    If the hash changes on re-scan, downstream layers (ontology, vectors)
    know to reprocess only this file — not the entire corpus.
    Reads in 64 KB chunks to handle arbitrarily large files.
    """
    h = hashlib.sha256()
    with open(filepath, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ProcessingResult:
    """
    Outcome of running one file through its preprocessing pipeline.
    Passed downstream to Step 4 (JSON storage) → Step 5 (ontology + vector DB).

    Fields:
        filepath        — absolute path to source file
        category        — one of the 21 DataCategory types
        success         — False if the pipeline raised an exception
        content_hash    — SHA-256 of raw file; drives incremental reprocessing
        output          — normalized JSON payload (schema varies per category)
        error           — exception message when success=False
    """
    filepath:       Path
    category:       DataCategory
    success:        bool
    content_hash:   str  = ""
    output:         dict = field(default_factory=dict)
    error:          str  = ""
