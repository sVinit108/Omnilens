"""
classifier.py
─────────────
Recursively walks a directory, identifies each file's true data type
(via magic bytes + extension fallback), and dispatches it to the
appropriate preprocessing pipeline.

Install dependency:
    pip install python-magic
"""

import os
import json
import magic  # python-magic: reads binary signatures (magic bytes)
from pathlib import Path


# ─────────────────────────────────────────────
# 1. CATEGORY DEFINITIONS
#    Maps your 21 data types → their file extensions.
#    Used as fallback when magic bytes are inconclusive.
# ─────────────────────────────────────────────

EXTENSION_MAP = {
    "Documents":                [".pdf", ".doc", ".docx", ".odt"],
    "Plain Text":               [".txt", ".md", ".rtf"],
    "Structured Text":          [".json", ".xml", ".yaml", ".yml", ".html"],
    "Tabular Data":             [".csv", ".xlsx", ".xls", ".tsv", ".parquet"],
    "Images":                   [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"],
    "Video":                    [".mp4", ".mkv", ".mov", ".avi", ".webm"],
    "Audio":                    [".mp3", ".wav", ".aac", ".flac", ".ogg"],
    "Archives":                 [".zip", ".tar", ".gz", ".rar", ".7z"],
    "Databases":                [".sqlite", ".db", ".sql"],
    "Code":                     [".py", ".cpp", ".java", ".js", ".sh", ".exe"],
    "Logs":                     [".log"],
    "ML Models":                [".pt", ".pth", ".onnx", ".h5", ".safetensors"],
    "Time-Series Data":         [],   # .csv/.json/.parquet shared — magic bytes decide
    "Geospatial Data":          [".geojson", ".shp", ".kml", ".gpx", ".geotiff"],
    "Graph Data":               [".graphml", ".gexf", ".rdf"],
    "Medical Imaging":          [".dcm", ".nii"],
    "Scientific Data":          [".hdf5", ".mat", ".nc"],
    "3D Data":                  [".obj", ".fbx", ".stl", ".gltf", ".ply"],
    "Point Cloud Data":         [".las", ".laz", ".pcd"],
    "CAD Data":                 [".dwg", ".dxf", ".step"],
    "eBooks":                   [".epub", ".mobi"],
}

# Reverse lookup: extension → category  (built once at startup)
EXT_TO_CATEGORY = {
    ext: category
    for category, exts in EXTENSION_MAP.items()
    for ext in exts
}


# ─────────────────────────────────────────────
# 2. MAGIC BYTES → CATEGORY
#    Maps MIME type substrings returned by python-magic
#    to your categories.  Most reliable identification method.
# ─────────────────────────────────────────────

MIME_TO_CATEGORY = {
    "application/pdf":          "Documents",
    "application/msword":       "Documents",
    "application/vnd.openxmlformats-officedocument.wordprocessingml": "Documents",
    "text/plain":               "Plain Text",
    "text/html":                "Structured Text",
    "application/json":         "Structured Text",
    "application/xml":          "Structured Text",
    "text/xml":                 "Structured Text",
    "text/csv":                 "Tabular Data",
    "application/vnd.ms-excel": "Tabular Data",
    "application/vnd.openxmlformats-officedocument.spreadsheetml": "Tabular Data",
    "application/parquet":      "Tabular Data",
    "image/":                   "Images",      # prefix match
    "video/":                   "Video",
    "audio/":                   "Audio",
    "application/zip":          "Archives",
    "application/x-tar":        "Archives",
    "application/gzip":         "Archives",
    "application/x-rar":        "Archives",
    "application/x-sqlite3":    "Databases",
    "application/x-executable": "Code",
    "application/x-hdf":        "Scientific Data",
    "model/":                   "3D Data",
}

def mime_to_category(mime: str) -> str | None:
    """Match a MIME type string to a category. Supports prefix matching."""
    for mime_key, category in MIME_TO_CATEGORY.items():
        if mime.startswith(mime_key):
            return category
    return None


# ─────────────────────────────────────────────
# 3. CORE CLASSIFIER
#    Layer 1: magic bytes  →  Layer 2: extension fallback
# ─────────────────────────────────────────────

def classify_file(filepath: Path) -> str:
    """
    Identify the data category of a single file.
    Returns a category string, or 'Unclassified' if unknown.
    """

    # --- Layer 1: Magic bytes (read actual binary content) ---
    try:
        mime = magic.from_file(str(filepath), mime=True)  # e.g. "image/png"
        category = mime_to_category(mime)
        if category:
            return category
    except Exception:
        pass  # If magic fails (permissions, corrupt file), fall through

    # --- Layer 2: Extension fallback ---
    ext = filepath.suffix.lower()
    if ext in EXT_TO_CATEGORY:
        return EXT_TO_CATEGORY[ext]

    return "Unclassified"


# ─────────────────────────────────────────────
# 4. DIRECTORY WALKER
#    Recursively finds all files and builds the manifest.
# ─────────────────────────────────────────────

def build_manifest(root_dir: str) -> dict[str, list[str]]:
    """
    Walk root_dir recursively.
    Returns a manifest: { category: [list of absolute file paths] }
    """
    manifest: dict[str, list[str]] = {}

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            filepath = Path(dirpath) / filename
            category = classify_file(filepath)

            # Group files by category (create key if first time)
            manifest.setdefault(category, []).append(str(filepath))

    return manifest


# ─────────────────────────────────────────────
# 5. PREPROCESSING PIPELINE STUBS
#    One function per category. Swap stubs with real logic later.
# ─────────────────────────────────────────────

def pipeline_documents(files):       print(f"  → Document pipeline:     {len(files)} files")
def pipeline_plain_text(files):      print(f"  → Plain Text pipeline:   {len(files)} files")
def pipeline_structured_text(files): print(f"  → Structured pipeline:   {len(files)} files")
def pipeline_tabular(files):         print(f"  → Tabular pipeline:      {len(files)} files")
def pipeline_images(files):          print(f"  → Image pipeline:        {len(files)} files")
def pipeline_video(files):           print(f"  → Video pipeline:        {len(files)} files")
def pipeline_audio(files):           print(f"  → Audio pipeline:        {len(files)} files")
def pipeline_archives(files):        print(f"  → Archive pipeline:      {len(files)} files")
def pipeline_databases(files):       print(f"  → Database pipeline:     {len(files)} files")
def pipeline_code(files):            print(f"  → Code pipeline:         {len(files)} files")
def pipeline_ml_models(files):       print(f"  → ML Model pipeline:     {len(files)} files")
def pipeline_medical(files):         print(f"  → Medical pipeline:      {len(files)} files")
def pipeline_scientific(files):      print(f"  → Scientific pipeline:   {len(files)} files")
def pipeline_geospatial(files):      print(f"  → Geospatial pipeline:   {len(files)} files")
def pipeline_3d(files):              print(f"  → 3D Data pipeline:      {len(files)} files")
def pipeline_unclassified(files):    print(f"  → Unclassified:          {len(files)} files (skipped)")

# Dispatcher: maps category name → pipeline function
PIPELINE_DISPATCHER = {
    "Documents":        pipeline_documents,
    "Plain Text":       pipeline_plain_text,
    "Structured Text":  pipeline_structured_text,
    "Tabular Data":     pipeline_tabular,
    "Images":           pipeline_images,
    "Video":            pipeline_video,
    "Audio":            pipeline_audio,
    "Archives":         pipeline_archives,
    "Databases":        pipeline_databases,
    "Code":             pipeline_code,
    "ML Models":        pipeline_ml_models,
    "Medical Imaging":  pipeline_medical,
    "Scientific Data":  pipeline_scientific,
    "Geospatial Data":  pipeline_geospatial,
    "3D Data":          pipeline_3d,
    "Unclassified":     pipeline_unclassified,
}


# ─────────────────────────────────────────────
# 6. DISPATCHER
#    Sends each category's file list to its pipeline.
# ─────────────────────────────────────────────

def dispatch_pipelines(manifest: dict[str, list[str]]):
    """Call the correct preprocessing pipeline for each category."""
    print("\n[DISPATCHING PIPELINES]")
    for category, files in manifest.items():
        pipeline_fn = PIPELINE_DISPATCHER.get(category, pipeline_unclassified)
        pipeline_fn(files)


# ─────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    ROOT_DIR = sys.argv[1] if len(sys.argv) > 1 else "./data"

    print(f"[SCANNING] {ROOT_DIR}\n")

    # Build the manifest (classify all files)
    manifest = build_manifest(ROOT_DIR)

    # Print summary
    print("[MANIFEST SUMMARY]")
    for category, files in manifest.items():
        print(f"  {category:<25} {len(files)} file(s)")

    # Save manifest to JSON (useful for debugging / next pipeline stage)
    with open("manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("\n[SAVED] manifest.json")

    # Dispatch each category to its pipeline
    dispatch_pipelines(manifest)
