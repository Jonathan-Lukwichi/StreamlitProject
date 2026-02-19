"""
Upload Pipeline Artifacts to Supabase Storage.

This script uploads all trained models from pipeline_artifacts/ to the Supabase
Storage bucket "models" for cloud persistence and reuse.

Features:
- Uploads all model files (.pkl, .keras, .json, .csv)
- Compresses large files (>50MB) with gzip
- Preserves folder structure in the cloud
- Reports success/failure for each file

Usage:
    python scripts/upload_models_to_supabase.py

Requirements:
    - Supabase credentials in .streamlit/secrets.toml
    - "models" bucket must exist in Supabase Storage
"""

import os
import sys
import gzip
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now we can import app_core modules
from app_core.data.model_storage_service import (
    get_model_storage_service,
    MODELS_BUCKET,
    MAX_FILE_SIZE_BYTES,
    SUPPORTED_EXTENSIONS
)

# =============================================================================
# CONFIGURATION
# =============================================================================

ARTIFACTS_DIR = project_root / "pipeline_artifacts"

# File extensions to upload (add .csv for metrics files)
UPLOAD_EXTENSIONS = set(SUPPORTED_EXTENSIONS.keys()) | {".csv"}


# =============================================================================
# UPLOAD FUNCTIONS
# =============================================================================

def format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def upload_file_with_compression(
    service,
    local_path: Path,
    remote_path: str
) -> tuple[bool, str]:
    """
    Upload a file, compressing if necessary.

    Args:
        service: ModelStorageService instance
        local_path: Path to local file
        remote_path: Remote path in Supabase Storage

    Returns:
        (success, message)
    """
    file_size = local_path.stat().st_size

    if file_size <= MAX_FILE_SIZE_BYTES:
        # File is under limit, upload directly
        success, msg = service.upload_model(str(local_path), remote_path)
        return success, msg
    else:
        # File exceeds limit, try compression
        print(f"   📦 Compressing {format_size(file_size)} file...")

        try:
            with open(local_path, "rb") as f:
                original_data = f.read()

            # Compress with maximum compression
            compressed_data = gzip.compress(original_data, compresslevel=9)
            compressed_size = len(compressed_data)

            compression_ratio = (1 - compressed_size / file_size) * 100
            print(f"   📉 Compressed: {format_size(file_size)} → {format_size(compressed_size)} ({compression_ratio:.1f}% reduction)")

            if compressed_size > MAX_FILE_SIZE_BYTES:
                return False, f"Still too large after compression: {format_size(compressed_size)}"

            # Upload compressed file with .gz extension
            remote_path_gz = remote_path + ".gz"

            # Direct upload via Supabase client
            try:
                # Remove existing file if it exists
                try:
                    service.client.storage.from_(service.bucket_name).remove([remote_path_gz])
                except Exception:
                    pass  # File might not exist

                # Upload compressed bytes
                service.client.storage.from_(service.bucket_name).upload(
                    path=remote_path_gz,
                    file=compressed_data,
                    file_options={"content-type": "application/gzip"}
                )

                return True, f"Uploaded compressed: {remote_path_gz}"

            except Exception as e:
                return False, f"Upload failed: {str(e)}"

        except Exception as e:
            return False, f"Compression failed: {str(e)}"


def upload_all_models():
    """Upload all models from pipeline_artifacts to Supabase Storage."""
    print("=" * 60)
    print("🚀 UPLOAD PIPELINE ARTIFACTS TO SUPABASE STORAGE")
    print("=" * 60)
    print(f"📁 Source: {ARTIFACTS_DIR}")
    print(f"☁️  Destination: Supabase Storage / {MODELS_BUCKET}")
    print(f"📏 Max file size: {MAX_FILE_SIZE_BYTES / (1024*1024):.0f} MB")
    print("=" * 60)

    # Initialize service
    print("\n🔌 Connecting to Supabase...")
    service = get_model_storage_service()

    if not service.is_connected():
        print("❌ Failed to connect to Supabase. Check your credentials.")
        return

    print("✅ Connected to Supabase\n")

    # Check if artifacts directory exists
    if not ARTIFACTS_DIR.exists():
        print(f"❌ Artifacts directory not found: {ARTIFACTS_DIR}")
        return

    # Collect all files to upload
    files_to_upload = []
    for root, dirs, files in os.walk(ARTIFACTS_DIR):
        for filename in files:
            local_path = Path(root) / filename
            ext = local_path.suffix.lower()

            if ext in UPLOAD_EXTENSIONS:
                # Calculate remote path (preserve folder structure)
                relative_path = local_path.relative_to(ARTIFACTS_DIR)
                # Use forward slashes for Supabase paths
                remote_path = str(relative_path).replace("\\", "/")
                files_to_upload.append((local_path, remote_path))

    if not files_to_upload:
        print("⚠️ No files found to upload.")
        return

    print(f"📊 Found {len(files_to_upload)} files to upload\n")

    # Upload each file
    success_count = 0
    failed_count = 0
    skipped_count = 0
    compressed_count = 0

    for i, (local_path, remote_path) in enumerate(files_to_upload, 1):
        file_size = local_path.stat().st_size
        size_str = format_size(file_size)

        print(f"[{i}/{len(files_to_upload)}] {remote_path} ({size_str})")

        if file_size > MAX_FILE_SIZE_BYTES:
            compressed_count += 1

        success, message = upload_file_with_compression(service, local_path, remote_path)

        if success:
            print(f"   ✅ {message}")
            success_count += 1
        else:
            if "too large" in message.lower():
                print(f"   ⚠️ SKIPPED: {message}")
                skipped_count += 1
            else:
                print(f"   ❌ FAILED: {message}")
                failed_count += 1

        print()

    # Summary
    print("=" * 60)
    print("📊 UPLOAD SUMMARY")
    print("=" * 60)
    print(f"✅ Successful: {success_count}")
    print(f"📦 Compressed: {compressed_count}")
    print(f"⚠️  Skipped (too large): {skipped_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"📁 Total: {len(files_to_upload)}")
    print("=" * 60)

    if skipped_count > 0:
        print("\n💡 Tip: Skipped files exceeded 50MB even after compression.")
        print("   Consider upgrading your Supabase plan for larger storage limits.")


def list_remote_files():
    """List all files currently in the Supabase models bucket."""
    print("=" * 60)
    print("📋 LISTING REMOTE FILES IN SUPABASE STORAGE")
    print("=" * 60)

    service = get_model_storage_service()
    if not service.is_connected():
        print("❌ Failed to connect to Supabase")
        return

    try:
        # List root folder
        response = service.client.storage.from_(service.bucket_name).list()
        print(f"\n📁 {MODELS_BUCKET}/")

        for item in response:
            name = item.get("name", "unknown")
            is_folder = item.get("id") is None
            if is_folder:
                print(f"   📂 {name}/")
            else:
                size = item.get("metadata", {}).get("size", 0)
                print(f"   📄 {name} ({format_size(size)})")

    except Exception as e:
        print(f"❌ Error listing files: {e}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload models to Supabase Storage")
    parser.add_argument("--list", action="store_true", help="List remote files instead of uploading")
    args = parser.parse_args()

    if args.list:
        list_remote_files()
    else:
        upload_all_models()
