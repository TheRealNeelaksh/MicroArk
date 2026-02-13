from pathlib import Path
import shutil
import sys

def setup():
    project_root = Path(__file__).resolve().parents[2]
    src_file = project_root / 'src' / 'data' / 'animal_sample_list.csv'
    dest_dir = project_root / 'data'
    dest_file = dest_dir / 'metalog_raw.csv'

    print(f"Project root: {project_root}")
    print(f"Checking existing source: {src_file}")

    if not src_file.exists():
        print(f"ERROR: Source file not found at {src_file}")
        sys.exit(1)

    if not dest_dir.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dest_dir}")

    try:
        shutil.copy2(src_file, dest_file)
        print(f"SUCCESS: Copied {src_file.name} to {dest_file}")
        print(f"Destination file size: {dest_file.stat().st_size} bytes")
    except Exception as e:
        print(f"ERROR: Failed to copy file. Reason: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup()
