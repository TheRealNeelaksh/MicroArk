from pathlib import Path
import shutil
import sys

def setup():
    base_dir = Path(__file__).parent
    src_file = base_dir / 'src' / 'data' / 'animal_sample_list.csv'
    dest_dir = base_dir / 'data'
    dest_file = dest_dir / 'metalog_raw.csv'

    print(f"Base directory: {base_dir}")
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
