import json
import re
from pathlib import Path


def compute_success_rate():
    results_dir = Path(__file__).parent / "results"

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    total = 0
    successes = 0

    for folder in sorted(results_dir.iterdir()):
        if not folder.is_dir() or not re.match(r"^webarena\.\d+$", folder.name):
            continue

        autoeval_file = folder / "gpt-3.5-turbo_autoeval.json"
        if not autoeval_file.exists():
            continue

        try:
            data = json.loads(autoeval_file.read_text())
            if isinstance(data, list):
                data = data[0]
            total += 1
            if data.get("rm") is True:
                successes += 1
        except (json.JSONDecodeError, IOError, IndexError) as e:
            print(f"Error reading {autoeval_file}: {e}")

    if total == 0:
        print("No results found.")
        return

    print(f"Total: {total}")
    print(f"Success: {successes}")
    print(f"Failure: {total - successes}")
    print(f"Success rate: {successes / total * 100:.2f}%")


if __name__ == "__main__":
    compute_success_rate()
