"""Create a tiny synthetic-only fixture for container integration tests."""

from __future__ import annotations

import argparse
from pathlib import Path

from triage.training import train_model

FIXTURE_CLASSES = {
    "cardiovascular": "cardiovascular demonstration marker",
    "digestive": "digestive demonstration marker",
    "general": "general pathology demonstration marker",
    "neoplasm": "neoplasm demonstration marker",
    "nervous": "nervous system demonstration marker",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("."),
        help="Directory for temporary training data and model artifacts.",
    )
    work_dir = parser.parse_args().work_dir
    root = work_dir / "data/integration-fixture"
    train_csv = root / "train.csv"
    test_csv = root / "test.csv"
    root.mkdir(parents=True, exist_ok=True)

    def write_fixture(path: Path, samples_per_class: int) -> None:
        with path.open("w", encoding="utf-8", newline="") as output:
            output.write("text,label\n")
            for label, marker in FIXTURE_CLASSES.items():
                for index in range(samples_per_class):
                    text = (
                        f"synthetic integration test only; {marker}; "
                        f"example {index:03d}"
                    )
                    output.write(f'"{text}",{label}\n')

    write_fixture(train_csv, samples_per_class=20)
    write_fixture(test_csv, samples_per_class=4)
    result = train_model(
        train_csv,
        work_dir / "artifacts",
        test_data_path=test_csv,
    )
    if result.samples != 100 or result.evaluation is None:
        raise RuntimeError("Integration fixture training did not complete")
    print(
        "Created test-only synthetic model fixture; "
        "it is not a clinical model or a benchmark dataset."
    )


if __name__ == "__main__":
    main()
