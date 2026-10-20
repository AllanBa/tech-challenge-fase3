"""Load medical-abstract classification CSVs, including the recommended corpus."""

from __future__ import annotations

import csv
from pathlib import Path

TEXT_COLUMNS = ("text", "medical_abstract", "abstract")
LABEL_COLUMNS = ("label", "condition_label")
LABEL_NAME_COLUMNS = ("condition_name", "class_name", "label_name", "name")


def _resolve_column(
    fieldnames: list[str] | None,
    candidates: tuple[str, ...],
    description: str,
) -> str:
    if not fieldnames:
        raise ValueError("CSV file must contain a header row")
    normalized = {name.strip().lower(): name for name in fieldnames if name}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    expected = ", ".join(f"'{name}'" for name in candidates)
    raise ValueError(f"CSV must contain a {description} column ({expected})")


def _read_label_names(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Label names CSV does not exist: {path}")

    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        label_column = _resolve_column(
            reader.fieldnames,
            LABEL_COLUMNS,
            "label ID",
        )
        name_column = _resolve_column(
            reader.fieldnames,
            LABEL_NAME_COLUMNS,
            "label name",
        )
        names: dict[str, str] = {}
        for row_number, row in enumerate(reader, start=2):
            label = (row.get(label_column) or "").strip()
            name = (row.get(name_column) or "").strip()
            if not label or not name:
                raise ValueError(
                    f"Label names CSV has an empty value on row {row_number}"
                )
            names[label] = name
    if not names:
        raise ValueError(f"Label names CSV contains no data rows: {path}")
    return names


def read_dataset(
    path: Path,
    labels_path: Path | None = None,
) -> tuple[list[str], list[str]]:
    """Read text/label columns and optionally resolve numeric IDs to class names.

    The Medical Abstracts TC Corpus uses ``medical_abstract`` and
    ``condition_label``; generic datasets may use ``text`` and ``label``.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Training CSV does not exist: {path}")

    label_names = _read_label_names(labels_path) if labels_path else {}
    texts: list[str] = []
    labels: list[str] = []
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        text_column = _resolve_column(reader.fieldnames, TEXT_COLUMNS, "text")
        label_column = _resolve_column(reader.fieldnames, LABEL_COLUMNS, "label")
        for row_number, row in enumerate(reader, start=2):
            text = (row.get(text_column) or "").strip()
            label_id = (row.get(label_column) or "").strip()
            if not text or not label_id:
                raise ValueError(
                    f"Training CSV has an empty text or label on row {row_number}"
                )
            if label_names:
                try:
                    label = label_names[label_id]
                except KeyError as error:
                    raise ValueError(
                        f"Label ID {label_id!r} on row {row_number} "
                        "is missing from the label names CSV"
                    ) from error
            else:
                label = label_id
            texts.append(text)
            labels.append(label)

    if not texts:
        raise ValueError(f"Training CSV contains no data rows: {path}")
    return texts, labels
