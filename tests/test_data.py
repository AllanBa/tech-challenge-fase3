import csv

import pytest

from triage.data import read_dataset


def test_read_medical_abstract_corpus_and_resolve_class_names(tmp_path):
    dataset_path = tmp_path / "medical_tc_train.csv"
    labels_path = tmp_path / "medical_tc_labels.csv"
    with dataset_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=("condition_label", "medical_abstract"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "condition_label": "1",
                "medical_abstract": "fictional abstract for a test",
            }
        )
    with labels_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=("condition_label", "condition_name"),
        )
        writer.writeheader()
        writer.writerow({"condition_label": "1", "condition_name": "Neoplasms"})

    texts, labels = read_dataset(dataset_path, labels_path)

    assert texts == ["fictional abstract for a test"]
    assert labels == ["Neoplasms"]


def test_read_dataset_rejects_missing_columns(tmp_path):
    path = tmp_path / "invalid.csv"
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=("message",))
        writer.writeheader()
        writer.writerow({"message": "sample"})

    with pytest.raises(ValueError, match="text column"):
        read_dataset(path)


def test_read_dataset_rejects_unmapped_label_id(tmp_path):
    dataset_path = tmp_path / "medical_tc_train.csv"
    labels_path = tmp_path / "medical_tc_labels.csv"
    dataset_path.write_text(
        "medical_abstract,condition_label\nfictional text,9\n",
        encoding="utf-8",
    )
    labels_path.write_text(
        "condition_label,condition_name\n1,Neoplasms\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Label ID '9'.*missing"):
        read_dataset(dataset_path, labels_path)
