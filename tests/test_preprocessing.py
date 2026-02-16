import pandas as pd
import pytest
from src.preprocessing import clean_data, prepare_real_data, _is_pdf_file, _is_zip_file


def test_clean_data_maps_and_dedupes_columns():
    df = pd.DataFrame(
        [["5,5", "Ametista", "4,0", "7,0", "8,0", "6,0", "5,0", "6,5", "7,5", "A", "B"]],
        columns=[
            "INDE 22",
            "Pedra 22",
            "IDA",
            "IEG",
            "IAA",
            "IPS",
            "IPP",
            "IPV",
            "IAN",
            "Destaque IPV",
            "Destaque IPV",
        ],
    )

    cleaned = clean_data(df)
    assert "INDE" in cleaned.columns
    assert "PEDRA" in cleaned.columns
    assert "IDA" in cleaned.columns

    destaque_cols = [c for c in cleaned.columns if c.startswith("destaque_ipv")]
    assert len(destaque_cols) == 2


def test_prepare_real_data_coerces_commas():
    df = pd.DataFrame({
        "INDE 22": ["5,5", "7,0"],
        "IDA": ["4,0", "6,0"],
        "IEG": ["7,0", "8,0"],
        "IAA": ["8,0", "9,0"],
        "IPS": ["6,0", "7,0"],
        "IPP": ["5,0", "6,0"],
        "IPV": ["6,5", "7,0"],
        "IAN": ["7,5", "8,0"],
    })

    X, y = prepare_real_data(df)
    assert "INDE" in X.columns
    assert "IDA" in X.columns
    assert y.tolist() == [1, 0]


def test_prepare_real_data_missing_columns():
    df = pd.DataFrame({"INDE 22": ["5,5"]})
    with pytest.raises(ValueError):
        prepare_real_data(df)


def test_file_signature_detection(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n")

    zip_path = tmp_path / "sample.zip"
    zip_path.write_bytes(b"PK\x03\x04")

    assert _is_pdf_file(str(pdf_path)) is True
    assert _is_zip_file(str(zip_path)) is True
