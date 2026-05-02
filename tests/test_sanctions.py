import csv

import pytest

from compliance.sanctions import SanctionsScreener


def write_sanctions_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["account_id", "list_name", "entity_name"])
        writer.writeheader()
        writer.writerows(rows)


def test_sanctions_screener_matches_sender_account(tmp_path):
    csv_path = tmp_path / "sanctions.csv"
    write_sanctions_csv(
        csv_path,
        [{"account_id": "ACC-SANCTIONED-1", "list_name": "UN_CONSOLIDATED", "entity_name": "Listed Entity"}],
    )
    screener = SanctionsScreener(str(csv_path))

    result = screener.screen({
        "transaction_id": "TXN_SANCTIONS_001",
        "sender_account": "ACC-SANCTIONED-1",
        "receiver_account": "ACC-CLEAR-1",
    })

    assert result.is_hit is True
    assert result.matched_account == "ACC-SANCTIONED-1"
    assert result.matched_field == "sender_account"
    assert result.list_name == "UN_CONSOLIDATED"


def test_sanctions_screener_matches_receiver_account(tmp_path):
    csv_path = tmp_path / "sanctions.csv"
    write_sanctions_csv(
        csv_path,
        [{"account_id": "ACC-SANCTIONED-2", "list_name": "EU", "entity_name": "Listed Beneficiary"}],
    )
    screener = SanctionsScreener(str(csv_path))

    result = screener.screen({
        "sender_account": "ACC-CLEAR-2",
        "receiver_account": "ACC-SANCTIONED-2",
    })

    assert result.is_hit is True
    assert result.matched_field == "receiver_account"
    assert result.entity_name == "Listed Beneficiary"


def test_sanctions_screener_no_hit_for_missing_file(tmp_path):
    screener = SanctionsScreener(str(tmp_path / "missing.csv"))

    result = screener.screen({"sender_account": "ACC-1", "receiver_account": "ACC-2"})

    assert result.is_hit is False
    assert screener.account_ids == frozenset()


def test_sanctions_csv_requires_account_id(tmp_path):
    csv_path = tmp_path / "invalid.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["name"])
        writer.writeheader()
        writer.writerow({"name": "Missing account id"})

    with pytest.raises(ValueError, match="account_id"):
        SanctionsScreener(str(csv_path))
