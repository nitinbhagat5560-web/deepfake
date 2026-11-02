"""Submission history: order, filtering, the SHA-256 chain, tamper detection, and 'forget me'."""
import sqlite3
import shutil

import pytest

from sourcediff import history


@pytest.fixture(autouse=True)
def _data(tmp_path, monkeypatch):
    monkeypatch.setenv("SOURCEDIFF_DATA", str(tmp_path / "data"))


def add(user="alice", name="a.jpg", verdict="authentic", sha="a" * 64, now=None):
    return history.add_submission(user, name, sha, "image", verdict, verdict.title(), "case-1", False, "r", "card",
                                  now=now)


def test_row_has_utc_time_user_file_and_hash():
    r = add()
    assert r["user_id"] == "alice" and r["filename"] == "a.jpg" and r["sha256"] == "a" * 64
    assert r["submitted_at"].endswith("Z") and len(r["row_hash"]) == 64


def test_newest_first_and_per_user_filter():
    add("alice", "1.jpg")
    add("bob", "2.jpg")
    add("alice", "3.jpg")
    assert [r["filename"] for r in history.list_submissions()] == ["3.jpg", "2.jpg", "1.jpg"]
    assert [r["filename"] for r in history.list_submissions("alice")] == ["3.jpg", "1.jpg"]
    assert history.list_submissions("nobody") == []
    assert len(history.list_submissions(limit=2)) == 2


def test_chain_verifies_and_links_rows():
    a, b = add(), add()
    assert b["prev_hash"] == a["row_hash"]
    assert history.verify_chain() == {"ok": True, "rows": 2, "first_bad_id": None}


def test_empty_history_verifies():
    assert history.verify_chain()["ok"]


@pytest.mark.parametrize("column,value", [("verdict", "authentic"), ("sha256", "f" * 64), ("submitted_at", "2020-01-01T00:00:00Z"),
                                          ("door", "video")])
def test_editing_an_old_row_is_detected(column, value):
    add(verdict="edited_picture")
    add(); add()
    con = sqlite3.connect(history.paths.history_db())
    con.execute(f"UPDATE submissions SET {column}=? WHERE id=1", (value,))
    con.commit(); con.close()
    assert history.verify_chain() == {"ok": False, "rows": 1, "first_bad_id": 1}


def test_deleting_a_middle_row_is_detected():
    add(); add(); add()
    con = sqlite3.connect(history.paths.history_db())
    con.execute("DELETE FROM submissions WHERE id=2")
    con.commit(); con.close()
    assert history.verify_chain()["ok"] is False


def test_forget_user_blanks_names_and_the_chain_still_verifies():
    add("alice", "secret.jpg"); add("bob", "other.jpg"); add("alice", "secret2.jpg")
    assert history.forget_user("alice") == 2
    rows = {r["id"]: r for r in history.list_submissions()}
    assert rows[1]["user_id"] == "[deleted]" and rows[1]["filename"] == "[deleted]"
    assert rows[2]["user_id"] == "bob" and rows[2]["filename"] == "other.jpg"
    assert history.verify_chain()["ok"]
    assert history.forget_user("alice") == 0


def test_two_users_with_similar_names_do_not_mix():
    add("alice", "1.jpg"); add("alice ", "2.jpg")
    assert len(history.list_submissions("alice")) == 1


def test_library_disk_entries_are_registered_in_sqlite(tmp_path, monkeypatch):
    import cv2
    from sourcediff import library

    monkeypatch.setenv("SOURCEDIFF_LIBRARY", str(tmp_path / "library"))
    source = tmp_path / "trusted.jpg"
    assert cv2.imwrite(str(source), __import__("numpy").zeros((20, 20, 3), dtype=__import__("numpy").uint8))

    library.add_item(source, source.name, "Trusted source", publisher="Museum",
                     license="Public domain", source_id="museum-source",
                     strict=False, rebuild=False)
    sync = history.sync_library_registry()
    rows = history.library_registry()

    assert sync["count"] == 1
    assert len(rows) == 1
    row = rows[0]
    assert row["kind"] == "trusted"
    assert row["door"] == "image"
    assert row["source_id"] == "museum-source"
    assert row["status"] == "verified"
    assert row["disk_path"].startswith("trusted/image/museum-source/")
