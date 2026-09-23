"""`LEDGER-001`: ledger row IDs are append-only.

Commit `0f7fa47` replaced a 62-line block of `docs/JAX_FIDELITY_LEDGER.md`
with one line and 50 rows vanished, while the dashboard and the code kept
citing them; two of the freed IDs were then reused for different findings.
`docs/ledger_ids.txt` is the committed manifest: every ID in it must still
have a row, and a new row must be added to the manifest (a deliberate act,
in the same commit). Removing an ID from the manifest is the only way to
retire a row, and that shows up in review as a deletion from this file.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROW = re.compile(r"^\| ([A-Z][A-Z0-9_-]+) \|", re.M)


def _ids_in_ledger() -> set[str]:
    return set(ROW.findall((ROOT / "docs" / "JAX_FIDELITY_LEDGER.md").read_text()))


def _manifest() -> list[str]:
    return [l.strip() for l in (ROOT / "docs" / "ledger_ids.txt").read_text().splitlines()
            if l.strip()]


def test_every_manifested_row_id_still_has_a_row():
    missing = sorted(set(_manifest()) - _ids_in_ledger())
    assert not missing, f"ledger rows deleted (restore them, or retire the ID in docs/ledger_ids.txt deliberately): {missing}"


def test_every_ledger_row_is_in_the_manifest():
    new = sorted(_ids_in_ledger() - set(_manifest()))
    assert not new, f"new ledger rows not in docs/ledger_ids.txt -- add them: {new}"


def test_the_manifest_is_sorted_and_unique():
    m = _manifest()
    assert m == sorted(set(m))
