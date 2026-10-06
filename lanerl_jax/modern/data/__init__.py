"""Pinned 26.19 patch data (``PATCH_DIR``), recorded-game oracles (``ORACLE_DIR``) and their loaders/builders.

The ``build_*`` modules regenerate the ``PATCH_DIR`` tables from the client research dumps in ``RESEARCH``.
"""
import argparse
import hashlib
import json
from pathlib import Path

PATCH = "26.19"
CLIENT_BUILD = "16.19.8230722"
PATCH_DIR = Path(__file__).resolve().parent / PATCH
ORACLE_DIR = Path(__file__).resolve().parent / "oracle"
RESEARCH = Path("/mnt/nfs/shared/modern-world-map-research")


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def walk_dicts(o):
    """Every dict nested in the JSON value ``o``, parents before children."""
    if isinstance(o, dict):
        yield o
        o = list(o.values())
    if isinstance(o, list):
        for v in o:
            yield from walk_dicts(v)


def build_main(doc, build, out, report, **dump):
    """CLI of the ``build_*`` tools: ``[--research DIR] [--out PATH]``, writing ``build(research)`` as JSON."""
    ap = argparse.ArgumentParser(description=doc.splitlines()[0])
    ap.add_argument("--research", type=Path, default=RESEARCH)
    ap.add_argument("--out", type=Path, default=out)
    args = ap.parse_args()
    payload = build(args.research)
    args.out.write_text(json.dumps(payload, **dump) + "\n")
    print(report(payload, args.out))
