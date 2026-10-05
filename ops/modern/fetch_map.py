"""TOOL: extract only requested Map11 assets from a pinned Riot manifest.

No game installation, launcher execution or account required. Uses the optional
riotmanifest==2.10.2 / league-tools==1.2.1 tooling in an isolated environment.
Call through ops/login_capped.sh; no training environment dependencies change.
The manifest URL/build are explicit: never resolve 'latest' during reproduction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

WAD = "DATA/FINAL/Maps/Shipping/Map11.wad.client"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, help="pinned Riot URL or local RMAN path")
    p.add_argument("--build", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--asset", action="append", required=True, help="internal WAD path; repeatable")
    a = p.parse_args()
    if len(set(a.asset)) != len(a.asset):
        p.error("duplicate assets")
    if any(Path(x).is_absolute() or ".." in Path(x).parts for x in a.asset):
        p.error("asset paths must be relative and may not contain ..")
    a.out.mkdir(parents=True, exist_ok=False)
    if a.manifest.startswith("https://"):
        with urllib.request.urlopen(a.manifest, timeout=60) as r:
            raw = r.read()
    else:
        raw = Path(a.manifest).read_bytes()
    manifest_path = a.out / "source.manifest"
    manifest_path.write_bytes(raw)
    from riotmanifest import PatcherManifest, WADExtractor
    manifest = PatcherManifest(str(manifest_path), path=str(a.out / "cache"),
                               concurrency_limit=2, max_retries=2)
    report = dict(build=a.build, manifest_source=a.manifest,
                  manifest_sha256=hashlib.sha256(raw).hexdigest(), wad=WAD,
                  retrieved_at=datetime.now(timezone.utc).isoformat(),
                  tools={k: version(k) for k in ("riotmanifest", "league-tools")},
                  assets={})
    # The library can return None for individual failures: preserve successes
    # for diagnosis but return a failure status, never claim a complete export.
    with WADExtractor(manifest, retry_limit=2, prefetch_chunk_concurrency=2) as extractor:
        extracted = extractor.extract_files({WAD: a.asset})
        for asset in a.asset:
            data = extracted.get(WAD, {}).get(asset)
            if data is None:
                report["assets"][asset] = dict(status="missing-or-failed")
                continue
            target = a.out / "assets" / asset
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            report["assets"][asset] = dict(status="extracted", bytes=len(data),
                                           sha256=hashlib.sha256(data).hexdigest())
        report["cache_stats"] = extractor.cache_stats()
    (a.out / "extraction.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    if any(x["status"] != "extracted" for x in report["assets"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
