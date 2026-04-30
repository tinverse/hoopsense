#!/usr/bin/env python3
"""Fetch external research material into a repo-local raw-source cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import re
import ssl
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DEFAULT_CACHE_ROOT = Path(".local_wiki/raw")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "source"


def _guess_extension(url: str, content_type: str | None) -> str:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix
    if content_type:
        mime = content_type.split(";", 1)[0].strip().lower()
        guessed = mimetypes.guess_extension(mime)
        if guessed:
            if not suffix or suffix[1:].isdigit():
                return guessed
    if suffix:
        return suffix
    return ".bin"


def fetch_url(url: str, insecure: bool = False) -> tuple[bytes, str, str | None]:
    request = Request(
        url,
        headers={
            "User-Agent": "HoopSenseResearchWiki/1.0",
            "Accept": "*/*",
        },
    )
    context = ssl._create_unverified_context() if insecure else None
    with urlopen(request, context=context) as response:
        payload = response.read()
        final_url = response.geturl()
        content_type = response.headers.get("Content-Type")
    return payload, final_url, content_type


def ingest(url: str, slug: str | None, cache_root: Path, insecure: bool = False) -> Path:
    payload, final_url, content_type = fetch_url(url, insecure=insecure)
    source_slug = _slugify(slug or Path(urlparse(final_url).path).stem or url)
    target_dir = cache_root / source_slug
    target_dir.mkdir(parents=True, exist_ok=True)
    ext = _guess_extension(final_url, content_type)
    payload_path = target_dir / f"payload{ext}"
    payload_path.write_bytes(payload)
    metadata = {
        "url": url,
        "final_url": final_url,
        "content_type": content_type,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "payload_path": str(payload_path),
        "size_bytes": len(payload),
    }
    (target_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return target_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Source URL to fetch into the local raw cache.")
    parser.add_argument("--slug", help="Stable local slug for the source directory.")
    parser.add_argument(
        "--cache-root",
        default=str(DEFAULT_CACHE_ROOT),
        help=f"Raw-source cache root. Default: {DEFAULT_CACHE_ROOT}",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification for local fetches in broken host environments.",
    )
    args = parser.parse_args()

    try:
        target_dir = ingest(
            url=args.url,
            slug=args.slug,
            cache_root=Path(args.cache_root),
            insecure=args.insecure,
        )
    except (HTTPError, URLError) as exc:
        print(f"error: failed to fetch {args.url}: {exc}", file=sys.stderr)
        return 1

    print(target_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
