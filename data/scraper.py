#!/usr/bin/env python3
"""
Async logo scraper for the ANN logo-classifier project.

Reads data/sample.csv (built by: python3 data/kaggle.py --build-sample),
fetches each company logo via og:image or Google favicon fallback,
letterboxes to 224×224 RGB PNG, and saves to data/logos/{sector_slug}/{domain}.png.

Usage (from project root):
    python3 data/scraper.py
"""

import asyncio
import csv
import os
import re
import sys
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
from bs4 import BeautifulSoup
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent
SAMPLE_CSV = DATA_DIR / "sample.csv"
LOG_CSV = DATA_DIR / "scrape_log.csv"
LOGOS_DIR = DATA_DIR / "logos"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONCURRENCY = 50
HEAD_CHUNK_LIMIT = 200_000  # max bytes to buffer when scanning for </head>

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sector_slug(sector: str) -> str:
    """'Sports, Recreation & Travel' → 'sports_recreation_travel'"""
    return re.sub(r"[^a-z0-9]+", "_", sector.lower()).strip("_")


def letterbox(img: Image.Image, size: int = 224) -> Image.Image:
    """Scale down preserving aspect ratio, pad remainder with white."""
    img = img.convert("RGB")
    img.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    offset = ((size - img.width) // 2, (size - img.height) // 2)
    canvas.paste(img, offset)
    return canvas


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

async def fetch_og_image(session: aiohttp.ClientSession, domain: str) -> bytes | None:
    """
    GET https://{domain}, stream until </head>, parse og:image meta tag,
    then download the image URL. Returns raw image bytes or None.
    """
    page_url = f"https://{domain}"
    base_url = page_url

    try:
        async with session.get(
            page_url,
            timeout=aiohttp.ClientTimeout(total=5),
            headers=BROWSER_HEADERS,
            allow_redirects=True,
            ssl=False,
        ) as resp:
            if resp.status >= 400:
                return None
            base_url = str(resp.url)

            buffer = b""
            async for chunk in resp.content.iter_chunked(8192):
                buffer += chunk
                if b"</head>" in buffer.lower():
                    break
                if len(buffer) >= HEAD_CHUNK_LIMIT:
                    break
    except Exception:
        return None

    # Parse og:image from buffered head HTML
    try:
        soup = BeautifulSoup(buffer, "lxml")
    except Exception:
        try:
            soup = BeautifulSoup(buffer, "html.parser")
        except Exception:
            return None

    meta = soup.find("meta", attrs={"property": "og:image"}) or soup.find(
        "meta", attrs={"name": "og:image"}
    )
    if not meta:
        return None

    img_url = (meta.get("content") or "").strip()
    if not img_url:
        return None

    # Resolve relative URLs
    if img_url.startswith("//"):
        img_url = "https:" + img_url
    elif img_url.startswith("/"):
        parsed = urlparse(base_url)
        img_url = f"{parsed.scheme}://{parsed.netloc}{img_url}"
    elif not img_url.startswith(("http://", "https://")):
        img_url = f"https://{domain}/{img_url}"

    # Download the image
    try:
        async with session.get(
            img_url,
            timeout=aiohttp.ClientTimeout(total=10),
            headers=BROWSER_HEADERS,
            ssl=False,
        ) as resp:
            if "image/" not in resp.headers.get("content-type", ""):
                return None
            return await resp.read()
    except Exception:
        return None


async def fetch_favicon(session: aiohttp.ClientSession, domain: str) -> bytes | None:
    """Google favicon service fallback — returns image bytes or None."""
    url = f"https://www.google.com/s2/favicons?domain={domain}&sz=256"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                data = await resp.read()
                return data or None
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

async def process_company(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    row: dict,
    log_file,
    log_writer: csv.writer,
    log_lock: asyncio.Lock,
    counter: dict,
) -> None:
    domain = str(row["domain"]).strip()
    name = str(row.get("name", "")).strip()
    sector = str(row["sector"]).strip()

    slug = sector_slug(sector)
    out_dir = LOGOS_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{domain}.png"

    async with sem:
        img_bytes: bytes | None = None
        source = ""
        status = "not_found"
        filepath = ""

        # Step A: og:image
        img_bytes = await fetch_og_image(session, domain)
        if img_bytes:
            source = "og_image"

        # Step B: Google favicon fallback
        if not img_bytes:
            img_bytes = await fetch_favicon(session, domain)
            if img_bytes:
                source = "favicon"

        # Step C: process and save
        if img_bytes:
            try:
                img = Image.open(BytesIO(img_bytes))
                img = letterbox(img)
                img.save(out_path, "PNG")
                status = "ok"
                filepath = str(out_path)
            except Exception:
                status = "error"
                source = ""

        async with log_lock:
            log_writer.writerow([domain, name, sector, status, source, filepath])
            counter[status] = counter.get(status, 0) + 1
            counter["done"] += 1
            done = counter["done"]

            if done % 1000 == 0:
                log_file.flush()
                total = counter["grand_total"]
                ok = counter.get("ok", 0)
                nf = counter.get("not_found", 0)
                err = counter.get("error", 0)
                print(f"[{done}/{total}] ok={ok} not_found={nf} errors={err}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run(
    rows: list[dict],
    log_is_new: bool,
    already_done: int,
    grand_total: int,
) -> None:
    counter: dict = {
        "done": already_done,
        "grand_total": grand_total,
    }
    log_lock = asyncio.Lock()
    sem = asyncio.Semaphore(CONCURRENCY)
    connector = aiohttp.TCPConnector(limit=CONCURRENCY, ttl_dns_cache=300)

    with open(LOG_CSV, "a", newline="") as log_file:
        log_writer = csv.writer(log_file)
        if log_is_new:
            log_writer.writerow(["domain", "name", "sector", "status", "source", "filepath"])

        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                process_company(session, sem, row, log_file, log_writer, log_lock, counter)
                for row in rows
            ]
            await asyncio.gather(*tasks)

        log_file.flush()

    this_run = counter["done"] - already_done
    ok = counter.get("ok", 0)
    nf = counter.get("not_found", 0)
    err = counter.get("error", 0)
    print(f"\nRun complete. Processed {this_run:,} companies — ok={ok} not_found={nf} errors={err}")


# ---------------------------------------------------------------------------
# Sector breakdown
# ---------------------------------------------------------------------------

def print_sector_breakdown() -> None:
    if not LOG_CSV.exists():
        return
    sector_ok: dict[str, int] = {}
    with open(LOG_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "ok":
                s = row.get("sector", "Unknown")
                sector_ok[s] = sector_ok.get(s, 0) + 1

    if not sector_ok:
        print("No successfully downloaded logos.")
        return

    print(f"\n{'SECTOR':<40} {'OK':>8}")
    print("=" * 50)
    for sector, count in sorted(sector_ok.items(), key=lambda x: -x[1]):
        print(f"{sector:<40} {count:>8,}")
    print("=" * 50)
    print(f"{'TOTAL':<40} {sum(sector_ok.values()):>8,}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas is required. Run: pip install pandas")
        sys.exit(1)

    if not SAMPLE_CSV.exists():
        print(f"ERROR: {SAMPLE_CSV} not found.")
        print("Build it first with:  python3 data/kaggle.py --build-sample")
        sys.exit(1)

    print(f"Loading {SAMPLE_CSV} ...")
    df = pd.read_csv(SAMPLE_CSV)
    print(f"Sample: {len(df):,} companies across {df['sector'].nunique()} sectors")

    # Load existing log — skip domains already marked ok or not_found
    done_domains: set[str] = set()
    log_is_new = not LOG_CSV.exists()
    if not log_is_new:
        with open(LOG_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") in ("ok", "not_found"):
                    done_domains.add(row["domain"])
        print(f"Resuming: {len(done_domains):,} domains already done (ok/not_found)")

    rows = [r for r in df.to_dict("records") if str(r["domain"]).strip() not in done_domains]
    already_done = len(done_domains)
    grand_total = len(df)

    print(f"To process: {len(rows):,} | Already done: {already_done:,} | Grand total: {grand_total:,}")

    LOGOS_DIR.mkdir(parents=True, exist_ok=True)

    asyncio.run(run(rows, log_is_new, already_done, grand_total))
    print_sector_breakdown()


if __name__ == "__main__":
    main()
