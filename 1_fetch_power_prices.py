"""
gpp_scrape_country_pages_numeric.py

Scrapes GlobalPetrolPrices country pages (pattern: https://www.globalpetrolprices.com/{country}/electricity_prices/)
and produces a numeric EUR/kWh mapping for business electricity prices for 2025.

Outputs:
 - gpp_2025_country_pages_numeric.json  (detailed per-country results + missed list)
 - RETAIL_PRICES_UPDATE.py              (a Python file you can import or paste: USD_TO_EUR + RETAIL_PRICES_UPDATE dict)
 - prints a ready-to-paste RETAIL_PRICES_UPDATE mapping to stdout

Requirements:
    pip install requests beautifulsoup4 pycountry

Usage:
    - Edit USD_TO_EUR at the top to your project's exchange rate (or leave as-is to compute).
    - Run: python gpp_scrape_country_pages_numeric.py
"""

import requests
from bs4 import BeautifulSoup
import pycountry
import re
import time
import json
from urllib.parse import urljoin
from config import USD_TO_EUR, OUT_JSON_POWER_PRICES

# ----------------- Configuration -----------------
BASE = "https://www.globalpetrolprices.com"
INDEX_URL = urljoin(BASE, "/electricity_prices/")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; GPP-Scraper/1.0)"}
SLEEP_SECONDS = 0.15  # polite delay between requests

# Manual corrections for country display names -> ISO2
MANUAL_ISO = {
    "UK": "GB",
    "United Kingdom": "GB",
    "U.K.": "GB",
    "USA": "US",
    "U.S.": "US",
    "United States": "US",
    "Russia": "RU",
    "S. Korea": "KR",
    "South Korea": "KR",
    "N. Maced.": "MK",
    "North Macedonia": "MK",
    "Ivory Coast": "CI",
    "Cabo Verde": "CV",
    "Cape Verde": "CV",
    "Czech Republic": "CZ",
    "Burma": "MM",
    "DR Congo": "CD",
    "Republic of the Congo": "CG",
    "Eswatini": "SZ",
    "Swaziland": "SZ",
    "Vatican": "VA",
}

# Manual slug overrides for country pages when the simple slug doesn't work
MANUAL_SLUG = {
    # Example: "United-States" or "USA" depending on site; add overrides if you encounter missing pages
    # "United States": "USA",
}

# ----------------- Helpers -----------------

def fetch(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text

def slugify_country(name):
    """
    Create a naive slug for the site pattern from a display name.
    The site appears to accept either capitalized or hyphen-delimited names, but some need overrides.
    """
    if name in MANUAL_SLUG:
        return MANUAL_SLUG[name]
    s = name.strip()
    # remove any parenthetical info: "Netherlands (NW)" -> "Netherlands"
    s = re.sub(r"\s*\(.*\)$", "", s).strip()
    s = s.replace(" & ", " and ").replace("&", "and")
    # remove punctuation except spaces/hyphens
    s = re.sub(r"[^\w\s\-]", "", s)
    # spaces -> hyphens
    s = re.sub(r"\s+", "-", s)
    return s

def parse_index_country_anchors(index_html):
    """
    Return a list of (display_name, href) tuples from the index page anchors
    that reference country-specific electricity pages.
    """
    soup = BeautifulSoup(index_html, "html.parser")
    anchors = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Accept both full and relative hrefs that contain the electricity_prices path
        if "/electricity_prices/" in href:
            txt = a.get_text(" ", strip=True)
            if not txt:
                # derive name from href fallback
                parts = href.strip("/").split("/")
                if len(parts) >= 1:
                    txt = parts[0].replace("-", " ").replace("_", " ").strip()
            # filter obviously non-country anchors
            if txt and len(txt) < 60:
                anchors.append((txt, href))
    # dedupe while preserving order
    seen = set()
    uniq = []
    for name, href in anchors:
        key = (name, href)
        if key not in seen:
            seen.add(key)
            uniq.append((name, href))
    return uniq

def extract_business_usd_from_html(html):
    """
    Try several robust patterns to find the business USD/kWh value in a country page.
    Returns a float USD value or None.
    """
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)

    # 1) common phrase: "electricity price for businesses is ... USD 0.xxx"
    m = re.search(r"electricity price for businesses.*?USD\s*([0-9]+\.[0-9]{2,4})", text, flags=re.I)
    if m:
        return float(m.group(1))

    # 2) "business ... USD 0.xxx" (shorter pattern)
    m = re.search(r"business(?:es|'s)?[^\d]{0,40}USD\s*([0-9]+\.[0-9]{2,4})", text, flags=re.I)
    if m:
        return float(m.group(1))

    # 3) pattern "Business ... 0.xxx USD" or "Business ... USD 0.xxx" - find nearby numbers
    m = re.search(r"Business[^\d]{0,40}([0-9]+\.[0-9]{2,4}).{0,40}USD", text, flags=re.I)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass

    # 4) find all USD numbers; heuristic: if two numbers found, assume first is residential, second is business
    usd_nums = re.findall(r"USD\s*([0-9]+\.[0-9]{2,4})", text, flags=re.I)
    if usd_nums:
        try:
            if len(usd_nums) >= 2:
                return float(usd_nums[1])
            return float(usd_nums[0])
        except Exception:
            pass

    # 5) fallback: find numbers close to the word 'business' (within 50 chars)
    m = re.search(r"business.{0,50}?([0-9]+\.[0-9]{2,4})", text, flags=re.I)
    if m:
        return float(m.group(1))

    return None

def name_to_iso2(display_name):
    """
    Map display name to ISO2 using manual overrides then pycountry fuzzy search.
    Returns a 2-letter ISO or None.
    """
    if display_name in MANUAL_ISO:
        return MANUAL_ISO[display_name]
    # strip qualifiers
    clean = re.sub(r"\s*\(.*\)$", "", display_name).strip()
    try:
        country = pycountry.countries.search_fuzzy(clean)[0]
        return country.alpha_2
    except Exception:
        # try a few heuristics
        alt = clean.replace(".", "").replace(" and ", " & ").replace("-", " ")
        try:
            country = pycountry.countries.search_fuzzy(alt)[0]
            return country.alpha_2
        except Exception:
            return None

# ----------------- Main scraping routine -----------------

def scrape_all_country_pages():
    print("Fetching index page:", INDEX_URL)
    idx_html = fetch(INDEX_URL)
    anchors = parse_index_country_anchors(idx_html)
    print(f"Found {len(anchors)} anchors on index (sample: {anchors[:6]})")

    results = {}   # iso2 -> {usd, eur, url, name}
    missed = []    # list of dicts with details for failed/parsing pages

    for i, (anchor_text, href) in enumerate(anchors, start=1):
        # normalize display name (use href if anchor text is odd)
        display_name = anchor_text.strip() or href
        # choose slug: prefer href-derived slug if href is relative like '/Greece/electricity_prices/'
        slug_candidate = None
        if href.startswith("/"):
            parts = href.strip("/").split("/")
            if parts and parts[0].lower() != "electricity_prices":
                slug_candidate = parts[0]
        # fallback slugify
        if not slug_candidate:
            slug_candidate = slugify_country(display_name)

        tried_slugs = [slug_candidate]
        # also try a lowercase version and a capitalized hyphen version
        tried_slugs.extend([
            slug_candidate.lower(),
            slug_candidate.replace(" ", "-"),
            slug_candidate.replace("-", " ").replace(" ", "-"),
        ])
        # dedupe slugs
        tried_slugs = list(dict.fromkeys(tried_slugs))

        usd_val = None
        used_url = None
        for slug in tried_slugs:
            url = f"{BASE}/{slug}/electricity_prices/"
            try:
                time.sleep(SLEEP_SECONDS)
                html = fetch(url)
                usd_val = extract_business_usd_from_html(html)
                used_url = url
                if usd_val is not None:
                    break
            except requests.HTTPError as e:
                # page may 404 for some slugs; continue trying others
                used_url = url
                continue
            except Exception as e:
                # log and break if it's a serious unexpected error
                used_url = url
                break

        iso = name_to_iso2(display_name)
        if usd_val is None:
            # record missed page with info for manual inspection
            missed.append({
                "name": display_name,
                "iso": iso,
                "tried_slugs": tried_slugs,
                "last_tried_url": used_url,
            })
            print(f"[{i}/{len(anchors)}] MISS: {display_name} (ISO: {iso}) tried {len(tried_slugs)} slugs.")
            continue

        # compute numeric EUR (rounded)
        try:
            eur_val = round(float(usd_val) * USD_TO_EUR, 3)
        except Exception:
            missed.append({
                "name": display_name,
                "iso": iso,
                "error": "invalid usd parse",
                "usd_raw": usd_val,
                "url": used_url,
            })
            print(f"[{i}] PARSE ERROR: {display_name} USD parse invalid: {usd_val}")
            continue

        if iso:
            results[iso] = {"usd": float(usd_val), "eur": eur_val, "url": used_url, "name": display_name}
            print(f"[{i}/{len(anchors)}] OK: {iso} ({display_name}) USD {usd_val} -> EUR {eur_val}")
        else:
            # store under display name so you can later map it manually
            results[display_name] = {"usd": float(usd_val), "eur": eur_val, "url": used_url, "name": display_name}
            print(f"[{i}/{len(anchors)}] OK (NO ISO): {display_name} USD {usd_val} -> EUR {eur_val} (check name->ISO)")

    return results, missed

# ----------------- Runner / Output -----------------

def write_outputs(results, missed, usd_to_eur):
    # produce simple iso -> eur numeric mapping for ready paste
    numeric_map = {}
    for k, v in results.items():
        # prefer ISO-2 keys
        if isinstance(k, str) and len(k) == 2 and k.isalpha():
            numeric_map[k] = v["eur"]
        else:
            # attempt to use v['name'] -> iso mapping (re-run name_to_iso2), else keep display name key
            iso_guess = name_to_iso2(v.get("name", k))
            if iso_guess:
                numeric_map[iso_guess] = v["eur"]
            else:
                numeric_map[k] = v["eur"]

    # Write JSON detail file
    with open(OUT_JSON_POWER_PRICES, "w", encoding="utf8") as f:
        json.dump({"results": results, "missed": missed, "numeric_map": numeric_map, 
                   "USD_TO_EUR_used": usd_to_eur}, f, indent=2, ensure_ascii=False)

    # Print ready-to-paste mapping to stdout
    print("\n# --- RETAIL_PRICES_UPDATE (paste into your code) ---")
    print("RETAIL_PRICES_UPDATE = {")
    for iso, eur in sorted(numeric_map.items()):
        print(f"    '{iso}': {eur},")
    print("}")
    print(f"\nWrote detailed JSON -> {OUT_JSON_POWER_PRICES}")

if __name__ == "__main__":
    results, missed = scrape_all_country_pages()
    write_outputs(results, missed, USD_TO_EUR)
    print("\nSummary:")
    print(f"  Parsed countries: {len(results)}")
    print(f"  Missed / requires manual check: {len(missed)}")
    if missed:
        print("  Missed examples (first 10):")
        for m in missed[:10]:
            print("   -", m)