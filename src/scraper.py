"""
Melbourne Real Estate Scraper & Dataset Compiler
Focuses on Burwood East (3151) and surrounding Eastern Suburbs:
Burwood, Mount Waverley, Glen Waverley, Box Hill, Box Hill South,
Blackburn, Blackburn South, Forest Hill, Vermont, Vermont South.

Collects property attributes, exact geospatial coordinates, and enriches
each record with local school zones, shopping hubs, public transport, and CBD distances.
"""

import os
import re
import time
import json
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup
from enrich_poi import enrich_property_with_pois, SUBURB_CENTROIDS

# Target Melbourne Eastern Suburbs
TARGET_SUBURBS = [
    {"slug": "burwood-east-vic-3151", "suburb": "Burwood East", "postcode": 3151},
    {"slug": "burwood-vic-3125", "suburb": "Burwood", "postcode": 3125},
    {"slug": "mount-waverley-vic-3149", "suburb": "Mount Waverley", "postcode": 3149},
    {"slug": "glen-waverley-vic-3150", "suburb": "Glen Waverley", "postcode": 3150},
    {"slug": "box-hill-vic-3128", "suburb": "Box Hill", "postcode": 3128},
    {"slug": "box-hill-south-vic-3128", "suburb": "Box Hill South", "postcode": 3128},
    {"slug": "blackburn-vic-3130", "suburb": "Blackburn", "postcode": 3130},
    {"slug": "blackburn-south-vic-3130", "suburb": "Blackburn South", "postcode": 3130},
    {"slug": "forest-hill-vic-3131", "suburb": "Forest Hill", "postcode": 3131},
    {"slug": "vermont-vic-3133", "suburb": "Vermont", "postcode": 3133},
    {"slug": "vermont-south-vic-3133", "suburb": "Vermont South", "postcode": 3133}
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/"
}


def parse_price(price_text: str) -> float | None:
    """
    Parses complex Australian real estate price expressions into clean numeric values ($AUD).
    Examples:
      - '$1,250,000' -> 1250000.0
      - '$1,200,000 - $1,300,000' -> 1250000.0 (midpoint)
      - '$1.25m' -> 1250000.0
      - '$850k' -> 850000.0
    """
    if not price_text or not isinstance(price_text, str):
        return None
    
    clean = price_text.replace(",", "").strip()
    
    # Range: e.g. $1200000 - $1300000
    range_match = re.findall(r"\$?(\d+(?:\.\d+)?)\s*(?:m|k|million)?", clean, re.IGNORECASE)
    
    # Try finding full dollar figures: e.g. $1,450,000
    full_dollars = re.findall(r"\$\s*(\d{6,8})", clean)
    if full_dollars:
        nums = [float(x) for x in full_dollars]
        return sum(nums) / len(nums)
    
    # Try finding millions: e.g. $1.25m or $1.2m - $1.3m
    m_matches = re.findall(r"\$?(\d+(?:\.\d+)?)\s*m\b", clean, re.IGNORECASE)
    if m_matches:
        nums = [float(x) * 1_000_000 for x in m_matches]
        return sum(nums) / len(nums)
        
    # Try finding thousands: e.g. $850k
    k_matches = re.findall(r"\$?(\d+(?:\.\d+)?)\s*k\b", clean, re.IGNORECASE)
    if k_matches:
        nums = [float(x) * 1_000 for x in k_matches]
        return sum(nums) / len(nums)
        
    return None


def parse_property_card(article_soup, json_ld_map: dict, default_suburb: str, default_postcode: int) -> dict | None:
    """
    Extracts structured features from a property listing card and links with its JSON-LD coordinate record.
    """
    text = article_soup.get_text(separator=" | ", strip=True)
    tokens = [t.strip() for t in text.split("|") if t.strip()]
    
    # Address extraction
    link_elem = article_soup.find("a", href=re.compile(r"/homes/"))
    href = link_elem.get("href") if link_elem else ""
    full_url = f"https://www.homely.com.au{href}" if href.startswith("/") else href
    
    # Extract address and match with JSON-LD geo coordinates
    geo_data = json_ld_map.get(full_url, {})
    lat = geo_data.get("lat")
    lon = geo_data.get("lon")
    address = geo_data.get("name", "")
    
    # If address wasn't in json-ld, find token matching postcode
    if not address:
        for t in tokens:
            if str(default_postcode) in t:
                address = t
                break
    if not address and len(tokens) > 2:
        address = tokens[2]

    # Price extraction
    price = None
    price_raw = ""
    for t in tokens:
        if "$" in t:
            p_val = parse_price(t)
            if p_val and p_val >= 200_000:  # Valid purchase price in Melbourne
                price = p_val
                price_raw = t
                break

    # Bed, Bath, Car, Size parsing
    # Look for numeric badges in text: e.g. ["4", "2", "2", "650m²"]
    beds = None
    baths = None
    cars = None
    land_size = None
    prop_type = "House"  # default
    
    # Find property type
    for p_type in ["House", "Townhouse", "Unit", "Apartment", "Villa", "Studio"]:
        if p_type.lower() in text.lower():
            prop_type = p_type
            break
            
    # Find land size: e.g. 650m² or 650 sqm
    land_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:m²|m2|sqm)", text, re.IGNORECASE)
    if land_match:
        land_size = float(land_match.group(1))

    # Sequential single digits often represent beds, baths, cars in card badges
    single_digits = []
    for t in tokens:
        if re.match(r"^\d{1,2}$", t):
            single_digits.append(int(t))
            
    if len(single_digits) >= 1:
        beds = single_digits[0]
    if len(single_digits) >= 2:
        baths = single_digits[1]
    if len(single_digits) >= 3:
        cars = single_digits[2]

    # Enrich with spatial POI metrics
    poi_metrics = enrich_property_with_pois(lat, lon, default_suburb)

    record = {
        "address": address,
        "suburb": default_suburb,
        "postcode": default_postcode,
        "property_type": prop_type,
        "bedrooms": beds,
        "bathrooms": baths,
        "parking_spaces": cars,
        "land_size_sqm": land_size,
        "price_aud": price,
        "price_raw": price_raw,
        "url": full_url,
        **poi_metrics
    }
    return record


def scrape_suburb_listings(suburb_info: dict, max_pages: int = 5) -> list[dict]:
    """
    Crawls listing pages for a given suburb, extracts JSON-LD and HTML card details.
    """
    slug = suburb_info["slug"]
    suburb = suburb_info["suburb"]
    postcode = suburb_info["postcode"]
    records = []
    
    for page in range(1, max_pages + 1):
        url = f"https://www.homely.com.au/buy/{slug}?page={page}"
        print(f"Scraping [{suburb}] Page {page}/{max_pages} -> {url}")
        
        try:
            resp = requests.get(url, headers=HEADERS, timeout=12)
            if resp.status_code != 200:
                print(f"  Status {resp.status_code}, skipping...")
                break
                
            soup = BeautifulSoup(resp.text, "html.parser")
            
            # 1. Parse JSON-LD coordinates map
            json_ld_map = {}
            json_scripts = soup.find_all("script", type="application/ld+json")
            for js in json_scripts:
                try:
                    data = json.loads(js.string)
                    if isinstance(data, dict) and data.get("@type") == "Residence":
                        p_url = data.get("url", "")
                        geo = data.get("geo", {})
                        if p_url and geo:
                            json_ld_map[p_url] = {
                                "name": data.get("name", ""),
                                "lat": geo.get("latitude"),
                                "lon": geo.get("longitude")
                            }
                except Exception:
                    continue
            
            # 2. Parse property articles
            articles = soup.find_all("article")
            page_records = 0
            for art in articles:
                rec = parse_property_card(art, json_ld_map, suburb, postcode)
                if rec and rec["address"]:
                    records.append(rec)
                    page_records += 1
            
            print(f"  Extracted {page_records} properties from page {page}")
            
            # Respectful delay between requests
            time.sleep(random.uniform(1.2, 2.5))
            
            if page_records == 0:
                # No more listings on subsequent pages
                break
                
        except Exception as e:
            print(f"  Error on page {page}: {e}")
            break
            
    return records


def compile_and_enrich_dataset(output_raw_path: str = "data/raw/melbourne_housing_raw.csv") -> pd.DataFrame:
    """
    Executes full web scraping cycle across all target Melbourne Eastern suburbs.
    Enriches with comprehensive POI attributes and guarantees a complete dataset.
    """
    all_properties = []
    print("=" * 70)
    print("STARTING MELBOURNE EASTERN SUBURBS REAL ESTATE WEB SCRAPER")
    print(f"Target Suburbs: {len(TARGET_SUBURBS)} centered around Burwood East (3151)")
    print("=" * 70)
    
    for sub in TARGET_SUBURBS:
        sub_props = scrape_suburb_listings(sub, max_pages=3)
        all_properties.extend(sub_props)
        print(f"Total collected so far: {len(all_properties)}")
    
    df_scraped = pd.DataFrame(all_properties)
    print(f"\nWeb scraping completed. Raw records scraped: {len(df_scraped)}")
    
    # Save raw scraped data
    os.makedirs(os.path.dirname(output_raw_path), exist_ok=True)
    df_scraped.to_csv(output_raw_path, index=False)
    print(f"Saved raw scraped data to {output_raw_path}")
    
    return df_scraped


if __name__ == "__main__":
    df = compile_and_enrich_dataset()
    print("Sample preview:")
    print(df.head(3))
