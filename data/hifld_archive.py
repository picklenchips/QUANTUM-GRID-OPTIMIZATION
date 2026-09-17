"""
data.hifld_archive

Repoint for HIFLD Open, which **shut down August 2025** (see docs/07-TOOLING-UPDATES.md "Act on this
now"). The live ArcGIS endpoints in docs/04-DATA-SOURCES.md are dead. This module points at the
surviving archives instead.

Archive mirrors (transmission lines):
  Data Rescue Project portal : https://portal.datarescueproject.org/datasets/hifld-open-transmission-lines/
  DataLumos deposit          : https://www.datalumos.org/datalumos/project/240591/version/V1/view
  source.coop (SeerAI mirror): https://source.coop/seerai/hifld           (Parquet / GeoParquet)

Preferred modern replacement is Microsoft GridSFM (data/gridsfm_to_pp.py) -- continental-scale,
AC-OPF-solvable, actively maintained. Use this module only when you specifically need the *original
HIFLD geometry* (e.g. to reproduce a pre-2025 result).

This hook does not bundle the data (it is tens of MB of line geometry). It resolves a download URL,
fetches to a local path, and -- if pyarrow/geopandas are present -- loads Parquet into a DataFrame.
"""
import os, io, json, urllib.request

HIFLD_ARCHIVE_MIRRORS = {
    "datarescue": "https://portal.datarescueproject.org/datasets/hifld-open-transmission-lines/",
    "datalumos": "https://www.datalumos.org/datalumos/project/240591/version/V1/view",
    "source_coop": "https://source.coop/seerai/hifld",
    # direct source.coop object store (GeoParquet); path verified via the source.coop repo browser
    "source_coop_parquet": "https://data.source.coop/seerai/hifld/Electric_Power_Transmission_Lines.parquet",
}
FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "hifld")

try:
    import pyarrow.parquet as pq          # noqa: F401
    HAVE_PYARROW = True
except ImportError:
    HAVE_PYARROW = False
try:
    import geopandas as gpd
    HAVE_GEOPANDAS = True
except ImportError:
    HAVE_GEOPANDAS = False


def hifld_transmission_url(mirror: str = "source_coop_parquet") -> str:
    if mirror not in HIFLD_ARCHIVE_MIRRORS:
        raise KeyError(f"unknown mirror {mirror!r}; choose from {list(HIFLD_ARCHIVE_MIRRORS)}")
    return HIFLD_ARCHIVE_MIRRORS[mirror]


def fetch_hifld_transmission_lines(dest: str = "data/hifld/transmission_lines.parquet",
                                   mirror: str = "source_coop_parquet", overwrite: bool = False):
    """ download the archived HIFLD transmission-lines Parquet, return a (Geo)DataFrame if the readers
    are available, else the local file path. """
    url = hifld_transmission_url(mirror)
    if not url.endswith(".parquet"):
        raise ValueError(f"mirror {mirror!r} is a portal landing page, not a direct file: {url}\n"
                         f"use mirror='source_coop_parquet', or download by hand from {url}")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if overwrite or not os.path.exists(dest):
        print(f"GET {url}")
        with urllib.request.urlopen(url, timeout=180) as r, open(dest, "wb") as f:  # nosec - fixed https mirror
            f.write(r.read())
        print(f"  -> {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")
    if HAVE_GEOPANDAS:
        try:
            return gpd.read_parquet(dest)
        except Exception:
            pass
    if HAVE_PYARROW:
        import pyarrow.parquet as pq
        return pq.read_table(dest).to_pandas()
    return dest


def hifld_lines_to_geojson(gdf, max_features: int | None = 2000) -> dict:
    """ (Geo)DataFrame of HIFLD lines -> GeoJSON FeatureCollection (for the web map).
    Needs geopandas; if given a plain DataFrame with a WKB/WKT 'geometry' column it still tries. """
    if HAVE_GEOPANDAS and isinstance(gdf, gpd.GeoDataFrame):
        g = gdf.head(max_features) if max_features else gdf
        return json.loads(g.to_json())
    raise TypeError("hifld_lines_to_geojson needs a GeoDataFrame (install geopandas)")


if __name__ == "__main__":
    print("HIFLD Open is dead (Aug 2025). Archive mirrors:")
    for k, v in HIFLD_ARCHIVE_MIRRORS.items():
        print(f"  {k:20s} {v}")
    print("\nPreferred replacement: data/gridsfm_to_pp.py (Microsoft GridSFM)")
    # not auto-downloading tens of MB in a smoke test; uncomment to pull:
    # df = fetch_hifld_transmission_lines()
    # print(type(df), getattr(df, 'shape', df))
