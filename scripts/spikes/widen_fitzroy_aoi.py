"""Widen the Fitzroy test data from the narrow lower-mainstem AOI (data/
fitzroy_kimberley_*) to the full Fitzroy River basin.

Not part of the package and not run by pytest -- a one-off data-preparation
script, like scripts/spikes/riverscape_phase0_spike.py. Requires:
  1. Network access to one BOM ArcGIS FeatureServer query (the basin
     boundary polygon; a few MB).
  2. A local copy of the national AHGF Surface Network file geodatabase
     (SH_Network.gdb, ~6.5 GB), which exists only on this development
     machine and is never added to the repo. Pass its path with --gdb if it
     is not at the default location.

Run:  python scripts/spikes/widen_fitzroy_aoi.py
Out:  data/fitzroy_basin_aoi.geojson, data/fitzroy_basin_drainage.gpkg
"""
from __future__ import annotations

import argparse
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import geopandas as gpd
import shapely

REPO = Path(__file__).resolve().parents[2]
DEFAULT_GDB = Path("D:/RLH/5.6/data_local/raw/extracted/SH_Network_GDB/SH_Network.gdb")
BASIN_QUERY_URL = (
    "https://hosting.wsapi.cloud.bom.gov.au/arcgis/rest/services/ahgf/"
    "Geofabric_V3x_All_Products/FeatureServer/13/query"
)
TARGET_CRS = "EPSG:3577"
AOI_OUT = REPO / "data" / "fitzroy_basin_aoi.geojson"
DRAINAGE_OUT = REPO / "data" / "fitzroy_basin_drainage.gpkg"


def fetch_basin_boundary(*, river_region_name: str) -> gpd.GeoDataFrame:
    """Query the BOM RiverRegion layer for one named basin polygon."""
    where = urllib.parse.quote(f"rivregname='{river_region_name}'")
    url = (
        f"{BASIN_QUERY_URL}?where={where}&outFields=*&f=geojson"
        f"&outSR={TARGET_CRS.split(':')[1]}"
    )
    raw = urllib.request.urlopen(url, timeout=120).read()
    gdf = gpd.read_file(raw.decode("utf-8"))
    if len(gdf) != 1:
        raise RuntimeError(
            f"expected exactly one RiverRegion feature for {river_region_name!r}, "
            f"got {len(gdf)}"
        )
    if gdf.crs is None or gdf.crs.to_string() != TARGET_CRS:
        gdf = gdf.set_crs(TARGET_CRS, allow_override=True)
    return gdf


def clip_network_to_basin(gdb_path: Path, basin: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Bbox-read AHGFNetworkStream from the local GDB, then exact-clip to the basin."""
    basin_native = basin.to_crs("EPSG:4283")
    bbox = tuple(basin_native.total_bounds)
    streams = gpd.read_file(str(gdb_path), layer="AHGFNetworkStream", bbox=bbox, engine="pyogrio")
    if streams.crs is None:
        streams = streams.set_crs("EPSG:4283", allow_override=True)
    streams = streams.to_crs(TARGET_CRS)
    basin_geom = basin.geometry.iloc[0]
    # Prepare the basin geometry before the vectorized intersects test below.
    # Deviation from the brief's verbatim script: without this, .intersects()
    # against the unprepared ~65k-vertex basin polygon was measured at ~27ms
    # per candidate reach (500-row batch took 13.55s), i.e. tens of minutes
    # for the full ~57k-row bbox superset -- not the "<2s" the brief's prior
    # interactive run reported. shapely.prepare() builds an index once so
    # each subsequent intersects() call against basin_geom is fast; it does
    # not change which reaches end up in the output, only how fast the
    # check runs.
    shapely.prepare(basin_geom)
    clipped = streams[streams.intersects(basin_geom)].copy()
    if clipped.empty:
        raise RuntimeError("basin clip produced zero reaches -- check the GDB path and basin polygon")
    return clipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gdb", type=Path, default=DEFAULT_GDB, help="path to SH_Network.gdb")
    parser.add_argument(
        "--river-region", default="FITZROY RIVER (WA)", help="BOM RiverRegion.rivregname value"
    )
    args = parser.parse_args()

    if not args.gdb.exists():
        print(f"GDB not found at {args.gdb} -- pass --gdb", file=sys.stderr)
        return 1

    basin = fetch_basin_boundary(river_region_name=args.river_region)
    area_km2 = float(basin.geometry.area.sum()) / 1e6
    print(f"basin polygon: {area_km2:,.1f} km2, bounds {tuple(basin.total_bounds)}")

    drainage = clip_network_to_basin(args.gdb, basin)
    print(f"clipped drainage: {len(drainage)} reaches")
    upstr = drainage["UpstrDArea"].astype(float)
    print(
        "UpstrDArea (m2) p1/p5/p25/p50/p75/p95/p99:",
        [round(v, 1) for v in upstr.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])],
    )

    AOI_OUT.parent.mkdir(parents=True, exist_ok=True)
    basin[["geometry"]].to_file(AOI_OUT, driver="GeoJSON")
    drainage.to_file(DRAINAGE_OUT, driver="GPKG")
    print(f"wrote {AOI_OUT}")
    print(f"wrote {DRAINAGE_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
