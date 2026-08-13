# Data & mapping sources

> Condensed from original planning doc's "Data & Resources" section. Reference catalog — check here before searching for grid dataset or map API from scratch. **Stops at ~mid-2024** — for updates/replacements since, see [07-TOOLING-UPDATES.md](07-TOOLING-UPDATES.md) (HIFLD Open is dead, read that first).

## Abstraction level (applies to all sources below)

Don't model individual power towers, solar panels, or turbines. Group power stations into single substation node; represent transmission lines as single edges; group same-voltage networks as microgrid. Level [01-ARCHITECTURE.md](01-ARCHITECTURE.md)'s node/edge model targets.

## Mapping / visualization

| Source | What | Note |
|---|---|---|
| [OpenStreetMap](https://wiki.openstreetmap.org/wiki/Key:power) `power` tag | Free, open grid infrastructure tags | Primary source. [Geofabrik extracts](https://www.geofabrik.de/data/energy-networks.html), [download docs](https://wiki.openstreetmap.org/wiki/Downloading_data) |
| [OpenInfraMap](https://openinframap.org/) | Renders OSM power/telecom/water/petroleum/microwave infra; shows substation names + max voltage | Open source ([github](https://github.com/openinframap/)) — candidate to fork rather than rebuild |
| [OpenGridMap](https://opengridmap.org/) | Same as OpenInfraMap, grid-only, Mapbox instead of OpenMapTiles | |
| [osm2pgsql](https://osm2pgsql.org/) | OSM → PostgreSQL ingestion | Feeds PostGIS layer in [01-ARCHITECTURE.md](01-ARCHITECTURE.md) |
| [Leaflet](https://leafletjs.com/) | Mobile-friendly map rendering, OSM-native | Used by [overpass-turbo](https://overpass-turbo.eu/) |
| [ArcGIS](https://www.esri.com/en-us/arcgis/geospatial-platform/overview) | Most capable GIS platform, any dataset as layer | 21-day trial only — overkill vs. OSM/Leaflet |
| [Google Maps JS API](https://developers.google.com/maps/documentation/javascript/get-api-key) | Roads/cities backdrop | Overkill |
| NREL [ARIES DERVE Phase 1](https://www.nrel.gov/docs/fy23osti/84079.pdf) (2023) + [climate projection layers](https://climrr.anl.gov/climateprojections) | Distributed energy resource visual emulation + climate overlay | Climate-impact visualization angle in [00-VISION.md](00-VISION.md) |

## Datasets

**Caveat**: substation pricing data readily available; full transmission-line topology findable (ArcGIS-hosted, see below); low-level consumer-side infrastructure often private or legally protected — may need synthetic/estimated data.

| Dataset | Scope | Link |
|---|---|---|
| U.S. Energy Atlas | All energy infra incl. oil/gas | [explorer](https://atlas.eia.gov/apps/eia::all-energy-infrastructure-and-resources/explore) |
| All U.S. power lines | Transmission | [dataset](https://atlas.eia.gov/datasets/bd24d1a282c54428b024988d32578e59_0/explore) |
| All U.S. power plants | Generation — **dead, HIFLD Open shut down Aug 2025**, see [07-TOOLING-UPDATES.md](07-TOOLING-UPDATES.md) for archive mirrors | [dataset](https://hifld-geoplatform.hub.arcgis.com/datasets/9dd630378fcf439999094a56c352670d_0/explore) |
| HIFLD transmission lines | Transmission, w/ API — **dead, HIFLD Open shut down Aug 2025**, see [07-TOOLING-UPDATES.md](07-TOOLING-UPDATES.md) for archive mirrors | [dataset](https://hifld-geoplatform.hub.arcgis.com/datasets/geoplatform::transmission-lines/about) · [API](https://hifld-geoplatform.hub.arcgis.com/datasets/geoplatform::transmission-lines/api) |
| HIFLD open catalog | Everything else background — **dead, HIFLD Open shut down Aug 2025**, see [07-TOOLING-UPDATES.md](07-TOOLING-UPDATES.md) for archive mirrors | [index](https://hifld-geoplatform.hub.arcgis.com/pages/hifld-open) |
| Microsoft GridSFM_US_power_grid | Transmission, continental scale (48 states + 6 multi-state regions, up to ~21,697-bus full Eastern Interconnection), AC-OPF-solvable | [dataset](https://huggingface.co/datasets/microsoft/GridSFM_US_power_grid) · [blog](https://www.microsoft.com/en-us/research/blog/building-realistic-electric-transmission-grid-dataset-at-scale-a-pipeline-from-open-dataset/) — HIFLD replacement, built same open-data way (OSM+EIA+Census); wired into `data/gridsfm_to_pp.py` |
| transnet-models | Node/edge CSVs, pre-built | [github](https://github.com/OpenGridMap/transnet-models) — already wired into `data/transnet_to_pp.py` |
| transnet Germany nodes | Corrected node CSV | [csv](https://github.com/OpenGridMap/power-grid-detection/blob/master/dataset/transnet_nodes_corrected_germany.csv) |
| Gridfinder | Satellite-predicted power networks incl. medium/low-voltage lines | [viz](https://gridfinder.rdrn.me/) · [Nature paper 2020](https://www.nature.com/articles/s41597-019-0347-4) |
| SciGrid | European grid, mapped | [downloads](https://www.power.scigrid.de/pages/downloads.html) |
| Texas synthetic grid | 7000-bus test case | [download](https://electricgrids.engr.tamu.edu/texas7k-td/) · [more cases](https://electricgrids.engr.tamu.edu/electric-grid-test-cases/) |
| NREL test grids | Up to 240-bus | [repo](https://www.nrel.gov/grid/test-case-repository.html) |
| GridStatus.io | Substation pricing, load, wind/solar output, DC tie flows, live frequency | [datasets](https://www.gridstatus.io/datasets) (API key required) · [example Plotly notebook](https://github.com/gridstatus/gridstatusio/blob/main/Examples/Stacked%20Net%20Load%20Visualization.ipynb) |
| Open Power System Data | Renewable plant lists, Czechia/Denmark/France/Germany/Poland/Sweden/Switzerland/UK | [dataset](https://data.open-power-system-data.org/renewable_power_plants/2020-08-25) |
| U.S. National Climate Map Explorer | Heat index, temp max/min, fire weather, precip, wind speed | [tool](https://climrr.anl.gov/mapexplorer) · [raw downloads](https://anl.app.box.com/s/hmkkgkrkzxxocfe9kpgrzk2gfc4gizp8) |

## Synthetic grids (fallback if real data is gated)

- [Generation Method of Power System Test Examples Based on Complex Network Theory](https://ieeexplore.ieee.org/document/9347043) (2022) — generate plausible synthetic grid from network theory instead of sourcing real topology.
- [PyPSA](https://pypsa.readthedocs.io/en/latest/) — alternative to pandapower worth a look if pandapower's model doesn't fit a given case.

---

*Related: [[01-ARCHITECTURE]] · [[00-VISION]] · [[07-TOOLING-UPDATES]] · [[versions/V0_SUMMARY]]*
