/* map-init.js -- builds the MapLibre map (OpenInfraMap vector tiles + Esri basemaps + pipeline
   overlays). Data is fetched first and inlined into the initial style: adding geojson sources
   after 'load' is unreliable under software GL. Exposes window.QPGrid.{map,ready,data,setNetwork}. */
(function () {
  const QPGrid = (window.QPGrid = window.QPGrid || {});
  const PAL = QPGrid.PAL;

  const OIM = "https://openinframap.org/tiles/{z}/{x}/{y}.pbf";
  const ESRI = (s) => `https://server.arcgisonline.com/ArcGIS/rest/services/${s}/MapServer/tile/{z}/{y}/{x}`;
  const BASEMAPS = {
    dark: ESRI("Canvas/World_Dark_Gray_Base"),
    light: ESRI("Canvas/World_Light_Gray_Base"),
    streets: ESRI("World_Street_Map"),
    outdoors: ESRI("World_Topo_Map"),
  };
  const ESRI_ATTR = "Esri, HERE, Garmin, © OpenStreetMap contributors";
  const V_TIERS = [
    [0, "#ff5a5a", 0.8],
    [25000, "#e03030", 1.1],
    [132000, "#a01818", 1.6],
    [330000, "#111111", 2.2],
  ];
  const EMPTY = { type: "FeatureCollection", features: [] };
  QPGrid.BASEMAPS = BASEMAPS;

  const loadJSON = async (u) => {
    try {
      const r = await fetch(u);
      return r.ok ? await r.json() : null;
    } catch (_) {
      return null;
    }
  };
  const fc = (g) => (g && g.type === "FeatureCollection" ? { type: "FeatureCollection", features: g.features } : EMPTY);

  /* partition colour for buses: white if map-selected, else PAL[partition%8] if assigned, else cyan.
     partition is null/absent -> coalesce to -1 -> match falls through to the fallback. */
  const partitionColor = (fallback) => [
    "case",
    ["boolean", ["feature-state", "selected"], false], "#ffffff",
    [
      "match", ["%", ["to-number", ["coalesce", ["get", "partition"], -1]], 8],
      0, PAL[0], 1, PAL[1], 2, PAL[2], 3, PAL[3], 4, PAL[4], 5, PAL[5], 6, PAL[6], 7, PAL[7],
      fallback,
    ],
  ];

  let resolveReady;
  QPGrid.ready = new Promise((res) => (resolveReady = res));

  (async function init() {
    const [netD, osmD, fwiD, status] = await Promise.all([
      loadJSON("data/demo_network.geojson"),
      loadJSON("data/openinframap_power.geojson"),
      loadJSON("data/climrr_fwi.geojson"),
      loadJSON("data/tooling_status.json"),
    ]);
    const net = fc(netD), osmh = fc(osmD), fwi = fc(fwiD);
    QPGrid.data = { net, osmh, fwi };

    const ovNote = document.getElementById("ovNote");
    if (ovNote)
      ovNote.textContent = `network ${net.features.length} feats · Overpass ${osmh.features.length} · ClimRR ${fwi.features.length} cells`;
    if (status) {
      const pass = status.rows.filter((r) => /pass|ok/i.test(r.status)).length;
      const stx = document.getElementById("stx");
      if (stx) stx.textContent = `${pass}/${status.rows.length} hooks passing`;
    } else {
      const s = document.getElementById("status");
      if (s) s.style.display = "none";
    }

    let center = [-98, 39], zoom = 3.4;
    if (net.features.length) {
      const pts = net.features.flatMap((f) =>
        f.geometry.type === "Point" ? [f.geometry.coordinates] : f.geometry.coordinates
      );
      const lon = pts.map((p) => p[0]), lat = pts.map((p) => p[1]);
      center = [(Math.min(...lon) + Math.max(...lon)) / 2, (Math.min(...lat) + Math.max(...lat)) / 2];
      zoom = 5.4;
    }

    const map = new maplibregl.Map({
      container: "map",
      center,
      zoom,
      attributionControl: true,
      style: {
        version: 8,
        glyphs: "https://fonts.openmaptiles.org/{fontstack}/{range}.pbf",
        sources: {
          base: { type: "raster", tiles: [BASEMAPS.dark], tileSize: 256, attribution: ESRI_ATTR },
          oim: { type: "vector", tiles: [OIM], minzoom: 2, maxzoom: 17, attribution: "© OpenInfraMap" },
          net: { type: "geojson", data: net, generateId: true },
          osmh: { type: "geojson", data: osmh },
          fwi: { type: "geojson", data: fwi },
          bbox: { type: "geojson", data: EMPTY },
        },
        layers: [
          { id: "base", type: "raster", source: "base" },
          {
            id: "fwi-fill", type: "fill", source: "fwi", layout: { visibility: "none" },
            paint: {
              "fill-color": ["interpolate", ["linear"], ["coalesce", ["to-number", ["get", "summer_Midc_Mean"]], 0],
                8, "#2c7fb8", 16, "#7fcdbb", 24, "#ffffb2", 32, "#fd8d3c", 42, "#bd0026"],
              "fill-opacity": 0.45,
            },
          },
          {
            id: "oim-sub-area", type: "fill", source: "oim", "source-layer": "power_substation",
            paint: { "fill-color": "#3d6bff", "fill-opacity": 0.35 },
          },
          {
            id: "oim-line-unknown", type: "line", source: "oim", "source-layer": "power_line",
            filter: ["!", ["has", "voltage"]],
            paint: { "line-color": "#ff9090", "line-width": 0.6, "line-opacity": 0.7 },
          },
          ...V_TIERS.map((t, i) => ({
            id: "oim-line-" + i, type: "line", source: "oim", "source-layer": "power_line",
            filter: i < V_TIERS.length - 1
              ? ["all", [">=", ["to-number", ["get", "voltage"]], t[0]], ["<", ["to-number", ["get", "voltage"]], V_TIERS[i + 1][0]]]
              : [">=", ["to-number", ["get", "voltage"]], t[0]],
            paint: { "line-color": t[1], "line-width": t[2], "line-opacity": 0.9 },
          })),
          {
            id: "oim-sub-pt", type: "circle", source: "oim", "source-layer": "power_substation_point",
            paint: { "circle-radius": 3.5, "circle-color": "#3d6bff" },
          },
          {
            id: "oim-plant-pt", type: "circle", source: "oim", "source-layer": "power_plant_point",
            paint: { "circle-radius": 4, "circle-color": "#0b7755" },
          },
          {
            id: "oim-line-label", type: "symbol", source: "oim", "source-layer": "power_line", minzoom: 6,
            layout: { "symbol-placement": "line", "text-font": ["Noto Sans Regular"], "text-size": 11,
              "text-field": ["coalesce", ["get", "voltage"], ""] },
            paint: { "text-color": "#e9e9e9", "text-halo-color": "#000", "text-halo-width": 1 },
          },
          {
            id: "osmh-line", type: "line", source: "osmh", filter: ["==", "$type", "LineString"],
            paint: { "line-color": "#ff9f1c", "line-width": 1.4 },
          },
          {
            id: "osmh-pt", type: "circle", source: "osmh", filter: ["==", "$type", "Point"],
            paint: { "circle-radius": 3, "circle-color": "#ff9f1c", "circle-opacity": 0.85 },
          },
          {
            id: "net-line", type: "line", source: "net", filter: ["==", "kind", "line"],
            paint: {
              "line-color": ["case", ["boolean", ["get", "cross_partition"], false], "#ff4d4d", "#22d3ee"],
              "line-width": ["case", ["boolean", ["feature-state", "selected"], false], 3.4, 1.8],
              "line-opacity": 0.95,
            },
          },
          {
            id: "net-bus", type: "circle", source: "net", filter: ["==", "kind", "bus"],
            paint: {
              "circle-radius": ["case", ["boolean", ["feature-state", "selected"], false], 6.5, 4],
              "circle-color": partitionColor("#22d3ee"),
              "circle-stroke-width": 1,
              "circle-stroke-color": ["case", ["boolean", ["feature-state", "selected"], false], "#000", "#04222a"],
            },
          },
          { id: "bbox-fill", type: "fill", source: "bbox", paint: { "fill-color": "#fff", "fill-opacity": 0.15 } },
          {
            id: "bbox-line", type: "line", source: "bbox",
            paint: { "line-color": "#fff", "line-width": 1.5, "line-dasharray": [2, 1] },
          },
        ],
      },
    });

    map.addControl(new maplibregl.NavigationControl(), "top-right");
    map.addControl(new maplibregl.ScaleControl(), "bottom-right");
    map.on("error", (e) => console.warn("map:", (e.error && e.error.message) || e.type));

    QPGrid.map = map;
    map.on("load", () => resolveReady(map));
  })();

  /* swap the converted-network layer's data (result GeoJSON or a freshly built network).
     'net' source exists from construction, so setData on it is the reliable path. */
  QPGrid.setNetwork = function (geojson, opts) {
    opts = opts || {};
    const src = QPGrid.map && QPGrid.map.getSource("net");
    if (!src) return;
    QPGrid.data.net = geojson;
    src.setData(geojson);
    if (opts.fit) {
      const pts = geojson.features.flatMap((f) =>
        f.geometry.type === "Point" ? [f.geometry.coordinates] : f.geometry.coordinates
      );
      if (pts.length) {
        const lon = pts.map((p) => p[0]), lat = pts.map((p) => p[1]);
        QPGrid.map.fitBounds(
          [[Math.min(...lon), Math.min(...lat)], [Math.max(...lon), Math.max(...lat)]],
          { padding: 60, animate: true, maxZoom: 12 }
        );
      }
    }
    QPGrid.emit("networkloaded", { geojson });
  };
})();
