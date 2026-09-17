/* selection-tools.js -- panel wiring (from the old inline wireControls) + interaction tools:
   Terra Draw rectangle/lasso, click/shift-click feature selection, in-view counter, bus drag. */
(function () {
  const QPGrid = (window.QPGrid = window.QPGrid || {});
  QPGrid.selectedBuses = new Set();
  QPGrid.mode = "select";
  let suppressEcho = false;
  let currentBox = null;
  let terra = null;

  const $ = (id) => document.getElementById(id);
  const pointInRing = (pt, ring) => {
    let inside = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const xi = ring[i][0], yi = ring[i][1], xj = ring[j][0], yj = ring[j][1];
      if (yi > pt[1] !== yj > pt[1] && pt[0] < ((xj - xi) * (pt[1] - yi)) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  };

  QPGrid.ready.then((map) => {
    wirePanels(map);
    wireSelection(map);
    wireInViewCounter(map);
    wireBoxTool(map);
    setupTerraDraw(map);
    wireModeButtons(map);
    wireBusDrag(map);
  });

  /* ------------------------------------------------------------------ panels (ex-wireControls) */
  function wirePanels(map) {
    const panels = { bFilter: "filter", bView: "view", bBox: "box", bOverlays: "overlays", bOpt: "optimize", bTopo: "topology" };
    for (const [btn, panel] of Object.entries(panels)) {
      const b = $(btn);
      if (!b) continue;
      b.onclick = () => {
        const el = $(panel), was = el.classList.contains("open");
        document.querySelectorAll(".panel").forEach((p) => p.classList.remove("open"));
        document.querySelectorAll(".btns button").forEach((x) => x.classList.remove("on"));
        if (!was) {
          el.classList.add("open");
          b.classList.add("on");
        }
      };
    }
    if ($("bLegend"))
      $("bLegend").onclick = (e) => {
        $("vlegend").classList.toggle("open");
        e.target.classList.toggle("on");
      };

    const setVis = (ids, on) =>
      ids.forEach((id) => map.getLayer(id) && map.setLayoutProperty(id, "visibility", on ? "visible" : "none"));
    const LINE_IDS = ["oim-line-unknown", "oim-line-0", "oim-line-1", "oim-line-2", "oim-line-3"];
    const bind = (id, fn) => $(id) && ($(id).onchange = fn);
    bind("fLines", (e) => setVis(LINE_IDS, e.target.checked));
    bind("fLabels", (e) => setVis(["oim-line-label"], e.target.checked));
    bind("fSubs", (e) => setVis(["oim-sub-area", "oim-sub-pt"], e.target.checked));
    bind("fPlants", (e) => setVis(["oim-plant-pt"], e.target.checked));
    bind("oNet", (e) => setVis(["net-line", "net-bus"], e.target.checked));
    bind("oOsm", (e) => setVis(["osmh-line", "osmh-pt"], e.target.checked));
    bind("oFwi", (e) => setVis(["fwi-fill"], e.target.checked));
    document.querySelectorAll("input[name=bm]").forEach((r) => (r.onchange = (e) => map.getSource("base").setTiles([QPGrid.BASEMAPS[e.target.value]])));
    if ($("globe"))
      $("globe").onchange = (e) => {
        try {
          map.setProjection({ type: e.target.checked ? "globe" : "mercator" });
        } catch (_) {}
      };
  }

  /* ------------------------------------------------------------------ click / lasso selection */
  function applySelection(map) {
    const feats = map.querySourceFeatures("net", { filter: ["==", "kind", "bus"] });
    const seen = new Set();
    for (const f of feats) {
      if (f.id == null) continue;
      seen.add(f.id);
      map.setFeatureState({ source: "net", id: f.id }, { selected: QPGrid.selectedBuses.has(f.properties.id) });
    }
    // lines: selected if both endpoints selected
    for (const f of map.querySourceFeatures("net", { filter: ["==", "kind", "line"] })) {
      if (f.id == null) continue;
      const on = QPGrid.selectedBuses.has(f.properties.from_bus) && QPGrid.selectedBuses.has(f.properties.to_bus);
      map.setFeatureState({ source: "net", id: f.id }, { selected: on });
    }
  }

  function emitSelection() {
    if (suppressEcho) return;
    QPGrid.emit("selectionchange", { ids: [...QPGrid.selectedBuses] });
  }

  function wireSelection(map) {
    map.on("click", "net-bus", (e) => {
      if (QPGrid.mode !== "select") return;
      const id = e.features[0].properties.id;
      if (!e.originalEvent.shiftKey) QPGrid.selectedBuses.clear();
      QPGrid.selectedBuses.has(id) ? QPGrid.selectedBuses.delete(id) : QPGrid.selectedBuses.add(id);
      applySelection(map);
      emitSelection();
      e.originalEvent.stopPropagation();
    });
    map.on("click", (e) => {
      if (QPGrid.mode !== "select") return;
      const hit = map.queryRenderedFeatures(e.point, { layers: ["net-bus"] });
      if (!hit.length && !e.originalEvent.shiftKey && QPGrid.selectedBuses.size) {
        QPGrid.selectedBuses.clear();
        applySelection(map);
        emitSelection();
      }
    });
    for (const id of ["net-bus", "net-line"]) {
      map.on("mouseenter", id, () => (map.getCanvas().style.cursor = "pointer"));
      map.on("mouseleave", id, () => (map.getCanvas().style.cursor = ""));
    }

    QPGrid.on("topologyselectionchange", ({ ids }) => {
      suppressEcho = true;
      QPGrid.selectedBuses = new Set(ids);
      applySelection(map);
      const pts = map
        .querySourceFeatures("net", { filter: ["==", "kind", "bus"] })
        .filter((f) => QPGrid.selectedBuses.has(f.properties.id))
        .map((f) => f.geometry.coordinates);
      if (pts.length) {
        const lon = pts.map((p) => p[0]), lat = pts.map((p) => p[1]);
        map.fitBounds([[Math.min(...lon), Math.min(...lat)], [Math.max(...lon), Math.max(...lat)]], {
          padding: 80, maxZoom: 12, animate: true,
        });
      }
      suppressEcho = false;
    });

    // re-apply selection state after a network swap (feature ids are regenerated)
    QPGrid.on("networkloaded", () => setTimeout(() => applySelection(map), 150));
  }

  /* ------------------------------------------------------------------ in-view counter */
  function wireInViewCounter(map) {
    const el = $("inview");
    if (!el) return;
    const update = () => {
      const subs = map.queryRenderedFeatures({ layers: ["oim-sub-pt", "oim-sub-area"] }).length;
      const lines = map.queryRenderedFeatures({ layers: ["oim-line-0", "oim-line-1", "oim-line-2", "oim-line-3", "oim-line-unknown"] }).length;
      el.textContent = `${subs} substations · ${lines} line segments in view`;
    };
    map.on("moveend", update);
    map.on("idle", update);
  }

  /* ------------------------------------------------------------------ bbox tool + load-detail */
  function drawBox(map, w, s, ee, n, fit) {
    const bad = $("boxbad");
    const ok =
      [w, s, ee, n].every(Number.isFinite) && w < ee && s < n &&
      Math.abs(w) <= 180 && Math.abs(ee) <= 180 && Math.abs(s) <= 90 && Math.abs(n) <= 90;
    if (bad) bad.style.display = ok ? "none" : "inline";
    currentBox = ok ? [w, s, ee, n] : null;
    QPGrid.currentBox = currentBox;
    map.getSource("bbox").setData(
      ok
        ? { type: "Feature", geometry: { type: "Polygon", coordinates: [[[w, s], [ee, s], [ee, n], [w, n], [w, s]]] } }
        : { type: "FeatureCollection", features: [] }
    );
    if (ok && fit) map.fitBounds([[w, s], [ee, n]], { padding: 40, animate: false });
    const btn = $("bxLoad");
    if (btn) btn.disabled = !ok || !QPGrid.api.online;
  }

  function wireBoxTool(map) {
    const fromInputs = (fit) =>
      drawBox(map, parseFloat($("bxW").value), parseFloat($("bxS").value), parseFloat($("bxE").value), parseFloat($("bxN").value), fit);
    ["bxW", "bxS", "bxE", "bxN"].forEach((id) => $(id) && ($(id).oninput = () => fromInputs(false)));
    if ($("bxFit")) $("bxFit").onclick = () => fromInputs(true);

    let dragStart = null;
    map.on("mousedown", (e) => {
      if (e.originalEvent.shiftKey && QPGrid.mode === "select") {
        dragStart = e.lngLat;
        map.dragPan.disable();
      }
    });
    map.on("mousemove", (e) => {
      if (!dragStart) return;
      const a = dragStart, b = e.lngLat;
      drawBox(map, Math.min(a.lng, b.lng), Math.min(a.lat, b.lat), Math.max(a.lng, b.lng), Math.max(a.lat, b.lat), false);
    });
    map.on("mouseup", () => {
      if (!dragStart) return;
      if (currentBox) {
        const [w, s, ee, n] = currentBox;
        $("bxW").value = w.toFixed(4);
        $("bxS").value = s.toFixed(4);
        $("bxE").value = ee.toFixed(4);
        $("bxN").value = n.toFixed(4);
      }
      dragStart = null;
      map.dragPan.enable();
    });

    if ($("bxLoad"))
      $("bxLoad").onclick = async () => {
        if (!currentBox) return;
        const btn = $("bxLoad");
        btn.disabled = true;
        btn.textContent = "loading Overpass…";
        try {
          const res = await QPGrid.api.networkFromSelection(currentBox);
          QPGrid.currentNetworkId = res.network_id;
          QPGrid.setNetwork(res.geojson, { fit: true });
          $("bxNote").textContent = `network ${res.network_id.slice(0, 8)} · ${res.n_bus} buses · ${res.n_line} lines`;
          QPGrid.emit("networkidchange", { network_id: res.network_id, n_bus: res.n_bus, n_line: res.n_line });
        } catch (err) {
          $("bxNote").textContent = "error: " + err.message;
        } finally {
          btn.textContent = "load full detail (Overpass)";
          setTimeout(() => (btn.disabled = !currentBox || !QPGrid.api.online), 4000);
        }
      };
  }

  /* ------------------------------------------------------------------ Terra Draw */
  function setupTerraDraw(map) {
    if (typeof terraDraw === "undefined" || typeof terraDrawMaplibreGlAdapter === "undefined") return;
    terra = new terraDraw.TerraDraw({
      adapter: new terraDrawMaplibreGlAdapter.TerraDrawMapLibreGLAdapter({ map, coordinatePrecision: 9 }),
      modes: [new terraDraw.TerraDrawRectangleMode(), new terraDraw.TerraDrawFreehandMode()],
    });
    terra.start();
    terra.setMode("static");
    terra.on("finish", (id) => {
      const snap = terra.getSnapshot().find((f) => f.id === id);
      if (!snap) return;
      const ring = snap.geometry.coordinates[0];
      if (QPGrid.mode === "draw-rect") {
        const lon = ring.map((p) => p[0]), lat = ring.map((p) => p[1]);
        const w = Math.min(...lon), s = Math.min(...lat), e = Math.max(...lon), n = Math.max(...lat);
        $("bxW").value = w.toFixed(4);
        $("bxS").value = s.toFixed(4);
        $("bxE").value = e.toFixed(4);
        $("bxN").value = n.toFixed(4);
        drawBox(map, w, s, e, n, false);
        terra.clear();
        setMode(map, "select");
        if ($("bBox")) $("bBox").click();
      } else if (QPGrid.mode === "draw-lasso") {
        QPGrid.selectedBuses.clear();
        for (const f of map.querySourceFeatures("net", { filter: ["==", "kind", "bus"] })) {
          if (pointInRing(f.geometry.coordinates, ring)) QPGrid.selectedBuses.add(f.properties.id);
        }
        applySelection(map);
        emitSelection();
        terra.clear();
        setMode(map, "select");
      }
    });
  }

  /* ------------------------------------------------------------------ mode toolbar */
  function setMode(map, m) {
    QPGrid.mode = m;
    document.querySelectorAll("#tools button").forEach((b) => b.classList.toggle("on", b.dataset.mode === m));
    if (!terra) return;
    if (m === "draw-rect") terra.setMode("rectangle");
    else if (m === "draw-lasso") terra.setMode("freehand");
    else terra.setMode("static");
  }
  function wireModeButtons(map) {
    document.querySelectorAll("#tools button").forEach((b) => (b.onclick = () => setMode(map, b.dataset.mode)));
    setMode(map, "select");
  }

  /* ------------------------------------------------------------------ bus drag (riskiest; opt-in) */
  function wireBusDrag(map) {
    let dragId = null;
    map.on("mousedown", "net-bus", (e) => {
      if (QPGrid.mode !== "drag-bus") return;
      dragId = e.features[0].properties.id;
      map.dragPan.disable();
      e.preventDefault();
    });
    map.on("mousemove", (e) => {
      if (dragId == null) return;
      const gj = QPGrid.data.net;
      const f = gj.features.find((x) => x.properties.kind === "bus" && x.properties.id === dragId);
      if (f) {
        f.geometry.coordinates = [e.lngLat.lng, e.lngLat.lat];
        for (const ln of gj.features) {
          if (ln.properties.kind !== "line") continue;
          if (ln.properties.from_bus === dragId) ln.geometry.coordinates[0] = f.geometry.coordinates;
          if (ln.properties.to_bus === dragId) ln.geometry.coordinates[1] = f.geometry.coordinates;
        }
        map.getSource("net").setData(gj);
      }
    });
    map.on("mouseup", () => {
      if (dragId == null) return;
      QPGrid.emit("busmoved", { busId: dragId });
      dragId = null;
      map.dragPan.enable();
    });
  }
})();
