/* topology-panel.js -- Cytoscape.js graph of the converted network. Shows electrical adjacency
   independent of geography; stays in two-way sync with the map selection. */
(function () {
  const QPGrid = (window.QPGrid = window.QPGrid || {});
  const PAL = QPGrid.PAL;
  let cy = null;
  let suppressEcho = false;

  /* returns {els, nBus, hidden}. Isolated buses (degree 0) are omitted from the graph -- a real
     grid selection can carry hundreds of unconnected generator points that swamp a cose layout. */
  const elFromNet = (gj) => {
    const buses = new Map();
    for (const f of gj.features) {
      const p = f.properties || {};
      if (p.kind === "bus") buses.set(p.id, { id: "bus-" + p.id, busId: p.id, label: p.name || "bus " + p.id, vn_kv: p.vn_kv });
    }
    const edges = [];
    const deg = new Set();
    for (const f of gj.features) {
      const p = f.properties || {};
      if (p.kind !== "line") continue;
      if (!buses.has(p.from_bus) || !buses.has(p.to_bus)) continue;
      edges.push({ data: { id: "line-" + p.id, source: "bus-" + p.from_bus, target: "bus-" + p.to_bus } });
      deg.add(p.from_bus);
      deg.add(p.to_bus);
    }
    const nodes = [...buses.values()].filter((b) => deg.has(b.busId)).map((d) => ({ data: d }));
    return { els: nodes.concat(edges), nBus: buses.size, shown: nodes.length, nEdge: edges.length };
  };

  function build(gj) {
    const host = document.getElementById("cy");
    if (!host || typeof cytoscape === "undefined") return;
    const { els, nBus, shown, nEdge } = elFromNet(gj);
    document.getElementById("topoNote").textContent =
      `${shown}/${nBus} connected buses · ${nEdge} edges` + (shown < nBus ? " (isolated hidden)" : "");
    if (cy) cy.destroy();
    cy = cytoscape({
      container: host,
      elements: els,
      style: [
        { selector: "node", style: { "background-color": "#22d3ee", label: "data(label)", "font-size": 7, color: "#aaa", "text-opacity": 0.8, width: 12, height: 12 } },
        { selector: "node:selected", style: { "background-color": "#fff", "border-width": 2, "border-color": "#000" } },
        { selector: "edge", style: { "line-color": "#557", width: 1.4, "curve-style": "haystack" } },
        { selector: "node[part]", style: { "background-color": "data(partColor)" } },
      ],
      layout: { name: "cose", animate: false, nodeRepulsion: 6000, idealEdgeLength: 40 },
    });
    cy.on("select unselect", "node", () => {
      if (suppressEcho) return;
      const ids = cy.$("node:selected").map((n) => n.data("busId"));
      QPGrid.emit("topologyselectionchange", { ids });
    });
  }

  QPGrid.on("networkloaded", ({ geojson }) => build(geojson));
  QPGrid.ready.then(() => {
    if (QPGrid.data && QPGrid.data.net && QPGrid.data.net.features.length) build(QPGrid.data.net);
  });

  QPGrid.on("selectionchange", ({ ids }) => {
    if (!cy) return;
    suppressEcho = true;
    cy.$("node:selected").unselect();
    const set = new Set(ids);
    cy.nodes().forEach((n) => set.has(n.data("busId")) && n.select());
    suppressEcho = false;
  });

  QPGrid.on("resultready", ({ payload }) => {
    if (!cy || !payload || !payload.bus_assignment) return;
    cy.nodes().forEach((n) => {
      const part = payload.bus_assignment[String(n.data("busId"))];
      if (part == null) {
        n.removeData("part");
        n.removeData("partColor");
      } else {
        n.data("part", part);
        n.data("partColor", PAL[part % PAL.length]);
      }
    });
  });
})();
