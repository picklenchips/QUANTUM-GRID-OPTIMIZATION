/* results-panel.js -- optimizer picker (from GET /optimizers), generic param form, run + SSE,
   Plotly convergence/comparison charts, run history in localStorage. */
(function () {
  const QPGrid = (window.QPGrid = window.QPGrid || {});
  const PAL = QPGrid.PAL;
  const $ = (id) => document.getElementById(id);
  const HIST_KEY = "qpgrid_runs";

  let optimizers = [];
  let history = [];
  try {
    history = JSON.parse(localStorage.getItem(HIST_KEY) || "[]");
  } catch (_) {}

  QPGrid.on("networkidchange", ({ network_id, n_bus, n_line }) => {
    QPGrid.currentNetworkId = network_id;
    $("optNet").textContent = `network ${network_id.slice(0, 8)} · ${n_bus} buses · ${n_line} lines`;
  });

  QPGrid.ready.then(async () => {
    const health = await QPGrid.api.health();
    renderBackendBadge(health);
    if (!health) return;
    try {
      optimizers = await QPGrid.api.optimizers();
    } catch (e) {
      $("optNote").textContent = "optimizers: " + e.message;
      return;
    }
    populatePicker();
    renderHistory();
  });

  function renderBackendBadge(health) {
    const b = $("backendBadge");
    if (!b) return;
    b.textContent = health ? `backend: connected (${health.optimizers} optimizers)` : "backend: offline";
    b.className = "badge " + (health ? "ok" : "off");
  }

  function populatePicker() {
    const kind = $("optKind").value;
    const sel = $("optPick");
    sel.innerHTML = "";
    optimizers
      .filter((o) => o.kinds.includes(kind))
      .forEach((o) => {
        const opt = document.createElement("option");
        opt.value = o.name;
        opt.textContent = o.name + (o.available ? "" : " (backend missing)");
        opt.disabled = !o.available;
        sel.appendChild(opt);
      });
    buildForm();
  }

  function buildForm() {
    const o = optimizers.find((x) => x.name === $("optPick").value);
    const host = $("optParams");
    host.innerHTML = "";
    if (!o) return;
    $("optDoc").textContent = (o.docstring || "").split("\n")[0];
    for (const p of o.params) {
      const row = document.createElement("label");
      row.className = "prow";
      const span = document.createElement("span");
      span.textContent = p.name;
      const inp = document.createElement("input");
      inp.dataset.pname = p.name;
      inp.dataset.ptype = p.type;
      if (p.type === "boolean") {
        inp.type = "checkbox";
        inp.checked = !!p.default;
      } else {
        inp.type = p.type === "integer" || p.type === "number" ? "number" : "text";
        if (p.default != null) inp.value = p.default;
        inp.placeholder = p.nullable ? "(auto)" : "";
      }
      row.appendChild(span);
      row.appendChild(inp);
      host.appendChild(row);
    }
  }

  function collectParams() {
    const out = {};
    document.querySelectorAll("#optParams input").forEach((inp) => {
      const t = inp.dataset.ptype;
      if (t === "boolean") out[inp.dataset.pname] = inp.checked;
      else if (inp.value === "") return;
      else if (t === "integer") out[inp.dataset.pname] = parseInt(inp.value, 10);
      else if (t === "number") out[inp.dataset.pname] = parseFloat(inp.value);
      else out[inp.dataset.pname] = inp.value;
    });
    return out;
  }

  async function run() {
    const kind = $("optKind").value;
    const netId = QPGrid.currentNetworkId || "demo";
    const problemParams = {};
    if (kind === "qubo") problemParams.lambd = parseFloat($("optLambd").value || "1");
    if (kind === "uc") {
      try {
        problemParams.demand = JSON.parse($("optDemand").value);
      } catch (_) {
        $("optNote").textContent = "demand must be a JSON list";
        return;
      }
    }
    const req = {
      network_id: netId,
      problem_kind: kind,
      problem_params: problemParams,
      optimizer: $("optPick").value,
      solver_params: collectParams(),
    };
    $("optRun").disabled = true;
    $("optNote").textContent = "submitting…";
    try {
      const jobId = await QPGrid.api.submitOptimize(req);
      QPGrid.api.streamOptimize(jobId, {
        status: (s) => ($("optNote").textContent = s.status + "…"),
        error: (e) => {
          $("optNote").textContent = "error: " + e.type + " — " + e.message;
          $("optRun").disabled = false;
        },
        done: (payload) => {
          $("optNote").textContent = `${payload.method}: obj ${fmt(payload.objective)} · ${(payload.runtime_s * 1000).toFixed(0)} ms · ${payload.feasible ? "feasible" : "infeasible"}`;
          $("optRun").disabled = false;
          applyResult(payload, req);
        },
      });
    } catch (e) {
      $("optNote").textContent = "error: " + e.message;
      $("optRun").disabled = false;
    }
  }

  function applyResult(payload, req) {
    if (payload.geojson && payload.geojson.features.length) QPGrid.setNetwork(payload.geojson);
    QPGrid.emit("resultready", { payload });
    const entry = {
      t: Date.now(),
      optimizer: payload.method,
      kind: req.problem_kind,
      network_id: req.network_id,
      params: req.solver_params,
      objective: payload.objective,
      runtime_s: payload.runtime_s,
      feasible: payload.feasible,
      trace: payload.trace || [],
      bus_assignment: payload.bus_assignment || null,
      geojson: payload.geojson || null,
    };
    history.unshift(entry);
    history = history.slice(0, 25);
    try {
      localStorage.setItem(HIST_KEY, JSON.stringify(history));
    } catch (_) {}
    renderHistory();
    drawCharts();
  }

  function drawCharts() {
    if (typeof Plotly === "undefined") return;
    const conv = history
      .filter((h) => h.trace && h.trace.length)
      .slice(0, 6)
      .map((h, i) => ({ y: h.trace, mode: "lines", name: h.optimizer, line: { color: PAL[i % PAL.length] } }));
    Plotly.react("optConv", conv, {
      margin: { t: 10, r: 10, b: 30, l: 40 }, height: 170,
      xaxis: { title: "checkpoint" }, yaxis: { title: "objective" },
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { size: 9, color: "#999" },
      showlegend: true, legend: { font: { size: 8 } },
    }, { displayModeBar: false });

    const scatter = [{
      x: history.map((h) => h.runtime_s * 1000),
      y: history.map((h) => h.objective),
      text: history.map((h) => h.optimizer),
      mode: "markers", type: "scatter", marker: { size: 8, color: PAL[1] },
    }];
    Plotly.react("optScatter", scatter, {
      margin: { t: 10, r: 10, b: 30, l: 40 }, height: 150,
      xaxis: { title: "runtime (ms)", type: "log" }, yaxis: { title: "objective" },
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { size: 9, color: "#999" },
    }, { displayModeBar: false });
  }

  function renderHistory() {
    const host = $("optHist");
    if (!host) return;
    host.innerHTML = "";
    history.forEach((h, i) => {
      const d = document.createElement("div");
      d.className = "histrow";
      d.textContent = `${h.optimizer} · ${fmt(h.objective)} · ${(h.runtime_s * 1000).toFixed(0)}ms`;
      d.title = new Date(h.t).toLocaleString() + "\n" + JSON.stringify(h.params);
      d.onclick = () => {
        if (h.geojson) QPGrid.setNetwork(h.geojson);
        QPGrid.emit("resultready", { payload: h });
      };
      host.appendChild(d);
    });
    drawCharts();
  }

  const fmt = (v) => (v == null ? "–" : Math.abs(v) >= 1000 || Math.abs(v) < 0.01 ? v.toExponential(2) : v.toFixed(3));

  QPGrid.ready.then(() => {
    $("optKind").onchange = () => {
      populatePicker();
      $("optQuboRow").style.display = $("optKind").value === "qubo" ? "flex" : "none";
      $("optUcRow").style.display = $("optKind").value === "uc" ? "flex" : "none";
    };
    $("optPick").onchange = buildForm;
    $("optRun").onclick = run;
    $("optClearHist") &&
      ($("optClearHist").onclick = () => {
        history = [];
        localStorage.removeItem(HIST_KEY);
        renderHistory();
      });
  });
})();
