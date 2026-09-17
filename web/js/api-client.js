/* api-client.js -- thin wrapper around the local api/ backend (uvicorn api.main:app).
   Everything no-ops gracefully when the backend is not running; the page stays usable. */
(function () {
  const QPGrid = (window.QPGrid = window.QPGrid || {});

  const BASE = localStorage.getItem("qpgrid_api") || "http://localhost:8000";

  const api = {
    base: BASE,
    online: false,

    async health() {
      try {
        const r = await fetch(BASE + "/health", { cache: "no-store" });
        this.online = r.ok;
        return r.ok ? await r.json() : null;
      } catch (_) {
        this.online = false;
        return null;
      }
    },

    async optimizers(kind) {
      const u = BASE + "/optimizers" + (kind ? "?kind=" + encodeURIComponent(kind) : "");
      const r = await fetch(u);
      if (!r.ok) throw new Error("optimizers: HTTP " + r.status);
      return (await r.json()).optimizers;
    },

    async networkFromSelection(bbox) {
      const r = await fetch(BASE + "/network/from-selection", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ bbox }),
      });
      const body = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(body.detail || "network: HTTP " + r.status);
      return body;
    },

    async submitOptimize(req) {
      const r = await fetch(BASE + "/optimize", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(req),
      });
      const body = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(body.detail || "optimize: HTTP " + r.status);
      return body.job_id;
    },

    /* opens an EventSource; calls handlers {status, done, error}. returns the EventSource. */
    streamOptimize(jobId, handlers) {
      const es = new EventSource(BASE + "/optimize/" + jobId + "/stream");
      es.addEventListener("status", (e) => handlers.status && handlers.status(JSON.parse(e.data)));
      es.addEventListener("done", (e) => {
        handlers.done && handlers.done(JSON.parse(e.data));
        es.close();
      });
      es.addEventListener("error", (e) => {
        if (e.data) handlers.error && handlers.error(JSON.parse(e.data));
        else handlers.error && handlers.error({ type: "ConnectionError", message: "stream lost" });
        es.close();
      });
      return es;
    },
  };

  QPGrid.api = api;
  QPGrid.PAL = ["#DC267F", "#648FFF", "#FE6100", "#785EF0", "#FFB000", "#009E73", "#3DDBD9", "#808080"];

  /* small event bus over window. Names are prefixed 'qpg:' to avoid colliding with native
     events (e.g. the browser's own 'selectionchange' on document). */
  QPGrid.emit = (name, detail) => window.dispatchEvent(new CustomEvent("qpg:" + name, { detail: detail || {} }));
  QPGrid.on = (name, fn) => window.addEventListener("qpg:" + name, (e) => fn(e.detail || {}));
})();
