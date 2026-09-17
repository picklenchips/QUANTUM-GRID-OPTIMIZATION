"""
api -- local FastAPI backend bridging the web map (web/opengridmap_source.html) to the
optimizer stack (algos/) and the OpenInfraMap hook (data/openinframap_hook.py).

Run:  uvicorn api.main:app --reload --port 8000
      (alongside `python -m http.server` at the repo root for the static page)

Local-only, single-user research tool: CORS is wide open, job/network state is in-memory.
The web page degrades to its static behaviour when this server is not running.
See docs/09-WEB-OVERHAUL.md.
"""
