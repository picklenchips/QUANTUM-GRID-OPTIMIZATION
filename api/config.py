"""api.config -- knobs for the local backend. No secrets; safe to commit."""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CORS: the static page is served from python -m http.server (usually :8000 or :5500),
# this API from uvicorn on another port. Local-only tool -> allow everything.
CORS_ORIGINS = ["*"]

# on-disk cache for Overpass responses (data/openinframap_hook.fetch_power_geojson)
CACHE_DIR = os.path.join(REPO_ROOT, ".cache", "overpass")

# guardrail against hammering Overpass with a huge interactive selection (deg^2).
# ~1.0 deg^2 is roughly a 100km x 100km box at mid latitudes -- generous for a research tool.
MAX_BBOX_DEG2 = float(os.environ.get("QPGRID_MAX_BBOX_DEG2", "1.0"))

# cap on solver wall time surfaced to the UI (informational; individual optimizers self-limit)
DEFAULT_SOLVE_TIMEOUT_S = 60.0
