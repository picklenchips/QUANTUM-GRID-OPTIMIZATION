"""
data.gridsfm_to_pp

Loader + pandapower converter for Microsoft's GridSFM_US_power_grid dataset:
  https://huggingface.co/datasets/microsoft/GridSFM_US_power_grid
  blog: https://www.microsoft.com/en-us/research/blog/building-realistic-electric-transmission-grid-dataset-at-scale-a-pipeline-from-open-dataset/

Replaces HIFLD Open (dead, shut down Aug 2025 -- see docs/07-TOOLING-UPDATES.md) as the modern
continental-scale transmission dataset. See docs/04-DATA-SOURCES.md for the catalog entry.

FORMAT (confirmed 2026-08-12 against the dataset's own README.md, dataset_metadata.json, AND a
downloaded sample file 16h/rhode_island_model.json -- fields below are verbatim, not guessed):

  - 54 instances: 48 contiguous US states + 6 multi-state regions (up to 'eastern' = full Eastern
    Interconnection, ~21,697 buses across 36 states), each with two hourly snapshots.
  - One self-contained JSON per {region} x {hour}, MATPOWER/PowerModels.jl-format:
        16h/{region}_model.json        full topology + params + demand + shunts (peak, 4pm)
        16h/{region}_dc_results.json   DC-OPF solution
        16h/{region}_ac_results.json   AC-OPF solution
        04h/...                        same, off-peak (4am)
        dataset_metadata.json          region list + 2-letter abbreviations + hour/file-type index
  - model.json top-level sections (each a dict keyed by stringified element id):
        bus, gen, branch, load, shunt, dcline, storage (always empty), switch (always empty)
    plus scalars baseMVA (=100.0), per_unit (=True), name, balancing_authority, target_datetime, ...
  - All electrical quantities are per-unit on baseMVA (NOT physical ohms/km/MW like transnet's CSVs) --
    see the de-normalization math in add_gridsfm_branch/gen/load/shunt below.
  - License: MIT. Confirmed via HF API (huggingface.co/api/datasets/microsoft/GridSFM_US_power_grid).

Download: `pip install huggingface_hub`, then either hf_hub_download() per-file (used below) or
`hf download --repo-type dataset microsoft/GridSFM_US_power_grid --local-dir ./gridsfm_data` for
the whole ~230MB set. There is also an official loader (gridsfm_pg_loader.py /
GridSFM_PG_Loader) in https://github.com/microsoft/GridSFM/tree/main/power_grid -- not used here
to avoid an extra non-PyPI dependency; this file talks to huggingface_hub directly instead, same
as how transnet_to_pp.py/psse_to_pp.py just read local files directly rather than wrapping a 3rd
library.

TODO (needs a real AC power-flow run to validate, not just a schema check):
  - add_gridsfm_branch's transformer vk_percent/vkr_percent formula uses the standard per-unit
    base-conversion identity (Z_pu_on_own_base = Z_pu_on_system_base * S_own_base/S_system_base,
    with S_own_base := rate_a*baseMVA) -- textbook-correct, but GridSFM's docs don't spell out a
    pandapower-specific recipe, so this hasn't been checked against pp.runpp() convergence on a
    real transformer-heavy region (rhode_island's 8 transformers construct without error in
    testing here, but AC-OPF convergence was not verified end-to-end).
  - dcline handling (add_gridsfm_dcline) is written but untested -- no sample file inspected here
    had any dclines (rhode_island's dcline section is empty). Structure follows the README's
    documented dcline fields (pf/pt/vf/vt/pmaxf/pmaxt/br_status) but verify against an actual
    dcline-bearing region (e.g. a multi-state region file) before relying on it.
"""
import sys, os, json, math
import pandas as pd
import pandapower as pp
import pandapower.auxiliary as aux  # for pandapowerNet typing
import pandapower.plotting.plotly as ppl
from dotenv import load_dotenv
load_dotenv()
from glob import glob
from collections import defaultdict
pd.options.display.max_rows = 20

# import modules from enclosing directory
pdir = sys.path[0] + '/..'
if pdir not in sys.path:
    sys.path.append(pdir)
from util import options_menu

try:
    from huggingface_hub import hf_hub_download
    HAVE_HF_HUB = True
except ImportError:
    HAVE_HF_HUB = False


### GRIDSFM DATA (huggingface JSON, one file per region x hour) ###

GRIDSFM_REPO_ID = "microsoft/GridSFM_US_power_grid"
GRIDSFM_HOURS = ('16h', '04h')  # peak (4pm) / off-peak (4am), both snapshotted 2024-07-15

# store one GridSFM model instance (region + hour), created by reading its *_model.json
class GridSFMOut():
    def __init__(self, name='', hour='16h', modelpath='', read=False):
        """ name: region name e.g. 'rhode_island', 'texas', 'eastern' (see get_gridsfm_regions)
            hour: '16h' (peak) or '04h' (off-peak)
            modelpath: local path to an already-downloaded {name}_model.json """
        self.name = name
        self.hour = hour
        self.modelpath = modelpath
        self.dataIsProcessed = False
        if read:
            self.read_data()

    def read_data(self, modelpath=''):
        """ read the *_model.json into pandas DataFrames, one per PowerModels.jl/MATPOWER section """
        if not modelpath: modelpath = self.modelpath
        with open(modelpath) as f:
            self.model = json.load(f)
        # each section is a dict keyed by stringified element id -- orient='index' -> one row per element
        self.bus    = pd.DataFrame.from_dict(self.model['bus'], orient='index')
        self.gen    = pd.DataFrame.from_dict(self.model['gen'], orient='index')
        self.branch = pd.DataFrame.from_dict(self.model['branch'], orient='index')
        self.load   = pd.DataFrame.from_dict(self.model['load'], orient='index')
        self.shunt  = pd.DataFrame.from_dict(self.model.get('shunt', {}), orient='index')
        self.dcline = pd.DataFrame.from_dict(self.model.get('dcline', {}), orient='index')
        self.baseMVA = self.model.get('baseMVA', 100.0)
        self.dataIsProcessed = True

    def __str__(self) -> str:
        if self.dataIsProcessed:
            return (f"{self.name} ({self.hour}) has {len(self.bus)} buses, {len(self.branch)} branches, "
                     f"{len(self.gen)} gens, {len(self.load)} loads, {len(self.shunt)} shunts")
        return f"GridSFMOut '{self.name}' ({self.hour}) at modelpath='{self.modelpath.split('/')[-1]}'"

    def __repr__(self) -> str:
        return self.__str__()


def get_gridsfm_regions(cache_dir=None) -> dict:
    """ download (or read from HF cache) dataset_metadata.json, return its 'regions' dict:
        {region_name: {'abbreviation': 'RI'|None, 'type': 'state'|'region', 'states': [...] if region}}
        54 entries: 48 contiguous states + 6 multi-state regions (up to 'eastern', ~21,697 buses) """
    if not HAVE_HF_HUB:
        raise ImportError("pip install huggingface_hub to fetch GridSFM data")
    path = hf_hub_download(repo_id=GRIDSFM_REPO_ID, filename="dataset_metadata.json",
                            repo_type="dataset", cache_dir=cache_dir)
    with open(path) as f:
        meta = json.load(f)
    return meta['regions']


def get_gridsfm_paths(pathname, hour='16h') -> dict[str, GridSFMOut]:
    """ mirror of transnet_to_pp.get_transnet_paths: scan a local directory (already populated via
    download_gridsfm_model() below, or `hf download ... --local-dir pathname`) for
    '{hour}/{region}_model.json' files """
    gridsfm_data = defaultdict(GridSFMOut)
    search_dir = pathname.rstrip('/') + f'/{hour}/'
    for path in glob(search_dir + "*_model.json"):
        fname = path.split('/')[-1]
        region = fname[:-len('_model.json')]
        gridsfm_data[region].name = region
        gridsfm_data[region].hour = hour
        gridsfm_data[region].modelpath = path
    return gridsfm_data


def download_gridsfm_model(region: str, hour='16h', local_dir=None) -> str:
    """ download a single {region}_model.json from huggingface, return its local path.
    region must be a lowercase_with_underscores name from get_gridsfm_regions() keys
    (e.g. 'rhode_island', 'texas', 'eastern') -- NOT the two-letter abbreviation. """
    if not HAVE_HF_HUB:
        raise ImportError("pip install huggingface_hub to fetch GridSFM data")
    filename = f"{hour}/{region}_model.json"
    return hf_hub_download(repo_id=GRIDSFM_REPO_ID, filename=filename, repo_type="dataset", local_dir=local_dir)


### CONVERSION TO PANDAPOWER ###

def add_gridsfm_bus(net: aux.pandapowerNet, bus: pd.Series) -> None:
    """ add one GridSFM bus (fields: bus_i, base_kv, lat, lon, vmax, vmin, name) as a pandapower busbar.
    GridSFM already emits one bus per voltage level per substation (OSM-derived), so unlike
    transnet_to_pp's separate_voltage logic there's no multi-voltage-per-substation splitting to do here. """
    pp.create_bus(net, index=int(bus['bus_i']), name=bus.get('name', ''), vn_kv=float(bus['base_kv']),
                   geodata=(bus['lat'], bus['lon']), type='b',
                   max_vm_pu=float(bus['vmax']), min_vm_pu=float(bus['vmin']))


def add_gridsfm_gen(net: aux.pandapowerNet, gen: pd.Series, baseMVA: float, bus_types: dict) -> None:
    """ add one GridSFM generator (fields: gen_bus, pg, pmax, pmin, qg, qmax, qmin, vg, fuel_type, ...).
    slack=True iff its bus is the PowerModels.jl reference bus (bus_type==3) -- see gridsfm_to_pp's
    ext_grid fallback for the (rare) case where the reference bus has no generator attached. """
    is_slack = bus_types.get(int(gen['gen_bus'])) == 3
    pp.create_gen(net, bus=int(gen['gen_bus']), p_mw=float(gen['pg']) * baseMVA,
                   vm_pu=float(gen['vg']) if pd.notna(gen.get('vg')) else 1.0,
                   name=gen.get('name', ''), slack=is_slack, in_service=bool(gen.get('gen_status', 1)),
                   max_p_mw=float(gen['pmax']) * baseMVA, min_p_mw=float(gen['pmin']) * baseMVA,
                   max_q_mvar=float(gen['qmax']) * baseMVA, min_q_mvar=float(gen['qmin']) * baseMVA,
                   controllable=True, type=gen.get('fuel_type', 'unknown'), index=int(gen['index']))


def add_gridsfm_branch(net: aux.pandapowerNet, branch: pd.Series, baseMVA: float, bus_kv: dict) -> None:
    """ add one GridSFM branch as a pandapower line (transmission) or transformer.
    GridSFM ships per-unit impedances (br_r, br_x on system baseMVA), not physical ohms/km or a
    transformer-own-base vk_percent like pandapower expects, so we de-normalize using the standard
    z_base = kv^2/MVA identity before handing off to pandapower's *_from_parameters constructors. """
    f_bus, t_bus = int(branch['f_bus']), int(branch['t_bus'])
    in_service = bool(branch.get('br_status', 1))
    index = int(branch['index'])
    ckey = branch.get('circuit_key')
    name = f"branch_{index}" + (f"_{ckey}" if pd.notna(ckey) else '')

    if bool(branch.get('transformer', False)):
        hv_kv = float(branch['transformer_hv_kv'])
        lv_kv = float(branch['transformer_lv_kv'])
        hv_bus, lv_bus = (f_bus, t_bus) if bus_kv.get(f_bus, hv_kv) >= bus_kv.get(t_bus, lv_kv) else (t_bus, f_bus)
        # rate_a is 0 for a handful of branches in some regions; floor sn_mva to avoid a div-by-0 below
        sn_mva = max(float(branch.get('rate_a', 0.0)) * baseMVA, 0.1)
        # br_x/br_r are per-unit on system baseMVA; vk%/vkr% pandapower wants are on the transformer's
        # OWN sn_mva base -- standard base-conversion: Z_pu_own = Z_pu_system * (S_own/S_system)
        vk_percent = max(abs(float(branch['br_x'])) * (sn_mva / baseMVA) * 100.0, 0.01)
        vkr_percent = max(abs(float(branch['br_r'])) * (sn_mva / baseMVA) * 100.0, 0.0)
        pp.create_transformer_from_parameters(
            net, hv_bus=hv_bus, lv_bus=lv_bus, sn_mva=sn_mva, vn_hv_kv=hv_kv, vn_lv_kv=lv_kv,
            vk_percent=vk_percent, vkr_percent=vkr_percent, pfe_kw=0.0, i0_percent=0.0,
            shift_degree=math.degrees(float(branch.get('shift', 0.0))),
            tap_pos=0, in_service=in_service, name=name, index=index)
        return

    vn_kv = bus_kv.get(f_bus) or bus_kv.get(t_bus) or 1.0
    z_base = vn_kv ** 2 / baseMVA          # ohms
    y_base = baseMVA / vn_kv ** 2          # siemens
    length_km = float(branch.get('length_km') or 0.0)
    if length_km <= 0:
        # a few branches (short substation jumpers) have length_km == 0; concentrate the per-unit
        # impedance onto a notional 1 km line instead of dividing by zero to get an ohm/km figure
        length_km = 1.0
    r_ohm_per_km = float(branch['br_r']) * z_base / length_km
    x_ohm_per_km = float(branch['br_x']) * z_base / length_km
    b_total_siemens = (float(branch.get('b_fr', 0.0)) + float(branch.get('b_to', 0.0))) * y_base
    c_nf_per_km = max(b_total_siemens / (2 * math.pi * 60) * 1e9 / length_km, 0.0)  # 60 Hz US grid
    rate_mva = float(branch.get('rate_a', 0.0)) * baseMVA
    max_i_ka = max(rate_mva / (3 ** 0.5 * vn_kv), 1e-3) if vn_kv > 0 else 1e-3
    pp.create_line_from_parameters(
        net, from_bus=f_bus, to_bus=t_bus, length_km=length_km,
        r_ohm_per_km=r_ohm_per_km, x_ohm_per_km=x_ohm_per_km, c_nf_per_km=c_nf_per_km,
        max_i_ka=max_i_ka, name=name, in_service=in_service, index=index)


def add_gridsfm_load(net: aux.pandapowerNet, load: pd.Series, baseMVA: float) -> None:
    """ fields: load_bus, pd, qd, status """
    pp.create_load(net, bus=int(load['load_bus']), p_mw=float(load['pd']) * baseMVA,
                    q_mvar=float(load['qd']) * baseMVA, in_service=bool(load.get('status', 1)),
                    index=int(load['index']))


def add_gridsfm_shunt(net: aux.pandapowerNet, shunt: pd.Series, baseMVA: float) -> None:
    """ fields: shunt_bus, gs, bs, status.
    pandapower q_mvar convention: positive = absorbing (inductive/reactor), negative = injecting (capacitive).
    GridSFM bs convention (per dataset README): positive bs = capacitor, negative bs = reactor -- negate. """
    pp.create_shunt(net, bus=int(shunt['shunt_bus']), q_mvar=-float(shunt['bs']) * baseMVA,
                     p_mw=float(shunt.get('gs', 0.0)) * baseMVA, in_service=bool(shunt.get('status', 1)),
                     index=int(shunt['index']))


def add_gridsfm_dcline(net: aux.pandapowerNet, dcline: pd.Series, baseMVA: float) -> None:
    """ fields: f_bus, t_bus, pf, pt, vf, vt, pmaxf, pmaxt, br_status.
    UNTESTED -- no sample file inspected during development had any dclines. See module TODO. """
    pf = float(dcline.get('pf', 0.0)) * baseMVA
    pt = float(dcline.get('pt', 0.0)) * baseMVA
    loss_mw = max(pf - pt, 0.0)  # HVDC loss = power in - power out; floor at 0 in case of data noise
    ckey = dcline.get('circuit_key')
    name = f"dcline_{int(dcline['index'])}" + (f"_{ckey}" if pd.notna(ckey) else '')
    pp.create_dcline(net, from_bus=int(dcline['f_bus']), to_bus=int(dcline['t_bus']),
                      p_mw=pf, loss_percent=0.0, loss_mw=loss_mw, name=name,
                      vm_from_pu=float(dcline.get('vf', 1.0)), vm_to_pu=float(dcline.get('vt', 1.0)),
                      max_p_mw=max(float(dcline.get('pmaxf', 0.0)), float(dcline.get('pmaxt', 0.0))) * baseMVA,
                      in_service=bool(dcline.get('br_status', 1)), index=int(dcline['index']))


def gridsfm_to_pp(bus: pd.DataFrame, gen: pd.DataFrame, branch: pd.DataFrame, load: pd.DataFrame,
                   shunt: pd.DataFrame = None, dcline: pd.DataFrame = None,
                   baseMVA: float = 100.0) -> aux.pandapowerNet:
    """ convert one GridSFM model instance (already split into DataFrames, e.g. via
    GridSFMOut.read_data()) into a pandapower network. One pandapower bus per GridSFM bus_i,
    one pandapower line/transformer per branch (split on the 'transformer' flag). """
    net = pp.create_empty_network()

    for _, row in bus.iterrows():
        add_gridsfm_bus(net, row)
    bus_kv = bus.set_index('bus_i')['base_kv'].to_dict()
    bus_types = bus.set_index('bus_i')['bus_type'].to_dict()

    for _, row in gen.iterrows():
        add_gridsfm_gen(net, row, baseMVA, bus_types)

    for _, row in branch.iterrows():
        add_gridsfm_branch(net, row, baseMVA, bus_kv)

    for _, row in load.iterrows():
        add_gridsfm_load(net, row, baseMVA)

    if shunt is not None:
        for _, row in shunt.iterrows():
            add_gridsfm_shunt(net, row, baseMVA)

    if dcline is not None:
        for _, row in dcline.iterrows():
            add_gridsfm_dcline(net, row, baseMVA)

    # PowerModels.jl guarantees exactly one bus_type==3 (reference/slack) bus per connected model,
    # but it isn't guaranteed to have a generator attached at that bus -- pandapower needs an
    # explicit slack, so fall back to an ext_grid at the reference bus if no gen claimed slack=True.
    if len(net.gen) == 0 or not net.gen['slack'].any():
        slack_candidates = [b for b, t in bus_types.items() if t == 3]
        slack_bus = slack_candidates[0] if slack_candidates else int(bus.iloc[0]['bus_i'])
        pp.create_ext_grid(net, bus=slack_bus, vm_pu=1.0, name='gridsfm_fallback_slack')

    return net


if __name__ == '__main__':
    save_net = True
    plot = False
    cwd = os.getcwd()
    gridsfm_dir = cwd + "/data/gridsfm/"
    if not os.path.exists(gridsfm_dir):
        os.makedirs(gridsfm_dir)
    save_dir = cwd + '/data/ppnets/'
    if save_net and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # SELECT AN HOUR (peak 16h vs off-peak 04h)
    hour = GRIDSFM_HOURS[options_menu('hour (16h=peak, 04h=off-peak)', list(GRIDSFM_HOURS))]

    # SELECT A GRIDSFM REGION -- prefer the live 54-region catalog from huggingface if reachable,
    # else fall back to whatever *_model.json files are already cached locally in data/gridsfm/{hour}/
    regions = None
    if HAVE_HF_HUB:
        try:
            regions = sorted(get_gridsfm_regions().keys())
        except Exception as e:
            print(f'could not reach huggingface ({e}); falling back to local cache')
    if regions is None:
        local = get_gridsfm_paths(gridsfm_dir, hour)
        regions = sorted(local.keys())
        if not regions:
            print(f'no local GridSFM data found in {gridsfm_dir}{hour}/ and huggingface_hub unavailable/unreachable')
            print('pip install huggingface_hub, or manually download *_model.json files there')
            sys.exit(1)
    print('available GridSFM regions:', ', '.join(regions))
    region = regions[options_menu('region', regions)]

    # DOWNLOAD (if needed) + READ
    local_model_path = f'{gridsfm_dir}{hour}/{region}_model.json'
    if not os.path.exists(local_model_path):
        if not HAVE_HF_HUB:
            print(f'{local_model_path} not found locally and huggingface_hub not installed')
            sys.exit(1)
        print(f'downloading {region} ({hour}) from huggingface...')
        local_model_path = download_gridsfm_model(region, hour)  # caches under ~/.cache/huggingface

    gridsfm_region = GridSFMOut(name=region, hour=hour, modelpath=local_model_path)
    gridsfm_region.read_data()
    print(gridsfm_region)

    print_info = True
    if print_info:
        print("\n\nbus columns:", list(gridsfm_region.bus.columns))
        print(gridsfm_region.bus.head())
        print("\n\nbranch columns:", list(gridsfm_region.branch.columns))
        print(gridsfm_region.branch.head())

    net = gridsfm_to_pp(gridsfm_region.bus, gridsfm_region.gen, gridsfm_region.branch,
                         gridsfm_region.load, gridsfm_region.shunt, gridsfm_region.dcline,
                         gridsfm_region.baseMVA)
    print(net)
    if save_net:
        name = f'{save_dir}gridsfm-{region}-{hour}.db'
        pp.to_sqlite(net, name, include_results=False)
        print('saved sqlite to ', name)

    if plot:
        # Plotly for all plotting (see repo CLAUDE.md). Do NOT hardcode a Mapbox token; set
        # MAPBOX_TOKEN yourself for the on_map basemap (on_map=False needs no token).
        token = os.environ.get('MAPBOX_TOKEN')
        if token:
            ppl.set_mapbox_token(token)
        ppl.simple_plotly(net, on_map=bool(token), use_line_geodata=False, figsize=1,
                          auto_open=True).show()
