import sys,os
import pandas as pd
import pandapower as pp
import pandapower.auxiliary as aux  # for pandapowerNet typing
import pandapower.plotting.plotly as ppl
pd.options.display.max_rows = 20
from glob import glob
from collections import defaultdict

# import modules from enclosing directory
pdir = sys.path[0] + '/..'
if pdir not in sys.path:
    sys.path.append(pdir)
from util import options_menu

### Siemens PSS/E DATA CONVERSIONS ###

# VIA PSS-E v30 user manual

# all measured in power, but refer to different things
# MW :  active power
# MVar : reactive power - megavolt-amperes reactive
# MVA - megavolt-amperes - total apparent power
# pu = define system in terms of a base power/voltage, and then express all other powers as fractions of that base power/voltage
col_defs = {
# bus data
    'baskv': 'bus base voltage (kV)',
    'gl': 'active shunt admittance (MW/V)',
    'bl': 'reactive shunt admittance (Mvar/V)',
    'vm': '|V| (pu)',
    'va': 'V angle (deg)',
# load data
    'pl': 'active power of MVA load (MW)',
    'ql': 'reactive power of MVA load (Mvar)',
    'ip': 'active power of I load (MW,pu)',
    'iq': 'reactive power of I load (Mvar,pu)',
    'yp': 'active power of shunt (MW,pu)',
    'yq': 'reactive power of shunt (Mvar,pu)',
# generator data
    'pg': 'active power of generator (MW)',
    'qg': 'reactive power of generator (Mvar)',
    'qt': 'maximum reactive power of generator (Mvar)',
    'qb': 'minimum reactive power of generator (Mvar)',
    'vs': 'voltage set point of generator (pu)',
    'ireg': 'voltage regulator status of generator',
    'mbase': 'base MVA of generator', # required for switchin studies, fault analysis, and dynamic simulation
    'zr': 'generator impedance real part (pu)',
    'zx': 'generator impedance imaginary part (pu)',
    'rt': 'step-up transformer impedance real part (pu)',
    'xt': 'step-up transformer impedance imaginary part (pu)',
    'gtap': 'transformer off-nominal turns ratio',
    'stat': 'machine status', # 1=in-service, 0=out-of-service
    'pt': 'maximum active power of generator (MW)',
    'pb': 'minimum active power of generator (MW)',
    'o[i]': 'owner number',
    'f[i]': 'fraction of ownership',
# branch data
    'r': 'resistance (pu)',   # Re(Z)
    'x': 'reactance (pu)',    # Im(Z)
    'b': 'susceptance (pu)',  # Im(Y)
    'rate-[i]': 'loading rating (MVA)', # entered as sqrt(3)*E*I*10^-6
    'gi': 'resistance of first bus (pu)',
    'bi': 'reactance of first bus (pu)',
    'gj': 'resistance of second bus (pu)',
    'bj': 'reactance of second bus (pu)',
    'len': 'line length (user-selected units)',
# transformer stuff
    'windv3': 'winding 3 voltage ratio (pu)',
    'nomv3': 'winding 3 nominal voltage (kV)',
    'ang3': 'winding 3-phase shift angle (deg)',
    'rate3-[i]': 'winding 3 power rating (MVA)',
    # 0=no cntrl, +-1 for voltage, +-2 for reactive power, +-3 for active power
    'cod3': 'transformer control mode during PF', 
    'cont3': 'number of bus to be  controlled by transformer',
    'rma3': 'upper limits of tap ratio | 3-voltage (kV) | shift angle',
    'rmi3': 'lower limits of tap ratio | 3-voltage (kV) | shift angle',
    'vma3': 'upper limit of voltage (pu)',
    'vmi3': 'lower limit of voltage (pu)',
    'ntp3': 'number of tap positions available when COD3 is 1|2',
    'tab3': 'index of transformer impedance correction table if impedence is a function of either tap ratio or phase shift angle',
    'cr3': 'load drop compensation resistance (pu)',
    'cx3': 'load drop compensation reactance (pu)',
}

def get_psse_paths(pathname) -> list[str]:
    found = defaultdict(int)
    ret = []
    for path in glob(pathname+'*'):
        if os.path.isdir(path):
            ret.extend(get_psse_paths(path+'/'))
            continue
        name = path.split('/')[-1]
        if '.' not in name:
            continue
        rpath, ext = tuple(path.split('.'))
        print(name, ext)
        if ext == 'raw':
            if found[rpath] in (0,2):
                found[rpath] += 1
        elif ext == 'dyr':
            if found[rpath] in (0,1):
                found[rpath] += 2
    print(ret)
    for k, v in found.items():
        if v == 3:
            ret.append(k)
    return ret


def parse_to_vars(parts: list[str]) -> None:
    """modifies parts --> list[str | int | float], converting from str whenever possible"""

    for i in range(len(parts)):
        parts[i] = parts[i].strip()
        # strings are encoded as 'STR     '
        if parts[i][0] == "'":
            parts[i] = parts[i][1:-1].strip()
        try:
            parts[i] = int(parts[i])  # type: ignore
        except:
            try:
                parts[i] = float(parts[i])  # type: ignore
            except:
                pass


def PSS_raw_reader(filepath, printInfo=False) -> dict[str, pd.DataFrame]:
    with open(filepath, 'r') as f:
        lines = f.readlines()
    DATA = {}
    # might be completely incorrect, idk
    gen_titles = ['I','Name','Gen MW','Gen MVar','Set volt','min MW','Max MW','Min mVar','Max mvar']
    bus_titles=['ID','NAME','BASKV', # bus base voltage
                'IDE', # bus type code. 
    #1=load bus, 2=gen bus, 3=swing bus (V,º), 4=isolated bus
                'AREA', 'ZONE','OWNER',
                'GL',  # MW/V 
                'BL', # Mvar, 1/V. positive for capacitor, negative for reactor or inductor
                'VM','VA','VM1','VA1']
    # just from what I've seen
    sections = ['bus', 'load', 'fixed shunt','generator','branch','system switching device','transformer','area','two-terminal dc','vsc dc','impedance correction','multi-terminal dc','multi-section line','zone','inter-area transfer','owner','facts device','switched shunt','gne','induction machine','substation']
    section = 'system'
    ret = ''
    i = -1
    rowsPerEntry = 0
    allTitles = []  # multi-line entries
    while i + 1 < len(lines):
        i += 1
        line = lines[i]

        # SECTION CHANGE
        if line.startswith("0 / "):
            msgs = line[4:].split(',')
            old = msgs[0][7:]  # skip "END OF "
            if len(ret):
                print(old+'\n'+ret)
            ret = ''
            if len(msgs) != 2:
                # we have reached the end -> no new section
                break
            new = msgs[1][7:].lower() # skip " BEGIN "
            section = new[:-6]  # skip " DATA"
            # print('now on section '+section)
            allTitles = [] # reset titles
            rowsPerEntry = 0
            entryIndex = 0
            # set column titles
            if section == 'bus':  # for some reason only the bus section has no titles
                allTitles = [bus_titles]
                rowsPerEntry = 1
            continue
        elif lines[i].startswith('@!'):
            rowsPerEntry += 1
            titles = lines[i][2:].rstrip().split(',')
            # overcome the wierd string formatting they have
            for j in range(len(titles)):
                titles[j]= titles[j].strip()
                if titles[j][0] == "'":
                    titles[j] = titles[j][1:-1].strip()
            allTitles.append(titles)
        elif section == 'system':
            # we don't care about eanything in the intial specifications,
            # so let's just print em all
            ret += line
        else:
            # PARSE ROW OF DATA
            parts = line.split(',')
            parse_to_vars(parts)
            # ADD TO DATA
            # some data is not specified (like generator data) but DataFrames must take
            # columns that have the same length, so we must fill in the blanks with None if so
            if entryIndex == 0:
                entry = {}
            # print(allTitles, entryIndex, rowsPerEntry)
            titles = allTitles[entryIndex]
            for j in range(len(parts)):
                if j < len(titles):
                    entry[titles[j]] = parts[j]
                else:
                    entry[str(j)] = parts[j]
            entryIndex += 1
            if entryIndex == rowsPerEntry:
                DATA[section] = DATA.get(section, [])
                DATA[section].append(entry)
                entryIndex = 0
    # convert all dictionaries to pandas dataframes
    for section, data in DATA.items():
        DATA[section] = pd.DataFrame(data)
    if printInfo:
        for section, frame in DATA.items():
            print(f"{len(frame)} {section}(s) w/ {len(frame.columns)} columns {', '.join(frame.columns)}")
    return DATA


def PSS_dyr_reader(filepath, printInfo=False) -> pd.DataFrame:
    # lines are separated by "/"
    with open(filepath, 'r') as f:
        lines = f.readlines()
    DATA = []
    titles = ['id','model','letter']
    parts = []  # to store a single data entry
    i = -1
    while i + 1 < len(lines):
        i += 1
        line = lines[i].strip()
        # starting messages
        if line.startswith("//"):
            print(line)
            continue
        # SECTION CHANGE
        if line.startswith("/"):
            continue
        # section 1 stores all generator components probably
        # PARSE LINE INTO PARTS
        atContent = finalPart = False
        j = 0
        while j < len(line):
            if line[j] == "/":
                finalPart = True
            elif line[j] in (' ',','):
                atContent = False
            else:
                if not atContent:
                    parts.append('')
                parts[-1] += line[j]
                atContent = True
            j += 1
        # ADD TO DATA IF DONE WITH TING
        if finalPart:
            parse_to_vars(parts)
            # data is id, model name, single-letter, then variables
            entry = {titles[j]: parts[j] for j in range(3)}
            entry['vars'] = parts[3:]
            DATA.append(entry)
            parts = []  # reset parts
    print('found',len(DATA),'entries in dyr file')
    DYR = pd.DataFrame(DATA).sort_values('id')
    if printInfo:
        for col in titles:
            uniques = DYR[col].unique()
            print(f"{len(uniques)} unique {col}s: {', '.join(map(str,uniques))}")
    return DYR

def raw_to_pp(RAW: dict[str, pd.DataFrame]) -> aux.pandapowerNet:
    """ convert PSS/E raw data (as parsed by PSS_raw_reader above) into a pandapower network,
    following the same "buses first, then loads/gens, then branches as lines" pattern as
    transnet_to_pp() in transnet_to_pp.py.

    Column names below were confirmed against the real WECC 240-bus test case
    (data/psse/WECC_240bus_2018summer_2021_IEEE-NASPI_OSL/240busWECC_2018_PSS.raw) by actually
    running PSS_raw_reader() on it -- NOT guessed from col_defs above (col_defs is just a field
    glossary; the real DataFrame columns are upper-case abbreviations like 'BASKV', 'PL', 'PG').

    Known simplifications:
     - Only the 'bus', 'load', 'generator', and 'branch' sections are converted. The
       'transformer' section (voltage-changing branches between different-BASKV buses) is NOT
       turned into pandapower trafo elements -- out of scope for this pass. In the WECC test
       file this means buses that are only reachable via a transformer (e.g. the 500/345/230kV
       levels of a multi-voltage substation) end up topologically disconnected from the 'line'
       elements built here, even though the bus objects themselves (with their loads/gens) are
       all still created correctly.
     - PSS/E branch R/X/B are lumped per-unit impedances for the whole branch (on the system
       MVA base, from the .raw SYSTEM-WIDE DATA header), not per-unit-length values, and this
       file's LEN/RATE1-12 fields are all 0 (no physical length or thermal rating given). Each
       branch is therefore modeled as a 1 km pandapower line whose std_type r/x/c *per km*
       equal that branch's *total* impedance (length_km=1) -- the standard trick for shoehorning
       a lumped-impedance branch into pandapower's per-km line model without inventing a length.
       max_i_ka has no source data at all in this file, so a generous placeholder (5 kA, i.e.
       effectively non-binding) is used; do not trust thermal-loading results from this network.
     - PSS/E's raw format models each load as a ZIP load: PL/QL (constant-MVA), IP/IQ
       (constant-current), YP/YQ (constant-admittance), all already expressed in MW/Mvar at
       rated voltage. In the WECC file every load's PL/QL is 0 and the actual demand lives in
       IP/YQ instead, so summing just PL+QL would create zero loads. All three ZIP components
       are summed into a single constant-MVA-equivalent p_mw/q_mvar per row (what
       pp.create_load models), which reproduces the file's real 139/139 nonzero loads.
     - Exactly one bus has IDE==3 (PSS/E's "swing"/slack bus code). Of the generators at that
       bus, only the single largest one (by PT, max active power) is marked slack=True --
       marking every generator at that bus as slack would over-determine the network's slack
       reference and isn't needed just to build the pandapower object.
    """
    net = pp.create_empty_network()
    if 'bus' not in RAW or not len(RAW['bus']):
        return net
    bus = RAW['bus']
    Sbase = 100.0  # MVA system base, from the .raw file's SYSTEM-WIDE DATA header (field 2)

    # 1. buses
    for _, row in bus.iterrows():
        pp.create_bus(net, index=int(row['ID']), name=row['NAME'], vn_kv=float(row['BASKV']),
                       zone=row['ZONE'], type='b')

    # 2. loads -- sum the PSS/E ZIP load components into one constant-MVA equivalent per row
    #    (see docstring: this file's demand is carried by IP/YQ, not PL/QL)
    if 'load' in RAW:
        for _, row in RAW['load'].iterrows():
            p_mw = float(row['PL']) + float(row['IP']) + float(row['YP'])
            q_mvar = float(row['QL']) + float(row['IQ']) + float(row['YQ'])
            if p_mw == 0 and q_mvar == 0:
                continue
            pp.create_load(net, bus=int(row['I']), p_mw=p_mw, q_mvar=q_mvar,
                            name=f"load {row['I']}-{row['ID']}", in_service=bool(row['STAT']))

    # 3. generators
    if 'generator' in RAW:
        gen = RAW['generator']
        swing_bus_ids = set(bus.loc[bus['IDE'] == 3, 'ID']) if 'IDE' in bus.columns else set()
        # pick a single slack generator per swing bus (the largest by max active power PT)
        slack_rows = set()
        for swing_id in swing_bus_ids:
            at_bus = gen[gen['I'] == swing_id]
            if len(at_bus):
                slack_rows.add(at_bus['PT'].astype(float).idxmax())
        for idx, row in gen.iterrows():
            pp.create_gen(net, bus=int(row['I']), p_mw=float(row['PG']), vm_pu=float(row['VS']),
                          name=f"gen {row['I']}-{row['ID']}", slack=(idx in slack_rows),
                          in_service=bool(row['STAT']), controllable=True,
                          max_p_mw=float(row['PT']), min_p_mw=float(row['PB']),
                          max_q_mvar=float(row['QT']), min_q_mvar=float(row['QB']))

    # 4. branches -> lines
    if 'branch' in RAW:
        vn_kv_of_bus = bus.set_index('ID')['BASKV']
        for _, row in RAW['branch'].iterrows():
            i, j = int(row['I']), int(row['J'])
            v = vn_kv_of_bus.get(i, vn_kv_of_bus.get(j))
            if not v:
                continue  # no base kv available on either end -- can't derive ohms from pu
            zbase = v ** 2 / Sbase   # ohms
            ybase = Sbase / v ** 2   # siemens
            r_ohm = float(row['R']) * zbase
            x_ohm = float(row['X']) * zbase
            # susceptance (pu) -> capacitance (nF), assuming 60 Hz like the rest of WECC
            c_nf = float(row['B']) * ybase / (2 * 3.141592653589793 * 60) * 1e9
            typedata = {'r_ohm_per_km': r_ohm, 'x_ohm_per_km': x_ohm, 'c_nf_per_km': c_nf,
                        'max_i_ka': 5.0}  # placeholder -- RATE1-12 are all 0 in this file
            fitting_types = pp.find_std_type_by_parameter(net, typedata, element='line', epsilon=0.001)
            if len(fitting_types):
                std_type = fitting_types[0]
            else:
                std_type = f"branch {i}-{j}-{row['CKT']}"
                pp.create_std_type(net, typedata, std_type)
            pp.create_line(net, from_bus=i, to_bus=j, length_km=1.0, std_type=std_type,
                            name=f"branch {i}-{j}-{row['CKT']}", in_service=bool(row['STAT']))
    return net


if __name__ == '__main__':
    save_net = False
    plot = False
    
    cwd = os.getcwd()
    psse_dir = cwd + "/data/psse/"
    if not os.path.exists(psse_dir):
        os.makedirs(psse_dir)
    save_dir = cwd + '/data/ppnets/'
    if save_net and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # SELECT A PSS/E DATA FILE (example IEEE systems)
    path = psse_dir
    validpaths = get_psse_paths(path)  # without extension
    fnames = list(map(lambda x: x.split('/')[-1], validpaths))
    print('found valid files:',', '.join(fnames))
    path = validpaths[options_menu('file', fnames)]
    print('\nloading raw file')
    RAW = PSS_raw_reader(path+'.raw', True)
    print('\nloading dyr file')
    DYR = PSS_dyr_reader(path+'.dyr', True)
    # TODO: implement this conversion
    net = raw_to_pp(RAW)
    if save_net:
        pp.to_sqlite(net, save_dir+path.split('/')[-1]+'.db')
    
    if plot:
        # Plotly for all plotting (see repo CLAUDE.md). on_map=True needs MAPBOX_TOKEN in the
        # environment (a token was previously hardcoded here + committed to a public repo --
        # compromised); on_map=False works with no token.
        token = os.environ.get('MAPBOX_TOKEN', '')
        if token:
            ppl.set_mapbox_token(token)
        ppl.simple_plotly(net, on_map=bool(token), use_line_geodata=False, figsize=1,
                          auto_open=True).show()
        # , on_map=True, use_line_geodata=False, filename='temp-plot.html')
