import sys,os
import pandas as pd
import pandapower as pp
import pandapower.auxiliary as aux  # for pandapowerNet typing
import pandapower.plotting as ppplot
pd.options.display.max_rows = 20
from glob import glob
from collections import defaultdict

# import modules from enclosing directory
pdir = sys.path[0] + '/..'
if pdir not in sys.path:
    sys.path.append(pdir)
from util import timeIt, Ith


### TRANSNET DATA (.csvs) ###

# store a transnet data instance, which is created from reading two .csv files
class TransnetOut():
    def __init__(self, name='', linepath='', nodepath='', read=False):
        """ input paths are paths to .csv files downloaded from transnet """
        self.name = name
        self.linepath = linepath
        self.nodepath = nodepath
        self.dataIsProcessed = False
        if read:
            self.read_data()
    
    def read_data(self, linepath='', nodepath=''):
        """ read the data from the csv files into pandas.Dataframes """
        if not linepath: linepath = self.linepath
        if not nodepath: nodepath = self.nodepath
        self.lines = pd.read_csv(linepath)
        self.nodes = pd.read_csv(nodepath)
        self.dataIsProcessed = True

    def __str__(self) -> str:
        if self.dataIsProcessed:
            ret = f"{self.name} has {len(self.lines)} lines and {len(self.nodes)} nodes"
            return ret
        return f"{self.name} at linepath='{self.linepath.split('/')[-1]}' and nodepath='{self.nodepath.split('/')[-1]}'"

    def __repr__(self) -> str:
        return self.__str__()

# get transnet data from folder
def get_transnet_paths(pathname) -> dict[str,TransnetOut]:
    transnet_data = defaultdict(TransnetOut)
    for path in glob(pathname + "*.csv"):
        name = path.split('/')[-1]
        realname = '_'.join(name.split('_')[:-1])  # like "california"
        if 'lines' in name:
            transnet_data[realname].name = realname
            transnet_data[realname].linepath = path
        elif 'nodes' in name:
            transnet_data[realname].name = realname
            transnet_data[realname].nodepath = path
    return transnet_data

def add_transnet_bus(net: aux.pandapowerNet, node: pd.Series|dict, separate_voltage=False) -> None:
    """ add a transnet node to the pandapower network """
    voltage = node['voltage']  # string of voltages like "115000;12000"
    if ';' in voltage:  # get maximum voltage if multiple
        voltage = max(map(int,voltage.split(';')))
    else:
        voltage = int(voltage)
    pp.create_bus(net, index=node['n_id'], name=node['name'], 
                  vn_kv=voltage, node_type=node['type'],
                  geodata=(node['latitude'], node['longitude']), type='b' )  #is busbar

def create_offshoot_bus(net: aux.pandapowerNet, bus_id: int, new_voltage: float) -> int:
    """ offshoot a new lower-voltage bus from an existing bus, adding a busbar and transformer 
      - shift the geodata of the new bus by a tiny bit east """
    bus = net['bus'].loc[bus_id].copy()
    bus['vn_kv'] = new_voltage
    # shift bus geodata by a tiny bit
    old_geo = net['bus_geodata'].loc[bus_id]
    bus['geodata'] = (old_geo['x']+0.005, old_geo['y'])
    new_id = int(pp.create_bus(net, **bus))
    # add a transformer from the highest bus to this new bus
    pp.create_transformer(net, hv_bus=bus_id, lv_bus=new_id, std_type='0.4 MVA 20/0.4 kV')
    return new_id

def add_transnet_gen(net: aux.pandapowerNet, node: pd.Series, p_mw=10) -> None:
    # add voltage-controlled PV node - we specify active power and voltage magnitude
    # if you want to specify only active + reactive power, use sgen instead
    pp.create_gen(net, node['n_id'], p_mw=p_mw, vm_pu=1.0, name=node['name'], 
                  slack=False, type='unkown', controllable=True, 
                  max_p_mw=p_mw, min_p_mw=0, max_q_mvar=0.01, min_q_mvar=0)

def transnet_to_pp(nodes: pd.DataFrame, lines: pd.DataFrame, separate_voltage=False) -> aux.pandapowerNet:
    """ convert transnet data to power plant data 
    - adds all nodes first
       - substations with more than one voltage are added as one bus with the highest voltage a
    if separate_voltage, substations with multiple voltages are realistically treated by:
     - creating one long busbar for the highest voltage
     - for each lower voltage, creating a new small bus and connect it to the higher voltage bus
       via a transformer
     - keep track of the equivalences using a dict of 
       {original_id: {highest: original_id, lower1: bus_id1, lower2: bus_id2, ...},
       {} ...}
    """
    net = pp.create_empty_network()
    # create all nodes
    for i, node in nodes.iterrows():
        add_transnet_bus(net, node, separate_voltage)
        if node['type'] == 'substation':
            pass
        elif node['type'] == 'generator':
            # just assume all generators are 10MW for now
            add_transnet_gen(net, node, 10)
        elif node['type'] == 'plant':
            # just assume all plants are 50MW for now
            add_transnet_gen(net, node, 50)
    
    #  add all transmission lines
    multiV = {}
    for i, line in lines.iterrows():
        # freq = line['frequency'] if line['frequency'] else 60.0
        # type is either line | cable
        # don't care about freq = line['frequency'] if line['frequency'] else 60.0
        # don't care about operator
        V = line['voltage']
        start = line['n_id_start']
        end = line['n_id_end']
        
        for j in (start, end):
            # keep track of the multiple voltages present in a substation
            if net['bus'].loc[j,'vn_kv'] < V:
                if separate_voltage:
                    if j not in multiV:
                        multiV[j] = {}
                    # new voltage is now the highest and takes the original index
                    multiV[j][V] = j
                    # if old voltage is zero, it has no connected lines yet
                    # if bus had some old voltage, it may have connected lines
                    if net['bus'].loc[j,'vn_kv'] != 0:
                        lowerV = net['bus'].loc[j,'vn_kv']
                        # make a new bus for the lower voltage by copying the old bus
                        multiV[j][lowerV] = create_offshoot_bus(net, j, lowerV)
                        # update any lines connected to this bus in the past
                        # to be connected to the new, lower voltage bus
                        for bus in ('from_bus', 'to_bus'):
                            for l in net['line'][net['line'][bus] == j].index:
                                net['line'].loc[l,bus] = multiV[j][lowerV]
                # update bus to hold the highest voltage among all cables
                net['bus'].loc[j,'vn_kv'] = V
            elif net['bus'].loc[j,'vn_kv'] > V and V:
                if separate_voltage:
                    if j not in multiV:
                        multiV[j] = {net['bus'].loc[j,'vn_kv']: j}
                    # add line to lower bus index
                    lower_bus = multiV[j].get(V, None)
                    if not lower_bus:  # need to create the lower bus index
                        lower_bus = create_offshoot_bus(net, j, V)
                        multiV[j][V] = lower_bus
                    if j == start: 
                        start = lower_bus
                    else:
                        end = lower_bus
        
        cables = line['cables'] if line['cables'] else 1
        # get or create standard type from line data
        std_type = "NAYY 4x150 SE"
        if line['r_ohm_km'] and line['x_ohm_km'] and line['c_nf_km']:
            typedata = {'r_ohm_per_km': line['r_ohm_km'], 'x_ohm_per_km': line['x_ohm_km'], 'c_nf_per_km': 0.0, 'max_i_ka': line['i_th_max_km']}
            fitting_types = pp.find_std_type_by_parameter(net, typedata, element='line', epsilon=0.001)
            if len(fitting_types):
                std_type = fitting_types[0]
            else:
                std_type = f"line {line['l_id']}"
                pp.create_std_type(net, typedata, std_type)
        id = pp.create_line(net, from_bus=start, to_bus=end, length_km=line['length_m']/1000, std_type=std_type,
                            vn_kv=line['voltage']/1000, parallel=cables, name=line['name'], index=line['l_id'])
    return net


if __name__ == '__main__':
    # create a pandapower network from transnet data
    # get data directory, or make if not there
    cwd = os.getcwd()
    transnet_dir = cwd + "/data/transnet/"
    if not os.path.exists(transnet_dir):
        os.makedirs(transnet_dir)
    transnet_data = get_transnet_paths(transnet_dir)
    print('found data for regions',', '.join([f for f in transnet_data.keys()]))
    while 1:
        region = input('which region would you like to load?\n>>').rstrip().lower()
        if region.isdigit():
            try:
                region = [f for f in transnet_data.keys()][int(region)]
            except IndexError:
                print('please enter a valid region index 0-{}'.format(len(transnet_data)-1))
                continue
        if region in transnet_data.keys():
            break
        print('please enter a valid region')
    transnet_region = transnet_data[region]
    transnet_region.read_data()
    
    print(transnet_region)
    print_info = True
    if print_info:
        n = transnet_region.nodes
        l = transnet_region.lines
        print("\n\nnodes")
        for k in n.columns:
            if k in ('type','voltage','frequency'):
                print(k, ', '.join(map(str,n[k].unique())))
        print(n.head())
        print("\n\nlines")
        for k in l.columns:
            if k in ('type','voltage','cables','frequency', 'r_ohm_km'):
                print(k, ', '.join(map(str,l[k].unique())))
        print(l.head())
    
    t = input('turn region into high-def network? (y/n)\n>>').rstrip().lower()
    separate_voltage = 'y' in t
    net = transnet_to_pp(transnet_region.nodes, transnet_region.lines, separate_voltage)
    print(net)
    ppplot.simple_plot(net, plot_gens=True, plot_loads=True)
    #ppplot.simple_plotly(net, on_map=True, use_line_geodata=False, figsize=1)
    # , on_map=True, use_line_geodata=False, filename='temp-plot.html')
