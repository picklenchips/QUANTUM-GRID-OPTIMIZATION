# the following is equivalent to the multivoltage example
import sys,os
import pandas as pd
import pandapower as pp
import pandapower.auxiliary as aux  # for pandapowerNet typing
import pandapower.plotting as ppplot
import pandapower.networks as ppnet
import plotly.express as px
import matplotlib.pyplot as plt
mapBoxToken = '***REMOVED-MAPBOX-TOKEN***'
publicToken = '***REMOVED-MAPBOX-TOKEN***'
allAccess = '***REMOVED-MAPBOX-TOKEN***'
#px.set_mapbox_access_token(token2)
fname = '/opt/miniconda3/envs/qpgrid/lib/python3.11/site-packages/pandapower/plotting/plotly/mapbox_token.txt'
if os.path.exists(fname):
    os.remove(fname)
with open(fname, 'w+') as f:
    f.write(mapBoxToken)

# netv = pandapower.networks.example_multivoltage()
nbusses = 3
def create_minimal_example(nbusses=3):
    net = pp.create_empty_network()
    # power plant
    planti = pp.create_bus(net, name = "110 kV plant", vn_kv = 110, type = 'b')
    pp.create_gen(net, planti, p_mw = 100, vm_pu = 1.0, name = "diesel gen")
    i = pp.create_bus(net, vn_kv = 110, type='n', name='lithium ion storage')
    pp.create_storage(net, i, p_mw = 10, max_e_mwh = 20, q_mvar = 0.01, name = "battery")
    pp.create_line(net, name = "plant to storage", from_bus = 0, to_bus = 1, length_km = 0.1, std_type = "NAYY 4x150 SE")
    # external grid
    exti = pp.create_bus(net, name = "110 kV bar out", vn_kv = 110, type = 'b')
    pp.create_ext_grid(net, exti, vm_pu = 1)
    pp.create_line(net, name = "plant to out", from_bus = planti, to_bus = exti, length_km = 2, std_type = "NAYY 4x150 SE")
    pp.create_switch(net, bus = planti, element = exti, et = 'b', closed = True)
    # city
    cityi = pp.create_bus(net, name = "110 kV city bar", vn_kv = 110, type = 'b')
    pp.create_line(net, name = "plant to city", from_bus = planti, to_bus = cityi, length_km = 1.5, std_type = "NAYY 4x150 SE")
    pp.create_switch(net, bus = planti, element = cityi, et = 'b', closed = True)
    # neighborhood
    neighbori = pp.create_bus(net, name = "20 kV bar", vn_kv = 20, type = 'b')
    previ = neighbori
    i = pp.create_transformer_from_parameters(net, hv_bus=cityi, lv_bus=neighbori, i0_percent=0.038, pfe_kw=11.6,
                                        vkr_percent=0.322, sn_mva=40, vn_lv_kv=22.0, vn_hv_kv=110.0, 
                                        vk_percent=17.8, name='city to n1 trafo')
    pp.create_switch(net, bus = cityi, element = i, et = 't', closed = True)
    # add 2 sections
    for i in range(nbusses):
        newi = pp.create_bus(net, name = f"bus {i+2}", vn_kv = 20, type = 'b')
        pp.create_line(net, name = f"line {previ}-{newi}", from_bus = previ, to_bus = newi, length_km = 0.3, std_type = "NAYY 4x150 SE")
        pp.create_load(net, newi, p_mw = 1, q_mvar = 0.2, name = f"load {newi}")
        previ = newi
    sec1i = newi
    previ = neighbori
    for i in range(nbusses):
        newi = pp.create_bus(net, name = f"bus {i+2+nbusses}", vn_kv = 20, type = 'b')
        pp.create_line(net, name = f"line {previ}-{newi}", from_bus = previ, to_bus = newi, length_km = 0.3, std_type = "NAYY 4x150 SE")
        pp.create_load(net, newi, p_mw = 1, q_mvar = 0.2, name = f"load {newi}")
        previ = newi
    # connect the 2 sections at the end
    i = pp.create_line(net, name = f"line {previ}-{sec1i}", from_bus = previ, to_bus = sec1i, length_km = 0.2, std_type = "NAYY 4x150 SE")
    pp.create_switch(net, bus = previ, element = i, et = 'l', closed = False)
    return net

if __name__ == '__main__':
    while 1:
        t = input('which example to try? (minimal, x)').rstrip().lower()
        if t == 'minimal':
            net = create_minimal_example(nbusses=3)
            print(net)
            print(net['bus'].head())
            ppplot.simple_plot(net, plot_loads = True, plot_gens=True)
            ppplot.simple_plotly(net)
        else:
            net = ppnet.mv_oberrhein(include_substations=True)
            print(net)
            ppplot.simple_plot(net, plot_loads = True, plot_gens=True)
            ppplot.simple_plotly(net, bus_size=1, aspectratio="original")