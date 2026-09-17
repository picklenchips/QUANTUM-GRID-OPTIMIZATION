# %% Pandapower — electrical grid modeling
"""
Most extensive Python package for electrical power systems formulation.
pandapipes also models heat and gas energy flows.
Can implement any electrical grid topologies and model busses interacting with an external grid.
"""

import pandapower as pp
import pandapower.networks as ppnet
import pandapower.topology as pptop
import pandapower.plotting as ppplot
import pandapower.converter as ppconv
import pandapower.estimation as ppest
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import numpy as np

# %% Create a simple network with transformer, external grid
nbusses = 5

net = pp.create_empty_network()

# power plant
pp.create_gen(net, 0, p_mw=100, vm_pu=1.0, name="power plant")
pp.create_bus(net, name="110 kV plant", vn_kv=110, type='b')
pp.create_bus(net, name="110 kV bar", vn_kv=110, type='b')
pp.create_bus(net, name="20 kV bar", vn_kv=20, type='b')

for i in range(nbusses):
    pp.create_bus(net, name=f"bus {i+2}", vn_kv=20, type='b')

pp.create_ext_grid(net, 0, vm_pu=1)

# transmission lines
pp.create_line(net, name="line 0", from_bus=0, to_bus=1, length_km=0.6, std_type="NAYY 4x150 SE")
pp.create_line(net, name="line 0", from_bus=1, to_bus=2, length_km=0.6, std_type="NAYY 4x150 SE")
pp.create_line(net, name="line 1", from_bus=2, to_bus=3, length_km=1, std_type="NAYY 4x150 SE")
pp.create_line(net, name="line 2", from_bus=3, to_bus=4, length_km=1, std_type="NAYY 4x150 SE")
pp.create_line(net, name="line 3", from_bus=4, to_bus=5, length_km=1, std_type="NAYY 4x150 SE")
pp.create_line(net, name="line 4", from_bus=5, to_bus=6, length_km=1, std_type="NAYY 4x150 SE")
pp.create_line(net, name="line 5", from_bus=6, to_bus=1, length_km=1, std_type="NAYY 4x150 SE")

# transformer
pp.create_transformer_from_parameters(
    net, hv_bus=0, lv_bus=1, i0_percent=0.038, pfe_kw=11.6,
    vkr_percent=0.322, sn_mva=40, vn_lv_kv=22.0, vn_hv_kv=110.0,
    vk_percent=17.8
)

# switches on lines
for i in range(nbusses):
    pp.create_switch(net, bus=i, element=i, et='l')

pp.create_switch(net, bus=1, element=0, et='l')
pp.create_switch(net, bus=2, element=0, et='l')
pp.create_switch(net, bus=2, element=1, et='l')
pp.create_switch(net, bus=3, element=1, et='l')
pp.create_switch(net, bus=3, element=2, et='l')
pp.create_switch(net, bus=4, element=2, et='l')
pp.create_switch(net, bus=4, element=3, et='l', closed=False)
pp.create_switch(net, bus=5, element=3, et='l')
pp.create_switch(net, bus=5, element=4, et='l')
pp.create_switch(net, bus=6, element=4, et='l')
pp.create_switch(net, bus=6, element=5, et='l')
pp.create_switch(net, bus=1, element=5, et='l')

# electrical loads
pp.create_load(net, 2, p_mw=1, q_mvar=0.2, name="load 0")
pp.create_load(net, 3, p_mw=1, q_mvar=0.2, name="load 1")
pp.create_load(net, 4, p_mw=1, q_mvar=0.2, name="load 2")
pp.create_load(net, 5, p_mw=1, q_mvar=0.2, name="load 3")
pp.create_load(net, 6, p_mw=1, q_mvar=0.2, name="load 4")

print(net)
print("\n=== Bus DataFrame ===")
print(net.bus)

# %% Run power flow and visualize
ppplot.simple_plotly(net)
pp.runpp(net)
ppplot.plot_voltage_profile(net)

# %% Inspect results
print("\n=== Network Contents ===")
for k, v in net.items():
    if v is None:
        continue
    try:
        if len(v):
            print(f'\n{k}')
            if isinstance(v, pd.DataFrame):
                print(v)
            else:
                print(v)
    except TypeError:
        print(v)

# %% NetworkX graph representation
mg = pptop.create_nxgraph(net, multi=True, calc_branch_impedances=True)
pos = nx.spring_layout(mg, seed=42)

edge_x, edge_y = [], []
for u, v in mg.edges():
    x0, y0 = pos[u]
    x1, y1 = pos[v]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

node_x, node_y, node_labels = [], [], []
for node in mg.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    node_labels.append(str(node))

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=edge_x, y=edge_y, mode='lines',
    line=dict(width=0.5, color='#888'),
    hoverinfo='none', showlegend=False
))
fig.add_trace(go.Scatter(
    x=node_x, y=node_y, mode='markers+text',
    text=node_labels, textposition="top center",
    marker=dict(size=10, color='#1f77b4'),
    showlegend=False
))
fig.update_layout(
    showlegend=False, hovermode='closest',
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    plot_bgcolor='white'
)
fig.show()

print(f"\nNetworkX graph: {mg}")
for edge, datadict in mg.edges.items():
    print(f"  {edge}: {datadict}")
for node, datadict in mg.nodes.items():
    print(f"  {node}: {datadict}")

# %% Example multivoltage network
netv = ppnet.example_multivoltage()
ppplot.simple_plotly(netv)
print(f"Type: {type(netv)}")

# %% Extreme network example
net_ext = ppnet.kb_extrem_vorstadtnetz_trafo_1()
ppplot.simple_plotly(net_ext)
