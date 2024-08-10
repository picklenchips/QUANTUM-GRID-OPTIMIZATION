# the following is equivalent to the multivoltage example
import sys,os
import pandas as pd
import pandapower as pp
import pandapower.auxiliary as aux  # for pandapowerNet typing
import pandapower.plotting as ppplot
import pandapower.networks as ppnet
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import pandapower as pp
import pandapower.toolbox as pptools
import pandapower.auxiliary as aux
import numpy as np
from numpy import ones, conj, nonzero, any, exp, pi, hstack, real, int64, errstate
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv
publicToken = '***REMOVED-MAPBOX-TOKEN***'
noWriting = '***REMOVED-MAPBOX-TOKEN***'
fullAccess = '***REMOVED-MAPBOX-TOKEN***'
ppplot.set_mapbox_token(fullAccess)

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


net = pp.from_sqlite('/Users/benkroul/Documents/Physics/womanium/QUANTUM-GRID-OPTIMIZATION/data/ppnets/transnet-california-n.db')
print(net)

def cut_net_to(net, nbusses) -> None:
    """
    Cuts the network to the first nbusses
    """
    if nbusses >= len(net.bus):
        print('this will delete the whole network...')
        return
    pptools.drop_buses(net, net.bus.index[nbusses:])

def add_admittance_impedance(net: aux.pandapowerNet) -> np.complex64:
    """ add admittance and impedance matrices to the network as the keys
        'Ybus' and 'Zbus' respectively, stored as csr_matrices
    1. compute the admittance matrix Y_ij by open-circuiting all loads
      Y_{ii} = \sum_{k \in N(i)} 1/Z_{ik}
      Y_{ij} = -1/Z_{ij} if i \neq j and (i, j) is a line
      Y_{ij} = 0 if i \neq j and (i, j) is not a line
    2. compute the impedance matrix Z_ij = inv(Y_ij)
    """
    buses = net.bus.index.to_numpy()
    # get addmittance from each (from, to) line
    from_bus = net.line['from_bus'].to_numpy()
    to_bus = net.line['to_bus'].to_numpy()
    r = net.line['r_ohm_per_km'].to_numpy()
    x = net.line['x_ohm_per_km'].to_numpy()
    l = net.line['length_km'].to_numpy()
    Y = (r - 1j*x)/(l*(r**2+x**2)) # = 1/Z = 1/(R+jX) = 1 / l*(r+jx)
    # get maximum admittance for normalization purposes
    Ymax = np.max(np.abs(Y))
    # all buses contained in (from, to)
    # compute the diagonal elements of the admittance matrix
    diag = np.zeros_like(buses)
    for i in range(len(buses)):
        msk = np.logical_or(from_bus == buses[i], to_bus == buses[i])
        diag[i] = np.sum(Y[msk])
    # diagonal, then off-diagonal elements
    # allow indexing both (from, to) and (to, from)
    row_indices = np.concatenate([buses, from_bus, to_bus])
    col_indices = np.concatenate([buses, to_bus, from_bus])
    data = np.concatenate([diag, -Y, -Y])
    n = len(buses)
    Y = csr_matrix((data, (row_indices, col_indices)), shape=(n, n), dtype=np.complex64)
    net['Ybus'] = Y
    net['Zbus'] = inv(Y)
    return Ymax

def power_transfer_distribution_factor(net: aux.pandapowerNet, a_line: int, t_line: int) -> float:
    """
    Calculates the PTDF between two busses i and j for the given line
    line_idx
    PTDF = (Z_im - Z_in - Z_jm + Z_jn) / X_ij
      for impedances Z, reactance X, and busses i, j, m, n
    assume power transfer is small and system is operating in linear regime
    - a_line: idx of affected line 
    - t_line: idx of transaction line to be perturbed
    """
    # calculate impedances if not already calculated
    if 'Zbus' not in net: add_admittance_impedance(net)
    ref_line = net.line.loc[t_line]
    i, j = ref_line.from_bus, ref_line.to_bus
    aff_line = net.line.loc[a_line]
    m, n = aff_line.from_bus, aff_line.to_bus
    X = net.res_line['x_ohm_per_km']*net.line['length_km']
    Z = net['Zbus']
    return (abs(Z[i,m]) - abs(Z[i,n]) - abs(Z[j,m]) + abs(Z[j,n])) / X[i,j]

def min_sensitivity_matrix(net: aux.pandapowerNet) -> csr_matrix:
    """
    Returns the (normalized) minimum sensitivity matrix for the network
      Used for subsequent microgrid optimization formulations
    C_{ij} = | PTDF_{ij} | if i \neq j
    C_{ij} = 0 if i = j
    """
    if 'Zbus' not in net: add_admittance_impedance(net)
    buses = net.bus.index.to_numpy()
    # get addmittance from each (from, to) line
    from_bus = net.line['from_bus'].to_numpy()
    to_bus = net.line['to_bus'].to_numpy()
    n = len(net.bus)
    C = np.zeros_like(from_bus)
    maxC = -np.inf
    for line in net.line.index:
        min_coeff = np.inf
        for line2 in net.line.index:
            if line == line2: continue
            c = line['vn_kv']*power_transfer_distribution_factor(net, line, line2)
            if c < min_coeff: min_coeff = c
        C[line] = min_coeff
        if min_coeff > maxC: maxC = min_coeff
    row_indices = np.concatenate([from_bus, to_bus])
    col_indices = np.concatenate([to_bus, from_bus])
    data = np.concatenate([C, C])/maxC
    ret = csr_matrix((data, (row_indices, col_indices)), shape=(n, n), dtype=np.complex64)
    return ret

def electrical_coupling_strength_matrix(net: aux.pandapowerNet, alpha=0.5) -> csr_matrix:
    """
    Returns the electrical coupling strength of the network
      Used for subsequent microgrid optimization formulations
    A_{ij} = | alpha Y_ij + beta C_ij |
    where Y_ij is the admittance matrix, C_ij is the 'sensitivity' matrix, and both are normalized
    here alpha = beta = 1/2
    """
    if alpha < 0 or alpha > 1: alpha = 0.5
    beta = 1-alpha
    if 'Zbus' not in net: add_admittance_impedance(net)
    n = len(net.bus)
    # skip all diagonal elements, which are the first n elements
    Y = net['Ybus'][n:]
    Y = Y / np.max(np.abs(Y))
    # get the normalized sensitivity matrix
    C = min_sensitivity_matrix(net)
    return np.abs(alpha * Y + beta * C)

if __name__ == '__main__':
    while 1:
        t = input('which example to try? (minimal, california)\n>>').rstrip().lower()
        if 'm' in t:
            net = create_minimal_example(nbusses=3)
            print(net)
            print(net['bus'].head())
            ppplot.simple_plot(net, plot_loads = True, plot_gens=True)
            ppplot.simple_plotly(net)
        else:
            net = pp.from_sqlite('/Users/benkroul/Documents/Physics/womanium/QUANTUM-GRID-OPTIMIZATION/data/ppnets/transnet-california-n.db')
            print(net)
            n = input('how many busses to keep?\n>>').rstrip().lower()
            