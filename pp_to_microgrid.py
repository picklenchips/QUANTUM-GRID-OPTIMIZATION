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
from util import timeIt

from dimod import BinaryQuadraticModel as BQM
from dimod import SimulatedAnnealingSampler
from collections import defaultdict
from dwave.system import DWaveSampler, EmbeddingComposite

import queue

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

@ timeIt
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

@ timeIt
def min_sensitivity_matrix(net: aux.pandapowerNet) -> csr_matrix:
    """
    Returns the (normalized) minimum sensitivity matrix for the network
      Used for subsequent microgrid optimization formulations
    C_{ij} = min_l P_l*PTDF_{ij}^l, for all lines (i,j)
    """
    if 'Zbus' not in net: add_admittance_impedance(net)
    # get addmittance from each (from, to) line
    from_bus = net.line['from_bus'].to_numpy()
    to_bus = net.line['to_bus'].to_numpy()
    n = len(net.bus)
    C = np.zeros_like(from_bus)
    maxC = -np.inf  # normalize sensitivity weighting
    for i, line in enumerate(net.line.index):
        min_coeff = np.inf
        for line2 in net.line.index:
            if line == line2:  # PTDF is 0 for the same line
                continue
            c = line['vn_kv']*power_transfer_distribution_factor(net, line, line2)
            c = min(c, min_coeff)
        C[i] = min_coeff
        maxC = max(maxC, min_coeff)
    row_indices = np.concatenate([from_bus, to_bus])
    col_indices = np.concatenate([to_bus, from_bus])
    data = np.concatenate([C, C]) / maxC
    ret = csr_matrix((data, (row_indices, col_indices)), shape=(n, n), dtype=np.complex64)
    return ret

@ timeIt
def electrical_coupling_strength_matrix(net: aux.pandapowerNet, alpha=0.5) -> csr_matrix:
    """
    Returns the electrical coupling strength of the network
         A_{ij} = | alpha Y_ij + beta C_ij |
      where Y_ij is the admittance matrix, C_ij is the 'sensitivity' matrix, and both are normalized
      here alpha = beta = 1/2
     - Used for subsequent microgrid optimization formulations
   
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

@ timeIt
def modularity_matrix(net: aux.pandapowerNet) -> csr_matrix:
    """
    Returns the modularity matrix of the network
      M_{ij} = 1/2m ( A_{ij} - k_i*k_j / 2m )
    where A is the electrical coupling strength matrix, k_i is the sum of weights of bus i, 
     and m is the sum of all edge weights (not double counted)
    """
    A = electrical_coupling_strength_matrix(net, alpha=0.5)
    k = A.sum(axis=1)
    m = k.sum()
    M = (A - np.outer(k, k) / m ) / m
    return M

@ timeIt
def self_reliance_matrix(net: aux.pandapowerNet) -> csr_matrix:
    """
    Returns the self-reliance matrix of the network, normalized by the maximum power
      S_{ij} = 2 p_i p_j / P    if i != j, else 0
    formatted for QUBO with offset \sum_{i,j} p_i p_j
    Returns the matrix S and the offset
    """
    # both positive
    n = len(net.bus)
    loads = []
    power_buses = []
    P = 0  # normalize powers
    for bus in net.bus.index:
        load = net.load.loc[net.load['bus'] == bus, 'p_mw'].sum() - net.gen.loc[net.gen['bus'] == bus, 'p_mw'].sum() - net.sgen.loc[net.sgen['bus'] == bus, 'p_mw'].sum()
        if load:
            P = max(P, load**2)
            loads.append(load)
            power_buses.append(bus)
    col, row = np.meshgrid(power_buses, power_buses, sparse=False)
    data = np.outer(loads, loads) / P
    # convert data to 1-D array
    data_1d = data.flatten()
    row_1d = row.flatten()
    col_1d = col.flatten()
    S = csr_matrix((data_1d, (row_1d, col_1d)), shape=(n, n), dtype=float)
    return S


@ timeIt
def microgrid_objective(net: aux.pandapowerNet, lambd = 1) -> csr_matrix:
    """ create microgrid objective weightings for the given network """
    M = modularity_matrix(net)
    S = self_reliance_matrix(net)
    # objective function to minimize, sum over all idx (i,j)
    f = lambd*S - M
    return f


def partition_objective(f: csr_matrix, buses: dict[int,bool]) -> tuple[csr_matrix, csr_matrix]:
    """
    Returns the partitioned objective matrix for the given partition
      f: the objective matrix relating bus i to bus j
        - f[f.row, f.col] = f.data
      partition: the partition of the network as a binary array\
        - partition[i] = 1 if bus i is in partition 1, 0 otherwise
    
    Returns the partitioned objective matrices f1, f2
    """
    # partition the objective matrix
    n1 = sum([v for v in buses.values()])
    n2 = len(buses) - n1
    row1, col1, data1 = [], [], []
    row2, col2, data2 = [], [], []
    for i, j in f.nonzero():
        d = f[i,j]
        if buses[i] and buses[j]:
            row1.append(i)
            col1.append(j)
            data1.append(d)
        elif not buses[i] and not buses[j]:
            row2.append(i)
            col2.append(j)
            data2.append(d)
    f1 = csr_matrix((data1, (row1, col1)), shape=(n1, n1), dtype=float)
    f2 = csr_matrix((data2, (row2, col2)), shape=(n2, n2), dtype=float)
    return f1, f2

def to_QUBO(f: csr_matrix) -> tuple[dict[tuple[int,int], float], float]:
    """
    Converts the given objective f to a QUBO representation with at most n(n-1)/2 variables
      (which is upper-trianglar), using the formula
          Q = sum_{i,j} f_{ij} (2 x_i x_j - x_i - x_j + 1)

    Returns the QUBO_dict and the offset
    """
    offset = np.sum(f.data)
    #TODO: compare speed with np.zeros(n,n)
    Q = defaultdict(int)
    for i, j in f.nonzero():
        d = f[i,j]
        if i > j:
            Q[(j,i)] += 2*d
        else:
            Q[(i,j)] += 2*d
        Q[(j,j)] -= d
        Q[(i,i)] -= d
    return dict(Q), offset

def QUBO_energy(Q: dict[tuple[int,int], float], x: dict[int, int]) -> float:
    """
    Returns the energy of the given QUBO and solution using the formula
        E = sum_{i,j} Q_{ij} x_i x_j = x.T Q x
    """
    energy = 0
    for (i, j), d in Q.items():
        energy += d*x[i]*x[j]
    return energy

def full_multivar_energy(f: csr_matrix, x: dict[int, int]) -> float:
    """
    Returns the energy of the full multivariable objective f and solution x
    where x corresponds to x_i,l for bus i and partition l
    where x is labeled as the partition number l. 
    need to create the full QUBO implementation for this
    """
    energy = 0
    for i, j in f.nonzero():
        energy += f[i,j]*x[i]*x[j]
    return energy


def simulate_anneal(bqm: BQM, num_reads=1000) -> tuple[dict[int, int], float]:
    """
    Return solution to any BinaryQuadraticModel problem
    """
    try:
        sampler = EmbeddingComposite(DWaveSampler())
    except:
        sampler = SimulatedAnnealingSampler()
    response = sampler.sample(bqm, num_reads=num_reads)
    lowest = response.lowest()
    return lowest.sample, lowest.energy


class PartitionStorage():
    def __init__(self, level: int, objective: csr_matrix, best_energy: float, buses: list[int]):
        self.level = level
        self.objective = objective
        self.best_energy = best_energy
        self.buses = buses
    
    def unpack(self) -> tuple[int, csr_matrix, float]:
        return self.level, self.objective, self.best_energy

@ timeIt
def microgrid_optimization(net: aux.pandapowerNet, lambd = 1, num_reads=1000) -> tuple[dict[int, int], float]:
    """
    Solves the microgrid optimization problem for the given network
     - net: the pandapower network
     - lambd: the weighting factor of the self-reliance matrix
     - num_reads: the number of reads to find the optimal partition
    Returns the optimal solution and the energy
    """
    # only need to calculate objective once
    objective = microgrid_objective(net, lambd)
    best_energy = np.inf
    # store the full solution as a list where 
    # fsol[i] = 
    full_solution = {}
    
    # initialize the queue with the full network
    # format is (level, objective matrix, best energy of section)
    Q = queue.Queue()
    Q.put(PartitionStorage(0,objective,best_energy,[i for i in net.bus.index]))
    queue_size = 1
    #TODO: actually, we do want a global best energy because we are optimizing objective over the whole network... 
    #TODO: thus, we need to write a way to evaluate the energy on the entire objective, with multiple groups?
    # TODO: this is because modularity is nonlinear when we partition the network as we lose edge weights
    # TODO: this is OK though because we just edit the to_QUBO
    while queue_size:
        item = Q.get()
        queue_size -= 1
        level, objective, best_energy = Q.unpack()

        q_dict, offset = to_QUBO(objective)
        bqm = BQM.from_qubo(q_dict, offset=offset)
        solution, energy = simulate_anneal(bqm, num_reads)
        if energy >= best_energy:
            # energy is worse than best energy of section of network
            continue
        best_energy = energy
        full_solution.append(solution)
        # partition the network
        ob1, ob2 = partition_objective(objective, solution)
    # repeat the optimization problem by binary partitioning the grid
    # according tot he optimal solution, and then running optimization again
    # on each sub-part until the parameters are completely optimized. 
    return solution

if __name__ == '__main__':
    while 1:
        t = input('which example to try? (minimal, california)\n>>').rstrip().lower()
        if 'm' in t:
            net = create_minimal_example(nbusses=3)
            
            print(net.load)
            print(net.gen)
            S = self_reliance_matrix(net)
            print(S)
            #ppplot.simple_plot(net, plot_loads = True, plot_gens=True)
            ppplot.simple_plotly(net)
        else:
            net = pp.from_sqlite('/Users/benkroul/Documents/Physics/womanium/QUANTUM-GRID-OPTIMIZATION/data/ppnets/transnet-california-n.db')
            print(net)
            n = input('how many busses to keep?\n>>').rstrip().lower()
            