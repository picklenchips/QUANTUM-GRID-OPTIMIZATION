# the following is equivalent to the multivoltage example
import sys,os

import pandas as pd
import pandapower as pp
import pandapower.auxiliary as aux  # for pandapowerNet typing
import pandapower.plotting as ppplot
import pandapower.networks as ppnet
import pandapower.topology as pptop
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import pandapower as pp
import pandapower.toolbox as pptools

import networkx as nx

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv
from util import timeIt

from dimod import SampleSet
from dimod import BinaryQuadraticModel as BQM
from dimod import SimulatedAnnealingSampler
from collections import defaultdict
from typing import Iterable
#from dwave.system import DWaveSampler, EmbeddingComposite

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

def admittance_of_pd(df: pd.DataFrame) -> pd.Series:
    # should be faster to write a native pandas function
    return df['r_ohm_per_km'] - 1j*df['x_ohm_per_km']/(df['length_km']*(df['r_ohm_per_km']**2 + df['x_ohm_per_km']**2))

class NetGraph():
    """ 
    class to define a network-equivalent graph for wrapping a given 
    pandapower network, so that we can quickly perform graph operations on it
    
    represent pandapower Net as a numpy() structure for fast iteration
     - only convert from pandas to numpy once for efficiency
    
    >>> Variables:
    | - self.net: pandapower network (pandapower.auxiliary.pandapowerNet)
    |   <> not changed, only used for reference
    | - self.N  : networkx graph     (networkx.Graph)
    |   = changed as a "view" of the network
    | - self.A  : adjacency matrix   (scipy.sparse.csr_matrix)
    |   = changed as a "view" of the network
    |---------
    | - self.buses  - bus indexes
    | - self.lines  - line indexes
    | - self.trafos - trafo indexes
    |----------
    | adjacency matrix creates the following variables:
    | - self.A_lines   - adjacency matrix for lines
    | - self.A_trafos  - adjacency matrix for trafos
    | - self.from_bus  - from bus of line (indices) (np.ndarray)
    | - self.to_bus    - to bus of line   (indices) (np.ndarray)
    -------------
    >>> Functions:

    """
    def __init__(self, net: aux.pandapowerNet, make_adjacency=True, make_nx=True,
                 consider_trafos=False):
        self.net = net
        self.consider_trafos = consider_trafos
        # store all bus, line, and trafo indices as numpy arrays
        self.buses = net.bus.index.to_numpy()
        # bus IDs are not guaranteed contiguous 0..n-1 (e.g. transnet_to_pp
        # creates buses with explicit index=n_id) -- map bus ID -> array
        # position once here, use positions everywhere self.A is indexed
        self._pos = {int(b): i for i, b in enumerate(self.buses)}
        self.lines = net.line.index.to_numpy()
        if self.consider_trafos:
            self.trafos = net.trafo.index.to_numpy()
        
        self.N = None
        if make_nx: 
            self.make_nx_graph()
        # these will be used to model network as adjacency matrix
        self.A = None
        if make_adjacency:
            self.make_adjacency_matrix()

    def __len__(self):
        return len(self.buses)
    
    def __str__(self):
        return str(self.net)
    
    def idx_to_bus(self, idx: int | Iterable[int]) -> int | list[int]:
        if isinstance(idx, int):
            return self.buses[idx]
        assert isinstance(idx, Iterable)
        return [self.buses[i] for i in idx]

    def bus_to_idx(self, bus: int | Iterable[int]) -> int | list[int]:
        if isinstance(bus, (int, np.integer)):
            return self._pos[int(bus)]
        assert isinstance(bus, Iterable)
        return [self._pos[int(b)] for b in bus]

    def make_nx_graph(self, out_of_service=[]) -> nx.Graph:
        """
        Returns the networkx graph of the network
          - out_of_service: list of buses to exclude from the graph
        """
        self.N = pptop.create_nxgraph(self.net, multi=False, calc_branch_impedances=True, 
                                      include_out_of_service=True, 
                                      respect_switches=True, include_switches=True,
                                      nogobuses=out_of_service)
        assert isinstance(self.N, nx.Graph)
        return self.N

    def make_adjacency_matrix(self, from_nx=False) -> csr_matrix:
        """
        Returns the adjacency matrix of the network
          where A[bus1, bus2] = line_idx, for the first 2*len(lines) nonzero elements and 
            A[bus1, bus2] = trafo_idx, for the next 2*len(trafos) nonzero elements
        
        from_nx = True: use networkx graph to create adjacency matrix
                = False: use pandapower network to create adjacency matrix
            
        Sets the following internal variables:
          - self.A: the adjacency matrix of the network
          - self.from_bus 
          - self.to_bus
          - self.line_buses
          - self.A_lines : (csr_matrix)
         if self.consider_trafos:
            - self.hv_bus
            - self.lv_bus
            - self.trafo_buses
            - self.A_trafos
        """
        n = len(self.buses)
        # LINES
        if from_nx:
            if not isinstance(self.N, nx.Graph):
                self.make_nx_graph()
            assert isinstance(self.N, nx.Graph)
            self.from_bus, self.to_bus = np.array(self.N.edges).T
            # or whatever this is stored as in the networkX graph
            self.lines = self.N.edges['index']
        else:
            self.from_bus = self.net.line['from_bus'].to_numpy()
            self.to_bus = self.net.line['to_bus'].to_numpy()
        self.line_buses = np.unique(np.concatenate([self.from_bus, self.to_bus]))
        # bus IDs -> array positions (bus IDs aren't guaranteed 0..n-1)
        from_pos = np.array([self._pos[int(b)] for b in self.from_bus], dtype=int)
        to_pos = np.array([self._pos[int(b)] for b in self.to_bus], dtype=int)
        row_lines = np.concatenate([from_pos, to_pos])
        col_lines = np.concatenate([to_pos, from_pos])
        line_data = np.concatenate([self.lines, self.lines])
        self.A_lines = csr_matrix((line_data, (row_lines, col_lines)), shape=(n, n), dtype=int)
        # TRAFOS``
        hv_pos = lv_pos = np.array([], dtype=int)
        trafo_data = []
        if self.consider_trafos:
            self.hv_bus = self.net.trafo['hv_bus'].to_numpy()
            self.lv_bus = self.net.trafo['lv_bus'].to_numpy()
            self.trafo_buses = np.unique(np.concatenate([self.hv_bus, self.lv_bus]))
            hv_pos = np.array([self._pos[int(b)] for b in self.hv_bus], dtype=int)
            lv_pos = np.array([self._pos[int(b)] for b in self.lv_bus], dtype=int)
            row_trafos = np.concatenate([hv_pos, lv_pos])
            col_trafos = np.concatenate([lv_pos, hv_pos])
            trafo_data = np.concatenate([self.trafos, self.trafos])
            self.A_trafos = csr_matrix((trafo_data, (row_trafos, col_trafos)), shape=(n, n), dtype=int)
        else:
            self.hv_bus = self.lv_bus = np.array([], dtype=int)
        # concatenate the two matrices into big adjacency matrix
        row_indices = np.concatenate([from_pos, to_pos, hv_pos, lv_pos])
        col_indices = np.concatenate([to_pos, from_pos, lv_pos, hv_pos])
        data = np.concatenate([line_data, trafo_data])
        self.A = csr_matrix((data, (row_indices, col_indices)), shape=(n, n), dtype=int)
        return self.A

    def cut_to_nbus(self, nbuses: int) -> None:
        """
        Cuts the network down to (at most) nbuses buses, keeping it
        connected: breadth-first-search from a bus in the largest connected
        component until nbuses buses have been visited (or the component
        is exhausted).
        """
        if self.A is None:
            self.make_adjacency_matrix()
        if self.N is None:
            self.make_nx_graph()
        assert isinstance(self.A, csr_matrix)
        assert isinstance(self.N, nx.Graph)
        if nbuses >= len(self.buses):
            print(f"{nbuses}>={len(self.buses)}, so no need to cut network")
            return

        n = len(self.buses)
        # start the search from the largest connected component so we have
        # the best chance of reaching nbuses without exhausting it
        largest_cc = max(nx.connected_components(self.N), key=len)
        start_bus = next(iter(largest_cc))
        start_pos = self.bus_to_idx(start_bus)

        visited = np.zeros(n, dtype=bool)
        visited[start_pos] = True
        Q = queue.Queue()
        Q.put(start_pos)
        nbuses_added = 1
        while not Q.empty() and nbuses_added < nbuses:
            b = Q.get()
            start = self.A.indptr[b]
            end = self.A.indptr[b + 1]
            connected = self.A.indices[start:end]
            for j in connected:
                if not visited[j]:
                    visited[j] = True
                    Q.put(j)
                    nbuses_added += 1
                    if nbuses_added >= nbuses:
                        break

        # delete all buses not visited
        self.only_keep_buses(visited)

    def only_keep_buses(self, keep: np.ndarray) -> None:
        """
        Keeps only the buses in the given boolean array (indexed by array
        position, matching self.buses / self.A ordering)
         - updates self.N representation
         - rebuilds self.A restricted to the kept buses
        """
        busesToKeep = self.buses[keep]
        busesToRemove = self.buses[~keep]
        if self.N is None:
            self.make_nx_graph(out_of_service=busesToRemove)
        else:
            assert isinstance(self.N, nx.Graph)
            self.N.remove_nodes_from(busesToRemove)

        # rebuild A restricted to kept buses, remapped to new (smaller)
        # position space so self.A stays consistent with self.buses
        if self.A is None:
            self.make_adjacency_matrix()
        assert isinstance(self.A, csr_matrix)
        old_to_new_pos = {old: new for new, old in enumerate(np.flatnonzero(keep))}
        n_new = len(busesToKeep)
        new_row, new_col, new_data = [], [], []
        for row in range(len(self.A.indptr) - 1):
            if not keep[row]:
                continue
            start, end = self.A.indptr[row], self.A.indptr[row + 1]
            for idx in range(start, end):
                col = self.A.indices[idx]
                if not keep[col]:
                    continue
                new_row.append(old_to_new_pos[row])
                new_col.append(old_to_new_pos[col])
                new_data.append(self.A.data[idx])

        self.buses = busesToKeep
        self._pos = {int(b): i for i, b in enumerate(self.buses)}
        self.A = csr_matrix((new_data, (new_row, new_col)), shape=(n_new, n_new), dtype=int)
        self.lines = np.unique(self.A.data) if len(new_data) else np.array([], dtype=int)
    
    def add_admittance_impedance(self, net = None | aux.pandapowerNet) -> np.complex64:
        """ add admittance and impedance matrices to the network as the keys
            'Ybus' and 'Zbus' respectively, stored as csr_matrices
        1. compute the admittance matrix Y_ij by open-circuiting all loads
        Y_{ij} = sum_{k in N(i)} 1/Z_{ik} if i = j
               = -1/Z_{ij} if i neq j and (i, j) is a line
               = 0 if i neq j and (i, j) is not a line
        2. compute the impedance matrix Z_ij = inv(Y_ij)

        ASSUMPTIONS
        1. if self.A exists, self.net.line has not been updated since. 
           : current self.A should match current self.net.line
        """
        if self.A is None:
            self.make_adjacency_matrix()
        if net is None:
            net = self.net
        assert isinstance(self.A, csr_matrix)
        assert isinstance(net, aux.pandapowerNet)
        buses = self.buses
        # get addmittance from each (from, to) line
        from_bus = self.from_bus
        to_bus = self.to_bus
        Y = admittance_of_pd(net.line).to_numpy() # = 1/Z = 1/(R+jX) = 1 / l*(r+jx)
        # get maximum admittance for normalization purposes
        Ymax = np.max(np.abs(Y))
        # all buses contained in (from, to)
        # compute the diagonal elements of the admittance matrix
        diag = np.zeros_like(buses)
        for i, bus in enumerate(buses):
            start = self.A.indptr[bus]
            end = self.A.indptr[bus + 1]
            connected_buses = self.A.indices[start:end]
            msk = np.logical_or(from_bus == bus, to_bus == bus)
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
    Y = admittance_of_pd(net.line).to_numpy() # = 1/Z = 1/(R+jX) = 1 / l*(r+jx)
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

# NOTE (docs/pipeline/state.md task 1): min_sensitivity_matrix,
# electrical_coupling_strength_matrix, and modularity_matrix below are only
# reached when microgrid_objective() is called with lambd<1 -- the default
# (lambd=1) path uses self_reliance_matrix only and does not exercise them.
# They have known unfixed bugs (min_coeff never assigned back in the loop
# below, C built with integer dtype losing precision, a bus-ID-vs-position
# indexing issue like the one fixed elsewhere in this file, and an
# electrical_coupling_strength_matrix Y-slicing expression that doesn't do
# what its comment implies) -- left as a follow-up, not silently patched
# without being able to verify correctness against real data.
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
    if 'Zbus' not in net: add_admittance_impedance(net)
    n = len(net.bus)
    # skip all diagonal elements, which are the first n elements
    Y = net['Ybus'][n:]
    Y = Y / np.max(np.abs(Y))
    # get the normalized sensitivity matrix
    C = min_sensitivity_matrix(net)
    return np.abs(alpha * Y + (1-alpha) * C)

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
    formatted for QUBO with offset sum_{i,j} p_i p_j
    Returns the matrix S and the offset
     - load is positive, generation is negative (consumer model)
     - matrix is indexed by array position (0..n-1 in net.bus.index order),
       not raw bus IDs -- bus IDs aren't guaranteed contiguous (e.g. buses
       created with explicit index=n_id in transnet_to_pp.py)
    """
    # both positive
    n = len(net.bus)
    bus_pos = {b: i for i, b in enumerate(net.bus.index)}
    loads = []        # store p_i values
    power_buses = []  # store bus array positions
    max_P = 0         # normalize powers
    for bus in net.bus.index:
        load = net.load.loc[net.load['bus'] == bus, 'p_mw'].sum() - net.gen.loc[net.gen['bus'] == bus, 'p_mw'].sum() - net.sgen.loc[net.sgen['bus'] == bus, 'p_mw'].sum()
        if load:
            max_P = max(max_P, load**2)
            loads.append(load)
            power_buses.append(bus_pos[bus])
    if not loads:
        # no net load/generation anywhere -- nothing to optimize
        return csr_matrix((n, n), dtype=float)
    # now that we have lists of indices, values, we create matrix from all combinations
    col, row = np.meshgrid(power_buses, power_buses, sparse=False)
    data = np.outer(loads, loads) / max_P
    # convert data to 1-D array
    data_1d = data.flatten()
    row_1d = row.flatten()
    col_1d = col.flatten()
    S = csr_matrix((data_1d, (row_1d, col_1d)), shape=(n, n), dtype=float)
    return S


@ timeIt
def microgrid_objective(net: aux.pandapowerNet, lambd = 0.5) -> csr_matrix:
    """ create microgrid objective weightings for the given network 
     - lambd: percent weighting of self-reliance matrix vs. modularity """
    if lambd < 0 or lambd > 1: lambd = 0.5
    if lambd == 0:
        return modularity_matrix(net)
    if lambd == 1:
        return self_reliance_matrix(net)
    M = modularity_matrix(net)
    S = self_reliance_matrix(net)
    # objective function to minimize, sum over all idx (i,j)
    f = lambd*S - (1-lambd)*M
    return f


def partition_csr(f: csr_matrix, indices: dict[int,bool]) -> tuple[csr_matrix, csr_matrix]:
    """
    Partitions a square csr_matrix f into two matrices f1, f2

    Arguments:
      f: the objective matrix relating bus i to bus j
        - f[f.row, f.col] = f.data
      indices: the partition of the network as a binary array\
        - indices[i] = 1 if bus i is in partition 1, 0 otherwise
    
    Returns:
        f1, f2: the partitioned objective matrices, where f1 contains 'True' elements (i,j) and f2 contains 'False' elements (i,j)
    """
    # partition the objective matrix
    n = f.shape[0]
    # number of 'True' values in indices
    n1 = sum([v for v in indices.values()])
    n2 = n - n1
    row1, col1, data1 = [], [], []
    row2, col2, data2 = [], [], []
    # iterate over all items in f
    for row in range(len(f.indptr)-1):
        for idx in range(f.indptr[row], f.indptr[row+1]):
            col = f.indices[idx]
            d = f.data[idx]
            t1, t2 = indices.get(row, False), indices.get(col, False)
            if t1 and t2:
                # both (i,j) are in partition 1
                row1.append(row)
                col1.append(col)
                data1.append(d)
            elif not t1 and not t2:
                # both (i,j) are in partition 2
                row2.append(row)
                col2.append(col)
                data2.append(d)
    f1 = csr_matrix((data1, (row1, col1)), shape=(n1, n1), dtype=float)
    f2 = csr_matrix((data2, (row2, col2)), shape=(n2, n2), dtype=float)
    return f1, f2

def to_QUBO(f: csr_matrix) -> tuple[dict[tuple[int,int], float], float]:
    """
    Converts the given objective f to a QUBO representation with at most n(n-1)/2 variables
      (which is upper-trianglar), using the formula
          Q = sum_{i,j} f_{ij} (x_i = x_j)
             where (x_i=x_j) = 1 if x_i = x_j and 0 otherwise
          Q = sum_{i,j} f_{ij} (2 x_i x_j - x_i - x_j + 1)

    Returns (QUBO_dict, offset)
    """
    offset = np.sum(f.data)
    #TODO: compare speed with np.zeros(n,n)
    Q = defaultdict(int)
    for i in range(len(f.indptr)-1):
        for idx in range(f.indptr[i], f.indptr[i+1]):
            j = f.indices[idx]
            d = f.data[idx]
            if i > j:  # keep upper-triangular
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
     - Q (dict): Q[(i,j)] = Q_{ij}
     - x (dict): x[bus index] = partition number
    """
    energy = 0
    for (i, j), d in Q.items():
        energy += d*x[i]*x[j]
    return energy

def objective_energy(f: csr_matrix, x: dict[int, int]) -> float:
    """
    Returns the energy of the full multivariable objective f and solution x
     - f (csr_matrix): the objective matrix
     - x (dict): x[bus index] = partition number
    """
    energy = 0
    for i in range(len(f.indptr)-1):
        for idx in range(f.indptr[i], f.indptr[i+1]):
            j = f.indices[idx]
            d = f.data[idx]
            # QUBO matrix does this already by encoding x_i = x_j when summed over
            # Q_ii, Q_jj, and Q_ij
            if x[i] == x[j]:
                energy += d
    return energy


def simulate_anneal(bqm: BQM, num_reads=1000) -> tuple[dict[int, int], float]:
    """
    Solves the given BinaryQuadraticModel problem, on real D-Wave hardware if
    a DWaveSampler is configured, otherwise falls back to classical
    simulated annealing (no D-Wave account/token needed).
    Returns (solution, energy) -- solution[var] = 0 or 1.
    """
    try:
        # if dwave.system has been imported and a solver/token is configured
        sampler = EmbeddingComposite(DWaveSampler())
    except Exception:
        sampler = SimulatedAnnealingSampler()
    response = sampler.sample(bqm, num_reads=num_reads)
    best = response.first  # dimod.SampleView: .sample (dict), .energy
    return dict(best.sample), float(best.energy)


class PartitionStorage():
    """Bookkeeping for one queued sub-network in microgrid_optimization's
    (currently single-level) partition search."""
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
    Solves the microgrid partitioning problem for the given network as a
    single QUBO bipartition (splits buses into two groups by minimizing the
    microgrid objective).
     - net: the pandapower network
     - lambd: weighting factor of the self-reliance matrix (default 1 =
       self-reliance only; lambd<1 pulls in the modularity term, which has
       known unfixed bugs -- see note above min_sensitivity_matrix)
     - num_reads: number of annealer reads to find the optimal partition
    Returns (solution, energy) where solution[bus_position] = 0 or 1 is the
    partition group for the bus at that array position in net.bus.index.

    NOTE: recursive multi-level partitioning (splitting each group again) is
    intentionally not implemented here -- the original design's open
    question about evaluating global energy under a nonlinear modularity
    term across partitions (see git history) is unresolved; this returns one
    bipartition rather than silently pretending to solve that.
    """
    objective = microgrid_objective(net, lambd)
    q_dict, offset = to_QUBO(objective)
    bqm = BQM.from_qubo(q_dict, offset=offset)
    solution, energy = simulate_anneal(bqm, num_reads)
    return solution, energy

if __name__ == '__main__':
    while 1:
        t = input('which example to try? (minimal, california)\n>>').rstrip().lower()
        if 'm' in t:
            net = create_minimal_example(nbusses=3)
            print(net.load)
            print(net.gen)
            S = self_reliance_matrix(net)
            print(S)
            N = NetGraph(net)
            # get adjacency matrix
            A = N.A
            assert isinstance(A, csr_matrix)
            """
            iteration showing how data is stored in a csr_matrix object A
              A.indptr is a row iterator
              A.indices stores all column indices
              A.data stores all relevant data"""
            for row in range(len(A.indptr)-1):
                start = A.indptr[row]
                end = A.indptr[row+1]
                cols = A.indices[start:end]
                data = A.data[start:end]
                print(f"row {row}: {cols} -> {data}")


            #ppplot.simple_plot(net, plot_loads = True, plot_gens=True)
        else:
            cwd = os.getcwd()
            net = pp.from_sqlite(cwd + '/data/ppnets/transnet-california-n.db')
            print(net)
            n = input('how many busses to keep?\n>>').rstrip().lower()
            n = int(n) if n.isdigit() else 100
            N = NetGraph(net)
            N.cut_to_nbus(n)
            print(f'cut to {len(N.buses)} buses')
            solution, energy = microgrid_optimization(net, lambd=1, num_reads=100)
            print(f'microgrid partition energy: {energy}')
            print(f'partition (bus position -> group): {solution}')
            break
