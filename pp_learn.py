# attempt to recreate the pandapower admittance matrix
# calculation

try:
    from pandapower.pf.makeYbus_numba import makeYbus
except ImportError:
    from pandapower.pypower.makeYbus import makeYbus


from pandapower.pypower.idx_brch_sc import K_ST
from pandapower.shortcircuit.ppc_conversion import _init_ppc, _create_k_updated_ppci, _get_is_ppci_bus
from pandapower.shortcircuit.impedance import _calc_zbus, _calc_ybus
"""
calculation method for single phase to ground short-circuit currents
"""
def get_Y_from_net(net: aux.pandapowerNet):
    """ aaa """
    # pos. seq bus impedance
    ppc, ppci = _init_ppc(net)
    # Create k updated ppci
    _calc_ybus(ppci)

    # zero seq bus impedance
    ppc_0, ppci_0 = _pd2ppc_zero(net, ppc['branch'][:, K_ST])
    _calc_ybus(ppci_0)
    


@errstate(all="raise")
def branch_vectors(branch, nl):
    stat = branch[:, BR_STATUS]  # ones at in-service branches
    Ysf = stat / (branch[:, BR_R] + 1j * branch[:, BR_X])  # series admittance
    if any(branch[:, BR_R_ASYM]) or any(branch[:, BR_X_ASYM]):
        Yst = stat / (branch[:, BR_R] + branch[:, BR_R_ASYM] +
                      1j * (branch[:, BR_X] + branch[:, BR_X_ASYM]))
    else:
        Yst = Ysf

    Bcf = stat * (branch[:, BR_G] + 1j * branch[:, BR_B])  # branch charging admittance
    if any(branch[:, BR_G_ASYM]) or any(branch[:, BR_B_ASYM]):
        Bct = stat * (branch[:, BR_G] + branch[:, BR_G_ASYM] +
                      1j * (branch[:, BR_B] + branch[:, BR_B_ASYM]))
    else:
        Bct = Bcf
    
    tap = ones(nl)  # default tap ratio = 1
    i = nonzero(real(branch[:, TAP]))  # indices of non-zero tap ratios
    tap[i] = real(branch[i, TAP])  # assign non-zero tap ratios
    tap = tap * exp(1j * pi / 180 * branch[:, SHIFT])  # add phase shifters

    Ytt = Yst + Bct / 2
    Yff = (Ysf + Bcf / 2) / (tap * conj(tap))
    Yft = - Ysf / conj(tap)
    Ytf = - Yst / tap
    return Ytt, Yff, Yft, Ytf

def get_admittance_matrix(net: aux.pandapowerNet) -> csr_matrix:
    """
    Returns the admittance matrix of the network, assuming that only transmission lines have impedances. 
    """
    F_BUS = 0; T_BUS = 1
    branch = net.line[['from_bus', 'to_bus', 'r_ohm_per_km', 'x_ohm_per_km', 'c_nf_per_km', 'g_us_per_km', 'max_i_ka', 'length_km', 'in_service']].to_numpy()
    branch[:, 2:5] = branch[:, 2:5] * branch[:, 7]  ## convert to total impedances
    bus = net.bus[['bus', 'p_kw', 'q_kvar', 'gs_us', 'bs_us', 'vm_pu', 'va_degree']].to_numpy()
    nb = bus.shape[0]  ## number of buses
    nl = branch.shape[0]  ## number of lines

    ## for each branch, compute the elements of the branch admittance matrix where
    ##
    ##      | If |   | Yff  Yft |   | Vf |
    ##      |    | = |          | * |    |
    ##      | It |   | Ytf  Ytt |   | Vt |
    ##
    Ytt, Yff, Yft, Ytf = branch_vectors(branch, nl)
    ## compute shunt admittance
    ## if Psh is the real power consumed by the shunt at V = 1.0 p.u.
    ## and Qsh is the reactive power injected by the shunt at V = 1.0 p.u.
    ## then Psh - j Qsh = V * conj(Ysh * V) = conj(Ysh) = Gs - j Bs,
    ## i.e. Ysh = Psh + j Qsh, so ...
    ## vector of shunt admittances
    #Ysh = (bus[:, GS] + 1j * bus[:, BS]) / baseMVA

    ## build connection matrices
    f = real(branch[:, F_BUS]).astype(int64)  ## list of "from" buses
    t = real(branch[:, T_BUS]).astype(int64)  ## list of "to" buses
    ## connection matrix for line & from buses
    Cf = csr_matrix((ones(nl), (range(nl), f)), (nl, nb))
    ## connection matrix for line & to buses
    Ct = csr_matrix((ones(nl), (range(nl), t)), (nl, nb))

    ## build Yf and Yt such that Yf * V is the vector of complex branch currents injected
    ## at each branch's "from" bus, and Yt is the same for the "to" bus end
    i = hstack([range(nl), range(nl)])  ## double set of row indices

    Yf = csr_matrix((hstack([Yff, Yft]), (i, hstack([f, t]))), (nl, nb))
    Yt = csr_matrix((hstack([Ytf, Ytt]), (i, hstack([f, t]))), (nl, nb))
    # Yf = spdiags(Yff, 0, nl, nl) * Cf + spdiags(Yft, 0, nl, nl) * Ct
    # Yt = spdiags(Ytf, 0, nl, nl) * Cf + spdiags(Ytt, 0, nl, nl) * Ct

    ## build Ybus
    Ybus = Cf.T * Yf + Ct.T * Yt

    # for canonical format
    for Y in (Ybus, Yf, Yt):
        Y.eliminate_zeros()
        Y.sum_duplicates()
        Y.sort_indices()
        del Y._has_canonical_format

    return Ybus