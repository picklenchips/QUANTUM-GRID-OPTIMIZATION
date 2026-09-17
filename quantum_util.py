"""
~~~~ Created by Ben Kroul, 2024 ~~~
Defines useful utility functions and constants for quantum physics related things. Of note:
- plot_op and plot_theory_exp for plotting 2D operators and comparing theoretical vs experimental operators
- pauli matrices defined like pauli_e, sigma_g, etc.
- number, phase, charge + x basis operators
- gates, rotation operators, common pulse sequences
- expectations of operators over states or density matrices
- full bloch sphere simulation/animation of quantum state under unitary evolution...
    see example at the end of the file
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Range-Kutta approximation for ODEs
from scipy.integrate import solve_ivp
from tqdm import tqdm
import sys
from util import uFormat, timeIt, PLOTDIR, SAVEEXT

# --- PLOTTING --- #

# --- ANIMATE AND PLOT A BLOCH SPHERE --- #


# plot all edges of wireframe cube
def plot_wf_cube(fig, corner1, corner2, color="orange", linestyle="dashed", linewidth=1):
    x1, y1, z1 = corner1
    x2, y2, z2 = corner2
    x_pts = [x1, x2, x2, x2, x2, x1, x1, x1, x1, x1, x1, x2, x2, x2, x2, x1]
    y_pts = [y1, y1, y1, y2, y2, y2, y2, y1, y1, y2, y2, y2, y2, y1, y1, y1]
    z_pts = [z1, z1, z2, z2, z1, z1, z2, z2, z1, z1, z2, z2, z1, z1, z2, z2]
    dash = "dash" if linestyle == "dashed" else "solid"
    fig.add_trace(go.Scatter3d(
        x=x_pts, y=y_pts, z=z_pts,
        mode="lines",
        line=dict(color=color, width=linewidth, dash=dash),
        showlegend=False,
        hoverinfo="skip"
    ))


def plot_bloch_sphere(fig, frame_number=False, init_angle=False, angle_step=0):
    """Plot bloch sphere and axes to Plotly figure. adds frame_number in top left corner if specified"""
    default_angle = 24
    if isinstance(init_angle, bool):
        init_angle = default_angle
    # format axis
    if frame_number and angle_step:
        init_angle += frame_number * angle_step

    # plot wireframe sphere
    sphere_res = 40j
    u, v = np.mgrid[0 : 2 * np.pi : sphere_res, 0 : np.pi : sphere_res]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale="Blues",
        opacity=0.1,
        showscale=False,
        hoverinfo="skip"
    ))

    # plot axis lines (x, y, z)
    fig.add_trace(go.Scatter3d(
        x=[1, -1], y=[0, 0], z=[0, 0],
        mode="lines",
        line=dict(color="black", width=2, dash="dash"),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[1, -1], z=[0, 0],
        mode="lines",
        line=dict(color="black", width=2, dash="dash"),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[1, -1],
        mode="lines",
        line=dict(color="black", width=2, dash="dash"),
        showlegend=False,
        hoverinfo="skip"
    ))

    # plot axis circles
    th = np.arange(0, 2 * np.pi + np.pi / 20, np.pi / 20)
    x_circle = np.cos(th)
    y_circle = np.sin(th)
    z_circle = np.zeros_like(th)

    fig.add_trace(go.Scatter3d(
        x=x_circle, y=y_circle, z=z_circle,
        mode="lines",
        line=dict(color="black", width=1, dash="dash"),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter3d(
        x=z_circle, y=x_circle, z=y_circle,
        mode="lines",
        line=dict(color="black", width=1, dash="dash"),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter3d(
        x=y_circle, y=z_circle, z=x_circle,
        mode="lines",
        line=dict(color="black", width=1, dash="dash"),
        showlegend=False,
        hoverinfo="skip"
    ))

    # add axis labels
    labels_pos = [
        (0, 0, -1, "$|1\\rangle$"),
        (0, 0, 1, "$|0\\rangle$"),
        (1, 0, 0, "$|+\\rangle$"),
        (-1, 0, 0, "$|-\\rangle$"),
        (0, 1, 0, "$|y\\rangle$"),
        (0, -1, 0, "$|-y\\rangle$"),
    ]
    for px, py, pz, lbl in labels_pos:
        fig.add_trace(go.Scatter3d(
            x=[px], y=[py], z=[pz],
            mode="text",
            text=[lbl],
            textposition="top center",
            showlegend=False,
            hoverinfo="skip"
        ))

    # label frame number in the top left of the plot
    if isinstance(frame_number, int):
        angle = np.pi * init_angle / 180 - 3 * np.pi / 15 - np.pi / 2
        frame_x = np.cos(angle)
        frame_y = np.sin(angle)
        fig.add_trace(go.Scatter3d(
            x=[frame_x], y=[frame_y], z=[0.8],
            mode="text",
            text=[f"frame: {frame_number}"],
            textposition="top center",
            showlegend=False,
            hoverinfo="skip"
        ))

    # set camera angle
    camera = dict(
        eye=dict(
            x=1.5*np.cos(np.pi*init_angle/180),
            y=1.5*np.sin(np.pi*init_angle/180),
            z=0.5
        )
    )
    fig.update_layout(scene_camera=camera)


def plot_bloch_vector(fig, vec, color="red", add_purity=True, scale_vector=False):
    """plot the 3-vector vec on Plotly figure with color color
    if add_purity, shows purity of state at top-left corner of plot
    if scale_vector, will scale vector magnitude with purity of state"""
    x, y, z = vec
    purity = np.sqrt(x**2 + y**2 + z**2)
    if purity:  # normalize vector
        xn = x / purity
        yn = y / purity
        zn = z / purity
    else:
        xn = 0
        yn = 0
        zn = 0

    if scale_vector:
        fig.add_trace(go.Scatter3d(
            x=[x, xn], y=[y, yn], z=[z, zn],
            mode="lines",
            line=dict(color="black", width=1),
            showlegend=False,
            hoverinfo="skip"
        ))

    # normalized vector endpoint
    fig.add_trace(go.Scatter3d(
        x=[xn], y=[yn], z=[zn],
        mode="markers",
        marker=dict(color=color, size=5),
        showlegend=False,
        hoverinfo="skip"
    ))

    # arrow from origin to vector
    if scale_vector:  # plot "actual" vector
        vec_len = np.sqrt(x**2 + y**2 + z**2)
        if vec_len > 0:
            fig.add_trace(go.Scatter3d(
                x=[0, x], y=[0, y], z=[0, z],
                mode="lines+markers",
                line=dict(color=color, width=4),
                marker=dict(size=[0, 8]),
                showlegend=False,
                hoverinfo="skip"
            ))
    else:  # plot the unit-normalized vector, which is easier to see
        fig.add_trace(go.Scatter3d(
            x=[0, xn], y=[0, yn], z=[0, zn],
            mode="lines+markers",
            line=dict(color=color, width=4),
            marker=dict(size=[0, 8]),
            showlegend=False,
            hoverinfo="skip"
        ))

    # plot cube as lighter color
    plot_wf_cube(fig, (0, 0, 0), (xn, yn, zn), color=color)

    if add_purity:
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0.8], z=[1],
            mode="text",
            text=[f"purity: {uFormat(purity,0)}"],
            textposition="top center",
            showlegend=False,
            hoverinfo="skip"
        ))


def animate_bloch(
    states,
    name: str,
    pbar=False,
    rot_vecs=[],
    fps=60,
    dpi=200,
    add_purity=True,
    angle_step=0,
):
    """Animates bloch sphere and saves to {PLOTDIR}{name}.html
    states[time] = (x,y,z).  state vector at each time step
    rot_vecs[time] = (x,y,z) Hamiltonian vector at each time step
    fps: frames per second to save animation at
    dpi: dots per inch for resolution of animation (ignored for Plotly HTML)
    add_purity: if True, labels the purity of each frame"""
    nframes = states.shape[0]

    # Create figure for each frame
    frames_list = []
    for frame_idx in range(nframes):
        fig_frame = go.Figure()

        # Plot sphere
        init_angle = 24 + frame_idx * angle_step if angle_step else 24
        plot_bloch_sphere(fig_frame, frame_number=frame_idx, init_angle=init_angle, angle_step=0)
        plot_bloch_vector(fig_frame, states[frame_idx], add_purity=add_purity)

        if len(rot_vecs):  # add arrow symbolizing rotation axis
            if len(rot_vecs.shape) > 1:
                plot_bloch_vector(fig_frame, rot_vecs[frame_idx], color="green", add_purity=False)
            else:  # single rotation specified
                plot_bloch_vector(fig_frame, rot_vecs, color="green", add_purity=False)

        # Extract frame data
        frames_list.append(go.Frame(data=fig_frame.data, name=str(frame_idx)))

        if not isinstance(pbar, bool):
            pbar.update(1)

    # Create main figure with first frame
    fig = frames_list[0].data
    final_fig = go.Figure(data=fig, frames=frames_list)

    # Add animation settings
    final_fig.update_layout(
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {'label': 'Play', 'method': 'animate',
                 'args': [None, {'frame': {'duration': 1000/fps, 'redraw': True},
                                'fromcurrent': True}]},
                {'label': 'Pause', 'method': 'animate',
                 'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                  'mode': 'immediate'}]}
            ]
        }],
        width=800, height=800,
        scene=dict(
            xaxis=dict(range=[-1.2, 1.2]),
            yaxis=dict(range=[-1.2, 1.2]),
            zaxis=dict(range=[-1.2, 1.2]),
            aspectmode='cube'
        )
    )

    plot_name = PLOTDIR + "/" + name + ".html"
    final_fig.write_html(plot_name)
    print("saved animation to", plot_name)
    return final_fig


def label_str_states(N):
    """return ordered string of |00>, |01>, |10>, |11> for N qubits"""
    ret = []
    for i in range(2**N):
        string = "$|"
        for n in range(N - 1, -1, -1):
            if i >= 2**n:
                string += "1"
                i -= 2**n
            else:
                string += "0"
        string += r"\rangle$"
        ret.append(string)
    return ret


def plot_op(
    operators: list | tuple | np.ndarray,
    titles=[],
    saveplot=False,
    cmap_name="RdBu",
    box_spec=False,
):
    """Plot 2-dimensional operators using Plotly
    Inputs:
    - operators: takes in ndarray or list/tuple of ndarrays
    - titles: title or list of titles for multiple ops
    - saveplot: True or string to name file of plot
    - cmap_name: name of Plotly colorscale to use (default "RdBu" for diverging)
    - box_spec: if True, will show numbers of matrix elements
    """
    if not isinstance(operators, list) and not isinstance(operators, tuple):
        operators = [operators]
    if not isinstance(titles, list) and not isinstance(operators, tuple):
        titles = [titles]
    nops = len(operators)

    # get abs max of all operators
    absmax = 0
    for op in operators:
        themin = min(op.real.min(), op.imag.min())
        themax = max(op.real.max(), op.imag.max())
        amax = max(abs(themin), abs(themax))
        absmax = max(absmax, amax)

    # Create subplot titles
    subplot_titles = []
    for i in range(nops):
        title = titles[i] if len(titles) > i else "_DEF_"
        subplot_titles.append(f"Re{{ {title} }}")
        subplot_titles.append(f"Im{{ {title} }}")

    # Create subplots: nops rows, 2 cols (real and imag)
    fig = make_subplots(
        rows=nops, cols=2,
        subplot_titles=subplot_titles,
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}]] * nops,
        horizontal_spacing=0.15,
        vertical_spacing=0.1
    )

    # plot operators
    for i in range(nops):
        operator = operators[i]

        # Real part heatmap
        fig.add_trace(
            go.Heatmap(
                z=operator.real,
                colorscale=cmap_name,
                zmid=0,
                zmin=-absmax,
                zmax=absmax,
                showscale=(i == 0),  # only first subplot shows colorbar
                colorbar=dict(x=1.02) if i == 0 else None,
                hovertemplate="Value: %{z:.3f}<extra></extra>",
                name="Real"
            ),
            row=i+1, col=1
        )

        # Imaginary part heatmap
        fig.add_trace(
            go.Heatmap(
                z=operator.imag,
                colorscale=cmap_name,
                zmid=0,
                zmin=-absmax,
                zmax=absmax,
                showscale=False,
                hovertemplate="Value: %{z:.3f}<extra></extra>",
                name="Imag"
            ),
            row=i+1, col=2
        )

        # Add text annotations if box_spec is True
        if box_spec:
            for (row_idx, col_idx), val in np.ndenumerate(operator.real):
                fig.add_annotation(
                    text=uFormat(val, 0),
                    x=col_idx, y=row_idx,
                    xref=f"x{i*2+1}", yref=f"y{i*2+1}",
                    showarrow=False,
                    font=dict(
                        color="black" if abs(val) < absmax / 2 else "white",
                        size=8
                    )
                )
            for (row_idx, col_idx), val in np.ndenumerate(operator.imag):
                fig.add_annotation(
                    text=uFormat(val, 0),
                    x=col_idx, y=row_idx,
                    xref=f"x{i*2+2}", yref=f"y{i*2+2}",
                    showarrow=False,
                    font=dict(
                        color="black" if abs(val) < absmax / 2 else "white",
                        size=8
                    )
                )

    # Update layout
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        height=400 * nops,
        width=900,
        showlegend=False,
        hovermode="closest"
    )

    if saveplot:
        if isinstance(saveplot, str):
            plt_name = (
                PLOTDIR
                + "/"
                + saveplot.replace(" ", "_").replace("$", "")
                + "_2d"
                + SAVEEXT
            )
        else:
            plt_name = (
                PLOTDIR
                + "/"
                + title.replace(" ", "_").replace("$", "")
                + "_2d"
                + SAVEEXT
            )
        fig.write_image(plt_name)
        print(f"saved figure {plt_name}")
    fig.show()


def plot_theory_exp(
    op_th, op_exp, title, saveplot=False, cmap_name="Turbo", labels=None
):
    """3-d comparison of theoretical vs. experimental operators using Plotly"""
    zmax = max(op_th.real.max(), op_th.imag.max())
    zmin = min(op_th.real.min(), op_th.imag.min())
    cmax = max(
        abs(max(op_exp.real.max(), op_exp.imag.max())),
        abs(min(op_exp.real.min(), op_exp.imag.min())),
    )
    cmax = max(cmax, max(abs(zmax), abs(zmin)))
    _labels = labels if labels else label_str_states(int(np.log2(op_th.shape[0])))
    xrange = range(op_th.shape[0])
    X, Y = np.meshgrid(xrange, xrange)
    X = X.flatten()
    Y = Y.flatten()
    Z = np.zeros_like(X)
    dx = 0.5 * np.ones_like(X)
    op_th_flat = op_th.flatten()
    op_exp_flat = op_exp.flatten()
    dy = dx.copy()

    # Create subplots for Real and Imaginary parts
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=(f"Re{{ {title} }}", f"Im{{ {title} }}")
    )

    # Real part - experimental (main bars)
    nreal = op_exp_flat.real / (2 * cmax) + 0.5
    colors_real = [cmap_name] * len(nreal)  # Plotly will handle colorscale
    fig.add_trace(
        go.Scatter3d(
            x=X, y=Y, z=op_exp_flat.real,
            mode="markers",
            marker=dict(
                size=6,
                color=op_exp_flat.real,
                colorscale=cmap_name,
                showscale=True,
                cmin=zmin,
                cmax=zmax,
                colorbar=dict(x=0.46, len=0.4)
            ),
            name="Experimental (Real)"
        ),
        row=1, col=1
    )

    # Real part - theoretical (outline)
    fig.add_trace(
        go.Scatter3d(
            x=X, y=Y, z=op_th_flat.real,
            mode="markers",
            marker=dict(size=6, color="black", symbol="diamond"),
            name="Theoretical (Real)",
            opacity=0.5
        ),
        row=1, col=1
    )

    # Imaginary part - experimental (main bars)
    nimag = op_exp_flat.imag / (2 * cmax) + 0.5
    fig.add_trace(
        go.Scatter3d(
            x=X, y=Y, z=op_exp_flat.imag,
            mode="markers",
            marker=dict(
                size=6,
                color=op_exp_flat.imag,
                colorscale=cmap_name,
                showscale=True,
                cmin=zmin,
                cmax=zmax,
                colorbar=dict(x=1.02, len=0.4)
            ),
            name="Experimental (Imag)"
        ),
        row=1, col=2
    )

    # Imaginary part - theoretical (outline)
    fig.add_trace(
        go.Scatter3d(
            x=X, y=Y, z=op_th_flat.imag,
            mode="markers",
            marker=dict(size=6, color="black", symbol="diamond"),
            name="Theoretical (Imag)",
            opacity=0.5
        ),
        row=1, col=2
    )

    # Update axes for both subplots
    for col_num in [1, 2]:
        col_str = "" if col_num == 1 else "2"
        fig.update_layout({
            f"scene{col_str}": dict(
                xaxis=dict(tickvals=list(xrange), ticktext=_labels, title="Input"),
                yaxis=dict(tickvals=list(xrange), ticktext=_labels, title="Output"),
                zaxis=dict(title=r"$\hat{\rho}$", range=[zmin, zmax]),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
            )
        })

    fig.update_layout(
        height=600,
        width=1200,
        showlegend=True,
        hovermode="closest"
    )

    if saveplot:
        if isinstance(saveplot, str):
            plt_name = (
                PLOTDIR
                + "/"
                + saveplot.replace(" ", "_").replace("$", "")
                + "_3d"
                + SAVEEXT
            )
        else:
            plt_name = (
                PLOTDIR
                + "/"
                + title.replace(" ", "_").replace("$", "")
                + "_3d"
                + SAVEEXT
            )
        fig.write_image(plt_name)
        print(f"saved figure {plt_name}")
    fig.show()


# --- OPERATOR FUNCTIONS --- #

sigma_x = np.array([[0, 1], [1, 0]])  # |0><1| + |1><0|
sigma_y = np.array([[0, -1j], [1j, 0]])  # i|0><1| - i|1><0|
sigma_z = np.array([[1, 0], [0, -1]])  # |0><0| - |1><1|
sigma_i = np.array([[1, 0], [0, 1]])  # |0><0| + |1><1|
sigma_g = np.array([[1, 0], [0, 0]])  # |0><0|
sigma_e = np.array([[0, 0], [0, 1]])  # |1><1|
pauli_x = sigma_x
pauli_y = sigma_y
pauli_z = sigma_z
pauli_i = sigma_i
pauli_g = sigma_g
pauli_e = sigma_e


# ---- NUMBER (FOCK) BASIS OPERATORS ---- #
# 0-1 number states
def number_op(N):  # n|n><n|
    return np.diag(range(N))


def create_op(N):  # sqrt(n)|n+1><n|
    return np.diag(np.sqrt(np.arange(1, N)), k=-1)


def destroy_op(N):  # sqrt(n)|n-1><n|
    return np.diag(np.sqrt(np.arange(1, N)), k=1)


def phase_op(N):  # a + a†
    return create_op(N) + destroy_op(N)


def charge_op(N):  # i(a - a†)
    return 1j * (destroy_op(N) - create_op(N))


def cosphi_op(N):  # 2 cos(i(a+a†)) = |n+1><n| + |n-1><n|
    return (np.diag(np.ones(N - 1), k=-1) + np.diag(np.ones(N - 1), k=1)) / 2


def sinphi_op(N):  # 2i sin(i(a+a†)) = |n+1><n| - |n+1><n|
    return (np.diag(np.ones(N - 1), k=-1) - np.diag(np.ones(N - 1), k=1)) / (2j)


# ---- X BASIS OPERATORS ---- #
def xop(xpts):
    return np.diag(xpts)


def dop_forward(xpts):
    # We will assume a uniform spacing for now
    dx = xpts[1] - xpts[0]
    return (
        np.diag(np.ones(len(xpts) - 1), k=1) - np.diag(np.ones(len(xpts)), k=0)
    ) / dx


def dop_central(xpts):
    # We will assume a uniform spacing for now
    dx = xpts[1] - xpts[0]
    return (
        np.diag(np.ones(len(xpts) - 1), k=1) - np.diag(np.ones(len(xpts) - 1), k=-1)
    ) / (2 * dx)


def d2op(xpts):
    # We will assume a uniform spacing for now
    dx = xpts[1] - xpts[0]
    return (
        np.diag(np.ones(len(xpts) - 1), k=1)
        - 2 * np.diag(np.ones(len(xpts)), k=0)
        + np.diag(np.ones(len(xpts) - 1), k=-1)
    ) / (dx**2)


# ------- GATE FUNCTIONS ------- #


# idx of qubit is 0 to N-1
def gate_on_nth(N, idx, gate):
    arr = np.array([1], dtype=complex)
    for i in range(N):
        arr = np.kron(arr, gate) if idx == i else np.kron(arr, np.identity(2))
    return arr


def gate_on_all(N, gate):
    arr = np.array([1], dtype=complex)
    for i in range(N):
        arr = np.kron(arr, gate)
    return arr


def mixed_on_all(N, gate):
    arr = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N - 1):
        arr += gate_on_nth(N, i, gate) @ gate_on_nth(N, i + 1, gate)
    return arr


def rotation_op(theta, vec, N=2):
    vec = vec / np.linalg.norm(vec)  # normalize vector (it should be tho)
    x, y, z = vec
    vec_matrix = x * sigma_x + y * sigma_y + z * sigma_z
    single_rot = (
        np.identity(2) * np.cos(theta / 2) - 1j * np.sin(theta / 2) * vec_matrix
    )
    return gate_on_all(N, single_rot)


# unitary operator for time-independent, diagonal hamiltonian
def unitary_diag(t, H):
    Ut = np.diag(np.exp(-1j * np.diag(H) * t))
    return Ut


# --- SEQUENCES --- #


def ramsey_sequence(T, H):  # pi/2 +y -> T -> pi/2 -y
    N = int(np.log2(H.shape[0]))
    op1 = rotation_op(np.pi / 2, (0, 1, 0), N)  # pi/2 +y
    op2 = unitary_diag(T, H)  # wait T
    op3 = rotation_op(np.pi / 2, (0, -1, 0), N)  # pi/2 -y
    return op3 @ op2 @ op1


def spin_echo_sequence(T, H):  # pi/2 +y -> T/2 -> pi +x -> T/2 -> pi/2 -y
    N = int(np.log2(H.shape[0]))
    op1 = rotation_op(np.pi / 2, (0, 1, 0), N)  # pi/2 +y
    op2 = unitary_diag(T / 2, H)  # wait T/2
    op3 = rotation_op(np.pi, (1, 0, 0), N)  # pi +x
    op4 = rotation_op(np.pi / 2, (0, -1, 0), N)  # pi/2 -y
    return op4 @ op2 @ op3 @ op2 @ op1


def CPMG_N_sequence(T, N, H):
    nqubits = int(np.log2(H.shape[0]))
    half_pi_y = rotation_op(np.pi / 2, (0, 1, 0), nqubits)  # pi/2 +y
    min_half_pi_y = rotation_op(np.pi / 2, (0, -1, 0), nqubits)  # pi/2 +y
    pi_x = rotation_op(np.pi, (1, 0, 0), nqubits)
    if N == 0:
        return min_half_pi_y @ unitary_diag(T, H) @ half_pi_y
    wait_half = unitary_diag(T / (2 * N), H)  # wait T/2N
    wait_full = unitary_diag(T / N, H)
    op = pi_x @ wait_half @ half_pi_y
    for i in range(N - 1):
        op = pi_x @ wait_full @ op
    return min_half_pi_y @ wait_half @ op


# --- OPERATOR FUNCTIONALS --- #
def complex_phase(complex_num):
    return np.arctan(complex_num.imag / complex_num.real)


def expect_op(operator, psi):
    """Expectation value of operator over state psi
    - psi can be either a single state or a matrix of multiple states
    - returns a vector of expectation values for each state in psi_t"""
    if len(psi.shape) == 1:  # vectorized to psi_t
        psi = psi[np.newaxis]
    if psi.shape[0] != operator.shape[0]:  # num rows must be the same
        psi = psi.T
    # now psi.shape is  (state_dimension, number_of_states)
    # basically compute <psi|operator|psi> for all psis in psi_t
    ret = np.sum(np.conj(psi) * (operator @ psi), axis=0)
    return np.real(ret)


def expect_op_2(operator, rho):
    """Expectation value of operator over VECTORIZED rho
    - can also be used to reverse-engineer hamiltonians
    - returns a vector of expectation values for each state in rho_t"""
    if len(rho.shape) == 2:
        rho = rho[np.newaxis]
    assert rho[0].shape == operator.shape
    # now rho.shape is (number_of_states, state_dimension, state_dimension)
    ret = np.trace(operator @ rho, axis1=1, axis2=2)
    return np.real(ret)


# @ timeIt
def bloch_from_psi(psi_t):
    """returns 2D array (len(times), 3)"""
    return np.array(
        [
            expect_op(sigma_x, psi_t),
            expect_op(sigma_y, psi_t),
            expect_op(sigma_z, psi_t),
        ]
    ).T


# @ timeIt
def bloch_from_op(rho_t):
    """returns 2D array (len(times), 3)"""
    return np.array(
        [
            expect_op_2(sigma_x, rho_t),
            expect_op_2(sigma_y, rho_t),
            expect_op_2(sigma_z, rho_t),
        ]
    ).T


# use eigenvectors and eigenvalues of diagonalized, time-independent hamiltonian
def time_independent_H(times, es, evs, psi_0_ev):
    """returns psi_t from diagonalized H"""
    return np.conj(evs).T @ np.diag(np.exp(-1j * es * times)) @ psi_0_ev


def rabi_hamiltonian(z_omega, x_omega, t, nu):
    if isinstance(t, np.ndarray):  # vectorize the mf
        return (
            2 * x_omega * np.cos(nu * t)[:, np.newaxis, np.newaxis] * sigma_x
            + z_omega * sigma_z
        )
    return 2 * x_omega * np.cos(nu * t) * sigma_x + z_omega * sigma_z


if __name__ == "__main__":
    # example bloch sphere
    z_omega = 0.1
    x_omega = 1
    nu = 1
    num_timesteps = 300
    num_periods = 1
    times = num_periods * 2 * np.pi / nu * np.linspace(0, 1, num_timesteps)

    def ddt_psi_t(t, psi):
        return -1j * rabi_hamiltonian(z_omega, x_omega, t, nu) @ psi

    # COMPUTE PSI_T
    psi_0 = np.array([1, 0], dtype=complex)
    psi_t = solve_ivp(ddt_psi_t, [times.min(), times.max()], psi_0, t_eval=times).y
    # psi_t.shape = (2, num_timesteps)
    bloch_states = bloch_from_psi(psi_t)
    Hs = rabi_hamiltonian(z_omega, x_omega, times, nu)
    rot_vecs = bloch_from_op(Hs)

    # animate 3D psi_t on bloch_sphere
    with tqdm(total=bloch_states.shape[0]) as pbar:
        animate_bloch(
            bloch_states,
            "hamiltonian_many",
            rot_vecs=rot_vecs,
            pbar=pbar,
            fps=15,
            dpi=100,
            add_purity=False,
            angle_step=1,
        )
