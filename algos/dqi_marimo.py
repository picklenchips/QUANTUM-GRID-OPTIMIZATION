# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair>=5.5,<7",
#     "marimo>=0.17,<1",
#     "numpy>=2.0,<3",
#     "pandas>=2.2,<4",
#     "qiskit>=2.1,<3",
#     "qiskit-aer>=0.17,<1",
# ]
# ///
# ruff: noqa: F811

import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import itertools
    from math import comb

    import altair as alt
    import marimo as mo
    import numpy as np
    import pandas as pd
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
    from qiskit.circuit.library import StatePreparation
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator

    return (
        AerSimulator,
        ClassicalRegister,
        QuantumCircuit,
        QuantumRegister,
        StatePreparation,
        Statevector,
        alt,
        comb,
        itertools,
        mo,
        np,
        pd,
        transpile,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        # Decoded Quantum Interferometry (DQI)

        This notebook implements the binary **max-XORSAT** specialization of
        Decoded Quantum Interferometry introduced by Jordan, Shutty, Wootters,
        and collaborators.

        Given \(B\in\mathbb{F}_2^{m\times n}\), \(v\in\mathbb{F}_2^m\), and
        \(x\in\mathbb{F}_2^n\), define

        \[
        f(x)=\sum_{i=1}^{m}(-1)^{v_i+b_i\cdot x}
            =2s(x)-m,
        \]

        where \(s(x)\) is the number of equations in \(Bx=v\pmod 2\) that
        \(x\) satisfies.

        The circuit below:

        1. prepares a superposition of error patterns \(y\) with
           \(|y|\leq\ell\);
        2. applies the phase \((-1)^{v\cdot y}\);
        3. computes the syndrome \(B^Ty\);
        4. reversibly decodes the syndrome and uncomputes \(y\);
        5. applies \(H^{\otimes n}\) and samples candidate assignments.

        For clarity, the notebook uses an exact lookup decoder. That is a
        faithful coherent decoder for small uniquely-decodable examples, but
        scalable DQI requires a structured reversible decoder such as one
        derived from an algebraic code or LDPC belief propagation.
        """
    )
    return


@app.cell
def _(
    AerSimulator,
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
    StatePreparation,
    Statevector,
    comb,
    itertools,
    np,
    transpile,
):
    def bounded_error_patterns(m: int, ell: int) -> list[tuple[int, ...]]:
        """Return all m-bit error patterns of Hamming weight at most ell."""
        return [
            bits
            for bits in itertools.product((0, 1), repeat=m)
            if sum(bits) <= ell
        ]

    def syndrome_of(
        B: np.ndarray, error: tuple[int, ...]
    ) -> tuple[int, ...]:
        """Compute B^T error over GF(2), in logical syndrome-wire order."""
        error_vector = np.asarray(error, dtype=int)
        return tuple(int(value) for value in (B.T @ error_vector) % 2)

    def validate_unique_syndromes(
        B: np.ndarray, patterns: list[tuple[int, ...]]
    ) -> dict[tuple[int, ...], tuple[int, ...]]:
        """Build a bounded-distance decoder table and reject collisions."""
        decoder: dict[tuple[int, ...], tuple[int, ...]] = {}
        for pattern in patterns:
            syndrome = syndrome_of(B, pattern)
            if syndrome in decoder and decoder[syndrome] != pattern:
                previous = decoder[syndrome]
                raise ValueError(
                    "The bounded decoder is ambiguous: "
                    f"{previous} and {pattern} both have syndrome {syndrome}."
                )
            decoder[syndrome] = pattern
        return decoder

    def optimal_binary_weights(m: int, ell: int) -> np.ndarray:
        """Optimal binary DQI weights from the principal tridiagonal eigenvector."""
        if not 0 <= ell <= m:
            raise ValueError("ell must satisfy 0 <= ell <= m.")

        matrix = np.zeros((ell + 1, ell + 1), dtype=float)
        for k in range(1, ell + 1):
            coupling = np.sqrt(k * (m - k + 1))
            matrix[k - 1, k] = coupling
            matrix[k, k - 1] = coupling

        _, eigenvectors = np.linalg.eigh(matrix)
        weights = np.asarray(eigenvectors[:, -1], dtype=float)
        if weights[0] < 0:
            weights = -weights
        return weights / np.linalg.norm(weights)

    def error_state_amplitudes(
        m: int, ell: int, weights: np.ndarray
    ) -> np.ndarray:
        """Amplitude-encode sum_k w_k |D^m_k> in Qiskit wire order."""
        if len(weights) != ell + 1:
            raise ValueError("weights must contain ell + 1 entries.")

        normalized_weights = np.asarray(weights, dtype=complex)
        normalized_weights /= np.linalg.norm(normalized_weights)
        amplitudes = np.zeros(2**m, dtype=complex)

        for pattern in bounded_error_patterns(m, ell):
            weight = sum(pattern)
            basis_index = sum(bit << wire for wire, bit in enumerate(pattern))
            amplitudes[basis_index] = normalized_weights[weight] / np.sqrt(
                comb(m, weight)
            )

        return amplitudes

    def pattern_controlled_x(
        circuit: QuantumCircuit,
        controls: list,
        target,
        pattern: tuple[int, ...],
    ) -> None:
        """Flip target exactly when controls equal the requested bit pattern."""
        if len(controls) != len(pattern):
            raise ValueError("Control register and pattern lengths differ.")

        zero_controls = [
            control for control, bit in zip(controls, pattern, strict=True) if bit == 0
        ]
        for control in zero_controls:
            circuit.x(control)

        if len(controls) == 1:
            circuit.cx(controls[0], target)
        else:
            circuit.mcx(controls, target)

        for control in reversed(zero_controls):
            circuit.x(control)

    def build_dqi_circuit(
        B: np.ndarray,
        v: np.ndarray,
        ell: int,
        weights: np.ndarray | None = None,
        *,
        measure: bool = False,
    ) -> QuantumCircuit:
        """Build exact small-instance DQI for binary max-XORSAT."""
        B_array = np.asarray(B, dtype=int) % 2
        v_array = np.asarray(v, dtype=int) % 2

        if B_array.ndim != 2:
            raise ValueError("B must be a two-dimensional binary matrix.")

        m, n = B_array.shape
        if v_array.shape != (m,):
            raise ValueError(f"v must have shape ({m},).")

        patterns = bounded_error_patterns(m, ell)
        decoder = validate_unique_syndromes(B_array, patterns)
        dqi_weights = (
            optimal_binary_weights(m, ell)
            if weights is None
            else np.asarray(weights, dtype=float)
        )

        error = QuantumRegister(m, "error")
        solution = QuantumRegister(n, "x")
        circuit = QuantumCircuit(error, solution, name=f"DQI_l{ell}")

        amplitudes = error_state_amplitudes(m, ell, dqi_weights)
        circuit.append(StatePreparation(amplitudes), list(error))
        circuit.barrier(label="Dicke superposition")

        for row, phase_bit in enumerate(v_array):
            if phase_bit:
                circuit.z(error[row])
        circuit.barrier(label="phase")

        for row in range(m):
            for column in range(n):
                if B_array[row, column]:
                    circuit.cx(error[row], solution[column])
        circuit.barrier(label="syndrome")

        syndrome_controls = list(solution)
        for syndrome, decoded_error in decoder.items():
            for row, error_bit in enumerate(decoded_error):
                if error_bit:
                    pattern_controlled_x(
                        circuit,
                        syndrome_controls,
                        error[row],
                        syndrome,
                    )
        circuit.barrier(label="decode")

        for qubit in solution:
            circuit.h(qubit)

        if measure:
            result = ClassicalRegister(n, "result")
            circuit.add_register(result)
            circuit.measure(solution, result)

        return circuit

    def logical_label(index: int, width: int) -> str:
        """Display logical bits as x0 x1 ... rather than Qiskit's c[n-1]...c[0]."""
        return "".join(str((index >> bit) & 1) for bit in range(width))

    def qiskit_key_to_logical(key: str) -> str:
        """Convert a Qiskit count key to x0 x1 ... logical order."""
        return key.replace(" ", "")[::-1]

    def exact_solution_probabilities(
        circuit: QuantumCircuit, m: int, n: int
    ) -> tuple[dict[str, float], float]:
        """Return exact solution probabilities and residual decoder leakage."""
        state = Statevector.from_instruction(circuit)
        solution_probabilities = state.probabilities(qargs=list(range(m, m + n)))
        error_probabilities = state.probabilities(qargs=list(range(m)))

        probabilities = {
            logical_label(index, n): float(probability)
            for index, probability in enumerate(solution_probabilities)
        }
        decoder_leakage = float(1.0 - error_probabilities[0])
        return probabilities, decoder_leakage

    def sample_solution_counts(
        circuit: QuantumCircuit, shots: int, seed: int
    ) -> dict[str, int]:
        """Sample a measured DQI circuit and normalize count-key endianness."""
        simulator = AerSimulator(seed_simulator=seed)
        compiled = transpile(
            circuit,
            simulator,
            optimization_level=1,
            seed_transpiler=seed,
        )
        raw_counts = simulator.run(
            compiled,
            shots=shots,
            seed_simulator=seed,
        ).result().get_counts()
        return {
            qiskit_key_to_logical(key): int(count)
            for key, count in raw_counts.items()
        }

    def assignment_statistics(
        B: np.ndarray, v: np.ndarray
    ) -> dict[str, tuple[int, int]]:
        """Map each logical assignment to (satisfied constraints, f(x))."""
        m, n = B.shape
        statistics: dict[str, tuple[int, int]] = {}

        for bits in itertools.product((0, 1), repeat=n):
            x = np.asarray(bits, dtype=int)
            satisfied = int(np.count_nonzero(((B @ x) % 2) == v))
            statistics["".join(map(str, bits))] = (
                satisfied,
                2 * satisfied - m,
            )

        return statistics

    return (
        assignment_statistics,
        bounded_error_patterns,
        build_dqi_circuit,
        exact_solution_probabilities,
        optimal_binary_weights,
        sample_solution_counts,
        syndrome_of,
        validate_unique_syndromes,
    )


@app.cell
def _(
    bounded_error_patterns,
    np,
    optimal_binary_weights,
    validate_unique_syndromes,
):
    # Three constraints on two variables:
    # x0 = 0, x1 = 0, and x0 XOR x1 = 0.
    B = np.asarray(
        [
            [1, 0],
            [0, 1],
            [1, 1],
        ],
        dtype=int,
    )
    v = np.asarray([0, 0, 0], dtype=int)
    ell = 1

    m, n = B.shape
    weights = optimal_binary_weights(m, ell)
    error_patterns = bounded_error_patterns(m, ell)
    decoder_table = validate_unique_syndromes(B, error_patterns)

    return B, decoder_table, ell, error_patterns, m, n, v, weights


@app.cell
def _(B, decoder_table, ell, mo, pd, v, weights):
    _constraint_rows = [
        {
            "constraint": index,
            "B row": "".join(map(str, row)),
            "v": int(v[index]),
        }
        for index, row in enumerate(B)
    ]
    _decoder_rows = [
        {
            "syndrome": "".join(map(str, syndrome)),
            "decoded error": "".join(map(str, error)),
        }
        for syndrome, error in sorted(decoder_table.items())
    ]

    mo.vstack(
        [
            mo.md(
                f"""
                ## Example instance

                We use decoding radius \(\ell={ell}\) and optimal binary
                DQI weights

                \[
                (w_0,w_1)=({weights[0]:.6f},{weights[1]:.6f}).
                \]

                All errors of weight at most one have distinct syndromes, so
                the coherent lookup decoder is exact on the prepared support.
                """
            ),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("**Constraints**"),
                            mo.ui.table(pd.DataFrame(_constraint_rows)),
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.md("**Bounded decoder**"),
                            mo.ui.table(pd.DataFrame(_decoder_rows)),
                        ]
                    ),
                ]
            ),
        ]
    )
    return


@app.cell
def _(B, build_dqi_circuit, ell, m, mo, n, v, weights):
    dqi_circuit = build_dqi_circuit(B, v, ell, weights, measure=False)
    measured_dqi_circuit = build_dqi_circuit(B, v, ell, weights, measure=True)

    _drawing = dqi_circuit.draw(output="text", fold=120)
    mo.vstack(
        [
            mo.md("## Coherent DQI circuit"),
            mo.md(f"```text\n{_drawing}\n```"),
            mo.md(
                f"""
                The circuit uses **{dqi_circuit.num_qubits} qubits**:
                {m} error/decode qubits and {n} solution qubits. Only the
                solution register is measured in the sampled circuit.
                """
            ),
        ]
    )
    return dqi_circuit, measured_dqi_circuit


@app.cell
def _(mo):
    shots = mo.ui.slider(
        start=1_000,
        stop=20_000,
        step=1_000,
        value=5_000,
        label="Simulator shots",
        show_value=True,
    )
    shots
    return (shots,)


@app.cell
def _(
    B,
    assignment_statistics,
    dqi_circuit,
    exact_solution_probabilities,
    m,
    measured_dqi_circuit,
    n,
    pd,
    sample_solution_counts,
    shots,
    v,
):
    exact_probabilities, decoder_leakage = exact_solution_probabilities(
        dqi_circuit, m, n
    )
    sampled_counts = sample_solution_counts(
        measured_dqi_circuit,
        shots=shots.value,
        seed=11,
    )
    statistics = assignment_statistics(B, v)
    uniform_probability = 1.0 / (2**n)

    results_df = pd.DataFrame(
        [
            {
                "x": label,
                "satisfied": statistics[label][0],
                "f(x)": statistics[label][1],
                "DQI exact": exact_probabilities[label],
                "DQI sampled": sampled_counts.get(label, 0) / shots.value,
                "uniform": uniform_probability,
                "counts": sampled_counts.get(label, 0),
            }
            for label in sorted(exact_probabilities)
        ]
    )

    expected_dqi = float(
        sum(
            exact_probabilities[label] * statistics[label][0]
            for label in exact_probabilities
        )
    )
    expected_uniform = float(
        sum(uniform_probability * value[0] for value in statistics.values())
    )
    best_satisfied = max(value[0] for value in statistics.values())
    optimal_probability = float(
        sum(
            exact_probabilities[label]
            for label, value in statistics.items()
            if value[0] == best_satisfied
        )
    )

    return (
        decoder_leakage,
        exact_probabilities,
        expected_dqi,
        expected_uniform,
        optimal_probability,
        results_df,
        sampled_counts,
    )


@app.cell
def _(
    decoder_leakage,
    expected_dqi,
    expected_uniform,
    mo,
    optimal_probability,
    results_df,
    shots,
):
    mo.vstack(
        [
            mo.md("## Results"),
            mo.callout(
                mo.md(
                    f"""
                    **Optimal assignment probability:** {optimal_probability:.3%}

                    **Expected satisfied constraints:** {expected_dqi:.6f} / 3
                    with DQI, versus {expected_uniform:.6f} / 3 under uniform
                    sampling.

                    **Decoder leakage:** {decoder_leakage:.3e}

                    The shot column uses {shots.value:,} seeded simulator shots.
                    """
                ),
                kind="success",
            ),
            mo.ui.table(
                results_df,
                format_mapping={
                    "DQI exact": "{:.6f}",
                    "DQI sampled": "{:.6f}",
                    "uniform": "{:.6f}",
                },
                selection=None,
            ),
        ]
    )
    return


@app.cell
def _(alt, results_df):
    _plot_data = results_df[["x", "DQI exact", "uniform"]].melt(
        id_vars="x",
        var_name="distribution",
        value_name="probability",
    )
    _chart = (
        alt.Chart(_plot_data)
        .mark_bar()
        .encode(
            x=alt.X("x:N", title="assignment x", sort=None),
            xOffset="distribution:N",
            y=alt.Y("probability:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("distribution:N", title=None),
            tooltip=[
                alt.Tooltip("x:N"),
                alt.Tooltip("distribution:N"),
                alt.Tooltip("probability:Q", format=".6f"),
            ],
        )
        .properties(
            title="DQI sampling bias versus uniform sampling",
            width=520,
            height=300,
        )
    )
    _chart
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## What the run demonstrates

        Here \(x=00\) satisfies all three constraints, while each other
        assignment satisfies one. The optimal degree-one weights create a
        polynomial amplitude proportional to

        \[
        P(f(x))=w_0+\frac{w_1}{\sqrt{m}}f(x).
        \]

        Consequently,

        \[
        \Pr(00)=\frac{2+\sqrt{3}}{4}\approx 0.933013,
        \]

        while each other assignment has probability approximately
        \(0.022329\). Uniform sampling would find \(00\) only \(25\%\) of the
        time.

        This is a pedagogical exact simulation, not a quantum-advantage
        demonstration. The nontrivial scaling question is whether the
        syndrome decoder can be implemented coherently and efficiently for
        a useful problem family and decoding radius.

        ## References

        - S. P. Jordan et al., *Optimization by Decoded Quantum
          Interferometry*, [arXiv:2408.08292](https://arxiv.org/abs/2408.08292).
        - Authors' supporting C++/Python material:
          [Zenodo record 16741169](https://zenodo.org/records/16741169).
        - PennyLane's independent DQI tutorial:
          [Decoded Quantum Interferometry](https://pennylane.ai/demos/tutorial_dqi).
        - Independent circuit implementation:
          [BankNatchapol/DQI-Circuit](https://github.com/BankNatchapol/DQI-Circuit).
        """
    )
    return


if __name__ == "__main__":
    app.run()
```__
