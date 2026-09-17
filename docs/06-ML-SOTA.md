# Classical ML SOTA for power flow (reference)

> Deep-research pass, Jun 2024 – Aug 2026. Not quantum — this is the bar this project's quantum work needs to beat on ACOPF, researched explicitly for that reference purpose. Extends the one ML paper already in [03-LITERATURE.md](03-LITERATURE.md) (arXiv:2406.10428, Jun 2024). Every citation fetched and verified against its primary source; anything not independently confirmed is flagged inline as unverified rather than presented as fact.

## Read this first

[**Revisiting Deep AC-OPF**](https://arxiv.org/abs/2509.00655) (Dada & Lawrence, Cambridge, Aug 2025) re-benchmarks published ML-OPF models against plain linear regression on IEEE Case 30/118:

| Test case | Linear regression MSE | DeepOPF-V MSE | Linear wins by |
|---|---|---|---|
| Case 30 | 4.484×10⁻⁷ | 3.272×10⁻⁶ | ~7× |
| Case 118 | 1.188×10⁻⁷ | 6.458×10⁻⁶ | ~54× |

Quote: *"the current stage of ML solvers offers only marginal improvements over benchmark linear methods."* A lot of this field's "deep learning beats classical" results don't survive a properly-tuned classical baseline. Read every speedup/accuracy claim below with that in mind — and hold quantum results to the same standard before believing them.

## Foundation models for grids

| Model | Org / date | Verified numbers |
|---|---|---|
| [GridSFM](https://www.microsoft.com/en-us/research/blog/gridsfm-a-new-small-foundation-model-for-the-electric-grid/) | Microsoft Research, May 2026 | 1,000× faster than full AC-OPF solve, ~100× faster than DC approximation; median cost gap 2.23% (mean 3.41%) vs. IPOPT ground truth, 83% of scenarios <5% gap; warm-start gives 1.66× geomean speedup; feasibility classifier 94.5–96.1% accuracy (ROC-AUC 0.986, Texas 2k-bus case). Trained on 150+ topologies, ~500K scenarios. Base model + weights open (GitHub/Hugging Face); 80,000-bus production tier commercial. **Single most important comparison point found.** |
| [GridFM](https://lfenergy.org/projects/gridfm/) | IBM + Imperial College London + LF Energy + Hydro-Québec, roadmap Oct 2024 | Targets pretraining on 300K+ solved OPF problems. "GridFM-v0" targeted Q2 2025 — **delivery/benchmarks not confirmed**; treat as roadmap, not result. |
| [gridfm-datakit-v1](https://arxiv.org/abs/2512.14658) | IBM/Argonne/Rice, Dec 2025 | Apache-2.0 data-generation library feeding the GridFM consortium — PF/OPF data up to 10,000 buses, N-k topology perturbation, infeasible/edge-case sampling |
| [LUMINA-Bench](https://arxiv.org/abs/2605.02133) | Argonne et al., May 2026 | Not a model — open benchmark suite for multi-topology pretraining/transfer evaluation of grid surrogates |
| [PowerPM](https://arxiv.org/abs/2408.04057) | Zhejiang University, NeurIPS 2024 | Pretrained on electricity time-series (load forecasting, demand-side mgmt) — **not an OPF solver**, adjacent use case |
| PowerGPT | OpenReview id=ntSP0bzr8Y | **Unverified** — page blocked every fetch attempt; one snippet suggests Oct 2023 origin, possibly pre-dating this survey's window entirely |

Field has a dedicated venue now: 3rd "Foundation Models for the Electric Grid" workshop, Argonne National Laboratory, Feb 2025, 70+ international stakeholders.

## Graph neural networks for power flow

- [**HH-MPNN**](https://arxiv.org/abs/2510.06860) (Arowolo & Cremer, Oct 2025/Apr 2026) — best verified generalization result in this survey. Heterogeneous GNN + scalable transformer + physics-informed positional encoding. **<1% optimality gap, 14–2,000 buses; up to 5,000× speedup vs. interior-point solvers; <3% gap in zero-shot transfer to thousands of unseen N-1 topologies** despite training only on default topology.
- [**PINCO**](https://arxiv.org/abs/2410.04818) (ETH Zurich + MIT/CMU, Oct 2024) — unsupervised, physics-informed (augmented-Lagrangian) GNN, IEEE 30/57-bus + Swiss transmission network. 2–3 orders of magnitude speedup vs. IPOPT, constraint satisfaction comparable to DeepOPF-FT, built-in feasible/infeasible clustering.
- [**Scalable Heterogeneous GNN Foundation Models**](https://arxiv.org/abs/2605.23194) (ORNL/Argonne, May 2026) — HydraGNN-based, preserves bus/generator/load/branch node-edge typing, trained across 10 PGLib-OPF cases (14–13,659 buses, 3M graph instances). Compact ~1.6–1.7M-param models found optimal; pretrain+fine-tune beats from-scratch on low data. No headline speedup number in accessible abstract.

## Physics-informed ML (successors to the OPF-DNN lineage)

The OPF-DNN approach already in [03-LITERATURE.md](03-LITERATURE.md) traces back to Nellikkath & Chatzivasileiadis (2021, arXiv:2110.02672). Direct continuations:

- [**Neural Networks for AC-OPF: Improving Worst-Case Guarantees during Training**](https://arxiv.org/abs/2510.23196) (DTU, Oct 2025) — verification-informed training bakes worst-case constraint violation into the loss; claims first fully-verified operational constraints for large-scale AC-OPF proxies (57–793 buses). Specific accuracy numbers not extractable from accessible text.
- [**Homotopy-Guided Self-Supervised Learning for AC-OPF**](https://arxiv.org/abs/2511.11677) (PNNL-affiliated, Nov 2025) — continuous deformation from relaxed to full nonconvex AC-OPF, no solver-labeled data needed. Higher feasibility than non-homotopy baselines; exact numbers not stated in abstract.
- [**SenseFlow**](https://arxiv.org/abs/2505.12302) (May 2025) — physics-informed FlowNet + iterative self-ensembling refinement. Claims outperform existing methods; no concrete numbers extracted.

**Gap found**: no paper directly benchmarks against and surpasses OPF-DNN's specific claimed numbers (104–10000× speedup, ~0.27% avg error) on the same terms. GridSFM's numbers are in a comparable range but on a different metric/test system — not a like-for-like successor.

## Novel architectures — transformers, diffusion, flow matching

- **OPFormer-V**, inside the *Revisiting Deep AC-OPF* paper above — transformer voltage predictor; only modestly beats plain linear regression in the paper's own tests (see "Read this first").
- [OPFormer (CNN-based transformer)](https://ieeexplore.ieee.org/document/10888727/) (ICASSP 2025) — title/venue confirmed via search; IEEE Xplore page paywalled, **results unverified**.
- [**FMOPF**](https://arxiv.org/abs/2607.22788) (Jul 2026) — latent flow matching + constraint-aware interaction priors; claims best Newton-Raphson warm-starts and lowest tail-risk among generative methods at several-hundred-bus scale "while preserving full feasibility" (first to do so, per authors) — no head-to-head numbers in accessible abstract.
- [**DiffOPF**](https://arxiv.org/abs/2510.14075) (Oct 2025/Mar 2026) — reframes OPF as conditional generative sampling; produces multiple statistically credible warm starts instead of one deterministic answer. No concrete numbers accessible.

## Differentiable optimization / learning-to-optimize

- [Differentiable DC-approximation enhancement](https://arxiv.org/abs/2504.01970) (Georgia Tech, Mar 2025) — NN predicts adjusted shunt/susceptance parameters so DC-OPF better approximates AC behavior, trained end-to-end through the optimization layer via implicit function theorem. "Significantly improved accuracy" claimed, no percentages extracted.
- [Constraint-Informed Active Learning for ACOPF Proxies](https://arxiv.org/abs/2511.06248) (Georgia Tech + Los Alamos, Nov 2025) — active-constraint-set sampling for training-data selection; claims better generalization per training budget, no numbers extracted.
- DiffOPF and the homotopy-SSL paper above are also L2O-flavored (generative warm-starts; continuation-method curriculum, respectively).

## Benchmark datasets — is there a standard yet?

Close to yes:

| Dataset/toolkit | Scale | Note |
|---|---|---|
| [**PGLearn**](https://arxiv.org/abs/2505.22825) (Georgia Tech, May 2025) | >10M OPF samples | Closest thing to a converged standard: AC/DC/SOC formulations, time-series for large systems, [Hugging Face-hosted](https://huggingface.co/PGLearn), CC BY 4.0. Explicitly built to fill the "no standardized datasets/metrics" gap the field kept flagging |
| [OPFData](https://arxiv.org/abs/2406.07234) (Google DeepMind, Jun 2024) | 13,000 solved instances, 8 topologies, 118–6,515 buses | Right at the edge of this project's "already known" cutoff — distinct from the OPF-DNN paper, flagged explicitly since it's easy to conflate. PyTorch Geometric-compatible |
| gridfm-datakit-v1 (above) | up to 10,000 buses | Part of GridFM tooling |
| LUMINA-Bench (above) | multi-topology | Transfer-evaluation focused |
| [Dataset-quality framework](https://arxiv.org/abs/2508.19083) (RSE/Polimi, Aug 2025) | — | Proposes standardized quality metrics (variability, constraint-activation diversity) + a sampling method beating uniform-random and 3 other generators |
| PGLib-OPF | legacy | Still the underlying test-case substrate nearly everything above builds on |

## Reinforcement learning for grid control

Consistent theme: **not production-ready**, but tooling has matured.

- [RL2Grid](https://arxiv.org/abs/2503.23101) (RTE France + MIT/CMU, Mar 2025) — standardized RL benchmark on RTE's Grid2Op simulator. Evaluated classic RL baselines; concludes current methods are inadequate for real systems.
- [Graph RL for Power Grids: Survey](https://arxiv.org/abs/2407.04522) (Jul 2024, updated through Jan 2026) — direct quote: *"not yet deployable to real-world applications."*
- [Safe RL for Power Systems: Review](https://arxiv.org/abs/2407.00304) (Jun 2024, rev. Jun 2025) — scalability + robustness under uncertainty remain unresolved.
- Context: Grid2Op / L2RPN (RTE France, hosted by LF Energy) remains the standard open RL environment/competition for grid topology control since 2020.
- Also found, not deep-verified: [cascading-failure mitigation RL](https://arxiv.org/abs/2505.09012) (May 2025); [RL-ADN for storage dispatch](https://arxiv.org/abs/2408.03685) (Aug 2024).

## Other honest limitations

- [**Scaling Laws of ML for OPF**](https://arxiv.org/abs/2601.02706) (Jan 2026) — first systematic scaling study (0.1K–40K samples). Finds power-law relationships for accuracy/constraint-violation/speed, but **prediction accuracy and constraint feasibility diverge under scale** — more data/compute doesn't proportionally buy more feasibility. Undercuts the "just scale it up" assumption.
- [Rethinking Neural Width for AC-OPF Proxies](https://arxiv.org/abs/2606.03125) (Jun 2026) — field has had no systematic method for choosing network width; their width-selection method matches literature baselines with up to 10× fewer neurons/layer, implying a lot of "SOTA" architectures were over-parameterized without earning it.
- Unverified/unsourced snippet, flagged not confirmed: worst-case ACOPF-proxy optimality gaps reportedly ~40× larger than mean-case gaps in some ICNN/DNN studies — could not trace to one fetchable source.

## Bottom line

No single blockbuster number defines "current best." Purpose-built architectures (HH-MPNN's sub-1% gap + 5,000× speedup, GridSFM's 1,000× speedup at ~2–3% cost gap) show real engineering progress since the OPF-DNN paper, and the field now has closer-to-standardized benchmarks (PGLearn, OPFData, LUMINA-Bench). But the field's own most rigorous 2025 self-audit found published "DL beats classical" results routinely lose to a properly-tuned linear regression baseline. **The bar for this project's quantum work is two-tiered**: the flashy claimed numbers a headline comparison would cite, and the more boring, more defensible well-tuned-classical-solver-plus-warm-start baseline that a claim of quantum advantage actually has to beat to survive scrutiny. Apply the same skepticism to quantum results that this field just applied to its own.

---

*Related: [[03-LITERATURE]] · [[02-ALGORITHMS]] · [[05-QUANTUM-UPDATES]] · [[versions/V0_SUMMARY]]*
