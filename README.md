# Circuits Crystallize: A Nucleation Theory of Conjunction-Circuit Formation

Code, data, and figures for the paper *Circuits Crystallize: A Nucleation Theory of Whether, When,
and How Conjunction Circuits Form, Compete, and Regenerate*.

We treat the training-time formation of a transformer circuit as **nucleation** — a metastable loss
plateau, a barrier crossed by stochastic ignition, and growth by a moving front. From quantities
computable *before training*, four laws forecast a conjunction circuit's developmental fate; each is
frozen on a calibration split and tested out of sample, on synthetic rigs and on Pythia (160M–1B).

## Layout

```
figures/      make_figs.py + the four figures      -- regenerates every figure from the data below
experiments/  <Experiment>/{<script>.py, data/*.json}
              _toy/data/                            -- toy JSONs read by two of the figures
```

Every script is self-contained and writes the raw JSON that its claims are computed from; every
number in the paper is recomputable from `experiments/`.

## Reproduce the figures

```
cd figures && python make_figs.py      # reads ../experiments/**, writes the four PDFs
```

## The four laws and where to find them

| Law | Claim | Key experiments | Figure |
|-----|-------|-----------------|--------|
| **1. Whether** | a capacity wall with a per-model density floor (held-out 0–3%) | `GeomInductionXval`, `DataDrivenWhen`, `DriveHaddr`, `NucleationFormula` | — |
| **2. When** | attempt frequency × combinatorial barrier $p^K$; held-out ±9% | `PythiaBarrier`, `KconjToy`, `BarrierExponent`, `BarrierLadder`, `NucleusGrowth` | `fig_law2_when` |
| **3. Interaction** | co-nucleation on capacity-rich models; suppression in the scarce limit | `SuppLaw`, `MoreCircuits` | — |
| **4. Regeneration** | seeded re-nucleation, basin-governed at scale (held-out 9%) | `ForceBasin`, `BasinCensus`, `SeedDissect`, `KscaleRegen`, `RegenMicroscope`, `RegenTwoLayer`, `SeedComponents` | `fig_law4_regen` |

Supporting evidence:

| Topic | Experiments | Figure |
|-------|-------------|--------|
| Identity (over-determination, TTT nose, melt) | `BathFDT`, `Goccupancy`, `TempNose`, `Melt`, `Langevin` | `fig_identity` |
| Large-model thermodynamics, cross-scale $K{=}3$ | `KscaleRegen`, `KscalePythia`, `DelocTimeline`, `CrossScaleRegen` | — |
| Grokking / Ostwald / boundary | `GrokIgnition`, `GrokNucleation`, `OstwaldLadder` | — |
| What crystallizes (QK/OV = associative memory) | `Q1Identity`, `Q1OVLn`, `Q1OVHopfield`, `QKgrow` | `fig_qkov` |

## Requirements

Python with `numpy`, `matplotlib`, `torch`, and `transformers` (for the real-LM scripts). Real-LM
experiments use the public Pythia checkpoints from Hugging Face.

## Scope

The demonstrated scope is conjunction-based circuits with the induction head as the primary instance.
The laws' functional forms are universal; their material constants are measured once per model rather
than derived from first principles. See §11 of the paper for the exact frontier.
