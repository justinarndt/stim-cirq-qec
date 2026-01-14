# Architecture

## Hybrid Adaptive QEC Stack

```
┌────────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACE                                 │
│                    AdaptiveSurfaceCode(config)                          │
└──────────────────────────────┬─────────────────────────────────────────┘
                               │
        ┌──────────────────────▼──────────────────────┐
        │            HYBRID INTEGRATION LAYER          │
        │                                              │
        │  ┌─────────────────────────────────────────┐ │
        │  │        HybridAdaptiveSampler            │ │
        │  │  - Combines Stim (fast) + Cirq (rich)   │ │
        │  │  - Provides unified sampling interface  │ │
        │  └─────────────────────────────────────────┘ │
        │                                              │
        │  ┌──────────────┐    ┌──────────────────┐   │
        │  │ StimCirqBridge│    │ NoiseModel      │   │
        │  │ - Convert    │    │ - Pauli (Stim)  │   │
        │  │ - Sample     │    │ - Kraus (Cirq)  │   │
        │  │ - Extract DEM│    │ - Drift (OU)    │   │
        │  └──────────────┘    └──────────────────┘   │
        └──────────────────────┬──────────────────────┘
                               │
   ┌───────────────────────────┼───────────────────────────┐
   │                           │                           │
   ▼                           ▼                           ▼
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│  FEEDBACK   │         │ DIAGNOSTICS │         │ REMEDIATION │
│             │         │             │         │             │
│ Syndrome    │         │ Hamiltonian │         │ Pulse       │
│ Feedback    │         │ Learner     │         │ Synthesizer │
│ Controller  │         │             │         │             │
│             │         │ - MBL trace │         │ - Optimal   │
│ - Integral  │         │ - L-BFGS-B  │         │   control   │
│ - Latency   │         │ - Defect    │         │ - Fidelity  │
│ - Bounds    │         │   detection │         │   recovery  │
└─────────────┘         └─────────────┘         └─────────────┘
```

## Data Flow

```
1. CIRCUIT GENERATION
   Cirq Circuit (coherent noise) → StimCirqBridge → Stim Circuit (Pauli noise)

2. SAMPLING
   Stim Circuit → Compile Sampler → Detection Events + Observable Flips

3. FEEDBACK CONTROL
   Detection Events → Syndrome Density → Controller → Correction Signal

4. DECODING
   Detection Events + Correction → Adaptive MWPM → Predictions

5. DIAGNOSTICS (offline)
   Experimental Trace → HamiltonianLearner → Recovered Couplings → Defect Map

6. REMEDIATION (offline)
   Defect Map → PulseSynthesizer → Optimized Control Pulse → Recovered Fidelity
```

## Component Responsibilities

| Component | Input | Output | Purpose |
|-----------|-------|--------|---------|
| StimCirqBridge | Cirq circuit, NoiseModel | Stim circuit, DEM | Cross-framework conversion |
| HybridAdaptiveSampler | Config, shots | SamplingResult | Unified sampling with feedback |
| SyndromeFeedbackController | Syndrome density | Correction signal | Real-time drift tracking |
| HamiltonianLearner | Imbalance trace | Coupling parameters | Hardware characterization |
| PulseSynthesizer | Diagnosed Hamiltonian | Optimal pulse | Fidelity recovery |
