Code Associated With:

## PI-JEPA: A Physics-Informed Joint Embedding Predictive Architecture for Multi-Step Coupled PDE Surrogate Modeling

#### Brandon Yee,<sup>*1</sup> Pairie Koh<sup>1,2</sup> Ryan Gomez<sup>1</sup>

<sup>1</sup> Physics Lab, Yee Collins Research Group

<sup>2</sup> Graduate School of Business, Stanford University

<sup>*</sup> Correspondance: b.yee@ycrg-labs.org

---

### Structure

```
PI-JEPA/          # Core library code
├── models/       # PIJEPA, encoder, decoder, predictor
├── training/     # Loss functions, EMA, training engine
├── data/         # Dataset and preprocessing
├── eval/         # Metrics and rollout evaluation
├── physics/      # Physics-informed loss (Darcy flow)
├── utils/        # Config, logging, checkpointing
└── benchmarks/   # Baseline model wrappers (FNO, DeepONet, etc.)

scripts/          # Runnable scripts
├── train.py      # Train PI-JEPA model
└── benchmark.py  # Run benchmark comparisons

configs/          # Configuration files
├── darcy.yaml    # Main training config
├── ablation.yaml # Ablation study configs
└── benchmark.yaml # Benchmark config
```

### Usage

**Sanity Check:**
```bash
python scripts/run_experiments.py --sanity-check
```

**Quick Start - Run Everything:**
```bash
python scripts/run_experiments.py --all
```

**Or Step-by-Step (Recommended):**

**1. Pretraining (~24-48 hours on GPU)**
```bash
python scripts/run_experiments.py --pretrain
# Or directly:
python scripts/train.py --config configs/darcy.yaml
```
This trains PI-JEPA for 500 epochs with the paper specifications (batch 64, AdamW lr=1.5e-4, EMA τ: 0.99→0.999).

**2. Fine-Tuning Data Efficiency Sweep (~6-12 hours)**
```bash
python scripts/run_experiments.py --finetune --checkpoint outputs/checkpoint_final.pt
```
Runs fine-tuning with N_l ∈ {10, 25, 50, 100, 250, 500} labeled samples to generate data efficiency curves.

**3. Rollout Evaluation (~1-2 hours)**
```bash
python scripts/run_experiments.py --evaluate --checkpoint outputs/checkpoint_final.pt
```
Evaluates at horizons T ∈ {1, 5, 10, 20, 40} with noise annealing.

**4. Benchmark Comparison (~12-24 hours)**
```bash
python scripts/run_experiments.py --benchmark
# Or directly:
python scripts/benchmark.py --config configs/benchmark.yaml
```
Compares against FNO, PINO, GeoFNO, DeepONet, PINN, PI-LatentNO.

**5. Ablation Studies**
```bash
python scripts/run_experiments.py --ablation
```
Generates configs for ablating physics loss, variance/covariance regularization, and K ∈ {1, 2, 3}.

**6. Generate Publication Figures**
```bash
python scripts/run_experiments.py --figures
```
Creates data efficiency curves, error accumulation plots, and rollout comparisons.