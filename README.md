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

Train PI-JEPA:
```bash
python scripts/train.py --config configs/darcy.yaml
```

Run benchmarks:
```bash
python scripts/benchmark.py --config configs/benchmark.yaml
```
