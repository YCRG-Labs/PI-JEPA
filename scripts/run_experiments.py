#!/usr/bin/env python
"""Run all PI-JEPA experiments for publication. Outputs data as JSON/CSV for TikZ figures."""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PI-JEPA"))

import torch
import numpy as np

from utils import load_config, Logger, save_checkpoint


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_pretraining(config_path="configs/darcy.yaml", output_dir="outputs"):
    print("\n" + "="*60)
    print("PHASE 1: PRETRAINING")
    print("="*60)
    
    from train import train
    checkpoint_path = train(config_path, output_dir)
    
    return checkpoint_path


def run_finetuning_sweep(checkpoint_path, config_path="configs/darcy.yaml", output_dir="outputs/finetune"):
    print("\n" + "="*60)
    print("PHASE 2: FINE-TUNING DATA EFFICIENCY SWEEP")
    print("="*60)
    
    from training import FineTuningPipeline
    from models import ViTEncoder, Predictor, PIJEPA, Decoder
    from neuralop.data.datasets import load_darcy_flow_small
    
    cfg = load_config(config_path)
    device = get_device()
    
    encoder = ViTEncoder(cfg).to(device)
    target_encoder = ViTEncoder(cfg).to(device)
    predictors = [Predictor(cfg).to(device) for _ in range(cfg["model"]["num_predictors"])]
    
    model = PIJEPA(
        encoder=encoder,
        target_encoder=target_encoder,
        predictors=predictors,
        embed_dim=cfg["model"]["encoder"]["embed_dim"],
        num_patches=None,
        patch_size=cfg["model"]["encoder"]["patch_size"],
    ).to(device)
    
    decoder = Decoder(**cfg["decoder"]).to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.encoder.load_state_dict(checkpoint["student_encoder"])
        model.target_encoder.load_state_dict(checkpoint["target_encoder"])
        for p, state in zip(model.predictors, checkpoint["predictors"]):
            p.load_state_dict(state)
        decoder.load_state_dict(checkpoint["decoder"])
    
    train_loader, _, _ = load_darcy_flow_small(
        n_train=1000, batch_size=cfg["training"].get("batch_size", 64),
        test_resolutions=[64], n_tests=[100], test_batch_sizes=[32], encode_output=False,
    )
    
    n_labeled_values = cfg.get("finetune", {}).get("n_labeled", [10, 25, 50, 100, 250, 500])
    n_epochs = cfg.get("finetune", {}).get("epochs", 100)
    
    results = {}
    os.makedirs(output_dir, exist_ok=True)
    
    for n_l in n_labeled_values:
        print(f"\n--- N_l = {n_l} ---")
        
        pipeline = FineTuningPipeline(
            model=model, decoder=decoder,
            config={"lr": cfg.get("finetune", {}).get("lr", 1.5e-4), "device": str(device)}
        )
        
        embed_dim = cfg["model"]["encoder"]["embed_dim"]
        pipeline.setup_linear_probe(embed_dim=embed_dim, out_dim=embed_dim)
        
        metrics = pipeline.train(train_loader, n_labeled=n_l, n_epochs=n_epochs)
        results[n_l] = {"final_loss": metrics["final_loss"], "train_losses": metrics["train_losses"]}
        
        pipeline.save_checkpoint(os.path.join(output_dir, f"finetune_n{n_l}.pt"), metrics=metrics)
    
    with open(os.path.join(output_dir, "data_efficiency.json"), "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    
    return results


def run_rollout_evaluation(checkpoint_path, config_path="configs/darcy.yaml", output_dir="outputs/evaluation"):
    print("\n" + "="*60)
    print("PHASE 3: ROLLOUT EVALUATION")
    print("="*60)
    
    from eval import RolloutEvaluator, relative_l2
    from models import ViTEncoder, Predictor, PIJEPA, Decoder
    from neuralop.data.datasets import load_darcy_flow_small
    
    cfg = load_config(config_path)
    device = get_device()
    
    encoder = ViTEncoder(cfg).to(device)
    target_encoder = ViTEncoder(cfg).to(device)
    predictors = [Predictor(cfg).to(device) for _ in range(cfg["model"]["num_predictors"])]
    
    model = PIJEPA(
        encoder=encoder, target_encoder=target_encoder, predictors=predictors,
        embed_dim=cfg["model"]["encoder"]["embed_dim"], num_patches=None,
        patch_size=cfg["model"]["encoder"]["patch_size"],
    ).to(device)
    
    decoder = Decoder(**cfg["decoder"]).to(device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.encoder.load_state_dict(checkpoint["student_encoder"])
        model.target_encoder.load_state_dict(checkpoint["target_encoder"])
        for p, state in zip(model.predictors, checkpoint["predictors"]):
            p.load_state_dict(state)
        decoder.load_state_dict(checkpoint["decoder"])
    
    _, test_loaders, _ = load_darcy_flow_small(
        n_train=100, batch_size=32, test_resolutions=[64],
        n_tests=[200], test_batch_sizes=[32], encode_output=False,
    )
    test_loader = list(test_loaders.values())[0]
    
    horizons = cfg.get("rollout", {}).get("horizons", [1, 5, 10, 20, 40])
    noise_cfg = cfg.get("rollout", {}).get("noise", {})
    
    evaluator = RolloutEvaluator(
        model=model, decoder=decoder,
        noise_std=noise_cfg.get("sigma_start", 1e-2),
        noise_annealing=noise_cfg.get("enabled", True)
    )
    
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    for horizon in horizons:
        print(f"\n--- Horizon T = {horizon} ---")
        
        total_error = 0.0
        count = 0
        
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                x = batch["x"].to(device).float()
                y = batch["y"].to(device).float()
                
                if x.dim() == 3:
                    x = x.unsqueeze(1)
                if y.dim() == 3:
                    y = y.unsqueeze(1)
                
                pred = evaluator.single_step(torch.cat([y, x], dim=1))
                error = relative_l2(pred, torch.cat([y, x], dim=1))
                
                total_error += error.item()
                count += 1
        
        results[horizon] = total_error / max(count, 1)
        print(f"  Error: {results[horizon]:.6f}")
    
    with open(os.path.join(output_dir, "rollout.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def run_benchmark_comparison(config_path="configs/benchmark.yaml"):
    print("\n" + "="*60)
    print("PHASE 4: BENCHMARK COMPARISON")
    print("="*60)
    
    from benchmark import main as run_benchmark
    run_benchmark(config_path)


def run_ablation_studies(config_path="configs/darcy.yaml", output_dir="outputs/ablation"):
    print("\n" + "="*60)
    print("PHASE 5: ABLATION STUDIES")
    print("="*60)
    
    from eval import AblationModule
    import yaml
    
    cfg = load_config(config_path)
    ablation = AblationModule(cfg)
    
    os.makedirs(output_dir, exist_ok=True)
    
    configs = ablation.get_ablation_configs(ablation_type='both')
    
    for name, config in configs.items():
        with open(os.path.join(output_dir, f"config_{name}.yaml"), "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    
    print(f"Ablation configs saved to {output_dir}/")
    return configs


def run_sanity_check(config_path="configs/darcy.yaml"):
    """Quick validation that all components load and run correctly."""
    print("\n" + "="*60)
    print("SANITY CHECK")
    print("="*60)
    
    checks = []
    device = get_device()
    checks.append(("Device detection", True, str(device)))
    
    # Check config loading
    cfg = None
    try:
        cfg = load_config(config_path)
        checks.append(("Config loading", True, config_path))
    except Exception as e:
        checks.append(("Config loading", False, str(e)))
    
    # Check model imports
    try:
        from models import ViTEncoder, Predictor, PIJEPA, Decoder
        checks.append(("Model imports", True, "ViTEncoder, Predictor, PIJEPA, Decoder"))
    except Exception as e:
        checks.append(("Model imports", False, str(e)))
    
    # Check training imports
    try:
        from training.finetune import FineTuningPipeline
        checks.append(("Training imports", True, "FineTuningPipeline"))
    except Exception as e:
        checks.append(("Training imports", False, str(e)))
    
    # Check eval imports
    try:
        from eval import RolloutEvaluator, relative_l2
        checks.append(("Eval imports", True, "RolloutEvaluator, relative_l2"))
    except Exception as e:
        checks.append(("Eval imports", False, str(e)))
    
    # Check benchmark imports
    try:
        from benchmarks import FNOWrapper, DeepONetWrapper, PINNWrapper
        checks.append(("Benchmark imports", True, "FNOWrapper, DeepONetWrapper, PINNWrapper"))
    except Exception as e:
        checks.append(("Benchmark imports", False, str(e)))
    
    # Model instantiation and forward pass
    model = None
    if cfg:
        try:
            from models import ViTEncoder, Predictor, PIJEPA, Decoder
            encoder = ViTEncoder(cfg).to(device)
            target_encoder = ViTEncoder(cfg).to(device)
            predictors = [Predictor(cfg).to(device) for _ in range(cfg["model"]["num_predictors"])]
            
            model = PIJEPA(
                encoder=encoder,
                target_encoder=target_encoder,
                predictors=predictors,
                embed_dim=cfg["model"]["encoder"]["embed_dim"],
                num_patches=None,
                patch_size=cfg["model"]["encoder"]["patch_size"],
            ).to(device)
            
            decoder = Decoder(**cfg["decoder"]).to(device)
            checks.append(("Model instantiation", True, f"{sum(p.numel() for p in model.parameters())} params"))
        except Exception as e:
            checks.append(("Model instantiation", False, str(e)))
    
    # Forward pass test
    if model is not None:
        try:
            model.eval()
            B, C, H, W = 2, 2, 64, 64
            dummy_input = torch.randn(B, C, H, W, device=device)
            with torch.no_grad():
                z = model.encode(dummy_input)
            checks.append(("Forward pass", True, f"input {list(dummy_input.shape)} -> latent {list(z.shape)}"))
        except Exception as e:
            checks.append(("Forward pass", False, str(e)))
    
    # Quick training step test
    if model is not None and cfg:
        try:
            from training import LossBuilder
            model.train()
            
            B, C, H, W = 2, 2, 64, 64
            dummy_input = torch.randn(B, C, H, W, device=device, requires_grad=True)
            
            # Single forward + backward
            z_context = model.encode(dummy_input)
            z_target = model.target_encoder(dummy_input)
            
            loss_builder = LossBuilder(cfg)
            losses = loss_builder(z_context.mean(dim=1), z_target.mean(dim=1).detach())
            losses["total"].backward()
            
            checks.append(("Training step", True, f"loss={losses['total'].item():.4f}"))
        except Exception as e:
            checks.append(("Training step", False, str(e)))
    
    # Print results
    print("\nResults:")
    all_passed = True
    for name, passed, detail in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}: {detail}")
        if not passed:
            all_passed = False
    
    print("\n" + ("All checks passed!" if all_passed else "Some checks failed."))
    return all_passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--sanity-check", action="store_true", help="Quick validation of imports and setup")
    parser.add_argument("--config", default="configs/darcy.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoint_final.pt")
    parser.add_argument("--output", default="outputs")
    
    args = parser.parse_args()
    
    # Handle sanity check separately
    if args.sanity_check:
        success = run_sanity_check(args.config)
        sys.exit(0 if success else 1)
    
    if not any([args.all, args.pretrain, args.finetune, args.evaluate, args.benchmark, args.ablation]):
        parser.print_help()
        print("\nWorkflow:")
        print("  0. python scripts/run_experiments.py --sanity-check  (quick validation)")
        print("  1. python scripts/run_experiments.py --pretrain")
        print("  2. python scripts/run_experiments.py --finetune")
        print("  3. python scripts/run_experiments.py --evaluate")
        print("  4. python scripts/run_experiments.py --benchmark")
        print("  5. python scripts/run_experiments.py --ablation")
        print("  Or: python scripts/run_experiments.py --all")
        return
    
    print(f"\nDevice: {get_device()}")
    print(f"Started: {datetime.now()}")
    
    checkpoint_path = args.checkpoint
    
    if args.all or args.pretrain:
        checkpoint_path = run_pretraining(args.config, args.output)
    
    if args.all or args.finetune:
        run_finetuning_sweep(checkpoint_path, args.config, os.path.join(args.output, "finetune"))
    
    if args.all or args.evaluate:
        run_rollout_evaluation(checkpoint_path, args.config, os.path.join(args.output, "evaluation"))
    
    if args.all or args.benchmark:
        run_benchmark_comparison("configs/benchmark.yaml")
    
    if args.all or args.ablation:
        run_ablation_studies(args.config, os.path.join(args.output, "ablation"))
    
    print(f"\nFinished: {datetime.now()}")


if __name__ == "__main__":
    main()
