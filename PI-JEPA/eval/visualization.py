"""Visualization and Ablation Tools for PI-JEPA."""

from typing import Dict, List, Optional, Union, Any, Literal
import copy

import torch
import numpy as np


class VisualizationModule:
    """Visualization tools for generating paper figures."""
    
    @staticmethod
    def plot_rollout_comparison(
        pred: torch.Tensor,
        target: torch.Tensor,
        timesteps: List[int],
        save_path: Optional[str] = None,
        channel_names: Optional[List[str]] = None,
        figsize: tuple = (16, 8),
        cmap: str = 'viridis',
        title: Optional[str] = None
    ) -> Optional[Any]:
        """Generate rollout prediction vs ground truth comparison plots."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")
        
        # Handle batch dimension
        if pred.dim() == 5:
            pred = pred[0]  # Take first sample
            target = target[0]
        
        # Convert to numpy
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        T, C, H, W = pred_np.shape
        
        # Default channel names
        if channel_names is None:
            if C == 2:
                channel_names = ['Pressure', 'Saturation']
            else:
                channel_names = [f'Channel {i}' for i in range(C)]
        
        # Filter valid timesteps
        valid_timesteps = [t for t in timesteps if 0 <= t < T]
        n_timesteps = len(valid_timesteps)
        
        if n_timesteps == 0:
            raise ValueError(f"No valid timesteps in {timesteps}. Available: 0 to {T-1}")
        
        # Create figure: rows = channels, cols = timesteps * 2 (pred + target)
        fig, axes = plt.subplots(C, n_timesteps * 2, figsize=figsize)
        
        if C == 1:
            axes = axes.reshape(1, -1)
        if n_timesteps == 1:
            axes = axes.reshape(C, -1)
        
        for c_idx, c_name in enumerate(channel_names):
            for t_idx, t in enumerate(valid_timesteps):
                # Prediction
                ax_pred = axes[c_idx, t_idx * 2]
                im_pred = ax_pred.imshow(pred_np[t, c_idx], cmap=cmap)
                ax_pred.set_title(f'Pred {c_name}\nt={t}', fontsize=10)
                ax_pred.axis('off')
                plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)
                
                # Ground truth
                ax_gt = axes[c_idx, t_idx * 2 + 1]
                im_gt = ax_gt.imshow(target_np[t, c_idx], cmap=cmap)
                ax_gt.set_title(f'GT {c_name}\nt={t}', fontsize=10)
                ax_gt.axis('off')
                plt.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)
        
        if title:
            fig.suptitle(title, fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return None
        
        return fig

    @staticmethod
    def plot_data_efficiency_curve(
        results: Dict[int, float],
        save_path: Optional[str] = None,
        figsize: tuple = (8, 6),
        xlabel: str = 'Number of Labeled Samples ($N_l$)',
        ylabel: str = 'Relative $\\ell_2$ Error',
        title: Optional[str] = None,
        log_scale_x: bool = True,
        log_scale_y: bool = False,
        marker: str = 'o',
        color: str = 'blue',
        label: Optional[str] = None,
        comparison_results: Optional[Dict[str, Dict[int, float]]] = None
    ) -> Optional[Any]:
        """Generate data efficiency curve: error vs N_l."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort by N_l
        n_labeled = sorted(results.keys())
        errors = [results[n] for n in n_labeled]
        
        # Plot main curve
        main_label = label or 'PI-JEPA'
        ax.plot(n_labeled, errors, marker=marker, color=color, label=main_label, linewidth=2, markersize=8)
        
        # Plot comparison methods if provided
        if comparison_results:
            colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']
            markers = ['s', '^', 'D', 'v', '<', '>']
            for i, (method_name, method_results) in enumerate(comparison_results.items()):
                method_n = sorted(method_results.keys())
                method_errors = [method_results[n] for n in method_n]
                ax.plot(method_n, method_errors, 
                       marker=markers[i % len(markers)], 
                       color=colors[i % len(colors)],
                       label=method_name, linewidth=2, markersize=8, linestyle='--')
        
        # Set scales
        if log_scale_x:
            ax.set_xscale('log')
        if log_scale_y:
            ax.set_yscale('log')
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14)
        
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return None
        
        return fig

    @staticmethod
    def plot_error_accumulation(
        errors: Dict[int, float],
        horizons: Optional[List[int]] = None,
        save_path: Optional[str] = None,
        figsize: tuple = (8, 6),
        xlabel: str = 'Rollout Horizon (T)',
        ylabel: str = 'Relative $\\ell_2$ Error',
        title: Optional[str] = None,
        marker: str = 'o',
        color: str = 'blue',
        label: Optional[str] = None,
        show_cumulative: bool = False,
        comparison_errors: Optional[Dict[str, Dict[int, float]]] = None
    ) -> Optional[Any]:
        """Generate per-timestep error accumulation plot."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get horizons to plot
        if horizons is None:
            horizons = sorted(errors.keys())
        else:
            horizons = [h for h in horizons if h in errors]
        
        error_values = [errors[h] for h in horizons]
        
        # Plot main curve
        main_label = label or 'PI-JEPA'
        ax.plot(horizons, error_values, marker=marker, color=color, 
               label=main_label, linewidth=2, markersize=8)
        
        # Plot cumulative error if requested
        if show_cumulative:
            cumulative = []
            running_sum = 0.0
            for e in error_values:
                running_sum += e
                cumulative.append(running_sum)
            ax.plot(horizons, cumulative, marker='s', color='gray',
                   label=f'{main_label} (cumulative)', linewidth=2, 
                   markersize=6, linestyle='--', alpha=0.7)
        
        # Plot comparison methods if provided
        if comparison_errors:
            colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']
            markers = ['s', '^', 'D', 'v', '<', '>']
            for i, (method_name, method_errors) in enumerate(comparison_errors.items()):
                method_horizons = [h for h in horizons if h in method_errors]
                method_values = [method_errors[h] for h in method_horizons]
                ax.plot(method_horizons, method_values,
                       marker=markers[i % len(markers)],
                       color=colors[i % len(colors)],
                       label=method_name, linewidth=2, markersize=8, linestyle='--')
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14)
        
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return None
        
        return fig

    @staticmethod
    def plot_latent_space(
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        method: Literal['tsne', 'umap'] = 'tsne',
        save_path: Optional[str] = None,
        figsize: tuple = (10, 8),
        perplexity: int = 30,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42,
        cmap: str = 'tab10',
        alpha: float = 0.7,
        point_size: int = 50,
        title: Optional[str] = None,
        label_names: Optional[Dict[int, str]] = None
    ) -> Optional[Any]:
        """Generate latent space visualization using t-SNE or UMAP."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")
        
        # Convert to numpy
        embeddings_np = embeddings.detach().cpu().numpy()
        
        if labels is not None:
            labels_np = labels.detach().cpu().numpy()
        else:
            labels_np = None
        
        # Perform dimensionality reduction
        if method == 'tsne':
            try:
                from sklearn.manifold import TSNE
            except ImportError:
                raise ImportError("scikit-learn is required for t-SNE. Install with: pip install scikit-learn")
            
            reducer = TSNE(
                n_components=2,
                perplexity=min(perplexity, len(embeddings_np) - 1),
                random_state=random_state,
                max_iter=1000
            )
            embeddings_2d = reducer.fit_transform(embeddings_np)
            
        elif method == 'umap':
            try:
                import umap
            except ImportError:
                raise ImportError("umap-learn is required for UMAP. Install with: pip install umap-learn")
            
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(n_neighbors, len(embeddings_np) - 1),
                min_dist=min_dist,
                random_state=random_state
            )
            embeddings_2d = reducer.fit_transform(embeddings_np)
        else:
            raise ValueError(f"method must be 'tsne' or 'umap', got '{method}'")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        if labels_np is not None:
            unique_labels = np.unique(labels_np)
            scatter = ax.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=labels_np,
                cmap=cmap,
                alpha=alpha,
                s=point_size
            )
            
            # Create legend
            if label_names:
                handles = []
                for label_val in unique_labels:
                    color = plt.cm.get_cmap(cmap)(label_val / max(unique_labels.max(), 1))
                    name = label_names.get(int(label_val), f'Class {int(label_val)}')
                    handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                             markerfacecolor=color, markersize=10, label=name))
                ax.legend(handles=handles, fontsize=10)
            else:
                plt.colorbar(scatter, ax=ax, label='Label')
        else:
            ax.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                alpha=alpha,
                s=point_size,
                color='blue'
            )
        
        method_name = 't-SNE' if method == 'tsne' else 'UMAP'
        ax.set_xlabel(f'{method_name} Dimension 1', fontsize=12)
        ax.set_ylabel(f'{method_name} Dimension 2', fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(f'Latent Space Visualization ({method_name})', fontsize=14)
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return None
        
        return fig


class AblationModule:
    """Ablation study support for PI-JEPA."""
    
    # Valid loss components that can be disabled
    VALID_COMPONENTS = {'physics', 'variance', 'covariance'}
    
    # Valid number of predictors
    VALID_K_VALUES = {1, 2, 3}
    
    def __init__(self, base_config: Dict[str, Any]):
        """Initialize ablation module with base configuration."""
        self.base_config = copy.deepcopy(base_config)
        self._validate_config(self.base_config)
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate that config has required structure for ablation."""
        # Check loss section exists
        if 'loss' not in config:
            config['loss'] = {}
        
        # Ensure physics section exists
        if 'physics' not in config['loss']:
            config['loss']['physics'] = {'weight': 0.1, 'enabled': True}
        
        # Ensure regularization section exists
        if 'regularization' not in config['loss']:
            config['loss']['regularization'] = {
                'variance': {'weight': 0.05},
                'covariance': {'weight': 0.01}
            }
        
        # Ensure variance and covariance exist
        if 'variance' not in config['loss']['regularization']:
            config['loss']['regularization']['variance'] = {'weight': 0.05}
        if 'covariance' not in config['loss']['regularization']:
            config['loss']['regularization']['covariance'] = {'weight': 0.01}
        
        # Check model section exists
        if 'model' not in config:
            config['model'] = {'num_predictors': 2}
        
        if 'num_predictors' not in config['model']:
            config['model']['num_predictors'] = 2

    def disable_loss_component(
        self,
        component: Literal['physics', 'variance', 'covariance']
    ) -> Dict[str, Any]:
        """Disable a specific loss component by setting its weight to 0."""
        if component not in self.VALID_COMPONENTS:
            raise ValueError(
                f"Invalid component '{component}'. "
                f"Must be one of: {self.VALID_COMPONENTS}"
            )
        
        # Create deep copy of config
        config = copy.deepcopy(self.base_config)
        
        if component == 'physics':
            # Disable physics loss
            config['loss']['physics']['weight'] = 0.0
            if 'enabled' in config['loss']['physics']:
                config['loss']['physics']['enabled'] = False
                
        elif component == 'variance':
            # Disable variance regularization
            config['loss']['regularization']['variance']['weight'] = 0.0
            
        elif component == 'covariance':
            # Disable covariance regularization
            config['loss']['regularization']['covariance']['weight'] = 0.0
        
        return config
    
    def set_num_predictors(self, k: int) -> Dict[str, Any]:
        """Set the number of predictors K for operator splitting."""
        if k not in self.VALID_K_VALUES:
            raise ValueError(
                f"Invalid number of predictors k={k}. "
                f"Must be one of: {self.VALID_K_VALUES}"
            )
        
        # Create deep copy of config
        config = copy.deepcopy(self.base_config)
        
        # Update number of predictors
        config['model']['num_predictors'] = k
        
        # Adjust predictor stages if present
        if 'predictor' in config['model'] and 'stages' in config['model']['predictor']:
            stages = config['model']['predictor']['stages']
            
            # Define default stage configurations
            default_stages = [
                {'name': 'pressure_step', 'depth': 4, 'heads': 6, 'hidden_dim': 384},
                {'name': 'transport_step', 'depth': 4, 'heads': 6, 'hidden_dim': 384},
                {'name': 'reaction_step', 'depth': 4, 'heads': 6, 'hidden_dim': 384}
            ]
            
            # Adjust stages to match K
            if k == 1:
                # Single predictor
                config['model']['predictor']['stages'] = [
                    {'name': 'combined_step', 'depth': 4, 'heads': 6, 'hidden_dim': 384}
                ]
            elif k == 2:
                # Two predictors (Darcy flow)
                config['model']['predictor']['stages'] = default_stages[:2]
            elif k == 3:
                # Three predictors (reactive transport)
                config['model']['predictor']['stages'] = default_stages[:3]
        
        return config

    def run_ablation(
        self,
        train_fn: callable,
        eval_fn: callable,
        ablation_type: Literal['loss_components', 'num_predictors', 'both'] = 'both',
        components_to_ablate: Optional[List[str]] = None,
        k_values: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """Run ablation study and collect results."""
        results = {
            'baseline': None,
            'loss_ablations': {},
            'k_ablations': {}
        }
        
        # Set defaults
        if components_to_ablate is None:
            components_to_ablate = list(self.VALID_COMPONENTS)
        if k_values is None:
            k_values = list(self.VALID_K_VALUES)
        
        # Run baseline
        print("Running baseline configuration...")
        baseline_model = train_fn(self.base_config)
        baseline_metrics = eval_fn(baseline_model)
        results['baseline'] = {
            'config': copy.deepcopy(self.base_config),
            'metrics': baseline_metrics
        }
        
        # Run loss component ablations
        if ablation_type in ('loss_components', 'both'):
            for component in components_to_ablate:
                if component not in self.VALID_COMPONENTS:
                    print(f"Skipping invalid component: {component}")
                    continue
                
                print(f"Running ablation: no_{component}...")
                config = self.disable_loss_component(component)
                model = train_fn(config)
                metrics = eval_fn(model)
                
                results['loss_ablations'][f'no_{component}'] = {
                    'config': config,
                    'metrics': metrics
                }
        
        # Run K ablations
        if ablation_type in ('num_predictors', 'both'):
            for k in k_values:
                if k not in self.VALID_K_VALUES:
                    print(f"Skipping invalid K value: {k}")
                    continue
                
                print(f"Running ablation: K={k}...")
                config = self.set_num_predictors(k)
                model = train_fn(config)
                metrics = eval_fn(model)
                
                results['k_ablations'][f'k={k}'] = {
                    'config': config,
                    'metrics': metrics
                }
        
        return results
    
    def get_ablation_configs(
        self,
        ablation_type: Literal['loss_components', 'num_predictors', 'both'] = 'both',
        components_to_ablate: Optional[List[str]] = None,
        k_values: Optional[List[int]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Get all ablation configurations without running experiments."""
        configs = {'baseline': copy.deepcopy(self.base_config)}
        
        # Set defaults
        if components_to_ablate is None:
            components_to_ablate = list(self.VALID_COMPONENTS)
        if k_values is None:
            k_values = list(self.VALID_K_VALUES)
        
        # Generate loss component ablation configs
        if ablation_type in ('loss_components', 'both'):
            for component in components_to_ablate:
                if component in self.VALID_COMPONENTS:
                    configs[f'no_{component}'] = self.disable_loss_component(component)
        
        # Generate K ablation configs
        if ablation_type in ('num_predictors', 'both'):
            for k in k_values:
                if k in self.VALID_K_VALUES:
                    configs[f'k={k}'] = self.set_num_predictors(k)
        
        return configs
    
    def get_component_weight(
        self,
        config: Dict[str, Any],
        component: str
    ) -> float:
        """Get the weight of a specific loss component from a configuration."""
        if component == 'physics':
            return config.get('loss', {}).get('physics', {}).get('weight', 0.0)
        elif component == 'variance':
            return config.get('loss', {}).get('regularization', {}).get('variance', {}).get('weight', 0.0)
        elif component == 'covariance':
            return config.get('loss', {}).get('regularization', {}).get('covariance', {}).get('weight', 0.0)
        else:
            raise ValueError(f"Unknown component: {component}")
