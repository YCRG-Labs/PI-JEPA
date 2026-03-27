"""
Dataset loaders for PI-JEPA benchmark datasets.

This module provides:
- HDF5DatasetMixin: Mixin for loading HDF5 files following PDEBench standardized API
- DatasetFactory: Factory pattern for creating datasets by name
- UFNODataset: U-FNO CO2-water multiphase dataset

Specific dataset implementations (PDEBenchADRDataset, SPE10Dataset, etc.)
will be added in subsequent tasks.
"""

import os
from abc import ABC
from typing import Any, Dict, List, Optional, Union

import h5py
import torch
from torch.utils.data import Dataset


class HDF5DatasetMixin:
    """
    Mixin for HDF5 format with PDEBench standardized API.
    
    Provides utilities for loading data from HDF5 files following
    the PDEBench conventions for PDE benchmark datasets.
    
    PDEBench HDF5 Structure:
    - Data is stored in datasets with standardized keys
    - Common keys: 'u' (solution), 'x' (spatial coords), 't' (time coords)
    - Attributes contain metadata (grid size, dt, dx, etc.)
    """
    
    def load_hdf5(self, path: str, keys: List[str]) -> Dict[str, torch.Tensor]:
        """
        Load data from HDF5 file following PDEBench conventions.
        
        Args:
            path: Path to the HDF5 file
            keys: List of dataset keys to load (e.g., ['u', 'x', 't'])
            
        Returns:
            Dictionary mapping keys to torch tensors
            
        Raises:
            FileNotFoundError: If the HDF5 file does not exist
            KeyError: If a requested key is not found in the file
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"HDF5 file not found: {path}")
        
        data = {}
        with h5py.File(path, 'r') as f:
            for key in keys:
                if key not in f:
                    available_keys = list(f.keys())
                    raise KeyError(
                        f"Key '{key}' not found in HDF5 file. "
                        f"Available keys: {available_keys}"
                    )
                # Load data and convert to torch tensor
                arr = f[key][:]
                data[key] = torch.from_numpy(arr).float()
        
        return data

    def load_hdf5_lazy(
        self, 
        path: str, 
        key: str, 
        indices: Optional[Union[int, slice, List[int]]] = None
    ) -> torch.Tensor:
        """
        Lazily load a subset of data from HDF5 file.
        
        Useful for large datasets that don't fit in memory.
        
        Args:
            path: Path to the HDF5 file
            key: Dataset key to load
            indices: Optional indices or slice to load (loads all if None)
            
        Returns:
            Torch tensor with the requested data
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"HDF5 file not found: {path}")
        
        with h5py.File(path, 'r') as f:
            if key not in f:
                raise KeyError(f"Key '{key}' not found in HDF5 file")
            
            if indices is None:
                arr = f[key][:]
            else:
                arr = f[key][indices]
            
            return torch.from_numpy(arr).float()
    
    def get_hdf5_metadata(self, path: str) -> Dict[str, Any]:
        """
        Extract metadata from HDF5 file attributes.
        
        Args:
            path: Path to the HDF5 file
            
        Returns:
            Dictionary of metadata attributes
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"HDF5 file not found: {path}")
        
        metadata = {}
        with h5py.File(path, 'r') as f:
            # Get file-level attributes
            for attr_name in f.attrs:
                metadata[attr_name] = f.attrs[attr_name]
            
            # Get dataset shapes and dtypes
            metadata['datasets'] = {}
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    metadata['datasets'][key] = {
                        'shape': f[key].shape,
                        'dtype': str(f[key].dtype)
                    }
        
        return metadata
    
    def get_hdf5_shape(self, path: str, key: str) -> tuple:
        """
        Get the shape of a dataset without loading it.
        
        Args:
            path: Path to the HDF5 file
            key: Dataset key
            
        Returns:
            Shape tuple of the dataset
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"HDF5 file not found: {path}")
        
        with h5py.File(path, 'r') as f:
            if key not in f:
                raise KeyError(f"Key '{key}' not found in HDF5 file")
            return f[key].shape


class DatasetFactory:
    """
    Factory for loading all benchmark datasets.
    
    Supports creating datasets by name with consistent configuration interface.
    
    Supported datasets:
    - 'ufno_co2': U-FNO CO2-water multiphase dataset (2000 trajectories, 64├ù64)
    - 'pdebench_adr': PDEBench Advection-Diffusion-Reaction dataset
    - 'spe10': SPE10 OOD test set from Tarbert formation
    - 'navier_stokes': Navier-Stokes 2D dataset at ╬╜=10^-4
    
    Example:
        >>> config = {'path': '/data/ufno', 'grid_size': 64}
        >>> dataset = DatasetFactory.create('ufno_co2', config, split='train')
    """
    
    # Registry of available dataset classes
    # Will be populated as dataset implementations are added
    _registry: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, dataset_class: type) -> None:
        """
        Register a dataset class with the factory.
        
        Args:
            name: Name to register the dataset under
            dataset_class: Dataset class to register
        """
        cls._registry[name] = dataset_class
    
    @classmethod
    def create(
        cls,
        dataset_name: str,
        config: Dict[str, Any],
        split: str = "train"
    ) -> Dataset:
        """
        Create dataset by name.
        
        Args:
            dataset_name: One of 'ufno_co2', 'pdebench_adr', 'spe10', 'navier_stokes'
            config: Dataset configuration dictionary containing:
                - path: Path to dataset files
                - Additional dataset-specific parameters
            split: Data split - 'train', 'pretrain', 'finetune', or 'test'
            
        Returns:
            PyTorch Dataset instance
            
        Raises:
            ValueError: If dataset_name is not recognized
            KeyError: If required config parameters are missing
        """
        # Normalize dataset name
        name_lower = dataset_name.lower().replace('-', '_')
        
        # Check if dataset is registered
        if name_lower not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown dataset: '{dataset_name}'. "
                f"Available datasets: {available}"
            )
        
        # Validate split
        valid_splits = ['train', 'pretrain', 'finetune', 'test']
        if split not in valid_splits:
            raise ValueError(
                f"Invalid split: '{split}'. "
                f"Valid splits: {valid_splits}"
            )
        
        # Validate config has required 'path' key
        if 'path' not in config:
            raise KeyError("Config must contain 'path' key")
        
        # Create and return dataset instance
        dataset_class = cls._registry[name_lower]
        return dataset_class(config=config, split=split)
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        """
        List all registered dataset names.
        
        Returns:
            List of available dataset names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def get_dataset_info(cls, dataset_name: str) -> Dict[str, Any]:
        """
        Get information about a registered dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with dataset information (docstring, class name)
        """
        name_lower = dataset_name.lower().replace('-', '_')
        
        if name_lower not in cls._registry:
            raise ValueError(f"Unknown dataset: '{dataset_name}'")
        
        dataset_class = cls._registry[name_lower]
        return {
            'name': dataset_name,
            'class': dataset_class.__name__,
            'description': dataset_class.__doc__ or "No description available"
        }



class BenchmarkDataset(Dataset, HDF5DatasetMixin, ABC):
    """
    Abstract base class for PI-JEPA benchmark datasets.
    
    Combines PyTorch Dataset interface with HDF5 loading utilities.
    Specific dataset implementations should inherit from this class.
    """
    
    def __init__(self, config: Dict[str, Any], split: str = "train"):
        """
        Initialize benchmark dataset.
        
        Args:
            config: Dataset configuration
            split: Data split ('train', 'pretrain', 'finetune', 'test')
        """
        super().__init__()
        self.config = config
        self.split = split
        self.path = config['path']
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        raise NotImplementedError("Subclasses must implement __len__")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a sample from the dataset."""
        raise NotImplementedError("Subclasses must implement __getitem__")


class UFNODataset(BenchmarkDataset):
    """
    U-FNO CO2-water multiphase dataset.
    
    Loads 2000 trajectories on 64├ù64 2D-radial grid for CO2-water
    multiphase flow simulation. This dataset is used for training
    and evaluating physics-informed neural operators on subsurface
    flow problems.
    
    Dataset Structure:
        - pressure: (N, T, H, W) - Pressure field over time
        - saturation: (N, T, H, W) - Water saturation field over time
        - permeability: (N, H, W) - Permeability field (static)
        - porosity: (N, H, W) - Porosity field (static)
    
    Split Configuration:
        - train: 1600 trajectories (indices 0-1599)
        - pretrain: 200 trajectories (indices 1600-1799)
        - finetune: Same as pretrain (indices 1600-1799)
        - test: 200 trajectories (indices 1800-1999)
    
    Args:
        config: Configuration dictionary containing:
            - path: Path to the HDF5 data file
            - grid_size: Grid size (default 64)
            - n_trajectories: Total number of trajectories (default 2000)
        split: Data split ('train', 'pretrain', 'finetune', 'test')
    
    Example:
        >>> config = {'path': '/data/ufno_co2.h5', 'grid_size': 64}
        >>> dataset = UFNODataset(config=config, split='train')
        >>> sample = dataset[0]
        >>> print(sample['pressure'].shape)  # (T, 64, 64)
    """
    
    # Default split sizes following paper specification
    SPLIT_SIZES = {
        'train': 1600,
        'pretrain': 200,
        'finetune': 200,  # Same as pretrain
        'test': 200
    }
    
    # HDF5 keys for the dataset
    HDF5_KEYS = ['pressure', 'saturation', 'permeability', 'porosity']
    
    def __init__(
        self,
        config: Dict[str, Any],
        split: str = "train",
    ):
        """
        Initialize U-FNO CO2-water multiphase dataset.
        
        Args:
            config: Configuration dictionary with:
                - path: Path to HDF5 file
                - grid_size: Grid size (default 64)
                - n_trajectories: Total trajectories (default 2000)
            split: Data split ('train', 'pretrain', 'finetune', 'test')
        """
        super().__init__(config, split)
        
        # Extract configuration with defaults
        self.grid_size = config.get('grid_size', 64)
        self.n_trajectories = config.get('n_trajectories', 2000)
        
        # Validate split
        if split not in self.SPLIT_SIZES:
            raise ValueError(
                f"Invalid split: '{split}'. "
                f"Valid splits: {list(self.SPLIT_SIZES.keys())}"
            )
        
        # Compute split indices
        self._compute_split_indices()
        
        # Data storage (lazy loading)
        self._data: Optional[Dict[str, torch.Tensor]] = None
        self._loaded = False
    
    def _compute_split_indices(self) -> None:
        """
        Compute start and end indices for the current split.
        
        Split layout:
            - train: [0, 1600)
            - pretrain: [1600, 1800)
            - finetune: [1600, 1800) (same as pretrain)
            - test: [1800, 2000)
        """
        if self.split == 'train':
            self._start_idx = 0
            self._end_idx = self.SPLIT_SIZES['train']
        elif self.split in ('pretrain', 'finetune'):
            self._start_idx = self.SPLIT_SIZES['train']
            self._end_idx = self._start_idx + self.SPLIT_SIZES['pretrain']
        else:  # test
            self._start_idx = self.SPLIT_SIZES['train'] + self.SPLIT_SIZES['pretrain']
            self._end_idx = self._start_idx + self.SPLIT_SIZES['test']
        
        # Validate indices don't exceed total trajectories
        if self._end_idx > self.n_trajectories:
            raise ValueError(
                f"Split '{self.split}' requires indices up to {self._end_idx}, "
                f"but only {self.n_trajectories} trajectories available"
            )

    def _load_data(self) -> None:
        """
        Load data from HDF5 file.
        
        Loads all required fields and slices to the current split.
        Uses lazy loading - data is only loaded on first access.
        """
        if self._loaded:
            return
        
        # Check if file exists
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Dataset file not found: {self.path}")
        
        # Load data from HDF5
        self._data = {}
        
        with h5py.File(self.path, 'r') as f:
            # Load time-varying fields: (N, T, H, W)
            for key in ['pressure', 'saturation']:
                if key not in f:
                    raise KeyError(
                        f"Required key '{key}' not found in HDF5 file. "
                        f"Available keys: {list(f.keys())}"
                    )
                # Load only the split indices
                arr = f[key][self._start_idx:self._end_idx]
                self._data[key] = torch.from_numpy(arr).float()
            
            # Load static fields: (N, H, W)
            for key in ['permeability', 'porosity']:
                if key not in f:
                    raise KeyError(
                        f"Required key '{key}' not found in HDF5 file. "
                        f"Available keys: {list(f.keys())}"
                    )
                arr = f[key][self._start_idx:self._end_idx]
                self._data[key] = torch.from_numpy(arr).float()
        
        # Validate shapes
        self._validate_shapes()
        
        self._loaded = True
    
    def _validate_shapes(self) -> None:
        """Validate that loaded data has expected shapes."""
        n_samples = self._end_idx - self._start_idx
        
        # Check time-varying fields
        for key in ['pressure', 'saturation']:
            shape = self._data[key].shape
            if len(shape) != 4:
                raise ValueError(
                    f"Expected 4D tensor for '{key}' (N, T, H, W), "
                    f"got shape {shape}"
                )
            if shape[0] != n_samples:
                raise ValueError(
                    f"Expected {n_samples} samples for '{key}', "
                    f"got {shape[0]}"
                )
            if shape[2] != self.grid_size or shape[3] != self.grid_size:
                raise ValueError(
                    f"Expected grid size {self.grid_size}├ù{self.grid_size} for '{key}', "
                    f"got {shape[2]}├ù{shape[3]}"
                )
        
        # Check static fields
        for key in ['permeability', 'porosity']:
            shape = self._data[key].shape
            if len(shape) != 3:
                raise ValueError(
                    f"Expected 3D tensor for '{key}' (N, H, W), "
                    f"got shape {shape}"
                )
            if shape[0] != n_samples:
                raise ValueError(
                    f"Expected {n_samples} samples for '{key}', "
                    f"got {shape[0]}"
                )
            if shape[1] != self.grid_size or shape[2] != self.grid_size:
                raise ValueError(
                    f"Expected grid size {self.grid_size}├ù{self.grid_size} for '{key}', "
                    f"got {shape[1]}├ù{shape[2]}"
                )
    
    def __len__(self) -> int:
        """Return the number of samples in the current split."""
        return self._end_idx - self._start_idx
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single trajectory sample.
        
        Args:
            idx: Index within the current split (0 to len-1)
            
        Returns:
            Dictionary containing:
                - pressure: (T, H, W) pressure field trajectory
                - saturation: (T, H, W) saturation field trajectory
                - permeability: (H, W) permeability field
                - porosity: (H, W) porosity field
        """
        # Ensure data is loaded
        self._load_data()
        
        # Validate index
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for split '{self.split}' "
                f"with {len(self)} samples"
            )
        
        return {
            'pressure': self._data['pressure'][idx],
            'saturation': self._data['saturation'][idx],
            'permeability': self._data['permeability'][idx],
            'porosity': self._data['porosity'][idx],
        }
    
    def get_all_data(self) -> Dict[str, torch.Tensor]:
        """
        Get all data for the current split.
        
        Returns:
            Dictionary containing all tensors for the split:
                - pressure: (N, T, H, W)
                - saturation: (N, T, H, W)
                - permeability: (N, H, W)
                - porosity: (N, H, W)
        """
        self._load_data()
        return self._data.copy()
    
    @property
    def n_timesteps(self) -> int:
        """Get the number of timesteps in trajectories."""
        self._load_data()
        return self._data['pressure'].shape[1]
    
    @property
    def spatial_shape(self) -> tuple:
        """Get the spatial shape (H, W) of the fields."""
        return (self.grid_size, self.grid_size)


# Register UFNODataset with the factory
DatasetFactory.register('ufno_co2', UFNODataset)


class PDEBenchADRDataset(BenchmarkDataset):
    """
    PDEBench Advection-Diffusion-Reaction dataset.

    Loads trajectories for advection-diffusion-reaction systems with
    configurable P├⌐clet and Damk├╢hler numbers. Supports 9 parameter
    regimes: Pe Γêê {0.1, 1, 10}, Da Γêê {0.01, 0.1, 1.0}.

    The dataset contains multi-species concentration fields that evolve
    according to the ADR equation:
        Γêéc/Γêét + Pe┬╖(v┬╖Γêçc) = Γêç┬▓c + Da┬╖R(c)

    Dataset Structure:
        - concentration: (N, n_species, T, H, W) - Species concentrations over time
        - velocity: (N, 2, H, W) - Velocity field (static, 2D)
        - diffusivity: (N,) - Diffusivity coefficient per trajectory
        - reaction_rate: (N,) - Reaction rate coefficient per trajectory

    Parameter Regimes:
        - P├⌐clet number (Pe): {0.1, 1.0, 10.0} - Advection/diffusion ratio
        - Damk├╢hler number (Da): {0.01, 0.1, 1.0} - Reaction/transport ratio

    Args:
        config: Configuration dictionary containing:
            - path: Path to the HDF5 data file
            - peclet: P├⌐clet number (default 1.0)
            - damkohler: Damk├╢hler number (default 0.1)
            - n_species: Number of chemical species (default 1)
        split: Data split ('train', 'pretrain', 'finetune', 'test')

    Example:
        >>> config = {'path': '/data/pdebench_adr.h5', 'peclet': 1.0, 'damkohler': 0.1}
        >>> dataset = PDEBenchADRDataset(config=config, split='train')
        >>> sample = dataset[0]
        >>> print(sample['concentration'].shape)  # (n_species, T, H, W)

    **Validates: Requirements 3.2, 3.7**
    """

    # Valid parameter regimes as specified in the paper
    VALID_PECLET = {0.1, 1.0, 10.0}
    VALID_DAMKOHLER = {0.01, 0.1, 1.0}

    # Default split ratios (can be overridden in config)
    DEFAULT_SPLIT_RATIOS = {
        'train': 0.7,
        'pretrain': 0.1,
        'finetune': 0.1,  # Same indices as pretrain
        'test': 0.2
    }

    # HDF5 keys for the dataset
    HDF5_KEYS = ['concentration', 'velocity', 'diffusivity', 'reaction_rate']

    def __init__(
        self,
        config: Dict[str, Any],
        split: str = "train",
    ):
        """
        Initialize PDEBench Advection-Diffusion-Reaction dataset.

        Args:
            config: Configuration dictionary with:
                - path: Path to HDF5 file
                - peclet: P├⌐clet number (default 1.0, must be in {0.1, 1.0, 10.0})
                - damkohler: Damk├╢hler number (default 0.1, must be in {0.01, 0.1, 1.0})
                - n_species: Number of chemical species (default 1)
                - n_trajectories: Total number of trajectories (optional, auto-detected)
            split: Data split ('train', 'pretrain', 'finetune', 'test')
        """
        super().__init__(config, split)

        # Extract configuration with defaults
        self.peclet = config.get('peclet', 1.0)
        self.damkohler = config.get('damkohler', 0.1)
        self.n_species = config.get('n_species', 1)
        self._n_trajectories = config.get('n_trajectories', None)

        # Validate parameter regime
        self._validate_parameter_regime()

        # Validate split
        valid_splits = ['train', 'pretrain', 'finetune', 'test']
        if split not in valid_splits:
            raise ValueError(
                f"Invalid split: '{split}'. "
                f"Valid splits: {valid_splits}"
            )

        # Data storage (lazy loading)
        self._data: Optional[Dict[str, torch.Tensor]] = None
        self._loaded = False
        self._start_idx = 0
        self._end_idx = 0

    def _validate_parameter_regime(self) -> None:
        """
        Validate that P├⌐clet and Damk├╢hler numbers are in valid regimes.

        Raises:
            ValueError: If parameters are not in valid regimes
        """
        # Allow small tolerance for floating point comparison
        def is_close_to_valid(value: float, valid_set: set) -> bool:
            return any(abs(value - v) < 1e-6 for v in valid_set)

        if not is_close_to_valid(self.peclet, self.VALID_PECLET):
            raise ValueError(
                f"Invalid P├⌐clet number: {self.peclet}. "
                f"Valid values: {sorted(self.VALID_PECLET)}"
            )

        if not is_close_to_valid(self.damkohler, self.VALID_DAMKOHLER):
            raise ValueError(
                f"Invalid Damk├╢hler number: {self.damkohler}. "
                f"Valid values: {sorted(self.VALID_DAMKOHLER)}"
            )

    def _compute_split_indices(self, n_total: int) -> None:
        """
        Compute start and end indices for the current split.

        Args:
            n_total: Total number of trajectories in the dataset
        """
        # Calculate split sizes
        n_train = int(n_total * self.DEFAULT_SPLIT_RATIOS['train'])
        n_pretrain = int(n_total * self.DEFAULT_SPLIT_RATIOS['pretrain'])
        n_test = int(n_total * self.DEFAULT_SPLIT_RATIOS['test'])

        # Assign indices based on split
        if self.split == 'train':
            self._start_idx = 0
            self._end_idx = n_train
        elif self.split in ('pretrain', 'finetune'):
            self._start_idx = n_train
            self._end_idx = n_train + n_pretrain
        else:  # test
            self._start_idx = n_train + n_pretrain
            self._end_idx = n_train + n_pretrain + n_test

        # Ensure we don't exceed total
        self._end_idx = min(self._end_idx, n_total)

    def _load_data(self) -> None:
        """
        Load data from HDF5 file.

        Loads all required fields and slices to the current split.
        Concatenates species concentration channels to input tensor.
        Uses lazy loading - data is only loaded on first access.
        """
        if self._loaded:
            return

        # Check if file exists
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

        self._data = {}

        with h5py.File(self.path, 'r') as f:
            # Determine total number of trajectories
            if 'concentration' not in f:
                raise KeyError(
                    f"Required key 'concentration' not found in HDF5 file. "
                    f"Available keys: {list(f.keys())}"
                )

            n_total = f['concentration'].shape[0]
            if self._n_trajectories is not None:
                n_total = min(n_total, self._n_trajectories)

            # Compute split indices
            self._compute_split_indices(n_total)

            # Load concentration field: (N, n_species, T, H, W)
            # This is the main field with species channels concatenated
            arr = f['concentration'][self._start_idx:self._end_idx]
            concentration = torch.from_numpy(arr).float()

            # Ensure concentration has species dimension
            # If loaded as (N, T, H, W), add species dimension
            if concentration.dim() == 4:
                # Single species case: (N, T, H, W) -> (N, 1, T, H, W)
                concentration = concentration.unsqueeze(1)

            # Validate and potentially adjust n_species
            actual_n_species = concentration.shape[1]
            if self.n_species != actual_n_species:
                # Update n_species to match actual data
                self.n_species = actual_n_species

            self._data['concentration'] = concentration

            # Load velocity field: (N, 2, H, W)
            if 'velocity' in f:
                arr = f['velocity'][self._start_idx:self._end_idx]
                self._data['velocity'] = torch.from_numpy(arr).float()
            else:
                # Create zero velocity if not present (pure diffusion-reaction)
                H, W = concentration.shape[-2], concentration.shape[-1]
                n_samples = self._end_idx - self._start_idx
                self._data['velocity'] = torch.zeros(n_samples, 2, H, W)

            # Load scalar parameters
            # Diffusivity: (N,)
            if 'diffusivity' in f:
                arr = f['diffusivity'][self._start_idx:self._end_idx]
                self._data['diffusivity'] = torch.from_numpy(arr).float()
            else:
                # Use default diffusivity based on P├⌐clet number
                n_samples = self._end_idx - self._start_idx
                self._data['diffusivity'] = torch.ones(n_samples) / self.peclet

            # Reaction rate: (N,)
            if 'reaction_rate' in f:
                arr = f['reaction_rate'][self._start_idx:self._end_idx]
                self._data['reaction_rate'] = torch.from_numpy(arr).float()
            else:
                # Use default reaction rate based on Damk├╢hler number
                n_samples = self._end_idx - self._start_idx
                self._data['reaction_rate'] = torch.ones(n_samples) * self.damkohler

        # Store parameter regime metadata
        self._data['peclet'] = self.peclet
        self._data['damkohler'] = self.damkohler

        self._loaded = True

    def __len__(self) -> int:
        """Return the number of samples in the current split."""
        if not self._loaded:
            # Need to load to determine split size
            self._load_data()
        return self._end_idx - self._start_idx

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single trajectory sample.

        Args:
            idx: Index within the current split (0 to len-1)

        Returns:
            Dictionary containing:
                - concentration: (n_species, T, H, W) species concentrations
                - velocity: (2, H, W) velocity field
                - diffusivity: scalar diffusivity coefficient
                - reaction_rate: scalar reaction rate coefficient
                - peclet: P├⌐clet number for this regime
                - damkohler: Damk├╢hler number for this regime
        """
        # Ensure data is loaded
        self._load_data()

        # Validate index
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for split '{self.split}' "
                f"with {len(self)} samples"
            )

        return {
            'concentration': self._data['concentration'][idx],
            'velocity': self._data['velocity'][idx],
            'diffusivity': self._data['diffusivity'][idx],
            'reaction_rate': self._data['reaction_rate'][idx],
            'peclet': torch.tensor(self.peclet),
            'damkohler': torch.tensor(self.damkohler),
        }

    def get_all_data(self) -> Dict[str, Any]:
        """
        Get all data for the current split.

        Returns:
            Dictionary containing all tensors for the split:
                - concentration: (N, n_species, T, H, W)
                - velocity: (N, 2, H, W)
                - diffusivity: (N,)
                - reaction_rate: (N,)
                - peclet: float
                - damkohler: float
        """
        self._load_data()
        return {
            'concentration': self._data['concentration'],
            'velocity': self._data['velocity'],
            'diffusivity': self._data['diffusivity'],
            'reaction_rate': self._data['reaction_rate'],
            'peclet': self.peclet,
            'damkohler': self.damkohler,
        }

    @property
    def n_timesteps(self) -> int:
        """Get the number of timesteps in trajectories."""
        self._load_data()
        return self._data['concentration'].shape[2]

    @property
    def spatial_shape(self) -> tuple:
        """Get the spatial shape (H, W) of the fields."""
        self._load_data()
        return tuple(self._data['concentration'].shape[-2:])

    @classmethod
    def get_all_regimes(cls) -> List[Dict[str, float]]:
        """
        Get all 9 valid parameter regimes.

        Returns:
            List of dictionaries with 'peclet' and 'damkohler' keys
        """
        regimes = []
        for pe in sorted(cls.VALID_PECLET):
            for da in sorted(cls.VALID_DAMKOHLER):
                regimes.append({'peclet': pe, 'damkohler': da})
        return regimes


# Register PDEBenchADRDataset with the factory
DatasetFactory.register('pdebench_adr', PDEBenchADRDataset)


class SPE10Dataset(BenchmarkDataset):
    """
    SPE10 OOD test set from Tarbert formation.

    Loads 500 trajectories from layers 60-85 of the Tarbert formation
    for out-of-distribution (OOD) evaluation. The SPE10 benchmark is a
    standard test case for subsurface flow simulation with highly
    heterogeneous permeability fields.

    The Tarbert formation (layers 60-85) represents a shallow-marine
    depositional environment with complex channelized permeability
    structures that differ significantly from the training distribution.

    Dataset Structure:
        - pressure: (N, T, H, W) - Pressure field over time (N=500)
        - saturation: (N, T, H, W) - Water saturation field over time
        - permeability: (N, H, W) - Permeability field from layers 60-85
        - layer_idx: (N,) - Layer index for each trajectory

    OOD Evaluation:
        This dataset is specifically designed for OOD evaluation and
        does not have train/test splits. All 500 trajectories are used
        for testing generalization to unseen geological formations.

    Args:
        config: Configuration dictionary containing:
            - path: Path to the HDF5 data file
            - layers: Tuple of (start, end) layer indices (default (60, 85))
            - n_trajectories: Number of trajectories to load (default 500)
        split: Data split - only 'test' is valid for OOD evaluation

    Example:
        >>> config = {'path': '/data/spe10_tarbert.h5', 'layers': (60, 85)}
        >>> dataset = SPE10Dataset(config=config, split='test')
        >>> sample = dataset[0]
        >>> print(sample['pressure'].shape)  # (T, H, W)
        >>> print(sample['layer_idx'])  # tensor with layer index

    **Validates: Requirements 3.3**
    """

    # Default configuration
    DEFAULT_LAYERS = (60, 85)
    DEFAULT_N_TRAJECTORIES = 500

    # HDF5 keys for the dataset
    HDF5_KEYS = ['pressure', 'saturation', 'permeability', 'layer_idx']

    def __init__(
        self,
        config: Dict[str, Any],
        split: str = "test",
    ):
        """
        Initialize SPE10 OOD test set from Tarbert formation.

        Args:
            config: Configuration dictionary with:
                - path: Path to HDF5 file
                - layers: Tuple of (start, end) layer indices (default (60, 85))
                - n_trajectories: Number of trajectories (default 500)
            split: Data split - only 'test' is valid for OOD evaluation,
                   but 'train', 'pretrain', 'finetune' are accepted for
                   compatibility (all map to the same OOD test data)
        """
        super().__init__(config, split)

        # Extract configuration with defaults
        self.layers = config.get('layers', self.DEFAULT_LAYERS)
        self.n_trajectories = config.get('n_trajectories', self.DEFAULT_N_TRAJECTORIES)

        # Validate layers
        self._validate_layers()

        # Validate split - SPE10 is OOD test only, but accept all splits
        # for compatibility with DatasetFactory
        valid_splits = ['train', 'pretrain', 'finetune', 'test']
        if split not in valid_splits:
            raise ValueError(
                f"Invalid split: '{split}'. "
                f"Valid splits: {valid_splits}"
            )

        # Note: For SPE10 OOD, all splits return the same data
        # This is intentional as SPE10 is used only for OOD evaluation
        if split != 'test':
            import warnings
            warnings.warn(
                f"SPE10Dataset is designed for OOD evaluation. "
                f"Split '{split}' will return the same data as 'test'.",
                UserWarning
            )

        # Data storage (lazy loading)
        self._data: Optional[Dict[str, torch.Tensor]] = None
        self._loaded = False

    def _validate_layers(self) -> None:
        """
        Validate layer configuration.

        Raises:
            ValueError: If layers are invalid
        """
        if not isinstance(self.layers, (tuple, list)) or len(self.layers) != 2:
            raise ValueError(
                f"layers must be a tuple of (start, end), got {self.layers}"
            )

        start_layer, end_layer = self.layers

        if not isinstance(start_layer, int) or not isinstance(end_layer, int):
            raise ValueError(
                f"Layer indices must be integers, got ({type(start_layer)}, {type(end_layer)})"
            )

        if start_layer < 0 or end_layer < 0:
            raise ValueError(
                f"Layer indices must be non-negative, got ({start_layer}, {end_layer})"
            )

        if start_layer >= end_layer:
            raise ValueError(
                f"Start layer must be less than end layer, got ({start_layer}, {end_layer})"
            )

        # SPE10 has 85 layers total (0-84), Tarbert is layers 35-84
        # But we allow any valid range for flexibility
        if end_layer > 85:
            raise ValueError(
                f"End layer {end_layer} exceeds SPE10 maximum (85 layers total)"
            )

    def _load_data(self) -> None:
        """
        Load data from HDF5 file.

        Loads all required fields for the SPE10 OOD test set.
        Uses lazy loading - data is only loaded on first access.
        """
        if self._loaded:
            return

        # Check if file exists
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

        self._data = {}

        with h5py.File(self.path, 'r') as f:
            # Load time-varying fields: (N, T, H, W)
            for key in ['pressure', 'saturation']:
                if key not in f:
                    raise KeyError(
                        f"Required key '{key}' not found in HDF5 file. "
                        f"Available keys: {list(f.keys())}"
                    )
                # Load up to n_trajectories
                n_available = f[key].shape[0]
                n_load = min(self.n_trajectories, n_available)
                arr = f[key][:n_load]
                self._data[key] = torch.from_numpy(arr).float()

            # Load permeability field: (N, H, W)
            if 'permeability' not in f:
                raise KeyError(
                    f"Required key 'permeability' not found in HDF5 file. "
                    f"Available keys: {list(f.keys())}"
                )
            n_available = f['permeability'].shape[0]
            n_load = min(self.n_trajectories, n_available)
            arr = f['permeability'][:n_load]
            self._data['permeability'] = torch.from_numpy(arr).float()

            # Load layer indices: (N,)
            if 'layer_idx' in f:
                arr = f['layer_idx'][:n_load]
                self._data['layer_idx'] = torch.from_numpy(arr).long()
            else:
                # Generate layer indices if not present
                # Distribute trajectories across layers 60-85
                n_layers = self.layers[1] - self.layers[0]
                layer_indices = torch.arange(self.layers[0], self.layers[1])
                # Repeat to cover all trajectories
                repeats = (n_load + n_layers - 1) // n_layers
                layer_indices = layer_indices.repeat(repeats)[:n_load]
                self._data['layer_idx'] = layer_indices

        # Update actual number of trajectories loaded
        self._actual_n_trajectories = self._data['pressure'].shape[0]

        # Validate shapes
        self._validate_shapes()

        self._loaded = True

    def _validate_shapes(self) -> None:
        """Validate that loaded data has expected shapes."""
        n_samples = self._actual_n_trajectories

        # Check time-varying fields
        for key in ['pressure', 'saturation']:
            shape = self._data[key].shape
            if len(shape) != 4:
                raise ValueError(
                    f"Expected 4D tensor for '{key}' (N, T, H, W), "
                    f"got shape {shape}"
                )
            if shape[0] != n_samples:
                raise ValueError(
                    f"Expected {n_samples} samples for '{key}', "
                    f"got {shape[0]}"
                )

        # Check permeability field
        shape = self._data['permeability'].shape
        if len(shape) != 3:
            raise ValueError(
                f"Expected 3D tensor for 'permeability' (N, H, W), "
                f"got shape {shape}"
            )
        if shape[0] != n_samples:
            raise ValueError(
                f"Expected {n_samples} samples for 'permeability', "
                f"got {shape[0]}"
            )

        # Check layer indices
        shape = self._data['layer_idx'].shape
        if len(shape) != 1:
            raise ValueError(
                f"Expected 1D tensor for 'layer_idx' (N,), "
                f"got shape {shape}"
            )
        if shape[0] != n_samples:
            raise ValueError(
                f"Expected {n_samples} samples for 'layer_idx', "
                f"got {shape[0]}"
            )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if not self._loaded:
            self._load_data()
        return self._actual_n_trajectories

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single trajectory sample.

        Args:
            idx: Index (0 to len-1)

        Returns:
            Dictionary containing:
                - pressure: (T, H, W) pressure field trajectory
                - saturation: (T, H, W) saturation field trajectory
                - permeability: (H, W) permeability field from Tarbert layer
                - layer_idx: scalar tensor with layer index (60-85)
        """
        # Ensure data is loaded
        self._load_data()

        # Validate index
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset "
                f"with {len(self)} samples"
            )

        return {
            'pressure': self._data['pressure'][idx],
            'saturation': self._data['saturation'][idx],
            'permeability': self._data['permeability'][idx],
            'layer_idx': self._data['layer_idx'][idx],
        }

    def get_all_data(self) -> Dict[str, torch.Tensor]:
        """
        Get all data for the OOD test set.

        Returns:
            Dictionary containing all tensors:
                - pressure: (N, T, H, W) - N=500
                - saturation: (N, T, H, W)
                - permeability: (N, H, W) - from layers 60-85
                - layer_idx: (N,) - layer index
        """
        self._load_data()
        return self._data.copy()

    @property
    def n_timesteps(self) -> int:
        """Get the number of timesteps in trajectories."""
        self._load_data()
        return self._data['pressure'].shape[1]

    @property
    def spatial_shape(self) -> tuple:
        """Get the spatial shape (H, W) of the fields."""
        self._load_data()
        return tuple(self._data['pressure'].shape[-2:])

    @property
    def layer_range(self) -> tuple:
        """Get the layer range (start, end) for this dataset."""
        return self.layers

    def get_samples_by_layer(self, layer: int) -> List[int]:
        """
        Get indices of samples from a specific layer.

        Args:
            layer: Layer index (should be in range [layers[0], layers[1]))

        Returns:
            List of sample indices from the specified layer
        """
        self._load_data()

        if layer < self.layers[0] or layer >= self.layers[1]:
            raise ValueError(
                f"Layer {layer} out of range [{self.layers[0]}, {self.layers[1]})"
            )

        indices = (self._data['layer_idx'] == layer).nonzero(as_tuple=True)[0]
        return indices.tolist()


# Register SPE10Dataset with the factory
DatasetFactory.register('spe10', SPE10Dataset)


class NavierStokesDataset(BenchmarkDataset):
    """
    Navier-Stokes 2D dataset at viscosity ╬╜=10^-4.

    Loads 1000 trajectories of 2D incompressible Navier-Stokes flow
    at viscosity ╬╜=10^-4. This dataset is used for evaluating
    physics-informed neural operators on fluid dynamics problems.

    The Navier-Stokes equations describe the motion of viscous fluid:
        Γêé╧ë/Γêét + (u┬╖Γêç)╧ë = ╬╜Γêç┬▓╧ë
    where ╧ë is vorticity and u = (u_x, u_y) is velocity.

    Dataset Structure:
        - vorticity: (N, T, H, W) - Vorticity field over time (N=1000)
        - velocity_x: (N, T, H, W) - x-component of velocity over time
        - velocity_y: (N, T, H, W) - y-component of velocity over time

    Split Configuration:
        - train: 700 trajectories (indices 0-699)
        - pretrain: 100 trajectories (indices 700-799)
        - finetune: Same as pretrain (indices 700-799)
        - test: 200 trajectories (indices 800-999)

    Args:
        config: Configuration dictionary containing:
            - path: Path to the HDF5 data file
            - viscosity: Kinematic viscosity (default 1e-4)
            - n_trajectories: Total number of trajectories (default 1000)
        split: Data split ('train', 'pretrain', 'finetune', 'test')

    Example:
        >>> config = {'path': '/data/navier_stokes.h5', 'viscosity': 1e-4}
        >>> dataset = NavierStokesDataset(config=config, split='train')
        >>> sample = dataset[0]
        >>> print(sample['vorticity'].shape)  # (T, H, W)
        >>> print(sample['velocity_x'].shape)  # (T, H, W)

    **Validates: Requirements 3.4**
    """

    # Default configuration
    DEFAULT_VISCOSITY = 1e-4
    DEFAULT_N_TRAJECTORIES = 1000

    # Default split ratios
    DEFAULT_SPLIT_RATIOS = {
        'train': 0.7,      # 700 trajectories
        'pretrain': 0.1,   # 100 trajectories
        'finetune': 0.1,   # Same as pretrain
        'test': 0.2        # 200 trajectories
    }

    # HDF5 keys for the dataset
    HDF5_KEYS = ['vorticity', 'velocity_x', 'velocity_y']

    def __init__(
        self,
        config: Dict[str, Any],
        split: str = "train",
    ):
        """
        Initialize Navier-Stokes 2D dataset.

        Args:
            config: Configuration dictionary with:
                - path: Path to HDF5 file
                - viscosity: Kinematic viscosity (default 1e-4)
                - n_trajectories: Total trajectories (default 1000)
            split: Data split ('train', 'pretrain', 'finetune', 'test')
        """
        super().__init__(config, split)

        # Extract configuration with defaults
        self.viscosity = config.get('viscosity', self.DEFAULT_VISCOSITY)
        self.n_trajectories = config.get('n_trajectories', self.DEFAULT_N_TRAJECTORIES)

        # Validate split
        valid_splits = ['train', 'pretrain', 'finetune', 'test']
        if split not in valid_splits:
            raise ValueError(
                f"Invalid split: '{split}'. "
                f"Valid splits: {valid_splits}"
            )

        # Data storage (lazy loading)
        self._data: Optional[Dict[str, torch.Tensor]] = None
        self._loaded = False
        self._start_idx = 0
        self._end_idx = 0

    def _compute_split_indices(self, n_total: int) -> None:
        """
        Compute start and end indices for the current split.

        Split layout for 1000 trajectories:
            - train: [0, 700)
            - pretrain: [700, 800)
            - finetune: [700, 800) (same as pretrain)
            - test: [800, 1000)

        Args:
            n_total: Total number of trajectories in the dataset
        """
        # Calculate split sizes
        n_train = int(n_total * self.DEFAULT_SPLIT_RATIOS['train'])
        n_pretrain = int(n_total * self.DEFAULT_SPLIT_RATIOS['pretrain'])
        n_test = int(n_total * self.DEFAULT_SPLIT_RATIOS['test'])

        # Assign indices based on split
        if self.split == 'train':
            self._start_idx = 0
            self._end_idx = n_train
        elif self.split in ('pretrain', 'finetune'):
            self._start_idx = n_train
            self._end_idx = n_train + n_pretrain
        else:  # test
            self._start_idx = n_train + n_pretrain
            self._end_idx = n_train + n_pretrain + n_test

        # Ensure we don't exceed total
        self._end_idx = min(self._end_idx, n_total)

    def _load_data(self) -> None:
        """
        Load data from HDF5 file.

        Loads vorticity, velocity_x, and velocity_y fields and slices
        to the current split. Uses lazy loading - data is only loaded
        on first access.
        """
        if self._loaded:
            return

        # Check if file exists
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Dataset file not found: {self.path}")

        self._data = {}

        with h5py.File(self.path, 'r') as f:
            # Determine total number of trajectories
            # Try vorticity first, then velocity_x
            primary_key = None
            for key in ['vorticity', 'velocity_x', 'u']:
                if key in f:
                    primary_key = key
                    break

            if primary_key is None:
                raise KeyError(
                    f"No recognized primary key found in HDF5 file. "
                    f"Expected one of: 'vorticity', 'velocity_x', 'u'. "
                    f"Available keys: {list(f.keys())}"
                )

            n_total = f[primary_key].shape[0]
            n_total = min(n_total, self.n_trajectories)

            # Compute split indices
            self._compute_split_indices(n_total)

            # Load vorticity field: (N, T, H, W)
            if 'vorticity' in f:
                arr = f['vorticity'][self._start_idx:self._end_idx]
                self._data['vorticity'] = torch.from_numpy(arr).float()
            elif 'w' in f:
                # Alternative key name for vorticity
                arr = f['w'][self._start_idx:self._end_idx]
                self._data['vorticity'] = torch.from_numpy(arr).float()
            else:
                raise KeyError(
                    f"Required key 'vorticity' (or 'w') not found in HDF5 file. "
                    f"Available keys: {list(f.keys())}"
                )

            # Load velocity_x field: (N, T, H, W)
            if 'velocity_x' in f:
                arr = f['velocity_x'][self._start_idx:self._end_idx]
                self._data['velocity_x'] = torch.from_numpy(arr).float()
            elif 'u' in f:
                # Alternative key name
                arr = f['u'][self._start_idx:self._end_idx]
                self._data['velocity_x'] = torch.from_numpy(arr).float()
            elif 'ux' in f:
                arr = f['ux'][self._start_idx:self._end_idx]
                self._data['velocity_x'] = torch.from_numpy(arr).float()
            else:
                raise KeyError(
                    f"Required key 'velocity_x' (or 'u', 'ux') not found in HDF5 file. "
                    f"Available keys: {list(f.keys())}"
                )

            # Load velocity_y field: (N, T, H, W)
            if 'velocity_y' in f:
                arr = f['velocity_y'][self._start_idx:self._end_idx]
                self._data['velocity_y'] = torch.from_numpy(arr).float()
            elif 'v' in f:
                # Alternative key name
                arr = f['v'][self._start_idx:self._end_idx]
                self._data['velocity_y'] = torch.from_numpy(arr).float()
            elif 'uy' in f:
                arr = f['uy'][self._start_idx:self._end_idx]
                self._data['velocity_y'] = torch.from_numpy(arr).float()
            else:
                raise KeyError(
                    f"Required key 'velocity_y' (or 'v', 'uy') not found in HDF5 file. "
                    f"Available keys: {list(f.keys())}"
                )

        # Validate shapes
        self._validate_shapes()

        self._loaded = True

    def _validate_shapes(self) -> None:
        """Validate that loaded data has expected shapes."""
        n_samples = self._end_idx - self._start_idx

        # All fields should be 4D: (N, T, H, W)
        for key in ['vorticity', 'velocity_x', 'velocity_y']:
            shape = self._data[key].shape
            if len(shape) != 4:
                raise ValueError(
                    f"Expected 4D tensor for '{key}' (N, T, H, W), "
                    f"got shape {shape}"
                )
            if shape[0] != n_samples:
                raise ValueError(
                    f"Expected {n_samples} samples for '{key}', "
                    f"got {shape[0]}"
                )

        # Verify all fields have same spatial dimensions
        vort_shape = self._data['vorticity'].shape
        for key in ['velocity_x', 'velocity_y']:
            if self._data[key].shape != vort_shape:
                raise ValueError(
                    f"Shape mismatch: 'vorticity' has shape {vort_shape}, "
                    f"but '{key}' has shape {self._data[key].shape}"
                )

    def __len__(self) -> int:
        """Return the number of samples in the current split."""
        if not self._loaded:
            self._load_data()
        return self._end_idx - self._start_idx

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single trajectory sample.

        Args:
            idx: Index within the current split (0 to len-1)

        Returns:
            Dictionary containing:
                - vorticity: (T, H, W) vorticity field trajectory
                - velocity_x: (T, H, W) x-velocity field trajectory
                - velocity_y: (T, H, W) y-velocity field trajectory
                - viscosity: scalar tensor with kinematic viscosity
        """
        # Ensure data is loaded
        self._load_data()

        # Validate index
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for split '{self.split}' "
                f"with {len(self)} samples"
            )

        return {
            'vorticity': self._data['vorticity'][idx],
            'velocity_x': self._data['velocity_x'][idx],
            'velocity_y': self._data['velocity_y'][idx],
            'viscosity': torch.tensor(self.viscosity),
        }

    def get_all_data(self) -> Dict[str, Any]:
        """
        Get all data for the current split.

        Returns:
            Dictionary containing all tensors for the split:
                - vorticity: (N, T, H, W)
                - velocity_x: (N, T, H, W)
                - velocity_y: (N, T, H, W)
                - viscosity: float
        """
        self._load_data()
        return {
            'vorticity': self._data['vorticity'],
            'velocity_x': self._data['velocity_x'],
            'velocity_y': self._data['velocity_y'],
            'viscosity': self.viscosity,
        }

    @property
    def n_timesteps(self) -> int:
        """Get the number of timesteps in trajectories."""
        self._load_data()
        return self._data['vorticity'].shape[1]

    @property
    def spatial_shape(self) -> tuple:
        """Get the spatial shape (H, W) of the fields."""
        self._load_data()
        return tuple(self._data['vorticity'].shape[-2:])

    @property
    def reynolds_number(self) -> float:
        """
        Get the Reynolds number based on viscosity.

        Assuming characteristic length L=1 and velocity U=1:
            Re = UL/╬╜ = 1/╬╜

        Returns:
            Reynolds number (inverse of viscosity)
        """
        return 1.0 / self.viscosity


# Register NavierStokesDataset with the factory
DatasetFactory.register('navier_stokes', NavierStokesDataset)


class UnlabeledDarcyDataset(Dataset):
    """
    Dataset that loads only coefficient fields for self-supervised pretraining.
    
    This dataset is designed for the self-supervised pretraining phase of PI-JEPA,
    where only unlabeled coefficient fields (permeability K) are needed. It does
    not require solution fields (pressure/saturation) to be present.
    
    Dataset Structure:
        - x: (N, 1, H, W) - Coefficient field (permeability)
    
    The dataset supports:
        - Loading from neuralop's darcy_flow_small format
        - Loading from HDF5 files with 'x' or 'permeability' keys
        - Loading from NPZ files with 'k' key (existing PI-JEPA format)
        - Configurable number of samples (N_u for unlabeled pretraining)
    
    Args:
        config: Configuration dictionary containing:
            - path: Path to the data file (HDF5, NPZ, or neuralop cache dir)
            - n_samples: Number of unlabeled samples to load (N_u)
            - resolution: Spatial resolution (default 64)
            - normalize: Whether to normalize coefficient fields (default True)
        split: Data split ('train', 'pretrain', 'test') - affects which portion to load
    
    Example:
        >>> config = {'path': '/data/darcy', 'n_samples': 1000, 'resolution': 64}
        >>> dataset = UnlabeledDarcyDataset(config=config, split='pretrain')
        >>> sample = dataset[0]
        >>> print(sample['x'].shape)  # (1, 64, 64)
    
    **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**
    """
    
    # Default split ratios
    DEFAULT_SPLIT_RATIOS = {
        'train': 0.7,
        'pretrain': 0.2,
        'finetune': 0.2,  # Same as pretrain
        'test': 0.1
    }
    
    def __init__(
        self,
        config: Dict[str, Any],
        split: str = "pretrain",
    ):
        """
        Initialize UnlabeledDarcyDataset.
        
        Args:
            config: Configuration dictionary with:
                - path: Path to data file or directory
                - n_samples: Number of samples to load (optional, loads all if not specified)
                - resolution: Spatial resolution (default 64)
                - normalize: Whether to normalize (default True)
            split: Data split ('train', 'pretrain', 'finetune', 'test')
        """
        self.config = config
        self.split = split
        self.path = config.get('path', '')
        self.n_samples = config.get('n_samples', None)
        self.resolution = config.get('resolution', 64)
        self.normalize = config.get('normalize', True)
        
        # Validate split
        valid_splits = ['train', 'pretrain', 'finetune', 'test']
        if split not in valid_splits:
            raise ValueError(
                f"Invalid split: '{split}'. "
                f"Valid splits: {valid_splits}"
            )
        
        # Data storage (lazy loading)
        self._data: Optional[torch.Tensor] = None
        self._loaded = False
        self._start_idx = 0
        self._end_idx = 0
        self._actual_n_samples = 0
    
    def _compute_split_indices(self, n_total: int) -> None:
        """
        Compute start and end indices for the current split.
        
        Args:
            n_total: Total number of samples in the dataset
        """
        # Calculate split sizes
        n_train = int(n_total * self.DEFAULT_SPLIT_RATIOS['train'])
        n_pretrain = int(n_total * self.DEFAULT_SPLIT_RATIOS['pretrain'])
        n_test = int(n_total * self.DEFAULT_SPLIT_RATIOS['test'])
        
        # Assign indices based on split
        if self.split == 'train':
            self._start_idx = 0
            self._end_idx = n_train
        elif self.split in ('pretrain', 'finetune'):
            self._start_idx = n_train
            self._end_idx = n_train + n_pretrain
        else:  # test
            self._start_idx = n_train + n_pretrain
            self._end_idx = n_train + n_pretrain + n_test
        
        # Ensure we don't exceed total
        self._end_idx = min(self._end_idx, n_total)
        
        # Apply n_samples limit if specified
        if self.n_samples is not None:
            available = self._end_idx - self._start_idx
            if self.n_samples < available:
                self._end_idx = self._start_idx + self.n_samples
    
    def _load_from_hdf5(self, path: str) -> torch.Tensor:
        """Load coefficient field from HDF5 file."""
        with h5py.File(path, 'r') as f:
            # Try different key names for coefficient field
            coeff_keys = ['x', 'permeability', 'k', 'coefficient', 'K']
            coeff_key = None
            for key in coeff_keys:
                if key in f:
                    coeff_key = key
                    break
            
            if coeff_key is None:
                raise KeyError(
                    f"No coefficient field found in HDF5 file. "
                    f"Expected one of: {coeff_keys}. "
                    f"Available keys: {list(f.keys())}"
                )
            
            # Get total samples and compute split indices
            n_total = f[coeff_key].shape[0]
            self._compute_split_indices(n_total)
            
            # Load coefficient field
            arr = f[coeff_key][self._start_idx:self._end_idx]
            x = torch.from_numpy(arr).float()
            
            # Ensure shape is (N, 1, H, W)
            if x.ndim == 3:
                x = x.unsqueeze(1)
            elif x.ndim == 2:
                x = x.unsqueeze(0).unsqueeze(0)
            
            return x
    
    def _load_from_npz(self, path: str) -> torch.Tensor:
        """Load coefficient field from NPZ file (existing PI-JEPA format)."""
        import numpy as np
        raw = np.load(path)
        
        # Try different key names
        coeff_keys = ['k', 'x', 'permeability', 'K']
        coeff_key = None
        for key in coeff_keys:
            if key in raw:
                coeff_key = key
                break
        
        if coeff_key is None:
            raise KeyError(
                f"No coefficient field found in NPZ file. "
                f"Expected one of: {coeff_keys}. "
                f"Available keys: {list(raw.keys())}"
            )
        
        # Get total samples and compute split indices
        arr = raw[coeff_key]
        n_total = arr.shape[0]
        self._compute_split_indices(n_total)
        
        # Load coefficient field
        x = torch.from_numpy(arr[self._start_idx:self._end_idx]).float()
        
        # Ensure shape is (N, 1, H, W)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        elif x.ndim == 4 and x.shape[1] != 1:
            # If shape is (N, T, H, W), take first timestep
            x = x[:, 0:1, :, :]
        
        return x
    
    def _load_from_generated(self, train_path: str, test_path: str) -> torch.Tensor:
        """
        Load coefficient field from generated Darcy data files.
        
        Args:
            train_path: Path to darcy_train.pt
            test_path: Path to darcy_test.pt
            
        Returns:
            Coefficient field tensor (N, 1, H, W)
        """
        if self.split in ('train', 'pretrain', 'finetune'):
            data = torch.load(train_path, weights_only=False)
        else:
            data = torch.load(test_path, weights_only=False)
        
        # 'x' contains coefficient fields (N, 1, H, W)
        x = data['x']
        
        # Compute split indices
        n_total = x.shape[0]
        self._compute_split_indices(n_total)
        x = x[self._start_idx:self._end_idx]
        
        return x
    
    def _load_from_neuralop(self) -> torch.Tensor:
        """Load coefficient field from neuralop's darcy_flow_small dataset."""
        try:
            from neuralop.data.datasets import load_darcy_flow_small
        except ImportError:
            try:
                from neuralop.datasets import load_darcy_flow_small
            except ImportError:
                raise ImportError(
                    "neuralop package is required to load darcy_flow_small dataset. "
                    "Install with: pip install neuralop"
                )
        
        # Determine data root - use default cache if not specified
        if self.path and os.path.exists(self.path):
            data_root = self.path
        else:
            # Use default cache directory
            data_root = os.path.join(os.path.expanduser("~"), ".cache", "neuralop", "darcy")
            os.makedirs(data_root, exist_ok=True)
        
        print(f"Loading Darcy flow data from: {data_root}")
        
        # Load the dataset at the correct resolution
        # Note: neuralop's darcy_flow_small training data is always 16x16
        # The test_resolutions parameter only affects test data
        train_loader, test_loader, data_processor = load_darcy_flow_small(
            n_train=1000,
            n_tests=[200],  # Must be a list
            batch_size=1,
            test_batch_sizes=[1],  # Must be a list
            data_root=data_root,
            test_resolutions=[self.resolution],  # Request correct resolution for testing
        )
        
        # Collect coefficient fields based on split
        if self.split in ('train', 'pretrain', 'finetune'):
            loader = train_loader
        else:
            loader = test_loader
        
        x_list = []
        for batch in loader:
            # neuralop returns dict with 'x' key for coefficient field
            x = batch['x']
            x_list.append(x)
        
        x = torch.cat(x_list, dim=0)
        
        # Apply n_samples limit
        n_total = x.shape[0]
        self._compute_split_indices(n_total)
        x = x[self._start_idx:self._end_idx]
        
        # Ensure shape is (N, 1, H, W)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        
        return x
    
    def _load_data(self) -> None:
        """
        Load coefficient field data.
        
        Supports multiple formats:
        - Generated .pt files (data/darcy/darcy_train.pt, darcy_test.pt)
        - HDF5 files (.h5, .hdf5)
        - NPZ files (.npz)
        - neuralop's darcy_flow_small (if path is empty or directory)
        """
        if self._loaded:
            return
        
        # First, check for generated data in data/darcy directory
        generated_train = os.path.join("data", "darcy", "darcy_train.pt")
        generated_test = os.path.join("data", "darcy", "darcy_test.pt")
        
        if os.path.exists(generated_train) and os.path.exists(generated_test):
            print(f"Loading from generated data: {generated_train}")
            self._data = self._load_from_generated(generated_train, generated_test)
        # Determine file type and load accordingly
        elif self.path and os.path.isfile(self.path):
            if self.path.endswith(('.h5', '.hdf5')):
                self._data = self._load_from_hdf5(self.path)
            elif self.path.endswith('.npz'):
                self._data = self._load_from_npz(self.path)
            elif self.path.endswith('.pt'):
                # Single .pt file - load directly
                data = torch.load(self.path, weights_only=False)
                self._data = data['x']
                n_total = self._data.shape[0]
                self._compute_split_indices(n_total)
                self._data = self._data[self._start_idx:self._end_idx]
            else:
                raise ValueError(
                    f"Unsupported file format: {self.path}. "
                    f"Supported formats: .h5, .hdf5, .npz, .pt"
                )
        elif self.path and os.path.isdir(self.path):
            # Check for generated .pt files in directory
            train_pt = os.path.join(self.path, "darcy_train.pt")
            test_pt = os.path.join(self.path, "darcy_test.pt")
            if os.path.exists(train_pt) and os.path.exists(test_pt):
                print(f"Loading from generated data: {train_pt}")
                self._data = self._load_from_generated(train_pt, test_pt)
            else:
                # Check for NPZ files in directory (existing PI-JEPA format)
                npz_file = os.path.join(self.path, f"{self.split}.npz")
                if os.path.exists(npz_file):
                    self._data = self._load_from_npz(npz_file)
                else:
                    # Try neuralop format
                    self._data = self._load_from_neuralop()
        elif self.path and not os.path.exists(self.path):
            # Path specified but doesn't exist - fall back to neuralop
            print(f"Info: Path '{self.path}' not found, using neuralop auto-download")
            self._data = self._load_from_neuralop()
        else:
            # No path specified, try neuralop
            self._data = self._load_from_neuralop()
        
        # Normalize if requested
        if self.normalize:
            mean = self._data.mean()
            std = self._data.std() + 1e-6
            self._data = (self._data - mean) / std
        
        self._actual_n_samples = self._data.shape[0]
        self._loaded = True
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if not self._loaded:
            self._load_data()
        return self._actual_n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single coefficient field sample.
        
        Args:
            idx: Index (0 to len-1)
        
        Returns:
            Dictionary containing:
                - x: (1, H, W) coefficient field tensor
        
        Note: This method intentionally does NOT return 'y' (solution field)
        as this dataset is designed for self-supervised pretraining where
        solutions are not available.
        """
        # Ensure data is loaded
        self._load_data()
        
        # Validate index
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset "
                f"with {len(self)} samples"
            )
        
        return {'x': self._data[idx]}
    
    def get_all_data(self) -> Dict[str, torch.Tensor]:
        """
        Get all coefficient field data.
        
        Returns:
            Dictionary containing:
                - x: (N, 1, H, W) all coefficient fields
        """
        self._load_data()
        return {'x': self._data.clone()}
    
    @property
    def spatial_shape(self) -> tuple:
        """Get the spatial shape (H, W) of the fields."""
        self._load_data()
        return tuple(self._data.shape[-2:])
    
    @property
    def n_channels(self) -> int:
        """Get the number of channels (always 1 for coefficient field)."""
        return 1


# Register UnlabeledDarcyDataset with the factory
DatasetFactory.register('darcy_unlabeled', UnlabeledDarcyDataset)
