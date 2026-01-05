"""Dataset classes and data loaders."""

from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from ase import Atoms
from ase.io import read


class MLIPDataset(Dataset):
    """Dataset class for MLIP training data."""

    def __init__(
        self,
        data_path: Union[str, Path],
        format: str = "xyz",
        energy_key: str = "energy",
        forces_key: str = "forces",
        stress_key: Optional[str] = "stress",
        transform: Optional[callable] = None
    ):
        """Initialize MLIP dataset.

        Args:
            data_path: Path to data file or directory
            format: Data format ('xyz', 'traj', etc.)
            energy_key: Key for energy values in atoms.info
            forces_key: Key for forces in atoms.arrays
            stress_key: Key for stress values in atoms.info
            transform: Optional data transformation function
        """
        self.data_path = Path(data_path)
        self.format = format
        self.energy_key = energy_key
        self.forces_key = forces_key
        self.stress_key = stress_key
        self.transform = transform

        self._load_data()

    def _load_data(self) -> None:
        """Load atomic structures from file."""
        if self.data_path.is_file():
            self.structures = read(str(self.data_path), ":", format=self.format)
        elif self.data_path.is_dir():
            # Load from directory of files
            self.structures = []
            for file_path in self.data_path.glob("*"):
                if file_path.suffix in [".xyz", ".traj"]:
                    structures = read(str(file_path), ":", format=self.format)
                    self.structures.extend(structures)
        else:
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

        print(f"Loaded {len(self.structures)} structures from {self.data_path}")

    def __len__(self) -> int:
        """Return number of structures in dataset."""
        return len(self.structures)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single data point.

        Args:
            idx: Index of the structure

        Returns:
            Dictionary containing structure data and targets
        """
        atoms = self.structures[idx]

        # Convert ASE atoms to tensor format
        data = self._atoms_to_tensor(atoms)

        if self.transform:
            data = self.transform(data)

        return data

    def _atoms_to_tensor(self, atoms: Atoms) -> Dict[str, torch.Tensor]:
        """Convert ASE atoms object to tensor format.

        Args:
            atoms: ASE atoms object

        Returns:
            Dictionary with tensors for model input
        """
        data = {
            "pos": torch.tensor(atoms.positions, dtype=torch.float32),
            "atomic_numbers": torch.tensor(atoms.numbers, dtype=torch.long),
            "cell": torch.tensor(atoms.cell.array, dtype=torch.float32),
            "pbc": torch.tensor(atoms.pbc, dtype=torch.bool),
        }

        # Add energy if available
        if self.energy_key in atoms.info:
            data["energy"] = torch.tensor(atoms.info[self.energy_key], dtype=torch.float32)

        # Add forces if available
        if self.forces_key in atoms.arrays:
            data["forces"] = torch.tensor(atoms.arrays[self.forces_key], dtype=torch.float32)

        # Add stress if available
        if self.stress_key and self.stress_key in atoms.info:
            stress = atoms.info[self.stress_key]
            if stress is not None:
                data["stress"] = torch.tensor(stress, dtype=torch.float32)

        return data


def create_dataloader(
    dataset: MLIPDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """Create a DataLoader for MLIP dataset.

    Args:
        dataset: MLIPDataset instance
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        **kwargs: Additional arguments for DataLoader

    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        **kwargs
    )


def _collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching atomic structures.

    Args:
        batch: List of data dictionaries

    Returns:
        Batched data dictionary
    """
    # TODO: Implement proper batching for variable-size structures
    # For now, return a simple concatenation
    if len(batch) == 1:
        return batch[0]

    # This is a simplified implementation
    # In practice, you'd need to handle variable numbers of atoms
    batched = {}
    for key in batch[0].keys():
        if key in ["energy", "stress"]:
            # Scalar properties
            batched[key] = torch.stack([item[key] for item in batch])
        else:
            # Handle other properties (positions, forces, etc.)
            batched[key] = torch.cat([item[key] for item in batch], dim=0)

    return batched