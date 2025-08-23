from dataclasses import dataclass
import numpy as np
import yaml

from quarc.settings import load as load_settings

cfg = load_settings()


@dataclass
class BinningConfig:
    """Configuration for binning across all stages."""

    temperature_bins: np.ndarray
    reactant_amount_bins: np.ndarray
    agent_amount_bins: np.ndarray

    @classmethod
    def default(cls):
        """Default binning configuration matching current setup."""
        temperature_bins = np.arange(-100, 201, 10) + 273.15
        reactant_amount_bins = np.array(
            [0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.75, 2.25, 2.75, 3.5, 4.5, 5.5, 6.5, 7.5]
        )

        substoichiometric = np.array([0, 0.075, 0.15, 0.25, 0.55, 0.95])
        stoichiometric = np.array([1.25, 1.75, 2.25, 2.75, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])
        superstoichiometric = np.array([15.5, 25.5, 35.5, 45.5, 55.5, 65.5, 75.5, 85.5, 100.5])
        agent_amount_bins = np.concatenate(
            [substoichiometric, stoichiometric, superstoichiometric]
        )

        return cls(temperature_bins, reactant_amount_bins, agent_amount_bins)

    @classmethod
    def from_config(cls, config_path: str):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        binning_config = config["binning"]

        temperature_bins = np.array(binning_config["temperature"]["bins"])
        reactant_amount_bins = np.array(binning_config["reactant_amount"]["bins"])
        agent_amount_bins = np.concatenate(
            [
                binning_config["agent_amount"]["substoichiometric"],
                binning_config["agent_amount"]["stoichiometric"],
                binning_config["agent_amount"]["superstoichiometric"],
            ]
        )

        return cls(temperature_bins, reactant_amount_bins, agent_amount_bins)

    def get_bin_labels(self, bin_type: str) -> dict[int, str]:
        """Generate human-readable labels for bins."""
        bin_labels = {}
        bins = None

        if bin_type == "temperature":
            bins_in_celsius = [bin - 273.15 for bin in self.temperature_bins]
            bins = bins_in_celsius
        elif bin_type == "reactant":
            bins = self.reactant_amount_bins
        elif bin_type == "agent":
            bins = self.agent_amount_bins
        else:
            raise ValueError(f"Unknown bin_type: {bin_type}")

        for i in range(len(bins) + 1):
            if i == 0:
                label = f"(-inf, {bins[i]:.2f})"
            elif i == len(bins):
                label = f"[{bins[i-1]:.2f}, inf)"
            else:
                label = f"[{bins[i-1]:.2f}, {bins[i]:.2f})"
            bin_labels[i] = label

        return bin_labels
