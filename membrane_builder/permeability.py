# Membrane permeability prediction based on PerMM method
# Lomize & Pogozheva, J Chem Inf Model 2019

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import math
import numpy as np

from .lipids import LipidLibrary, LipidCategory

# Physical constants
R = 8.314
KB = 1.380649e-23
T_STANDARD = 310.15
RT = R * T_STANDARD / 1000


class AtomType(Enum):
    C_SP3 = "C_sp3"
    C_SP2 = "C_sp2"
    C_AROMATIC = "C_aromatic"
    N_AMINE = "N_amine"
    N_AMIDE = "N_amide"
    N_AROMATIC = "N_aromatic"
    O_HYDROXYL = "O_hydroxyl"
    O_ETHER = "O_ether"
    O_CARBONYL = "O_carbonyl"
    O_CARBOXYL = "O_carboxyl"
    S_THIOL = "S_thiol"
    S_THIOETHER = "S_thioether"
    F = "F"
    CL = "Cl"
    BR = "Br"
    P = "P"


class MembraneType(Enum):
    BLM = "BLM"
    PAMPA_DS = "PAMPA-DS"
    BBB = "BBB"
    CACO2 = "Caco-2"


@dataclass
class AtomDescriptor:
    element: str
    atom_type: AtomType
    x: float
    y: float
    z: float
    asa: float = 0.0
    charge: float = 0.0
    e_param: float = 0.0
    a_param: float = 0.0
    b_param: float = 0.0


@dataclass
class PolarGroup:
    name: str
    dipole_moment: float
    center_x: float
    center_y: float
    center_z: float


@dataclass
class IonizableGroup:
    name: str
    pka: float
    charge_when_ionized: float
    center_x: float
    center_y: float
    center_z: float
    is_acid: bool = True


@dataclass
class MoleculeDescriptor:
    name: str
    atoms: List[AtomDescriptor] = field(default_factory=list)
    polar_groups: List[PolarGroup] = field(default_factory=list)
    ionizable_groups: List[IonizableGroup] = field(default_factory=list)
    molecular_weight: float = 0.0
    total_asa: float = 0.0

    def __post_init__(self):
        if self.total_asa == 0.0 and self.atoms:
            self.total_asa = sum(a.asa for a in self.atoms)

    @classmethod
    def simple(cls, name, molecular_weight, total_asa, n_hbd=0, n_hba=0, charge=0.0, pka=None):
        mol = cls(name=name, molecular_weight=molecular_weight, total_asa=total_asa)

        for i in range(n_hbd):
            mol.polar_groups.append(PolarGroup(
                name="HBD_{}".format(i),
                dipole_moment=1.5,
                center_x=0, center_y=0, center_z=0
            ))

        for i in range(n_hba):
            mol.polar_groups.append(PolarGroup(
                name="HBA_{}".format(i),
                dipole_moment=1.2,
                center_x=0, center_y=0, center_z=0
            ))

        if pka is not None:
            is_acid = charge < 0
            mol.ionizable_groups.append(IonizableGroup(
                name="ionizable",
                pka=pka,
                charge_when_ionized=charge,
                center_x=0, center_y=0, center_z=0,
                is_acid=is_acid
            ))

        return mol

    @classmethod
    def water(cls):
        return cls.simple(name="water", molecular_weight=18.015, total_asa=40.0, n_hbd=2, n_hba=1)

    @classmethod
    def glucose(cls):
        return cls.simple(name="glucose", molecular_weight=180.16, total_asa=180.0, n_hbd=5, n_hba=6)

    @classmethod
    def ethanol(cls):
        return cls.simple(name="ethanol", molecular_weight=46.07, total_asa=80.0, n_hbd=1, n_hba=1)


@dataclass
class MembraneProfiles:
    z_positions: np.ndarray
    epsilon: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    core_boundary: float = 15.0

    def get_epsilon_at_z(self, z):
        return float(np.interp(abs(z), self.z_positions, self.epsilon))

    def get_alpha_at_z(self, z):
        return float(np.interp(abs(z), self.z_positions, self.alpha))

    def get_beta_at_z(self, z):
        return float(np.interp(abs(z), self.z_positions, self.beta))


@dataclass
class PermeabilityResults:
    log_p: float
    permeability_cm_s: float
    energy_profile: Dict[str, np.ndarray]
    partition_coefficients: Dict[str, np.ndarray]
    membrane_type: str
    membrane_bound_energy: float
    binding_position: float

    def summary(self):
        lines = [
            "Permeability prediction",
            "-" * 40,
            "Membrane type:     {}".format(self.membrane_type),
            "log P:             {:.2f}".format(self.log_p),
            "Permeability:      {:.2e} cm/s".format(self.permeability_cm_s),
            "",
            "Binding energy:    {:.1f} kJ/mol".format(self.membrane_bound_energy),
            "Binding position:  {:.1f} A from center".format(self.binding_position)
        ]
        return "\n".join(lines)


# Atomic solvation parameters from Table S1
ATOMIC_SOLVATION_PARAMS = {
    AtomType.C_SP3: {"sigma_0": 0.012, "e": 0.0, "a": 0.0, "b": 0.0},
    AtomType.C_SP2: {"sigma_0": 0.004, "e": 0.0, "a": 0.0, "b": 0.0},
    AtomType.C_AROMATIC: {"sigma_0": 0.000, "e": 0.0, "a": 0.0, "b": 0.0},
    AtomType.N_AMINE: {"sigma_0": -0.100, "e": -3.0, "a": 5.5, "b": 0.0},
    AtomType.N_AMIDE: {"sigma_0": -0.060, "e": -2.0, "a": 3.0, "b": 0.0},
    AtomType.N_AROMATIC: {"sigma_0": -0.040, "e": -1.5, "a": 0.0, "b": 2.0},
    AtomType.O_HYDROXYL: {"sigma_0": -0.070, "e": -2.0, "a": 4.0, "b": 3.0},
    AtomType.O_ETHER: {"sigma_0": -0.020, "e": -1.0, "a": 0.0, "b": 2.5},
    AtomType.O_CARBONYL: {"sigma_0": -0.050, "e": -1.5, "a": 0.0, "b": 4.0},
    AtomType.O_CARBOXYL: {"sigma_0": -0.080, "e": -2.5, "a": 2.0, "b": 4.5},
    AtomType.S_THIOL: {"sigma_0": 0.005, "e": -0.5, "a": 1.0, "b": 0.5},
    AtomType.S_THIOETHER: {"sigma_0": 0.008, "e": -0.3, "a": 0.0, "b": 0.5},
    AtomType.F: {"sigma_0": -0.010, "e": -0.8, "a": 0.0, "b": 0.5},
    AtomType.CL: {"sigma_0": 0.005, "e": -0.5, "a": 0.0, "b": 0.3},
    AtomType.BR: {"sigma_0": 0.008, "e": -0.3, "a": 0.0, "b": 0.2},
    AtomType.P: {"sigma_0": -0.030, "e": -1.0, "a": 0.0, "b": 2.0}
}

# Membrane calibration constants
CALIBRATION_PARAMS = {
    MembraneType.BLM: {"slope": 1.063, "intercept": 3.669},
    MembraneType.PAMPA_DS: {"slope": 0.981, "intercept": 2.159},
    MembraneType.BBB: {"slope": 0.375, "intercept": -1.600},
    MembraneType.CACO2: {"slope": 0.272, "intercept": -2.541}
}


class MembraneProfileGenerator:
    def __init__(self):
        self.library = LipidLibrary()

    def get_default_profiles(self):
        z = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])

        epsilon = np.array([
            2.0, 2.0, 2.0, 2.1, 2.3, 2.8,
            4.5, 8.0, 15.0, 25.0,
            40.0, 55.0, 68.0, 75.0, 77.5, 78.4
        ])

        alpha = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.02,
            0.08, 0.20, 0.35, 0.50,
            0.62, 0.72, 0.78, 0.80, 0.81, 0.82
        ])

        beta = np.array([
            0.0, 0.0, 0.0, 0.0, 0.0, 0.02,
            0.08, 0.20, 0.35, 0.50,
            0.62, 0.72, 0.78, 0.80, 0.81, 0.82
        ])

        return MembraneProfiles(z_positions=z, epsilon=epsilon, alpha=alpha, beta=beta, core_boundary=15.0)

    def get_profiles_for_composition(self, composition, thickness=36.5):
        profiles = self.get_default_profiles()

        if not composition:
            return profiles

        total = sum(composition.values())
        if total == 0:
            return profiles

        chol_frac = 0.0
        sm_frac = 0.0
        unsat_frac = 0.0
        charged_frac = 0.0

        for name, count in composition.items():
            frac = count / total
            lipid = self.library.get(name)
            if lipid is None:
                continue

            if lipid.category == LipidCategory.STEROL:
                chol_frac += frac
            elif lipid.category == LipidCategory.SPHINGOMYELIN:
                sm_frac += frac

            total_unsat = sum(lipid.tail_unsaturations)
            if total_unsat > 0:
                unsat_frac += frac

            if lipid.charge != 0:
                charged_frac += frac

        if chol_frac > 0:
            transition_factor = 1.0 - 0.3 * chol_frac
            profiles.epsilon = self._sharpen_profile(profiles.epsilon, profiles.z_positions, transition_factor)
            profiles.alpha = self._sharpen_profile(profiles.alpha, profiles.z_positions, transition_factor)
            profiles.beta = self._sharpen_profile(profiles.beta, profiles.z_positions, transition_factor)

        if sm_frac > 0:
            shift = -1.0 * sm_frac
            profiles.z_positions = profiles.z_positions + shift

        if unsat_frac > 0.5:
            broaden_factor = 1.0 + 0.2 * (unsat_frac - 0.5)
            profiles.z_positions = profiles.z_positions * broaden_factor

        thickness_ratio = thickness / 36.5
        profiles.z_positions = profiles.z_positions * (thickness_ratio ** 0.5)
        profiles.core_boundary = 15.0 * (thickness_ratio ** 0.5)

        return profiles

    def _sharpen_profile(self, profile, z, factor):
        mid_val = (profile[0] + profile[-1]) / 2
        mid_idx = np.argmin(np.abs(profile - mid_val))
        z_shifted = z - z[mid_idx]
        z_compressed = z_shifted * factor + z[mid_idx]
        return np.interp(z, z_compressed, profile)


class TransferEnergyCalculator:
    EPSILON_WATER = 78.4
    ALPHA_WATER = 0.82
    BETA_WATER = 0.82

    def __init__(self, profiles):
        self.profiles = profiles

    def calculate_transfer_energy(self, molecule, z_position, pH=7.4, temperature=T_STANDARD):
        energy = 0.0

        if molecule.atoms:
            for atom in molecule.atoms:
                sigma = self._get_solvation_parameter(atom, z_position)
                energy += sigma * atom.asa
        else:
            energy += self._simplified_solvation_energy(molecule, z_position)

        for group in molecule.polar_groups:
            eta = self._get_dipole_penalty(z_position)
            energy += eta * group.dipole_moment

        for ionizable in molecule.ionizable_groups:
            energy += self._ionization_energy(ionizable, z_position, pH, temperature)

        return energy

    def _get_solvation_parameter(self, atom, z):
        epsilon_bil = self.profiles.get_epsilon_at_z(z)
        alpha_bil = self.profiles.get_alpha_at_z(z)
        beta_bil = self.profiles.get_beta_at_z(z)

        params = ATOMIC_SOLVATION_PARAMS.get(atom.atom_type)
        if params is None:
            return 0.0

        sigma = params["sigma_0"] + params["e"] * (1/epsilon_bil - 1/self.EPSILON_WATER)
        sigma += params["a"] * (alpha_bil - self.ALPHA_WATER)
        sigma += params["b"] * (beta_bil - self.BETA_WATER)

        return sigma

    def _simplified_solvation_energy(self, molecule, z):
        epsilon_bil = self.profiles.get_epsilon_at_z(z)
        alpha_bil = self.profiles.get_alpha_at_z(z)

        hydrophobic_asa = molecule.total_asa * 0.6
        polar_asa = molecule.total_asa * 0.4

        hydrophobic_energy = -0.012 * hydrophobic_asa * (1.0 - epsilon_bil / self.EPSILON_WATER)
        polar_penalty = 0.08 * polar_asa * (1.0 - alpha_bil / self.ALPHA_WATER)

        return hydrophobic_energy + polar_penalty

    def _get_dipole_penalty(self, z):
        epsilon_bil = self.profiles.get_epsilon_at_z(z)
        return 0.5 * (self.EPSILON_WATER / epsilon_bil - 1.0)

    def _ionization_energy(self, group, z, pH, temperature):
        RT_local = R * temperature / 1000

        if group.is_acid:
            delta_G_ion = 2.303 * RT_local * (pH - group.pka)
        else:
            delta_G_ion = 2.303 * RT_local * (group.pka - pH)

        epsilon_bil = self.profiles.get_epsilon_at_z(z)
        born_energy = 332.0 * (group.charge_when_ionized ** 2) * (1/epsilon_bil - 1/self.EPSILON_WATER) / 4.0

        neutral_energy = 0.0
        ionized_energy = born_energy + delta_G_ion

        return min(neutral_energy, ionized_energy)


class PermeabilityPredictor:
    def __init__(self, composition=None, membrane_thickness=36.5, temperature=T_STANDARD):
        self.composition = composition or {}
        self.thickness = membrane_thickness
        self.temperature = temperature

        profile_gen = MembraneProfileGenerator()
        if composition:
            self.profiles = profile_gen.get_profiles_for_composition(composition, membrane_thickness)
        else:
            self.profiles = profile_gen.get_default_profiles()

        self.energy_calc = TransferEnergyCalculator(self.profiles)

    def calculate(self, molecule, membrane_type=MembraneType.BLM, pH=7.4, n_points=31):
        if isinstance(membrane_type, str):
            membrane_type = MembraneType(membrane_type)

        z_range = np.linspace(-self.profiles.core_boundary, self.profiles.core_boundary, n_points)

        energy_profile = np.array([
            self.energy_calc.calculate_transfer_energy(molecule, z, pH, self.temperature)
            for z in z_range
        ])

        RT_val = R * self.temperature / 1000
        K_profile = np.exp(-energy_profile / RT_val)

        K_safe = np.maximum(K_profile, 1e-10)
        # Use trapezoid (numpy 2.0+) or trapz (older numpy)
        try:
            integral = np.trapezoid(1.0 / K_safe, z_range)
        except AttributeError:
            integral = np.trapz(1.0 / K_safe, z_range)

        asa = molecule.total_asa if molecule.total_asa > 0 else 100.0
        log_P_sigma = -np.log10(max(integral, 1e-20)) - np.log10(asa)

        calib = CALIBRATION_PARAMS[membrane_type]
        log_P_calc = calib["slope"] * log_P_sigma + calib["intercept"]

        min_energy_idx = np.argmin(energy_profile)
        membrane_bound_energy = energy_profile[min_energy_idx]
        binding_position = z_range[min_energy_idx]

        return PermeabilityResults(
            log_p=log_P_calc,
            permeability_cm_s=10 ** log_P_calc,
            energy_profile={"z": z_range, "energy_kJ_mol": energy_profile},
            partition_coefficients={"z": z_range, "K": K_profile},
            membrane_type=membrane_type.value,
            membrane_bound_energy=membrane_bound_energy,
            binding_position=binding_position
        )

    @classmethod
    def quick_calculate(cls, molecule, membrane_type=MembraneType.BLM):
        predictor = cls()
        result = predictor.calculate(molecule, membrane_type)
        return result.log_p


class MembraneCompositionOptimizer:
    def __init__(self, target_molecule, base_composition=None):
        self.target_molecule = target_molecule
        self.base_composition = base_composition or {"POPC": 128}
        self.library = LipidLibrary()

    def predict_permeability_for_composition(self, composition, membrane_type=MembraneType.BLM):
        total = sum(composition.values())
        if total == 0:
            return -10.0

        weighted_thickness = 0.0
        for name, count in composition.items():
            lipid = self.library.get(name)
            if lipid:
                weighted_thickness += lipid.thickness_contribution * count

        thickness = weighted_thickness / total if total > 0 else 36.5

        predictor = PermeabilityPredictor(composition=composition, membrane_thickness=thickness)
        result = predictor.calculate(self.target_molecule, membrane_type)
        return result.log_p

    def optimize_for_target_permeability(self, target_log_p, membrane_type=MembraneType.BLM, max_iterations=100):
        best_composition = dict(self.base_composition)
        best_diff = float("inf")

        for chol_frac in np.linspace(0, 0.45, 10):
            composition = self._adjust_cholesterol(self.base_composition, chol_frac)
            log_p = self.predict_permeability_for_composition(composition, membrane_type)
            diff = abs(log_p - target_log_p)

            if diff < best_diff:
                best_diff = diff
                best_composition = composition

        return best_composition, self.predict_permeability_for_composition(best_composition, membrane_type)

    def _adjust_cholesterol(self, base, chol_fraction):
        total = sum(base.values())
        chol_count = int(total * chol_fraction)
        remaining = total - chol_count

        result = {}
        other_total = sum(c for n, c in base.items() if self.library.get(n) and self.library.get(n).category != LipidCategory.STEROL)

        for name, count in base.items():
            lipid = self.library.get(name)
            if lipid and lipid.category == LipidCategory.STEROL:
                result[name] = chol_count
            elif other_total > 0:
                result[name] = int(remaining * count / other_total)

        return result


def quick_permeability(molecule, composition=None, membrane_type="BLM"):
    mt = MembraneType(membrane_type)

    if composition:
        predictor = PermeabilityPredictor(composition=composition)
    else:
        predictor = PermeabilityPredictor()

    return predictor.calculate(molecule, mt).log_p
