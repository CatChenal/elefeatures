"""
MCCE Feature Extractor
This module contains the core logic for extracting electrostatic features from several
MCCE output files (listed in SOURCE_FILES).
"""
from collections import OrderedDict
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


ACIDS = {"ASP", "GLU", "CTR"}
BASES = {"LYS", "ARG", "HIS", "NTR", "NTG"}

RESIDUE_ALIASES = {
    "NTR": ["NTR", "NTG"],
    "CYS": ["CYS", "CYD", "CYL", "CYX"],
    "HIS": ["HIS", "HIL"],
}

PK0_VALUES = {
    "ASP": 3.90,
    "GLU": 4.10,
    "LYS": 10.4,
    "ARG": 12.5,
    "HIS": 6.98,
    "NTR": 8.00, 
    "NTG": 8.00, 
    "CTR": 3.75, 
}


# Representative side-chain anchor atoms
# If multiple atoms are listed, use their geometric midpoint.
#
# Design philosophy for preliminary neighbor graphs:
# - charged/polar groups -> functional atom(s)
# - aromatic groups -> ring center approximated by distal atoms
# - hydrophobics -> terminal carbon(s)
# - glycines -> CA fallback

ANCHOR_ATOM_NAMES = {
    # Charged acidic
    "ASP": ["OD1", "OD2"],
    "GLU": ["OE1", "OE2"],

    # Charged basic
    "LYS": ["NZ"],
    "ARG": ["NH1", "NH2"],   # better charge center than CZ
    "HIS": ["ND1", "NE2"],

    # Polar uncharged
    "SER": ["OG"],
    "THR": ["OG1"],
    "ASN": ["OD1", "ND2"],
    "GLN": ["OE1", "NE2"],
    "CYS": ["SG"],
    "CYD": ["SG"],           # deprotonated cysteine if used
    "CYL": ["SG"],           # ligand/disulfide naming variants
    "CYX": ["SG"],           # disulfide cystine
    "TYR": ["OH"],

    # Hydrophobic / aliphatic
    "ALA": ["CB"],
    "VAL": ["CG1", "CG2"],
    "LEU": ["CD1", "CD2"],
    "ILE": ["CD1"],
    "MET": ["SD", "CE"],     # midpoint better than CE alone
    "PRO": ["CG"],

    # Aromatic
    "PHE": ["CZ"],
    "TRP": ["CZ2", "CH2"],

    # Small / special
    "GLY": ["CA"],

    # Backbone-like / termini
    "NTR": ["N"],
    "NTG": ["N"],

    # Carboxyl terminus
    "CTR": ["O", "OXT"],     # midpoint of terminal oxygens
}


SOURCE_FILES = [
    "step1_out.pdb",
    "acc.res",
    "sum_crg.out",
    "pK.out",
]

CHARGE_PH_VALUES = (6.0, 7.0, 8.0)
CUSTOM_FEATURES_FILE = "mcce-features.txt"

class Atom:
    """A class representing an atom in the protein structure."""
    def __init__(self):
        self.name = ""                      # Atom name, stripped of the spaces used for alignment in PDB files, e.g., "CA", "OD1"
        self.xyz = np.zeros(3)              # 3D coordinates of the atom, placeholder value

class Residue:
    """A class representing a residue in the protein structure."""
    def __init__(self):
        self.residue_id = ""                # Unique identifier for the residue, e.g., "ASP_A_25"
        self.name = ""                      # Residue name, e.g., "ASP", "GLU", "LYS"
        self.charge = 0.0                   # Placeholder value, should be set based on MCCE output
        self.charges_by_ph = {}             # Charge values keyed by pH label, e.g., "7.0"
        self.sasa = 0.0                     # Solvent Accessible Surface Area, placeholder value
        self.sasa_fraction = 0.0            # Fraction of the residue's surface area that is solvent-accessible, placeholder value
        self.pka0 = 0.0                     # Intrinsic pKa value, placeholder value
        self.pka = 0.0                      # pKa value in the protein environment, placeholder value
        self.is_acidic = False              # Whether the residue is acidic, placeholder value
        self.is_basic = False               # Whether the residue is basic, placeholder value
        self.atoms = []                     # List of Atom objects representing the atoms in the residue
        self.anchor_point = np.zeros(3)     # 3D coordinates of the anchor point for the residue, placeholder value


def _identify_patches(residues: List[Residue]) -> Tuple[List[str], List[str]]:
    """
    Identify electrostatic charge patches by grouping spatially connected
    same-sign charged residues.

    Input:
        residues: list of Residue objects with:
            - residue.charge
            - residue.anchor_point

    Returns:
        A 2-tuple where each item is a list of residues defining a patch, e.g:
        (positive_patches, negative_patches)
    """
    logger = logging.getLogger(__name__)

    neighbor_distance_threshold = 10.0
    net_charge_threshold = 0.25
    positive_patches = []
    negative_patches = []

    def distance(r1, r2):
        p1 = np.asarray(r1.anchor_point, dtype=float)
        p2 = np.asarray(r2.anchor_point, dtype=float)
        return np.linalg.norm(p1 - p2)

    def build_patches(charged_residues):
        patches = []
        visited = set()

        for residue in charged_residues:
            if id(residue) in visited:
                continue

            patch = []
            stack = [residue]
            visited.add(id(residue))

            while stack:
                current = stack.pop()
                patch.append(current)

                for neighbor in charged_residues:
                    if id(neighbor) in visited:
                        continue

                    if distance(current, neighbor) <= neighbor_distance_threshold:
                        visited.add(id(neighbor))
                        stack.append(neighbor)

            patches.append(patch)

        return patches

    positive_residues = []
    negative_residues = []

    for residue in residues:
        charge = getattr(residue, "charge", None)
        anchor = getattr(residue, "anchor_point", None)

        if charge is None or anchor is None:
            continue

        if charge > net_charge_threshold:
            positive_residues.append(residue)

        elif charge < -net_charge_threshold:
            negative_residues.append(residue)

    if not positive_residues and not negative_residues:
        logger.warning("No significantly charged residues found for patch detection")

    positive_patches = build_patches(positive_residues)
    negative_patches = build_patches(negative_residues)

    return positive_patches, negative_patches


def _ph_label(ph: float) -> str:
    return f"{ph:.1f}"


def test_pkout(pka_file) -> bool:
    """pK.out can contain bytecodes: unreadable in normal read mode.
    This can append when the protein has no ionizable residues.
    """
    try:
        with open(pka_file) as fh:
            _ = fh.readline()
        return True
    except UnicodeDecodeError:
        return False


class MCCEFeatureExtractor:
    """
    A feature extractor for MCCE output files.

    The method extract_all_features() returns a list of floats ordered by feature names
    """
    def __init__(self):
        self.mcce_folder = None
        # REGISTER ALL FEATURE NAMES HERE
        self.standard_feature_names = [
            # Global composition and pH-7 net-charge summary.
            "net_charge",
            "isoelectric_point",
            "acid_fraction_all_residues",
            "glu_fraction_all_residues",
            "asp_fraction_all_residues",
            "base_fraction_all_residues",
            "lys_fraction_all_residues",
            "arg_fraction_all_residues",
            "acid_to_base_ratio",

            # pKa perturbation prevalence across acidic and basic residue classes.
            "acid_big_pka_shift_fraction_all_residues",
            "glu_big_pka_shift_fraction_all_residues",
            "asp_big_pka_shift_fraction_all_residues",
            "base_big_pka_shift_fraction_all_residues",
            "lys_big_pka_shift_fraction_all_residues",
            "arg_big_pka_shift_fraction_all_residues",
            "acid_big_pka_shift_fraction_acids_only",
            "glu_big_pka_shift_fraction_acids_only",
            "asp_big_pka_shift_fraction_acids_only",
            "base_big_pka_shift_fraction_bases_only",
            "lys_big_pka_shift_fraction_bases_only",
            "arg_big_pka_shift_fraction_bases_only",
            "mean_abs_pka_shift",
            "max_abs_pka_shift",

            # Solvent-exposed charge balance and charge density at target pH values.
            "surface_net_charge_6.0",
            "surface_acid_to_base_ratio_6.0",
            "surface_positive_charge_density_6.0",
            "surface_negative_charge_density_6.0",
            "surface_net_charge_7.0",
            "surface_acid_to_base_ratio_7.0",
            "surface_positive_charge_density_7.0",
            "surface_negative_charge_density_7.0",
            "surface_net_charge_8.0",
            "surface_acid_to_base_ratio_8.0",
            "surface_positive_charge_density_8.0",
            "surface_negative_charge_density_8.0",            

            # Buried charge balance and charge density at target pH values.
            "buried_net_charge_6.0",
            "buried_acid_to_base_ratio_6.0",
            "buried_positive_charge_density_6.0",
            "buried_negative_charge_density_6.0",
            "buried_net_charge_7.0",
            "buried_acid_to_base_ratio_7.0",
            "buried_positive_charge_density_7.0",
            "buried_negative_charge_density_7.0",
            "buried_net_charge_8.0",
            "buried_acid_to_base_ratio_8.0",
            "buried_positive_charge_density_8.0",
            "buried_negative_charge_density_8.0",

            # Largest same-sign charge patch size, charge, and density at target pH values.
            "largest_positive_patch_area_6.0",
            "largest_negative_patch_area_6.0",
            "largest_positive_patch_charge_6.0",
            "largest_negative_patch_charge_6.0",
            "largest_positive_patch_density_6.0",
            "largest_negative_patch_density_6.0",
            "largest_positive_patch_area_7.0",
            "largest_negative_patch_area_7.0",
            "largest_positive_patch_charge_7.0",
            "largest_negative_patch_charge_7.0",
            "largest_positive_patch_density_7.0",
            "largest_negative_patch_density_7.0",
            "largest_positive_patch_area_8.0",
            "largest_negative_patch_area_8.0",
            "largest_positive_patch_charge_8.0",
            "largest_negative_patch_charge_8.0",
            "largest_positive_patch_density_8.0",
            "largest_negative_patch_density_8.0",

            # Charge asymmetry, spatial moment, and positive-negative separation at target pH values.
            "all_charge_spatial_moment_magnitude_6.0",
            "surface_charge_spatial_moment_magnitude_6.0",
            "all_charge_spatial_moment_normalized_6.0",
            "surface_charge_spatial_moment_normalized_6.0",
            "charge_separation_magnitude_6.0",
            "all_charge_spatial_moment_magnitude_7.0",
            "surface_charge_spatial_moment_magnitude_7.0",
            "all_charge_spatial_moment_normalized_7.0",
            "surface_charge_spatial_moment_normalized_7.0",
            "charge_separation_magnitude_7.0",
            "all_charge_spatial_moment_magnitude_8.0",
            "surface_charge_spatial_moment_magnitude_8.0",
            "all_charge_spatial_moment_normalized_8.0",
            "surface_charge_spatial_moment_normalized_8.0",
            "charge_separation_magnitude_8.0",

            # Nearest charged-residue anchor distances at pH 7.0.
            "nearest_opposite_charge_distance",
            "nearest_positive_charge_distance",
            "nearest_negative_charge_distance",

        ]
        self.feature_names = list(self.standard_feature_names)

    def missing_sources(self) -> bool:
        if self.mcce_folder is None:
            return False
        bad = False
        fld = Path(self.mcce_folder)
        for source in SOURCE_FILES:
            if not fld.joinpath(source).exists():
                logging.debug(f"Missing: {source}")
                bad = True
                break
        return bad

    def _apply_charges_for_ph(self, ph_label: str):
        """Set residue.charge values from cached charges for one pH."""
        for residue in self.residues:
            if not hasattr(residue, "_fallback_charge_7_0"):
                residue._fallback_charge_7_0 = residue.charge
            charges_by_ph = getattr(residue, "charges_by_ph", {})
            if ph_label in charges_by_ph:
                residue.charge = charges_by_ph[ph_label]
            elif ph_label == "7.0":
                residue.charge = residue._fallback_charge_7_0
            else:
                residue.charge = 0.0


    def extract_composition_features(self) -> Dict[str, float]:
        """
        Extract composition-based electrostatic features.

        Features:
            - net_charge
            - isoelectric_point
            - acid_fraction_all_residues
            - glu_fraction_all_residues
            - asp_fraction_all_residues
            - base_fraction_all_residues
            - lys_fraction_all_residues
            - arg_fraction_all_residues
            - acid_to_base_ratio

        Returns:
            Dictionary mapping feature names to float values.
        """

        logger = logging.getLogger(__name__)

        features = {}

        if not self.residues:
            logger.warning("No residues loaded; returning zero composition features")
            return {
                "net_charge": 0.0,
                "isoelectric_point": 0.0,
                "acid_fraction_all_residues": 0.0,
                "glu_fraction_all_residues": 0.0,
                "asp_fraction_all_residues": 0.0,
                "base_fraction_all_residues": 0.0,
                "lys_fraction_all_residues": 0.0,
                "arg_fraction_all_residues": 0.0,
                "acid_to_base_ratio": 0.0,
            }

        total_residues = len(self.residues)
        acidic_residues = [
            residue for residue in self.residues
            if residue.is_acidic
        ]
        basic_residues = [
            residue for residue in self.residues
            if residue.is_basic
        ]
        acidic_count = len(acidic_residues)
        basic_count = len(basic_residues)
        glu_count = sum(
            1 for residue in self.residues
            if residue.name == "GLU"
        )
        asp_count = sum(
            1 for residue in self.residues
            if residue.name == "ASP"
        )
        lys_count = sum(
            1 for residue in self.residues
            if residue.name == "LYS"
        )
        arg_count = sum(
            1 for residue in self.residues
            if residue.name == "ARG"
        )

        # ------------------------------------------------------------
        # Net charge and isoelectric point from sum_crg.out
        # ------------------------------------------------------------
        net_charge = 0.0
        isoelectric_point = 0.0
        ph_values = None
        net_charge_values = None
        sum_charge_file = f"{self.mcce_folder}/sum_crg.out"

        if not Path(sum_charge_file).exists():
            logger.critical("Not found: sum_crg.out in %s", self.mcce_folder)
            # FIX: should return None
            return {"net_charge": 0.0,
                    "isoelectric_point": 0.0,
                    "acid_fraction_all_residues": 0.0,
                    "glu_fraction_all_residues": 0.0,
                    "asp_fraction_all_residues": 0.0,
                    "base_fraction_all_residues": 0.0,
                    "lys_fraction_all_residues": 0.0,
                    "arg_fraction_all_residues": 0.0,
                    "acid_to_base_ratio": 0.0
                    }

        with open(sum_charge_file) as f:
            for line in f:
                fields = line.split()

                if not fields:
                    continue

                if fields[0] == "ph":
                    ph_values = [float(x) for x in fields[1:]]

                elif fields[0] == "Net_Charge":
                    net_charge_values = [float(x) for x in fields[1:]]

        if ph_values is None or net_charge_values is None:
            logger.warning(
                "Could not find pH header or Net_Charge line in %s",
                sum_charge_file,
            )
        else:
            if len(ph_values) != len(net_charge_values):
                logger.warning(
                    "pH column count does not match Net_Charge column count in %s",
                    sum_charge_file,
                )
            elif len(net_charge_values) == 1 and net_charge_values[0] == 0.0:
                # case for protein with non-ionizable res, e.g. Collagen:
                # sum_crg.out has no data rows and Net crg is 0.0:
                # no need for calculation
                isoelectric_point = ph_values[0]
            else:
                # Net charge at pH 7.0
                try:
                    ph7_index = ph_values.index(7.0)
                    net_charge = net_charge_values[ph7_index]
                except ValueError:
                    logger.warning("Could not find pH 7.0 column in %s", sum_charge_file)

                # Estimate pI by linear interpolation where Net_Charge crosses zero
                for i in range(len(ph_values) - 1):
                    ph1, ph2 = ph_values[i], ph_values[i + 1]
                    q1, q2 = net_charge_values[i], net_charge_values[i + 1]

                    if q1 == 0.0:
                        isoelectric_point = ph1
                        break

                    if q1 * q2 < 0:
                        # Linear interpolation:
                        # q = q1 + slope * (pH - ph1)
                        # solve q = 0
                        isoelectric_point = ph1 + (0.0 - q1) * (ph2 - ph1) / (q2 - q1)
                        break
                else:
                    logger.warning(
                        "Net_Charge does not cross zero in sampled pH range %.1f-%.1f",
                        ph_values[0],
                        ph_values[-1],
                    )

                    # Fallback: use pH with smallest absolute net charge
                    min_abs_index = min(
                        range(len(net_charge_values)),
                        key=lambda i: abs(net_charge_values[i]),
                    )
                    isoelectric_point = ph_values[min_abs_index]

        # ------------------------------------------------------------
        # Composition fractions
        # ------------------------------------------------------------
        acid_fraction = acidic_count / total_residues
        glu_fraction = glu_count / total_residues
        asp_fraction = asp_count / total_residues
        base_fraction = basic_count / total_residues
        lys_fraction = lys_count / total_residues
        arg_fraction = arg_count / total_residues

        # ------------------------------------------------------------
        # Acid/base ratio
        # ------------------------------------------------------------
        if basic_count > 0:
            acid_to_base_ratio = acidic_count / basic_count
        elif acidic_count == 0 and basic_count == 0:
            acid_to_base_ratio = 1.0
        else:
            acid_to_base_ratio = 999.0   # Arbitrary large number to indicate all acids and no bases

        # ------------------------------------------------------------
        # Store features
        # ------------------------------------------------------------
        features["net_charge"] = float(net_charge)
        features["isoelectric_point"] = float(isoelectric_point)
        features["acid_fraction_all_residues"] = float(acid_fraction)
        features["glu_fraction_all_residues"] = float(glu_fraction)
        features["asp_fraction_all_residues"] = float(asp_fraction)
        features["base_fraction_all_residues"] = float(base_fraction)
        features["lys_fraction_all_residues"] = float(lys_fraction)
        features["arg_fraction_all_residues"] = float(arg_fraction)
        features["acid_to_base_ratio"] = float(acid_to_base_ratio)

        logger.debug(
            (
                "Composition features: "
                "residues=%d, acidic=%d, glu=%d, asp=%d, basic=%d, lys=%d, arg=%d, "
                "net_charge=%.3f, pI=%.3f, "
                "acid_fraction=%.3f, glu_fraction=%.3f, asp_fraction=%.3f, "
                "base_fraction=%.3f, lys_fraction=%.3f, arg_fraction=%.3f, "
                "acid_to_base_ratio=%.3f"
            ),
            total_residues,
            acidic_count,
            glu_count,
            asp_count,
            basic_count,
            lys_count,
            arg_count,
            net_charge,
            isoelectric_point,
            acid_fraction,
            glu_fraction,
            asp_fraction,
            base_fraction,
            lys_fraction,
            arg_fraction,
            acid_to_base_ratio,
        )

        return features


    def extract_pka_perturbation_features(self) -> Dict[str, float]:
        """
        Extract pKa perturbation features based on the difference between
        intrinsic pKa and pKa in the protein environment.

        Features:
            - acid_big_pka_shift_fraction_all_residues
            - glu_big_pka_shift_fraction_all_residues
            - asp_big_pka_shift_fraction_all_residues
            - base_big_pka_shift_fraction_all_residues
            - lys_big_pka_shift_fraction_all_residues
            - arg_big_pka_shift_fraction_all_residues
            - acid_big_pka_shift_fraction_acids_only
            - glu_big_pka_shift_fraction_acids_only
            - asp_big_pka_shift_fraction_acids_only
            - base_big_pka_shift_fraction_bases_only
            - lys_big_pka_shift_fraction_bases_only
            - arg_big_pka_shift_fraction_bases_only
            - mean_abs_pka_shift
            - max_abs_pka_shift

        Returns:
            Dictionary mapping feature names to float values.
        """

        logger = logging.getLogger(__name__)
        big_shift_threshold = 1.0
        features = {}

        if not self.residues:
            logger.warning("No residues loaded; returning zero pKa perturbation features")

            return {
                "acid_big_pka_shift_fraction_all_residues": 0.0,
                "glu_big_pka_shift_fraction_all_residues": 0.0,
                "asp_big_pka_shift_fraction_all_residues": 0.0,
                "base_big_pka_shift_fraction_all_residues": 0.0,
                "lys_big_pka_shift_fraction_all_residues": 0.0,
                "arg_big_pka_shift_fraction_all_residues": 0.0,
                "acid_big_pka_shift_fraction_acids_only": 0.0,
                "glu_big_pka_shift_fraction_acids_only": 0.0,
                "asp_big_pka_shift_fraction_acids_only": 0.0,
                "base_big_pka_shift_fraction_bases_only": 0.0,
                "lys_big_pka_shift_fraction_bases_only": 0.0,
                "arg_big_pka_shift_fraction_bases_only": 0.0,
                "mean_abs_pka_shift": 0.0,
                "max_abs_pka_shift": 0.0,
            }

        total_residues = len(self.residues)

        acidic_residues = [
            residue for residue in self.residues
            if residue.is_acidic
        ]

        basic_residues = [
            residue for residue in self.residues
            if residue.is_basic
        ]

        titratable_residues = acidic_residues + basic_residues

        pka_shifts = []
        acid_big_shift_count = 0
        glu_big_shift_count = 0
        asp_big_shift_count = 0
        base_big_shift_count = 0
        lys_big_shift_count = 0
        arg_big_shift_count = 0

        for residue in titratable_residues:
            pka_shift = residue.pka - residue.pka0
            abs_pka_shift = abs(pka_shift)

            pka_shifts.append(abs_pka_shift)

            if abs_pka_shift >= big_shift_threshold:
                logger.debug(f"Large pKa shift found for residue {residue.residue_id}: abs_pka_shift {abs_pka_shift:.3f} = |pKa {residue.pka:.3f} - pKa0 {residue.pka0:.3f}|")
                if residue.is_acidic:
                    acid_big_shift_count += 1
                    if residue.name == "GLU":
                        glu_big_shift_count += 1
                    elif residue.name == "ASP":
                        asp_big_shift_count += 1
                elif residue.is_basic:
                    base_big_shift_count += 1
                    if residue.name == "LYS":
                        lys_big_shift_count += 1
                    elif residue.name == "ARG":
                        arg_big_shift_count += 1

        acid_count = len(acidic_residues)
        base_count = len(basic_residues)

        features["acid_big_pka_shift_fraction_all_residues"] = acid_big_shift_count/total_residues
        features["glu_big_pka_shift_fraction_all_residues"] = glu_big_shift_count/total_residues
        features["asp_big_pka_shift_fraction_all_residues"] = asp_big_shift_count/total_residues
        features["base_big_pka_shift_fraction_all_residues"] = base_big_shift_count/total_residues
        features["lys_big_pka_shift_fraction_all_residues"] = lys_big_shift_count/total_residues
        features["arg_big_pka_shift_fraction_all_residues"] = arg_big_shift_count/total_residues
        features["acid_big_pka_shift_fraction_acids_only"] = (
            acid_big_shift_count/acid_count
            if acid_count > 0
            else 0.0
        )
        features["glu_big_pka_shift_fraction_acids_only"] = (
            glu_big_shift_count/acid_count
            if acid_count > 0
            else 0.0
        )
        features["asp_big_pka_shift_fraction_acids_only"] = (
            asp_big_shift_count/acid_count
            if acid_count > 0
            else 0.0
        )
        features["base_big_pka_shift_fraction_bases_only"] = (
            base_big_shift_count / base_count
            if base_count > 0
            else 0.0
        )
        features["lys_big_pka_shift_fraction_bases_only"] = (
            lys_big_shift_count / base_count
            if base_count > 0
            else 0.0
        )
        features["arg_big_pka_shift_fraction_bases_only"] = (
            arg_big_shift_count / base_count
            if base_count > 0
            else 0.0
        )
        features["mean_abs_pka_shift"] = (
            float(np.mean(pka_shifts))
            if pka_shifts
            else 0.0
        )
        features["max_abs_pka_shift"] = (
            float(np.max(pka_shifts))
            if pka_shifts
            else 0.0
        )
        logger.debug(
            (
                "pKa perturbation features: "
                "residues=%d, acids=%d, bases=%d, "
                "acid_big_shift=%d, glu_big_shift=%d, asp_big_shift=%d, "
                "base_big_shift=%d, lys_big_shift=%d, arg_big_shift=%d, "
                "mean_abs_shift=%.3f, max_abs_shift=%.3f"
            ),
            total_residues,
            acid_count,
            base_count,
            acid_big_shift_count,
            glu_big_shift_count,
            asp_big_shift_count,
            base_big_shift_count,
            lys_big_shift_count,
            arg_big_shift_count,
            features["mean_abs_pka_shift"],
            features["max_abs_pka_shift"],
        )

        return features

    
    def extract_surface_charge_features(self) -> Dict[str, float]:
        """
        Extract surface charge features based on the charges of residues and their solvent accessibility.

        Features:
        - surface_net_charge_<pH>
        - surface_acid_to_base_ratio_<pH>
        - surface_positive_charge_density_<pH>
        - surface_negative_charge_density_<pH>

        Data source:
        - Residue exposed surface area in residue.sasa
        - Residue exposed surface area fraction in residue.sasa_fraction
        - Residue charge in residue.charge
        - positive and negative charge densities are computed separately as
          base_charge / solvent-accessible surface area and
          acid_charge / solvent-accessible surface area, using a small
          pseudocount to avoid division by zero

        Returns:
        Dictionary mapping feature names to float values.
        """
        logger = logging.getLogger(__name__)
        surface_sasa_percentage_threshold = 0.1
        sasa_pseudocount = 1e-6

        def acid_to_base_ratio(acid_charge: float, base_charge: float) -> float:
            if base_charge > 0:
                return acid_charge / base_charge
            if acid_charge == 0 and base_charge == 0:
                return 1.0
            return 999.0

        def zero_features() -> Dict[str, float]:
            features = {}
            for ph in CHARGE_PH_VALUES:
                ph_label = _ph_label(ph)
                features[f"surface_net_charge_{ph_label}"] = 0.0
                features[f"surface_acid_to_base_ratio_{ph_label}"] = 0.0
                features[f"surface_positive_charge_density_{ph_label}"] = 0.0
                features[f"surface_negative_charge_density_{ph_label}"] = 0.0
            return features

        if not self.residues:
            logger.warning("No residues loaded; returning zero surface charge features")
            return zero_features()

        surface_residues = [
            residue for residue in self.residues
            if residue.sasa_fraction is not None
            and residue.sasa_fraction > surface_sasa_percentage_threshold
        ]

        if not surface_residues:
            logger.warning("No surface residues found; returning zero surface charge features")
            return zero_features()

        surface_total_sasa = sum(
            getattr(residue, "sasa", 0.0) or 0.0
            for residue in surface_residues
        )

        features = {}
        for ph in CHARGE_PH_VALUES:
            ph_label = _ph_label(ph)
            self._apply_charges_for_ph(ph_label)

            surface_net_charge = sum(getattr(residue, "charge", 0.0) or 0.0 for residue in surface_residues)
            surface_acid_charge = sum(
                abs(getattr(residue, "charge", 0.0) or 0.0)
                for residue in surface_residues
                if (getattr(residue, "charge", 0.0) or 0.0) < 0
            )
            surface_base_charge = sum(
                getattr(residue, "charge", 0.0) or 0.0
                for residue in surface_residues
                if (getattr(residue, "charge", 0.0) or 0.0) > 0
            )
            surface_acid_to_base_ratio = acid_to_base_ratio(surface_acid_charge, surface_base_charge)
            surface_positive_charge_density = surface_base_charge/(surface_total_sasa + sasa_pseudocount)
            surface_negative_charge_density = surface_acid_charge/(surface_total_sasa + sasa_pseudocount)

            features[f"surface_net_charge_{ph_label}"] = surface_net_charge
            features[f"surface_acid_to_base_ratio_{ph_label}"] = surface_acid_to_base_ratio
            features[f"surface_positive_charge_density_{ph_label}"] = surface_positive_charge_density
            features[f"surface_negative_charge_density_{ph_label}"] = surface_negative_charge_density

        self._apply_charges_for_ph("7.0")
        return features

    def extract_buried_charge_features(self) -> Dict[str, float]:
        """
        Extract buried charge features based on charges of residues below the surface cutoff.

        Features:
        - buried_net_charge_<pH>
        - buried_acid_to_base_ratio_<pH>
        - buried_positive_charge_density_<pH>
        - buried_negative_charge_density_<pH>

        Data source:
        - Residue exposed surface area in residue.sasa
        - Residue exposed surface area fraction in residue.sasa_fraction
        - Residue charge in residue.charge

        Returns:
        Dictionary mapping feature names to float values.
        """
        logger = logging.getLogger(__name__)
        surface_sasa_percentage_threshold = 0.1
        sasa_pseudocount = 1e-6

        def acid_to_base_ratio(acid_charge: float, base_charge: float) -> float:
            if base_charge > 0:
                return acid_charge / base_charge
            if acid_charge == 0 and base_charge == 0:
                return 1.0
            return 999.0

        def zero_features() -> Dict[str, float]:
            features = {}
            for ph in CHARGE_PH_VALUES:
                ph_label = _ph_label(ph)
                features[f"buried_net_charge_{ph_label}"] = 0.0
                features[f"buried_acid_to_base_ratio_{ph_label}"] = 0.0
                features[f"buried_positive_charge_density_{ph_label}"] = 0.0
                features[f"buried_negative_charge_density_{ph_label}"] = 0.0
            return features

        if not self.residues:
            logger.warning("No residues loaded; returning zero buried charge features")
            return zero_features()

        buried_residues = [
            residue for residue in self.residues
            if residue.sasa_fraction is not None
            and residue.sasa_fraction <= surface_sasa_percentage_threshold
        ]

        if not buried_residues:
            logger.warning("No buried residues found; returning zero buried charge features")
            return zero_features()

        buried_total_sasa = sum(
            getattr(residue, "sasa", 0.0) or 0.0
            for residue in buried_residues
        )

        features = {}
        for ph in CHARGE_PH_VALUES:
            ph_label = _ph_label(ph)
            self._apply_charges_for_ph(ph_label)

            buried_net_charge = sum(getattr(residue, "charge", 0.0) or 0.0 for residue in buried_residues)
            buried_acid_charge = sum(
                abs(getattr(residue, "charge", 0.0) or 0.0)
                for residue in buried_residues
                if (getattr(residue, "charge", 0.0) or 0.0) < 0
            )
            buried_base_charge = sum(
                getattr(residue, "charge", 0.0) or 0.0
                for residue in buried_residues
                if (getattr(residue, "charge", 0.0) or 0.0) > 0
            )
            buried_acid_to_base_ratio = acid_to_base_ratio(buried_acid_charge, buried_base_charge)
            buried_positive_charge_density = buried_base_charge/(buried_total_sasa + sasa_pseudocount)
            buried_negative_charge_density = buried_acid_charge/(buried_total_sasa + sasa_pseudocount)

            features[f"buried_net_charge_{ph_label}"] = buried_net_charge
            features[f"buried_acid_to_base_ratio_{ph_label}"] = buried_acid_to_base_ratio
            features[f"buried_positive_charge_density_{ph_label}"] = buried_positive_charge_density
            features[f"buried_negative_charge_density_{ph_label}"] = buried_negative_charge_density

        self._apply_charges_for_ph("7.0")
        return features

    def extract_dipole_features(self) -> Dict[str, float]:
        """
        Extract charge-separation-based dipole features.

        Method:
            1. Compute positive and negative charge centroids
            2. Compute separation vector between centroids
            3. Scale by effective charge

        Feature:
            charge_separation_magnitude_<pH>

        Notes:
            - Uses residue.anchor_point as coordinates
            - Uses residue.charge as charge
            - Neutral or one-sided systems return 0
            - More biologically interpretable than strict physics dipole
            for proteins with net charge
        """
        logger = logging.getLogger(__name__)

        if not self.residues:
            logger.warning("No residues loaded; returning zero dipole features")
            return {
                f"charge_separation_magnitude_{_ph_label(ph)}": 0.0
                for ph in CHARGE_PH_VALUES
            }

        def calculate_charge_separation():
            positive_coords = []
            positive_charges = []
            negative_coords = []
            negative_charges = []

            for residue in self.residues:
                q = getattr(residue, "charge", 0.0)
                if q == 0:
                    continue

                anchor = getattr(residue, "anchor_point", None)
                if anchor is None:
                    continue

                r = np.asarray(anchor, dtype=float)
                if r.shape != (3,):
                    logger.warning(
                        "Invalid anchor point shape for residue %s: %s",
                        getattr(residue, "resid", "?"),
                        r.shape,
                    )
                    continue

                if q > 0:
                    positive_coords.append(r)
                    positive_charges.append(q)
                else:
                    negative_coords.append(r)
                    negative_charges.append(abs(q))

            if not positive_coords or not negative_coords:
                return 0.0

            positive_coords = np.asarray(positive_coords)
            positive_charges = np.asarray(positive_charges)
            negative_coords = np.asarray(negative_coords)
            negative_charges = np.asarray(negative_charges)

            r_plus = np.average(positive_coords, axis=0, weights=positive_charges)
            r_minus = np.average(negative_coords, axis=0, weights=negative_charges)
            separation_distance = np.linalg.norm(r_plus - r_minus)
            q_eff = min(positive_charges.sum(), negative_charges.sum())
            return float(q_eff * separation_distance)

        features = {}
        for ph in CHARGE_PH_VALUES:
            ph_label = _ph_label(ph)
            self._apply_charges_for_ph(ph_label)
            features[f"charge_separation_magnitude_{ph_label}"] = calculate_charge_separation()

        self._apply_charges_for_ph("7.0")
        return features

    def extract_nearest_charge_distance_features(self) -> Dict[str, float]:
        """
        Extract nearest-neighbor distances among charged residues at pH 7.0.

        Features:
        - nearest_opposite_charge_distance
        - nearest_positive_charge_distance
        - nearest_negative_charge_distance

        Charged residues are selected with the same absolute charge threshold
        used for patch localization. Distances use residue anchor points.
        """
        logger = logging.getLogger(__name__)
        net_charge_threshold = 0.25

        features = {
            "nearest_opposite_charge_distance": 0.0,
            "nearest_positive_charge_distance": 0.0,
            "nearest_negative_charge_distance": 0.0,
        }

        if not self.residues:
            logger.warning("No residues loaded; returning zero nearest charge distance features")
            return features

        self._apply_charges_for_ph("7.0")

        positive_residues = []
        negative_residues = []

        for residue in self.residues:
            charge = getattr(residue, "charge", None)
            anchor = getattr(residue, "anchor_point", None)

            if charge is None or anchor is None:
                continue

            anchor = np.asarray(anchor, dtype=float)
            if anchor.shape != (3,):
                logger.warning(
                    "Invalid anchor point shape for residue %s: %s",
                    getattr(residue, "residue_id", "?"),
                    anchor.shape,
                )
                continue

            if charge > net_charge_threshold:
                positive_residues.append((residue, anchor))
            elif charge < -net_charge_threshold:
                negative_residues.append((residue, anchor))

        def nearest_distance(group_a, group_b=None):
            if group_b is None:
                if len(group_a) < 2:
                    return 0.0

                best = None
                for i in range(len(group_a) - 1):
                    for j in range(i + 1, len(group_a)):
                        distance = np.linalg.norm(group_a[i][1] - group_a[j][1])
                        if best is None or distance < best:
                            best = distance

                return float(best) if best is not None else 0.0

            if not group_a or not group_b:
                return 0.0

            best = None
            for _, anchor_a in group_a:
                for _, anchor_b in group_b:
                    distance = np.linalg.norm(anchor_a - anchor_b)
                    if best is None or distance < best:
                        best = distance

            return float(best) if best is not None else 0.0

        features["nearest_opposite_charge_distance"] = nearest_distance(
            positive_residues,
            negative_residues,
        )
        features["nearest_positive_charge_distance"] = nearest_distance(positive_residues)
        features["nearest_negative_charge_distance"] = nearest_distance(negative_residues)

        return features


    def extract_all_features(self, folder: str) -> List[float]:
        """
        Extract all electrostatic features from the MCCE output files in the specified folder.

        Returns:
            A list of floats representing the extracted features, ordered by feature names.
        """
        self.mcce_folder = folder

        logger = logging.getLogger(__name__)
        logger.debug(f"Extracting features from MCCE folder: {self.mcce_folder}")
        self.load_protein_structure()

        logger.debug(f"Initialize residue properties from MCCE output ...")
        self.initialize_residue_properties()
        features = {}
        features.update(self.extract_composition_features())
        features.update(self.extract_pka_perturbation_features())
        features.update(self.extract_surface_charge_features())
        features.update(self.extract_buried_charge_features())
        features.update(self.extract_patch_localization_features())
        features.update(self.extract_asymmetry_features())
        features.update(self.extract_dipole_features())
        features.update(self.extract_nearest_charge_distance_features())
        custom_features = self.extract_custom_features()
        self.feature_names = list(self.standard_feature_names) + list(custom_features)
        features.update(custom_features)

        # Guardrail 1: missing registered features
        missing = set(self.feature_names) - set(features)
        if missing:
            raise ValueError(f"Missing features: {sorted(missing)}")

        # Guardrail 2: unregistered extra features
        extra = set(features) - set(self.feature_names)
        if extra:
            raise ValueError(f"Unregistered features: {sorted(extra)}")

        # The list order is now guaranteed by feature_names
        return [
            float(f"{v:.3g}") if abs(v) < 1.0 else round(v, 3)
            for v in (features[name] for name in self.feature_names)
        ]

    def extract_custom_features(self) -> Dict[str, float]:
        """
        Read optional user-defined numeric features from mcce-features.txt.

        Each non-empty line must have the form:
            feature_name: numeric_value
        """
        custom_features = OrderedDict()
        custom_feature_file = Path(self.mcce_folder) / CUSTOM_FEATURES_FILE
        if not custom_feature_file.exists():
            return custom_features

        logger = logging.getLogger(__name__)
        logger.info("Reading custom features from %s", custom_feature_file)

        for line_number, line in enumerate(custom_feature_file.read_text().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            if ":" not in line:
                logger.warning(
                    "Skipping malformed custom feature line %s:%d: %r",
                    custom_feature_file,
                    line_number,
                    line,
                )
                continue

            name, value_text = (part.strip() for part in line.split(":", 1))
            if not name:
                logger.warning(
                    "Skipping custom feature with empty name in %s:%d",
                    custom_feature_file,
                    line_number,
                )
                continue
            if name in self.standard_feature_names:
                logger.warning(
                    "Skipping custom feature %r in %s:%d because it duplicates a standard feature",
                    name,
                    custom_feature_file,
                    line_number,
                )
                continue

            try:
                value = float(value_text)
            except ValueError:
                logger.warning(
                    "Skipping custom feature %r in %s:%d because value is not numeric: %r",
                    name,
                    custom_feature_file,
                    line_number,
                    value_text,
                )
                continue

            if name in custom_features:
                logger.warning(
                    "Duplicate custom feature %r in %s:%d; using the last value",
                    name,
                    custom_feature_file,
                    line_number,
                )
            custom_features[name] = value

        return custom_features
    
    def load_protein_structure(self):
        """Load the protein structure from MCCE output files."""
        step1_file = f"{self.mcce_folder}/step1_out.pdb"
        self.residues = self.parse_step1_pdb(step1_file)

    def parse_step1_pdb(self, pdb_file: str) -> List[Residue]:
        """Parse step1_out.pdb and extract residues, atoms, and anchor points."""

        logger = logging.getLogger(__name__)

        if not Path(pdb_file).exists():
            logger.critical("Not found: step1_out.pdb in %s", self.mcce_folder)
            return []

        logger.debug("Parsing step1_out.pdb: %s", pdb_file)

        def residue_base_id(mcce_res_id: str) -> str:
            # Example: A0001_001 -> A0001
            return mcce_res_id.split("_")[0]

        def is_hydrogen(atom_name: str) -> bool:
            # Handles H, HA, HZ1, 1H, 2HD, etc.
            name = atom_name.strip()
            return (
                name.startswith("H")
                or (len(name) > 1 and name[0].isdigit() and name[1] == "H")
            )

        def compute_anchor_point(residue: Residue) -> np.ndarray:
            atom_by_name = {atom.name: atom for atom in residue.atoms}
            anchor_names = ANCHOR_ATOM_NAMES.get(residue.name, [])

            xyz_list = [
                atom_by_name[name].xyz
                for name in anchor_names
                if name in atom_by_name
            ]
            # Preferred: midpoint of configured anchor atoms
            if xyz_list:
                return np.mean(xyz_list, axis=0)

            # Fallback 1: CA if available
            if "CA" in atom_by_name:
                logger.debug(
                    "Anchor fallback to CA for residue %s",
                    residue.residue_id,
                )
                return atom_by_name["CA"].xyz

            # Fallback 2: average of heavy atoms
            heavy_xyz = [
                atom.xyz for atom in residue.atoms
                if not is_hydrogen(atom.name)
            ]
            if heavy_xyz:
                logger.debug(
                    "Anchor fallback to heavy-atom centroid for residue %s",
                    residue.residue_id,
                )
                return np.mean(heavy_xyz, axis=0)

            # Fallback 3: average of all atoms
            if residue.atoms:
                logger.debug(
                    "Anchor fallback to all-atom centroid for residue %s",
                    residue.residue_id,
                )
                return np.mean(
                    [atom.xyz for atom in residue.atoms],
                    axis=0,
                )

            logger.warning(
                "Residue %s has no atoms; using origin as anchor point",
                residue.residue_id,
            )

            return np.zeros(3)

        residues_by_id = OrderedDict()
        total_lines = 0
        skipped_short = 0
        skipped_bad_xyz = 0
        total_atoms = 0

        with open(pdb_file, "r") as f:
            for line in f:
                total_lines += 1
                if not line.startswith(("ATOM", "HETATM")):
                    continue

                fields = line.split()
                if len(fields) < 8:
                    skipped_short += 1
                    logger.debug(
                        "Skipping malformed atom line: %s",
                        line.rstrip(),
                    )
                    continue

                atom_name = fields[2].strip()
                res_name = fields[3].strip()
                mcce_res_id = fields[4].strip()
                try:
                    xyz = np.array(
                        [
                            float(fields[5]),
                            float(fields[6]),
                            float(fields[7]),
                        ],
                        dtype=float,
                    )
                except ValueError:
                    skipped_bad_xyz += 1
                    logger.debug(
                        "Skipping atom with invalid coordinates: %s",
                        line.rstrip(),
                    )
                    continue

                base_id = residue_base_id(mcce_res_id)
                residue_id = f"{res_name}_{base_id}"
                if residue_id not in residues_by_id:
                    residue = Residue()
                    residue.residue_id = residue_id
                    residue.name = res_name
                    residue.is_acidic = res_name in ACIDS
                    residue.is_basic = res_name in BASES
                    residue.pka0 = PK0_VALUES.get(res_name, 0.0)
                    residue.atoms = []
                    residues_by_id[residue_id] = residue

                    logger.debug("Created residue %s", residue_id,)

                atom = Atom()
                atom.name = atom_name
                atom.xyz = xyz
                residues_by_id[residue_id].atoms.append(atom)
                total_atoms += 1

        residues = list(residues_by_id.values())

        logger.debug(
            "Parsed %d residues and %d atoms from %s",
            len(residues),
            total_atoms,
            pdb_file,
        )

        acidic_count = 0
        basic_count = 0
        for residue in residues:
            residue.anchor_point = compute_anchor_point(residue)
            if residue.is_acidic:
                acidic_count += 1

            if residue.is_basic:
                basic_count += 1

        logger.debug(
            (
                "lines=%d, "
                "residues=%d, atoms=%d, "
                "acidic=%d, basic=%d, "
                "skipped_short=%d, skipped_bad_xyz=%d"
            ),
            total_lines,
            len(residues),
            total_atoms,
            acidic_count,
            basic_count,
            skipped_short,
            skipped_bad_xyz,
        )

        return residues
    
    def initialize_residue_properties(self):
        """Initialize residue charge, SASA, SASA fraction,
        and pKa from MCCE output files.
        """
        logger = logging.getLogger(__name__)

        sum_charge_file = f"{self.mcce_folder}/sum_crg.out"
        sasa_file = f"{self.mcce_folder}/acc.res"
        pka_file = f"{self.mcce_folder}/pK.out"
        pka_file_ok = test_pkout(pka_file)

        logger.debug("Initializing residue properties from MCCE output files")
        logger.debug("Charge file: %s", sum_charge_file)
        logger.debug("SASA file:   %s", sasa_file)
        logger.debug("pKa file:    %s; file OK: %s", pka_file, pka_file_ok)

        def normalize_mcce_residue_id(raw_id: str) -> str:
            """
            Convert MCCE residue IDs to the Residue.residue_id format.

            Examples:
                LYS+A0001_ -> LYS_A0001
                GLU-A0007_ -> GLU_A0007
                NTR+A0001_ -> NTR_A0001
            """
            raw_id = raw_id.strip()
            raw_id = raw_id.rstrip("_")

            if "+" in raw_id:
                res_name, res_num = raw_id.split("+", 1)
            elif "-" in raw_id:
                res_name, res_num = raw_id.split("-", 1)
            else:
                return raw_id

            return f"{res_name}_{res_num}"

        def safe_float(value: str):
            """Return float(value), or None if it is not a plain number."""
            try:
                return float(value)
            except ValueError:
                return None

        def find_ph_indices(ph_values):
            """Return target pH column indexes for variants like 7, 7.0, and 7.00."""
            ph_indices = {}
            for index, value in enumerate(ph_values):
                ph = safe_float(value)
                for target_ph in CHARGE_PH_VALUES:
                    if ph == target_ph:
                        ph_indices[_ph_label(target_ph)] = index
            return ph_indices

        residue_by_id = {
            residue.residue_id: residue
            for residue in self.residues
        }

        charge_by_id = {}
        charges_by_id_by_ph = {
            _ph_label(ph): {}
            for ph in CHARGE_PH_VALUES
        }
        sasa_by_id = {}
        pka_by_id = {}

        # ------------------------------------------------------------
        # Parse sum_crg.out
        # ------------------------------------------------------------
        charge_lines = 0
        charge_values_loaded = 0
        skipped_charge_lines = 0
        ph_indices = {}

        with open(sum_charge_file, "r") as f:
            for line in f:
                fields = line.split()

                if not fields:
                    continue

                if fields[0] == "ph":
                    ph_values = fields[1:]

                    ph_indices = find_ph_indices(ph_values)
                    for ph_label in (_ph_label(ph) for ph in CHARGE_PH_VALUES):
                        if ph_label in ph_indices:
                            logger.debug(
                                "Found pH %s charge column at index %d",
                                ph_label,
                                ph_indices[ph_label],
                            )
                        else:
                            logger.warning(
                                "Could not find pH %s column in %s; charge values will not be loaded for that pH",
                                ph_label,
                                sum_charge_file,
                            )

                    continue

                if not ph_indices:
                    continue

                raw_res_id = fields[0]

                # Skip separator or malformed lines
                if raw_res_id.startswith("-"):
                    skipped_charge_lines += 1
                    continue

                residue_id = normalize_mcce_residue_id(raw_res_id)
                loaded_any_charge = False

                for ph_label, ph_index in ph_indices.items():
                    if len(fields) <= ph_index + 1:
                        continue

                    charge = safe_float(fields[ph_index + 1])
                    if charge is None:
                        continue

                    charges_by_id_by_ph[ph_label][residue_id] = charge
                    loaded_any_charge = True

                    if ph_label == "7.0":
                        charge_by_id[residue_id] = charge

                if not loaded_any_charge:
                    skipped_charge_lines += 1
                    continue

                charge_lines += 1
                charge_values_loaded += 1

        logger.debug(
            "Loaded target-pH charge values for %d residues from %s",
            charge_values_loaded,
            sum_charge_file,
        )

        # ------------------------------------------------------------
        # Parse acc.res
        # ------------------------------------------------------------
        sasa_lines = 0
        sasa_values_loaded = 0
        skipped_sasa_lines = 0

        with open(sasa_file, "r") as f:
            for line in f:
                fields = line.split()

                if len(fields) < 5:
                    skipped_sasa_lines += 1
                    continue

                if fields[0] != "RES":
                    skipped_sasa_lines += 1
                    continue

                res_name = fields[1]
                res_num = fields[2]
                residue_id = f"{res_name}_{res_num}"

                sasa = safe_float(fields[3])
                sasa_fraction = safe_float(fields[4])

                if sasa is None or sasa_fraction is None:
                    skipped_sasa_lines += 1
                    continue

                sasa_by_id[residue_id] = (sasa, sasa_fraction)

                sasa_lines += 1
                sasa_values_loaded += 1

        logger.debug(
            "Loaded SASA values for %d residues from %s",
            sasa_values_loaded,
            sasa_file,
        )

        # ------------------------------------------------------------
        # Parse pK.out
        # ------------------------------------------------------------
        pka_lines = 0
        pka_values_loaded = 0
        skipped_pka_lines = 0
        non_numeric_pka_count = 0

        if pka_file_ok:
            with open(pka_file, "r") as f:
                for line in f:
                    fields = line.split()

                    if not fields:
                        continue

                    # Skip header
                    if fields[0] == "pH":
                        continue

                    raw_res_id = fields[0]

                    # Skip separator or malformed lines
                    if raw_res_id.startswith("-"):
                        skipped_pka_lines += 1
                        continue

                    if len(fields) < 2:
                        skipped_pka_lines += 1
                        continue

                    residue_id = normalize_mcce_residue_id(raw_res_id)

                    # Normal case:
                    # LYS+A0001_  9.547  0.978  0.008
                    #
                    # Non-number cases may appear as:
                    # ASP-A0018_  < 0.0
                    # LYS+A0013_  > 14.0
                    pka = safe_float(fields[1])

                    if pka is None:
                        non_numeric_pka_count += 1

                        residue = residue_by_id.get(residue_id)

                        if residue is not None and residue.is_acidic:
                            pka = 0.0
                        elif residue is not None and residue.is_basic:
                            pka = 14.0
                        else:
                            skipped_pka_lines += 1
                            logger.debug(
                                "Skipping non-numeric pKa for non-acid/basic residue %s: %s",
                                residue_id,
                                line.rstrip(),
                            )
                            continue

                    pka_by_id[residue_id] = pka

                    pka_lines += 1
                    pka_values_loaded += 1

            logger.debug(
                "Loaded pKa values for %d residues from %s",
                pka_values_loaded,
                pka_file,
            )

        # ------------------------------------------------------------
        # Adding pKa0 values to residues based on their type
        # ------------------------------------------------------------
        for residue in self.residues:
            residue.pka0 = PK0_VALUES.get(residue.name, 0.0)

        # ------------------------------------------------------------
        # Apply properties to residues
        # ------------------------------------------------------------
        charge_assigned = 0
        sasa_assigned = 0
        pka_assigned = 0

        missing_charge = 0
        missing_sasa = 0
        missing_pka = 0

        for residue in self.residues:
            residue.charges_by_ph = {
                ph_label: charges_by_id.get(residue.residue_id, 0.0)
                for ph_label, charges_by_id in charges_by_id_by_ph.items()
            }

            if residue.residue_id in charge_by_id:
                residue.charge = charge_by_id[residue.residue_id]
                charge_assigned += 1
            else:
                missing_charge += 1

            # We need to be flexible in matching residue IDs for SASA, allowing for different naming conventions or aliases, 
            matched_id = None

            if residue.residue_id in sasa_by_id:
                matched_id = residue.residue_id
            else:
                res_name, res_num = residue.residue_id.split("_", 1)

                for _, aliases in RESIDUE_ALIASES.items():
                    if res_name in aliases:
                        for alias in aliases:
                            alias_id = f"{alias}_{res_num}"
                            if alias_id in sasa_by_id:
                                matched_id = alias_id
                                break

                    if matched_id is not None:
                        break

            if matched_id is not None:
                residue.sasa, residue.sasa_fraction = sasa_by_id[matched_id]
                sasa_assigned += 1
            else:
                missing_sasa += 1
                logger.debug(
                    "Missing SASA for residue %s; setting to 0",
                    residue.residue_id,
                )
                residue.sasa = 0.0
                residue.sasa_fraction = 0.0

            if residue.residue_id in pka_by_id:
                residue.pka = pka_by_id[residue.residue_id]
                pka_assigned += 1
            else:
                missing_pka += 1

        logger.debug(
            (
                "Residue property initialization summary: "
                "residues=%d, "
                "charge_assigned=%d, sasa_assigned=%d, pka_assigned=%d, "
                "missing_charge=%d, missing_sasa=%d, missing_pka=%d, "
                "skipped_charge_lines=%d, skipped_sasa_lines=%d, skipped_pka_lines=%d, "
                "non_numeric_pka=%d"
            ),
            len(self.residues),
            charge_assigned,
            sasa_assigned,
            pka_assigned,
            missing_charge,
            missing_sasa,
            missing_pka,
            skipped_charge_lines,
            skipped_sasa_lines,
            skipped_pka_lines,
            non_numeric_pka_count,
        )

    def extract_patch_localization_features(self) -> Dict[str, float]:
        """
        Extract features related to the localization of charged patches on the protein surface.

        Features:
        - largest_positive_patch_area_<pH>
        - largest_negative_patch_area_<pH>
        - largest_positive_patch_charge_<pH>
        - largest_negative_patch_charge_<pH>
        - largest_positive_patch_density_<pH>
        - largest_negative_patch_density_<pH>

        Notes:
        Patch density is defined as total charge in the patch divided by the solvent-accessible surface area of the patch.
        positive and negative patches contain residues with positive and negative charge.

        Returns:
            Dictionary mapping feature names to float values.
        """

        logger = logging.getLogger(__name__)
        sasa_pseudocount = 1e-6

        features = {}
        for ph in CHARGE_PH_VALUES:
            ph_label = _ph_label(ph)
            features[f"largest_positive_patch_area_{ph_label}"] = 0.0
            features[f"largest_negative_patch_area_{ph_label}"] = 0.0
            features[f"largest_positive_patch_charge_{ph_label}"] = 0.0
            features[f"largest_negative_patch_charge_{ph_label}"] = 0.0
            features[f"largest_positive_patch_density_{ph_label}"] = 0.0
            features[f"largest_negative_patch_density_{ph_label}"] = 0.0

        if not self.residues:
            logger.warning("No residues loaded; returning zero patch localization features")
            return features

        def summarize_largest_patch(patches, use_abs_charge: bool = True):
            """
            Return area, charge, and density for the largest patch by SASA area.
            """
            if not patches:
                return 0.0, 0.0, 0.0

            def patch_area(patch):
                return sum(getattr(residue, "sasa", 0.0) or 0.0 for residue in patch)

            largest_patch = max(patches, key=patch_area)

            area = patch_area(largest_patch)
            charge = sum(getattr(residue, "charge", 0.0) or 0.0 for residue in largest_patch)

            if use_abs_charge:
                charge_for_density = abs(charge)
            else:
                charge_for_density = charge

            density = charge_for_density / (area + sasa_pseudocount)

            return area, charge, density

        for ph in CHARGE_PH_VALUES:
            ph_label = _ph_label(ph)
            self._apply_charges_for_ph(ph_label)
            positive_patches, negative_patches = _identify_patches(self.residues)

            (
                largest_positive_patch_area,
                largest_positive_patch_charge,
                largest_positive_patch_density,
            ) = summarize_largest_patch(positive_patches)

            (
                largest_negative_patch_area,
                largest_negative_patch_charge,
                largest_negative_patch_density,
            ) = summarize_largest_patch(negative_patches)

            features.update(
                {
                    f"largest_positive_patch_area_{ph_label}": largest_positive_patch_area,
                    f"largest_negative_patch_area_{ph_label}": largest_negative_patch_area,
                    f"largest_positive_patch_charge_{ph_label}": largest_positive_patch_charge,
                    f"largest_negative_patch_charge_{ph_label}": largest_negative_patch_charge,
                    f"largest_positive_patch_density_{ph_label}": largest_positive_patch_density,
                    f"largest_negative_patch_density_{ph_label}": largest_negative_patch_density,
                }
            )

        self._apply_charges_for_ph("7.0")
        return features
    
    def extract_asymmetry_features(self) -> Dict[str, float]:
        """
        Extract features related to the asymmetry of charge distribution on the protein surface.

        Features:
        - all_charge_dipole_magnitude
        - surface_charge_dipole_magnitude
        - all_charge_dipole_normalized
        - surface_charge_dipole_normalized

        Notes:
        Dipole vector is calculated as:

            p = sum(q_i * (r_i - r_center))

        where:
            q_i = residue charge
            r_i = residue anchor point
            r_center = geometric center of selected residues

        Normalized dipole is:

            |p| / ((sum |q_i|) * Rg)

        Returns:
            Dictionary mapping feature names to float values.
        """

        logger = logging.getLogger(__name__)
        surface_sasa_percentage_threshold = 0.1
        pseudocount = 1e-6

        features = {}
        for ph in CHARGE_PH_VALUES:
            ph_label = _ph_label(ph)
            features[f"all_charge_spatial_moment_magnitude_{ph_label}"] = 0.0
            features[f"surface_charge_spatial_moment_magnitude_{ph_label}"] = 0.0
            features[f"all_charge_spatial_moment_normalized_{ph_label}"] = 0.0
            features[f"surface_charge_spatial_moment_normalized_{ph_label}"] = 0.0

        if not self.residues:
            logger.warning("No residues loaded; returning zero asymmetry features")
            return features

        def get_valid_residues(residues):
            valid = []

            for residue in residues:
                charge = getattr(residue, "charge", None)
                anchor_point = getattr(residue, "anchor_point", None)

                if charge is None or anchor_point is None:
                    continue

                valid.append(residue)

            return valid

        def calculate_dipole_features(residues):
            residues = get_valid_residues(residues)

            if not residues:
                return 0.0, 0.0

            coords = np.array(
                [residue.anchor_point for residue in residues],
                dtype=float,
            )

            charges = np.array(
                [getattr(residue, "charge", 0.0) or 0.0 for residue in residues],
                dtype=float,
            )

            center = coords.mean(axis=0)

            centered_coords = coords - center

            dipole_vector = np.sum(charges[:, np.newaxis] * centered_coords, axis=0)

            dipole_magnitude = float(np.linalg.norm(dipole_vector))

            squared_distances = np.sum(centered_coords ** 2, axis=1)
            rg = float(np.sqrt(np.mean(squared_distances)))

            total_abs_charge = float(np.sum(np.abs(charges)))

            dipole_normalized = dipole_magnitude / (
                total_abs_charge * rg + pseudocount
            )

            return dipole_magnitude, dipole_normalized

        for ph in CHARGE_PH_VALUES:
            ph_label = _ph_label(ph)
            self._apply_charges_for_ph(ph_label)
            all_residues = get_valid_residues(self.residues)

            surface_residues = [
                residue for residue in all_residues
                if (getattr(residue, "sasa_fraction", 0.0) or 0.0)
                > surface_sasa_percentage_threshold
            ]

            (
                features[f"all_charge_spatial_moment_magnitude_{ph_label}"],
                features[f"all_charge_spatial_moment_normalized_{ph_label}"],
            ) = calculate_dipole_features(all_residues)

            (
                features[f"surface_charge_spatial_moment_magnitude_{ph_label}"],
                features[f"surface_charge_spatial_moment_normalized_{ph_label}"],
            ) = calculate_dipole_features(surface_residues)

        self._apply_charges_for_ph("7.0")
        return features
