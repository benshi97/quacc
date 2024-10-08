from __future__ import annotations

import gzip
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import compare_atoms
from ase.io import read
from numpy.testing import assert_allclose, assert_equal

from quacc.atoms.skzcam import (
    CreateSKZCAMClusters,
    MRCCInputGenerator,
    ORCAInputGenerator,
    SKZCAMInputSet,
    _get_atom_distances,
    create_atom_coord_string,
)

FILE_DIR = Path(__file__).parent


@pytest.fixture
def skzcam_clusters():
    return CreateSKZCAMClusters(
        adsorbate_indices=[0, 1],
        slab_center_indices=[32],
        atom_oxi_states={"Mg": 2.0, "O": -2.0},
        adsorbate_slab_file=Path(FILE_DIR, "skzcam_files", "CO_MgO.poscar.gz"),
        pun_file=None,
    )


@pytest.fixture
def slab_embedded_cluster(skzcam_clusters):
    return skzcam_clusters._convert_pun_to_atoms(
        pun_file=Path(FILE_DIR, "skzcam_files", "ChemShell_Cluster.pun.gz")
    )


@pytest.fixture
def distance_matrix(slab_embedded_cluster):
    return slab_embedded_cluster.get_all_distances()


@pytest.fixture
def adsorbate_slab_embedded_cluster():
    with gzip.open(
        Path(FILE_DIR, "skzcam_files", "adsorbate_slab_embedded_cluster.npy.gz"), "r"
    ) as file:
        return np.load(file, allow_pickle=True).item()["atoms"]


@pytest.fixture
def mrcc_input_generator(adsorbate_slab_embedded_cluster, element_info):
    return MRCCInputGenerator(
        adsorbate_slab_embedded_cluster=adsorbate_slab_embedded_cluster,
        quantum_cluster_indices=[0, 1, 2, 3, 4, 5, 6, 7],
        ecp_region_indices=[8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24],
        element_info=element_info,
        include_cp=True,
        multiplicities={"adsorbate_slab": 3, "adsorbate": 1, "slab": 2},
    )


@pytest.fixture
def orca_input_generator(adsorbate_slab_embedded_cluster, element_info):
    pal_nprocs_block = {"nprocs": 1, "maxcore": 5000}

    method_block = {"Method": "hf", "RI": "on", "RunTyp": "Energy"}

    scf_block = {
        "HFTyp": "rhf",
        "Guess": "MORead",
        "MOInp": '"orca_svp_start.gbw"',
        "SCFMode": "Direct",
        "sthresh": "1e-6",
        "AutoTRAHIter": 60,
        "MaxIter": 1000,
    }

    ecp_info = {
        "Mg": """NewECP
N_core 0
lmax f
s 1
1      1.732000000   14.676000000 2
p 1
1      1.115000000    5.175700000 2
d 1
1      1.203000000   -1.816000000 2
f 1
1      1.000000000    0.000000000 2
end"""
    }
    return ORCAInputGenerator(
        adsorbate_slab_embedded_cluster=adsorbate_slab_embedded_cluster,
        quantum_cluster_indices=[0, 1, 2, 3, 4, 5, 6, 7],
        ecp_region_indices=[8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24],
        element_info=element_info,
        include_cp=True,
        multiplicities={"adsorbate_slab": 3, "adsorbate": 1, "slab": 2},
        pal_nprocs_block=pal_nprocs_block,
        method_block=method_block,
        scf_block=scf_block,
        ecp_info=ecp_info,
    )


@pytest.fixture
def element_info():
    return {
        "C": {
            "basis": "aug-cc-pVDZ",
            "core": 2,
            "ri_scf_basis": "def2/J",
            "ri_cwft_basis": "aug-cc-pVDZ/C",
        },
        "O": {
            "basis": "aug-cc-pVDZ",
            "core": 2,
            "ri_scf_basis": "def2/JK",
            "ri_cwft_basis": "aug-cc-pVDZ/C",
        },
        "Mg": {
            "basis": "cc-pVDZ",
            "core": 2,
            "ri_scf_basis": "def2/J",
            "ri_cwft_basis": "cc-pVDZ/C",
        },
    }


@pytest.fixture
def skzcam_clusters_output(adsorbate_slab_embedded_cluster):
    return {
        "adsorbate_slab_embedded_cluster": adsorbate_slab_embedded_cluster,
        "quantum_cluster_indices_set": [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                16,
                17,
                18,
                19,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
            ],
        ],
        "ecp_region_indices_set": [
            [8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24],
            [
                12,
                13,
                14,
                15,
                20,
                21,
                22,
                23,
                24,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
                55,
                56,
                78,
                79,
                80,
                81,
                82,
                83,
                84,
                85,
            ],
        ],
    }


def test_SKZCAMInputSet_init(skzcam_clusters_output):
    # Try SKZCAMInputSet specifying minimal amount of inputs with MRCC as code
    skzcam_input_set = SKZCAMInputSet(
        adsorbate_slab_embedded_cluster=skzcam_clusters_output[
            "adsorbate_slab_embedded_cluster"
        ],
        quantum_cluster_indices_set=skzcam_clusters_output[
            "quantum_cluster_indices_set"
        ],
        ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
        mp2_oniom1_ll={
            "max_cluster_num": 2,
            "frozencore": "semicore",
            "basis": "def2-SVP",
            "code": "mrcc",
        },
    )

    assert skzcam_input_set.skzcam_input_sets == {
        "mp2_oniom1_ll": {
            "max_cluster_num": 2,
            "frozencore": "semicore",
            "basis": "def2-SVP",
            "code": "mrcc",
            "multiplicities": {"adsorbate_slab": 1, "adsorbate": 1, "slab": 1},
            "ecp": {},
            "ri_scf_basis": None,
            "ri_cwft_basis": None,
            "nprocs": 1,
            "max_memory": 1000,
            "mrcc_calc_inputs": {},
        }
    }

    # Try SKZCAMInputSet by specifying all inputs needed for MRCC as code
    skzcam_input_set = SKZCAMInputSet(
        adsorbate_slab_embedded_cluster=skzcam_clusters_output[
            "adsorbate_slab_embedded_cluster"
        ],
        quantum_cluster_indices_set=skzcam_clusters_output[
            "quantum_cluster_indices_set"
        ],
        ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
        mp2_oniom1_ll={
            "max_cluster_num": 2,
            "frozencore": "semicore",
            "basis": "def2-SVP",
            "code": "mrcc",
            "nprocs": 4,
            "max_memory": 2000,
            "mrcc_calc_inputs": {"calc": "B2PLYP", "symm": "off"},
            "ecp": {"C": "ECP10SDF", "O": "ECP10SDF", "Mg": "ECP10SDF"},
            "ri_scf_basis": "def2-SVP-RI-JK",
            "ri_cwft_basis": "def2-SVPD-RI",
            "multiplicities": {"adsorbate_slab": 3, "adsorbate": 2, "slab": 1},
        },
    )

    assert skzcam_input_set.skzcam_input_sets == {
        "mp2_oniom1_ll": {
            "max_cluster_num": 2,
            "frozencore": "semicore",
            "basis": "def2-SVP",
            "code": "mrcc",
            "nprocs": 4,
            "max_memory": 2000,
            "mrcc_calc_inputs": {"calc": "B2PLYP", "symm": "off"},
            "ecp": {"C": "ECP10SDF", "O": "ECP10SDF", "Mg": "ECP10SDF"},
            "ri_scf_basis": "def2-SVP-RI-JK",
            "ri_cwft_basis": "def2-SVPD-RI",
            "multiplicities": {"adsorbate_slab": 3, "adsorbate": 2, "slab": 1},
        }
    }

    # Try SKZCAMInputSet specifying minimal amount of inputs with ORCA as code
    skzcam_input_set = SKZCAMInputSet(
        adsorbate_slab_embedded_cluster=skzcam_clusters_output[
            "adsorbate_slab_embedded_cluster"
        ],
        quantum_cluster_indices_set=skzcam_clusters_output[
            "quantum_cluster_indices_set"
        ],
        ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
        mp2_oniom1_ll={
            "max_cluster_num": 2,
            "frozencore": "semicore",
            "basis": "def2-SVP",
            "code": "orca",
        },
    )

    assert skzcam_input_set.skzcam_input_sets == {
        "mp2_oniom1_ll": {
            "max_cluster_num": 2,
            "frozencore": "semicore",
            "basis": "def2-SVP",
            "code": "orca",
            "multiplicities": {"adsorbate_slab": 1, "adsorbate": 1, "slab": 1},
            "ecp": {},
            "ri_scf_basis": None,
            "ri_cwft_basis": None,
            "nprocs": 1,
            "max_memory": 1000,
            "orca_method_block": {"RI": "on", "RunTyp": "Energy"},
            "orca_scf_block": {
                "HFTyp": "rhf",
                "Guess": "PAtom",
                "SCFMode": "Direct",
                "sthresh": "1e-6",
                "AutoTRAHIter": "60",
                "MaxIter": "1000",
            },
            "orca_cation_cap_ecp": {
                "Ti": "NewECP\nN_core 0\n  lmax f\n  s 2\n   1      0.860000       9.191690  2\n   2      0.172000       0.008301  2\n  p 2\n   1      0.860000      17.997720  2\n   2      0.172000      -0.032600  2\n  d 2\n   1      1.600000      -9.504310  2\n   2      0.320000      -0.151370  2\n  f 1\n   1      1.000000000    0.000000000 2\nend",
                "Mg": "NewECP\nN_core 0\nlmax f\ns 1\n1      1.732000000   14.676000000 2\np 1\n1      1.115000000    5.175700000 2\nd 1\n1      1.203000000   -1.816000000 2\nf 1\n1      1.000000000    0.000000000 2\nend",
            },
        }
    }

    # Try SKZCAMInputSet by specifying all inputs needed for ORCA as code
    skzcam_input_set = SKZCAMInputSet(
        adsorbate_slab_embedded_cluster=skzcam_clusters_output[
            "adsorbate_slab_embedded_cluster"
        ],
        quantum_cluster_indices_set=skzcam_clusters_output[
            "quantum_cluster_indices_set"
        ],
        ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
        mp2_oniom1_ll={
            "max_cluster_num": 2,
            "frozencore": "semicore",
            "basis": "def2-SVP",
            "code": "orca",
            "nprocs": 4,
            "max_memory": 2000,
            "ecp": {"C": "ECP10SDF", "O": "ECP10SDF", "Mg": "ECP10SDF"},
            "ri_scf_basis": "def2-SVP-RI-JK",
            "ri_cwft_basis": "def2-SVPD-RI",
            "multiplicities": {"adsorbate_slab": 3, "adsorbate": 2, "slab": 1},
            "orca_method_block": {"RI": "off"},
            "orca_scf_block": {"HFType": "uhf"},
            "orca_cation_cap_ecp": {"Mg": "NewECP\nECP10SDF\n"},
        },
    )

    assert skzcam_input_set.skzcam_input_sets == {
        "mp2_oniom1_ll": {
            "max_cluster_num": 2,
            "frozencore": "semicore",
            "basis": "def2-SVP",
            "code": "orca",
            "nprocs": 4,
            "max_memory": 2000,
            "ecp": {"C": "ECP10SDF", "O": "ECP10SDF", "Mg": "ECP10SDF"},
            "ri_scf_basis": "def2-SVP-RI-JK",
            "ri_cwft_basis": "def2-SVPD-RI",
            "multiplicities": {"adsorbate_slab": 3, "adsorbate": 2, "slab": 1},
            "orca_method_block": {"RI": "off"},
            "orca_scf_block": {"HFType": "uhf"},
            "orca_cation_cap_ecp": {"Mg": "NewECP\nECP10SDF\n"},
        }
    }

    # Check that all attributes of the SKZCAMInputSet object are correct
    assert (
        skzcam_input_set.adsorbate_slab_embedded_cluster
        == skzcam_clusters_output["adsorbate_slab_embedded_cluster"]
    )
    assert (
        skzcam_input_set.quantum_cluster_indices_set
        == skzcam_clusters_output["quantum_cluster_indices_set"]
    )
    assert (
        skzcam_input_set.ecp_region_indices_set
        == skzcam_clusters_output["ecp_region_indices_set"]
    )

    # Try specifying multiple ONIOM levels
    skzcam_input_set = SKZCAMInputSet(
        adsorbate_slab_embedded_cluster=skzcam_clusters_output[
            "adsorbate_slab_embedded_cluster"
        ],
        quantum_cluster_indices_set=skzcam_clusters_output[
            "quantum_cluster_indices_set"
        ],
        ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
        mp2_oniom1_ll={
            "max_cluster_num": 2,
            "frozencore": "valence",
            "basis": "def2-SVP",
            "code": "mrcc",
        },
        mp2_oniom1_hl={
            "max_cluster_num": 2,
            "frozencore": "valence",
            "basis": "def2-SVPD",
            "code": "mrcc",
        },
        mp2_oniom2_hl={
            "max_cluster_num": 2,
            "frozencore": "semicore",
            "basis": "def2-TZVP",
            "code": "orca",
        },
        mp2_oniom3_hl={
            "max_cluster_num": 1,
            "frozencore": "semicore",
            "basis": "def2-TZVPD",
            "code": "mrcc",
        },
        mp2_oniom4_hl={
            "max_cluster_num": 1,
            "frozencore": "semicore",
            "basis": "def2-QZVPP",
            "code": "orca",
        },
    )

    assert skzcam_input_set.skzcam_input_sets == {
        "mp2_oniom1_ll": {
            "max_cluster_num": 2,
            "frozencore": "valence",
            "basis": "def2-SVP",
            "code": "mrcc",
            "multiplicities": {"adsorbate_slab": 1, "adsorbate": 1, "slab": 1},
            "ecp": {},
            "ri_scf_basis": None,
            "ri_cwft_basis": None,
            "nprocs": 1,
            "max_memory": 1000,
            "mrcc_calc_inputs": {},
        },
        "mp2_oniom1_hl": {
            "max_cluster_num": 2,
            "frozencore": "valence",
            "basis": "def2-SVPD",
            "code": "mrcc",
            "multiplicities": {"adsorbate_slab": 1, "adsorbate": 1, "slab": 1},
            "ecp": {},
            "ri_scf_basis": None,
            "ri_cwft_basis": None,
            "nprocs": 1,
            "max_memory": 1000,
            "mrcc_calc_inputs": {},
        },
        "mp2_oniom2_hl": {
            "max_cluster_num": 2,
            "frozencore": "semicore",
            "basis": "def2-TZVP",
            "code": "orca",
            "multiplicities": {"adsorbate_slab": 1, "adsorbate": 1, "slab": 1},
            "ecp": {},
            "ri_scf_basis": None,
            "ri_cwft_basis": None,
            "nprocs": 1,
            "max_memory": 1000,
            "orca_method_block": {"RI": "on", "RunTyp": "Energy"},
            "orca_scf_block": {
                "HFTyp": "rhf",
                "Guess": "PAtom",
                "SCFMode": "Direct",
                "sthresh": "1e-6",
                "AutoTRAHIter": "60",
                "MaxIter": "1000",
            },
            "orca_cation_cap_ecp": {
                "Ti": "NewECP\nN_core 0\n  lmax f\n  s 2\n   1      0.860000       9.191690  2\n   2      0.172000       0.008301  2\n  p 2\n   1      0.860000      17.997720  2\n   2      0.172000      -0.032600  2\n  d 2\n   1      1.600000      -9.504310  2\n   2      0.320000      -0.151370  2\n  f 1\n   1      1.000000000    0.000000000 2\nend",
                "Mg": "NewECP\nN_core 0\nlmax f\ns 1\n1      1.732000000   14.676000000 2\np 1\n1      1.115000000    5.175700000 2\nd 1\n1      1.203000000   -1.816000000 2\nf 1\n1      1.000000000    0.000000000 2\nend",
            },
        },
        "mp2_oniom3_hl": {
            "max_cluster_num": 1,
            "frozencore": "semicore",
            "basis": "def2-TZVPD",
            "code": "mrcc",
            "multiplicities": {"adsorbate_slab": 1, "adsorbate": 1, "slab": 1},
            "ecp": {},
            "ri_scf_basis": None,
            "ri_cwft_basis": None,
            "nprocs": 1,
            "max_memory": 1000,
            "mrcc_calc_inputs": {},
        },
        "mp2_oniom4_hl": {
            "max_cluster_num": 1,
            "frozencore": "semicore",
            "basis": "def2-QZVPP",
            "code": "orca",
            "multiplicities": {"adsorbate_slab": 1, "adsorbate": 1, "slab": 1},
            "ecp": {},
            "ri_scf_basis": None,
            "ri_cwft_basis": None,
            "nprocs": 1,
            "max_memory": 1000,
            "orca_method_block": {"RI": "on", "RunTyp": "Energy"},
            "orca_scf_block": {
                "HFTyp": "rhf",
                "Guess": "PAtom",
                "SCFMode": "Direct",
                "sthresh": "1e-6",
                "AutoTRAHIter": "60",
                "MaxIter": "1000",
            },
            "orca_cation_cap_ecp": {
                "Ti": "NewECP\nN_core 0\n  lmax f\n  s 2\n   1      0.860000       9.191690  2\n   2      0.172000       0.008301  2\n  p 2\n   1      0.860000      17.997720  2\n   2      0.172000      -0.032600  2\n  d 2\n   1      1.600000      -9.504310  2\n   2      0.320000      -0.151370  2\n  f 1\n   1      1.000000000    0.000000000 2\nend",
                "Mg": "NewECP\nN_core 0\nlmax f\ns 1\n1      1.732000000   14.676000000 2\np 1\n1      1.115000000    5.175700000 2\nd 1\n1      1.203000000   -1.816000000 2\nf 1\n1      1.000000000    0.000000000 2\nend",
            },
        },
    }

    # Check if errors are raised if length of quantum_cluster_indices_set is different from ecp_region_indices_set
    with pytest.raises(
        ValueError,
        match="The quantum_cluster_indices_set and ecp_region_indices_set must be the same length.",
    ):
        skzcam_input_set = SKZCAMInputSet(
            adsorbate_slab_embedded_cluster=[[0], [0, 1, 2]],
            quantum_cluster_indices_set=[[0]],
            ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
            mp2_oniom1_ll={
                "max_cluster_num": 2,
                "frozencore": "semicore",
                "basis": "def2-SVP",
                "code": "orca",
                "nprocs": 4,
                "max_memory": 2000,
                "ecp": {"C": "ECP10SDF", "O": "ECP10SDF", "Mg": "ECP10SDF"},
                "ri_scf_basis": "def2-SVP-RI-JK",
                "ri_cwft_basis": "def2-SVPD-RI",
                "multiplicities": {"adsorbate_slab": 3, "adsorbate": 2, "slab": 1},
                "orca_method_block": {"RI": "off"},
                "orca_scf_block": {"HFType": "uhf"},
                "orca_cation_cap_ecp": {"Mg": "NewECP\nECP10SDF\n"},
            },
        )

    # Check if errors are raised if variables aren't set during initialization
    with pytest.raises(ValueError, match="The code must be specified."):
        skzcam_input_set = SKZCAMInputSet(
            adsorbate_slab_embedded_cluster=skzcam_clusters_output[
                "adsorbate_slab_embedded_cluster"
            ],
            quantum_cluster_indices_set=skzcam_clusters_output[
                "quantum_cluster_indices_set"
            ],
            ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
            mp2_oniom1_ll={
                "max_cluster_num": 2,
                "frozencore": "semicore",
                "basis": "def2-SVP",
            },
        )

    with pytest.raises(ValueError, match="The code must be either 'mrcc' or 'orca'."):
        skzcam_input_set = SKZCAMInputSet(
            adsorbate_slab_embedded_cluster=skzcam_clusters_output[
                "adsorbate_slab_embedded_cluster"
            ],
            quantum_cluster_indices_set=skzcam_clusters_output[
                "quantum_cluster_indices_set"
            ],
            ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
            mp2_oniom1_ll={
                "max_cluster_num": 2,
                "frozencore": "semicore",
                "basis": "def2-SVP",
                "code": "asdf",
            },
        )

    with pytest.raises(
        ValueError,
        match="The maximum cluster number must be provided for all ONIOM levels.",
    ):
        skzcam_input_set = SKZCAMInputSet(
            adsorbate_slab_embedded_cluster=skzcam_clusters_output[
                "adsorbate_slab_embedded_cluster"
            ],
            quantum_cluster_indices_set=skzcam_clusters_output[
                "quantum_cluster_indices_set"
            ],
            ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
            mp2_oniom1_ll={
                "frozencore": "semicore",
                "basis": "def2-SVP",
                "code": "mrcc",
            },
        )

    with pytest.raises(
        ValueError, match="The frozencore must be provided for all ONIOM levels."
    ):
        skzcam_input_set = SKZCAMInputSet(
            adsorbate_slab_embedded_cluster=skzcam_clusters_output[
                "adsorbate_slab_embedded_cluster"
            ],
            quantum_cluster_indices_set=skzcam_clusters_output[
                "quantum_cluster_indices_set"
            ],
            ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
            mp2_oniom1_ll={"max_cluster_num": 2, "basis": "def2-SVP", "code": "mrcc"},
        )

    with pytest.raises(
        ValueError, match="The basis must be provided for all ONIOM levels."
    ):
        skzcam_input_set = SKZCAMInputSet(
            adsorbate_slab_embedded_cluster=skzcam_clusters_output[
                "adsorbate_slab_embedded_cluster"
            ],
            quantum_cluster_indices_set=skzcam_clusters_output[
                "quantum_cluster_indices_set"
            ],
            ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
            mp2_oniom1_ll={
                "max_cluster_num": 2,
                "frozencore": "semicore",
                "code": "mrcc",
            },
        )

    # Check if errors are raise if max_cluster_num is more than length of quantum_cluster_indices_set
    with pytest.raises(
        ValueError,
        match="The maximum cluster number for all ONIOM levels must be less than or equal to the number of quantum clusters.",
    ):
        skzcam_input_set = SKZCAMInputSet(
            adsorbate_slab_embedded_cluster=skzcam_clusters_output[
                "adsorbate_slab_embedded_cluster"
            ],
            quantum_cluster_indices_set=skzcam_clusters_output[
                "quantum_cluster_indices_set"
            ],
            ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
            mp2_oniom1_ll={
                "max_cluster_num": 2,
                "frozencore": "semicore",
                "basis": "def2-SVP",
                "code": "mrcc",
            },
            mp2_oniom1_hl={
                "max_cluster_num": 3,
                "frozencore": "semicore",
                "basis": "def2-SVP",
                "code": "mrcc",
            },
        )

    # Check if errors are raised when higher ONIOM levels are specified withou lower ONIOM levels
    with pytest.raises(
        ValueError,
        match="The low-level ONIOM1 MP2 calculation must be provided if the high-level ONIOM1 is to be used.",
    ):
        skzcam_input_set = SKZCAMInputSet(
            adsorbate_slab_embedded_cluster=skzcam_clusters_output[
                "adsorbate_slab_embedded_cluster"
            ],
            quantum_cluster_indices_set=skzcam_clusters_output[
                "quantum_cluster_indices_set"
            ],
            ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
            mp2_oniom1_hl={
                "max_cluster_num": 2,
                "frozencore": "semicore",
                "basis": "def2-SVP",
                "code": "mrcc",
            },
        )

    with pytest.raises(
        ValueError,
        match="The high-level ONIOM1 MP2 calculation must be provided if the high-level ONIOM2 is to be used.",
    ):
        skzcam_input_set = SKZCAMInputSet(
            adsorbate_slab_embedded_cluster=skzcam_clusters_output[
                "adsorbate_slab_embedded_cluster"
            ],
            quantum_cluster_indices_set=skzcam_clusters_output[
                "quantum_cluster_indices_set"
            ],
            ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
            mp2_oniom2_hl={
                "max_cluster_num": 2,
                "frozencore": "semicore",
                "basis": "def2-SVP",
                "code": "mrcc",
            },
        )

    with pytest.raises(
        ValueError,
        match="The high-level ONIOM2 MP2 calculation must be provided if the high-level ONIOM3 is to be used.",
    ):
        skzcam_input_set = SKZCAMInputSet(
            adsorbate_slab_embedded_cluster=skzcam_clusters_output[
                "adsorbate_slab_embedded_cluster"
            ],
            quantum_cluster_indices_set=skzcam_clusters_output[
                "quantum_cluster_indices_set"
            ],
            ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
            mp2_oniom3_hl={
                "max_cluster_num": 2,
                "frozencore": "semicore",
                "basis": "def2-SVP",
                "code": "mrcc",
            },
        )

    with pytest.raises(
        ValueError,
        match="The high-level ONIOM3 MP2 calculation must be provided if the high-level ONIOM4 is to be used.",
    ):
        skzcam_input_set = SKZCAMInputSet(
            adsorbate_slab_embedded_cluster=skzcam_clusters_output[
                "adsorbate_slab_embedded_cluster"
            ],
            quantum_cluster_indices_set=skzcam_clusters_output[
                "quantum_cluster_indices_set"
            ],
            ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
            mp2_oniom4_hl={
                "max_cluster_num": 2,
                "frozencore": "semicore",
                "basis": "def2-SVP",
                "code": "mrcc",
            },
        )


def test_SKZCAMInputSet_create_element_info(skzcam_clusters_output):
    # Check SKZCAMInputSet when presets are used based on 'basis' in ['DZ', 'TZ', 'QZ'] and 'frozencore' in ['valence', 'semicore']
    # First for 'DZ' and 'semicore' for MRCC
    skzcam_input_set = SKZCAMInputSet(
        adsorbate_slab_embedded_cluster=skzcam_clusters_output[
            "adsorbate_slab_embedded_cluster"
        ],
        quantum_cluster_indices_set=skzcam_clusters_output[
            "quantum_cluster_indices_set"
        ],
        ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
        mp2_oniom1_ll={
            "max_cluster_num": 2,
            "frozencore": "semicore",
            "basis": "DZ",
            "code": "mrcc",
        },
    )
    skzcam_input_set.skzcam_input_sets["mp2_oniom1_ll"]

    element_info = skzcam_input_set.create_element_info(
        basis="DZ",
        frozencore="semicore",
        code="mrcc",
        ecp={},
        ri_scf_basis=None,
        ri_cwft_basis=None,
    )
    assert element_info == {
        "C": {
            "core": 2,
            "basis": "aug-cc-pVDZ",
            "ecp": "none",
            "ri_scf_basis": "def2-QZVPP-RI-JK",
            "ri_cwft_basis": "aug-cc-pVDZ-RI",
        },
        "O": {
            "core": 2,
            "basis": "aug-cc-pVDZ",
            "ecp": "none",
            "ri_scf_basis": "def2-QZVPP-RI-JK",
            "ri_cwft_basis": "aug-cc-pVDZ-RI",
        },
        "Mg": {
            "core": 2,
            "basis": "cc-pwCVDZ",
            "ecp": "none",
            "ri_scf_basis": "def2-QZVPP-RI-JK",
            "ri_cwft_basis": "cc-pwCVDZ-RI",
        },
    }

    # Then for 'QZ' and 'valence' for ORCA
    element_info = skzcam_input_set.create_element_info(
        basis="QZ",
        frozencore="valence",
        code="orca",
        ecp={},
        ri_scf_basis=None,
        ri_cwft_basis=None,
    )

    assert element_info == {
        "C": {
            "core": 2,
            "basis": "aug-cc-pVQZ",
            "ecp": "none",
            "ri_scf_basis": "def2/J",
            "ri_cwft_basis": "aug-cc-pVQZ/C",
        },
        "O": {
            "core": 2,
            "basis": "aug-cc-pVQZ",
            "ecp": "none",
            "ri_scf_basis": "def2/J",
            "ri_cwft_basis": "aug-cc-pVQZ/C",
        },
        "Mg": {
            "core": 10,
            "basis": "cc-pVQZ",
            "ecp": "none",
            "ri_scf_basis": "def2/J",
            "ri_cwft_basis": "cc-pVQZ/C",
        },
    }

    # Specifying ecp for MRCC
    element_info = skzcam_input_set.create_element_info(
        basis="DZ",
        frozencore="semicore",
        code="mrcc",
        ecp={"Mg": "ECP10SDF"},
        ri_scf_basis=None,
        ri_cwft_basis=None,
    )
    assert element_info == {
        "C": {
            "core": 2,
            "basis": "aug-cc-pVDZ",
            "ecp": "none",
            "ri_scf_basis": "def2-QZVPP-RI-JK",
            "ri_cwft_basis": "aug-cc-pVDZ-RI",
        },
        "O": {
            "core": 2,
            "basis": "aug-cc-pVDZ",
            "ecp": "none",
            "ri_scf_basis": "def2-QZVPP-RI-JK",
            "ri_cwft_basis": "aug-cc-pVDZ-RI",
        },
        "Mg": {
            "core": 2,
            "basis": "cc-pwCVDZ",
            "ecp": "ECP10SDF",
            "ri_scf_basis": "def2-QZVPP-RI-JK",
            "ri_cwft_basis": "cc-pwCVDZ-RI",
        },
    }

    # Now testing the custom inputs
    element_info = skzcam_input_set.create_element_info(
        basis="def2-SVP",
        frozencore={"C": 1, "O": 2, "Mg": 6},
        code="mrcc",
        ecp={},
        ri_scf_basis=None,
        ri_cwft_basis=None,
    )
    assert element_info == {
        "C": {
            "core": 1,
            "basis": "def2-SVP",
            "ecp": "none",
            "ri_scf_basis": "def2-QZVPP-RI-JK",
            "ri_cwft_basis": "def2-SVP-RI",
        },
        "O": {
            "core": 2,
            "basis": "def2-SVP",
            "ecp": "none",
            "ri_scf_basis": "def2-QZVPP-RI-JK",
            "ri_cwft_basis": "def2-SVP-RI",
        },
        "Mg": {
            "core": 6,
            "basis": "def2-SVP",
            "ecp": "none",
            "ri_scf_basis": "def2-QZVPP-RI-JK",
            "ri_cwft_basis": "def2-SVP-RI",
        },
    }

    element_info = skzcam_input_set.create_element_info(
        basis="def2-SVP",
        frozencore="valence",
        code="mrcc",
        ecp={},
        ri_scf_basis=None,
        ri_cwft_basis=None,
    )
    assert element_info == {
        "C": {
            "core": 2,
            "basis": "def2-SVP",
            "ecp": "none",
            "ri_scf_basis": "def2-QZVPP-RI-JK",
            "ri_cwft_basis": "def2-SVP-RI",
        },
        "O": {
            "core": 2,
            "basis": "def2-SVP",
            "ecp": "none",
            "ri_scf_basis": "def2-QZVPP-RI-JK",
            "ri_cwft_basis": "def2-SVP-RI",
        },
        "Mg": {
            "core": 10,
            "basis": "def2-SVP",
            "ecp": "none",
            "ri_scf_basis": "def2-QZVPP-RI-JK",
            "ri_cwft_basis": "def2-SVP-RI",
        },
    }

    element_info = skzcam_input_set.create_element_info(
        basis="def2-SVP",
        frozencore="semicore",
        code="mrcc",
        ecp={},
        ri_scf_basis=None,
        ri_cwft_basis=None,
    )
    assert element_info == {
        "C": {
            "core": 2,
            "basis": "def2-SVP",
            "ecp": "none",
            "ri_scf_basis": "def2-QZVPP-RI-JK",
            "ri_cwft_basis": "def2-SVP-RI",
        },
        "O": {
            "core": 2,
            "basis": "def2-SVP",
            "ecp": "none",
            "ri_scf_basis": "def2-QZVPP-RI-JK",
            "ri_cwft_basis": "def2-SVP-RI",
        },
        "Mg": {
            "core": 2,
            "basis": "def2-SVP",
            "ecp": "none",
            "ri_scf_basis": "def2-QZVPP-RI-JK",
            "ri_cwft_basis": "def2-SVP-RI",
        },
    }

    # Check if errors are raised when frozencore is not specified correctly
    with pytest.raises(
        ValueError,
        match="frozencore must be provided for all elements in the quantum cluster when provided as a dictionary.",
    ):
        element_info = skzcam_input_set.create_element_info(
            basis="def2-SVP",
            frozencore={"C": 1, "O": 5},
            code="mrcc",
            ecp={},
            ri_scf_basis=None,
            ri_cwft_basis=None,
        )

    with pytest.raises(
        ValueError,
        match="frozencore must be provided as either 'valence' or 'semicore' if provided as a string.",
    ):
        element_info = skzcam_input_set.create_element_info(
            basis="def2-SVP",
            frozencore="supercore",
            code="mrcc",
            ecp={},
            ri_scf_basis=None,
            ri_cwft_basis=None,
        )

    with pytest.raises(
        ValueError, match="frozencore must be provided as a string or as a dictionary."
    ):
        element_info = skzcam_input_set.create_element_info(
            basis="def2-SVP",
            frozencore=[2, 2, 10],
            code="mrcc",
            ecp={},
            ri_scf_basis=None,
            ri_cwft_basis=None,
        )

    # Test custom basis set inputs
    element_info = skzcam_input_set.create_element_info(
        basis="ano-SVP",
        frozencore="semicore",
        code="mrcc",
        ecp={},
        ri_scf_basis=None,
        ri_cwft_basis=None,
    )

    assert element_info == {
        "C": {
            "core": 2,
            "basis": "ano-SVP",
            "ecp": "none",
            "ri_scf_basis": "def2-QZVPP-RI-JK",
            "ri_cwft_basis": "ano-SVP-RI",
        },
        "O": {
            "core": 2,
            "basis": "ano-SVP",
            "ecp": "none",
            "ri_scf_basis": "def2-QZVPP-RI-JK",
            "ri_cwft_basis": "ano-SVP-RI",
        },
        "Mg": {
            "core": 2,
            "basis": "ano-SVP",
            "ecp": "none",
            "ri_scf_basis": "def2-QZVPP-RI-JK",
            "ri_cwft_basis": "ano-SVP-RI",
        },
    }

    element_info = skzcam_input_set.create_element_info(
        basis={"C": "def2-SVP", "O": "def2-SVPD", "Mg": "def2-TZVP"},
        frozencore="semicore",
        code="mrcc",
        ecp={},
        ri_scf_basis=None,
        ri_cwft_basis=None,
    )

    assert element_info == {
        "C": {
            "core": 2,
            "basis": "def2-SVP",
            "ecp": "none",
            "ri_scf_basis": "def2-QZVPP-RI-JK",
            "ri_cwft_basis": "def2-SVP-RI",
        },
        "O": {
            "core": 2,
            "basis": "def2-SVPD",
            "ecp": "none",
            "ri_scf_basis": "def2-QZVPP-RI-JK",
            "ri_cwft_basis": "def2-SVPD-RI",
        },
        "Mg": {
            "core": 2,
            "basis": "def2-TZVP",
            "ecp": "none",
            "ri_scf_basis": "def2-QZVPP-RI-JK",
            "ri_cwft_basis": "def2-TZVP-RI",
        },
    }

    with pytest.raises(
        ValueError,
        match="basis must be provided for all elements in the quantum cluster when provided as a dictionary.",
    ):
        element_info = skzcam_input_set.create_element_info(
            basis={"C": "def2-SVP", "O": "def2-SVPD"},
            frozencore="semicore",
            code="mrcc",
            ecp={},
            ri_scf_basis=None,
            ri_cwft_basis=None,
        )

    element_info = skzcam_input_set.create_element_info(
        basis="def2-SVP",
        frozencore="semicore",
        code="mrcc",
        ecp={"Mg": "ECP10SDF"},
        ri_scf_basis=None,
        ri_cwft_basis=None,
    )

    assert element_info == {
        "C": {
            "core": 2,
            "basis": "def2-SVP",
            "ecp": "none",
            "ri_scf_basis": "def2-QZVPP-RI-JK",
            "ri_cwft_basis": "def2-SVP-RI",
        },
        "O": {
            "core": 2,
            "basis": "def2-SVP",
            "ecp": "none",
            "ri_scf_basis": "def2-QZVPP-RI-JK",
            "ri_cwft_basis": "def2-SVP-RI",
        },
        "Mg": {
            "core": 2,
            "basis": "def2-SVP",
            "ecp": "ECP10SDF",
            "ri_scf_basis": "def2-QZVPP-RI-JK",
            "ri_cwft_basis": "def2-SVP-RI",
        },
    }

    # Test if ri_scf_basis and ri_cwft_basis are set correctly
    element_info = skzcam_input_set.create_element_info(
        basis="def2-SVP",
        frozencore="semicore",
        code="mrcc",
        ecp={},
        ri_scf_basis="def2-SVP-RI-JK",
        ri_cwft_basis="def2-SVP-RI",
    )

    assert element_info == {
        "C": {
            "core": 2,
            "basis": "def2-SVP",
            "ecp": "none",
            "ri_scf_basis": "def2-SVP-RI-JK",
            "ri_cwft_basis": "def2-SVP-RI",
        },
        "O": {
            "core": 2,
            "basis": "def2-SVP",
            "ecp": "none",
            "ri_scf_basis": "def2-SVP-RI-JK",
            "ri_cwft_basis": "def2-SVP-RI",
        },
        "Mg": {
            "core": 2,
            "basis": "def2-SVP",
            "ecp": "none",
            "ri_scf_basis": "def2-SVP-RI-JK",
            "ri_cwft_basis": "def2-SVP-RI",
        },
    }

    element_info = skzcam_input_set.create_element_info(
        basis={"C": "def2-SVP", "O": "def2-SVPD", "Mg": "def2-TZVP"},
        frozencore="semicore",
        code="mrcc",
        ecp={},
        ri_scf_basis={
            "C": "def2-SVP-RI-JK",
            "O": "def2-SVPD-RI-JK",
            "Mg": "def2-QZVP-RI-JK",
        },
        ri_cwft_basis={"C": "def2-TZVP-RI", "O": "def2-QZVP-RI", "Mg": "def2-QZVPP-RI"},
    )

    assert element_info == {
        "C": {
            "core": 2,
            "basis": "def2-SVP",
            "ecp": "none",
            "ri_scf_basis": "def2-SVP-RI-JK",
            "ri_cwft_basis": "def2-TZVP-RI",
        },
        "O": {
            "core": 2,
            "basis": "def2-SVPD",
            "ecp": "none",
            "ri_scf_basis": "def2-SVPD-RI-JK",
            "ri_cwft_basis": "def2-QZVP-RI",
        },
        "Mg": {
            "core": 2,
            "basis": "def2-TZVP",
            "ecp": "none",
            "ri_scf_basis": "def2-QZVP-RI-JK",
            "ri_cwft_basis": "def2-QZVPP-RI",
        },
    }

    # Test if errors are raised when ri_scf_basis and ri_cwft_basis are not provided correctly
    with pytest.raises(
        ValueError,
        match="ri_scf_basis must be provided for all elements in the quantum cluster when provided as a dictionary.",
    ):
        element_info = skzcam_input_set.create_element_info(
            basis={"C": "def2-SVP", "O": "def2-SVPD", "Mg": "def2-TZVP"},
            frozencore="semicore",
            code="mrcc",
            ecp={},
            ri_scf_basis={"C": "def2-SVP-RI-JK", "O": "def2-SVPD-RI-JK"},
            ri_cwft_basis={
                "C": "def2-TZVP-RI",
                "O": "def2-QZVP-RI",
                "Mg": "def2-QZVPP-RI",
            },
        )

    with pytest.raises(
        ValueError,
        match="ri_scf_basis must be provided as a string or dictionary of elements.",
    ):
        element_info = skzcam_input_set.create_element_info(
            basis={"C": "def2-SVP", "O": "def2-SVPD", "Mg": "def2-TZVP"},
            frozencore="semicore",
            code="mrcc",
            ecp={},
            ri_scf_basis=["def2-SVP-RI-JK", "def2-SVPD-RI-JK"],
            ri_cwft_basis={
                "C": "def2-TZVP-RI",
                "O": "def2-QZVP-RI",
                "Mg": "def2-QZVPP-RI",
            },
        )

    with pytest.raises(
        ValueError,
        match="ri_cwft_basis must be provided for all elements in the quantum cluster when provided as a dictionary.",
    ):
        element_info = skzcam_input_set.create_element_info(
            basis={"C": "def2-SVP", "O": "def2-SVPD", "Mg": "def2-TZVP"},
            frozencore="semicore",
            code="mrcc",
            ecp={},
            ri_scf_basis={
                "C": "def2-SVP-RI-JK",
                "O": "def2-SVPD-RI-JK",
                "Mg": "def2-QZVP-RI-JK",
            },
            ri_cwft_basis={"C": "def2-TZVP-RI", "O": "def2-QZVP-RI"},
        )

    with pytest.raises(
        ValueError,
        match="ri_cwft_basis must be provided as a string or dictionary of elements.",
    ):
        element_info = skzcam_input_set.create_element_info(
            basis={"C": "def2-SVP", "O": "def2-SVPD", "Mg": "def2-TZVP"},
            frozencore="semicore",
            code="mrcc",
            ecp={},
            ri_scf_basis={
                "C": "def2-SVP-RI-JK",
                "O": "def2-SVPD-RI-JK",
                "Mg": "def2-QZVP-RI-JK",
            },
            ri_cwft_basis=["def2-TZVP-RI", "def2-QZVP-RI"],
        )


def test_SKZCAMInputSet_generate_input(skzcam_clusters_output, tmp_path):
    skzcam_input_set = SKZCAMInputSet(
        adsorbate_slab_embedded_cluster=skzcam_clusters_output[
            "adsorbate_slab_embedded_cluster"
        ],
        quantum_cluster_indices_set=skzcam_clusters_output[
            "quantum_cluster_indices_set"
        ],
        ecp_region_indices_set=skzcam_clusters_output["ecp_region_indices_set"],
        mp2_oniom1_ll={
            "max_cluster_num": 2,
            "frozencore": "semicore",
            "basis": "DZ",
            "code": "mrcc",
        },
        mp2_oniom1_hl={
            "max_cluster_num": 1,
            "frozencore": "valence",
            "basis": "CBS(DZ/TZ)",
            "code": "orca",
        },
    )

    skzcam_input_set.generate_input(tmp_path)
    tmp_path_files = os.listdir(tmp_path)
    tmp_path_files.sort()
    assert tmp_path_files == [
        "MRCC_MINP_MP2_cluster_1_awCVDZ_adsorbate",
        "MRCC_MINP_MP2_cluster_1_awCVDZ_adsorbate_slab",
        "MRCC_MINP_MP2_cluster_1_awCVDZ_slab",
        "MRCC_MINP_MP2_cluster_2_awCVDZ_adsorbate",
        "MRCC_MINP_MP2_cluster_2_awCVDZ_adsorbate_slab",
        "MRCC_MINP_MP2_cluster_2_awCVDZ_slab",
        "ORCA_MP2_cluster_1_aVDZ.pc",
        "ORCA_MP2_cluster_1_aVDZ_adsorbate.inp",
        "ORCA_MP2_cluster_1_aVDZ_adsorbate_slab.inp",
        "ORCA_MP2_cluster_1_aVDZ_slab.inp",
        "ORCA_MP2_cluster_1_aVTZ.pc",
        "ORCA_MP2_cluster_1_aVTZ_adsorbate.inp",
        "ORCA_MP2_cluster_1_aVTZ_adsorbate_slab.inp",
        "ORCA_MP2_cluster_1_aVTZ_slab.inp",
    ]

    # Check that the input files are correct
    with open(Path(tmp_path, "ORCA_MP2_cluster_1_aVDZ_adsorbate_slab.inp")) as f:
        orca_adsorbate_slab_lines = f.readlines()[::10]

    assert orca_adsorbate_slab_lines == [
        "! TightSCF RI-MP2 RIJCOSX SlowConv DIIS \n",
        "end\n",
        "sthresh 1e-6\n",
        "C                       0.00000000000    0.00000000000    2.00000000000\n",
        "N_core 0\n",
        "end\n",
        "1      1.203000000   -1.816000000 2\n",
        "p 1\n",
        "lmax f\n",
        "Mg>    2.00000000000   -2.10705287155    0.00000000000   -2.14155206950\n",
        "f 1\n",
        "1      1.115000000    5.175700000 2\n",
        "s 1\n",
        "NewECP\n",
        "1      1.000000000    0.000000000 2\n",
        "d 1\n",
        "1      1.732000000   14.676000000 2\n",
        "N_core 0\n",
        "end\n",
        "1      1.203000000   -1.816000000 2\n",
        "p 1\n",
    ]

    with open(Path(tmp_path, "MRCC_MINP_MP2_cluster_1_awCVDZ_adsorbate")) as f:
        mrcc_adsorbate_lines = f.readlines()[::2]

    assert mrcc_adsorbate_lines == [
        "calc=DF-MP2\n",
        "verbosity=3\n",
        "symm=off\n",
        "scfiguess=small\n",
        "scfalg=locfit1\n",
        "def2-SVP\n",
        "def2-SVP\n",
        "def2-SVP\n",
        "def2-SVP\n",
        "\n",
        "basis=atomtype\n",
        "aug-cc-pVDZ\n",
        "aug-cc-pVDZ\n",
        "aug-cc-pVDZ\n",
        "aug-cc-pVDZ\n",
        "\n",
        "def2-QZVPP-RI-JK\n",
        "def2-QZVPP-RI-JK\n",
        "def2-QZVPP-RI-JK\n",
        "def2-QZVPP-RI-JK\n",
        "\n",
        "dfbasis_cor=atomtype\n",
        "aug-cc-pVDZ-RI\n",
        "aug-cc-pVDZ-RI\n",
        "aug-cc-pVDZ-RI\n",
        "aug-cc-pVDZ-RI\n",
        "\n",
        "none\n",
        "none\n",
        "none\n",
        "none\n",
        "\n",
        "mult=1\n",
        "geom=xyz\n",
        "\n",
        "O                       0.00000000000    0.00000000000    3.12800000000\n",
        "O                      -2.12018425659    0.00000000000    0.00567209089\n",
        "O                       2.12018425659    0.00000000000    0.00567209089\n",
        "O                       0.00000000000    0.00000000000   -2.14129966123\n",
        "ghost=serialno\n",
        "\n",
    ]


def test_MRCCInputGenerator_init(adsorbate_slab_embedded_cluster, element_info):
    # Check what happens if multiplicities is not provided
    mrcc_input_generator = MRCCInputGenerator(
        adsorbate_slab_embedded_cluster=adsorbate_slab_embedded_cluster,
        quantum_cluster_indices=[0, 1, 2, 3, 4, 5, 6, 7],
        ecp_region_indices=[8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24],
        element_info=element_info,
        include_cp=True,
    )

    assert mrcc_input_generator.multiplicities == {
        "adsorbate_slab": 1,
        "adsorbate": 1,
        "slab": 1,
    }

    mrcc_input_generator = MRCCInputGenerator(
        adsorbate_slab_embedded_cluster=adsorbate_slab_embedded_cluster,
        quantum_cluster_indices=[0, 1, 2, 3, 4, 5, 6, 7],
        ecp_region_indices=[8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24],
        element_info=element_info,
        include_cp=True,
        multiplicities={"adsorbate_slab": 3, "adsorbate": 1, "slab": 2},
    )

    assert not compare_atoms(
        mrcc_input_generator.adsorbate_slab_embedded_cluster,
        adsorbate_slab_embedded_cluster,
    )
    assert_equal(mrcc_input_generator.quantum_cluster_indices, [0, 1, 2, 3, 4, 5, 6, 7])
    assert_equal(mrcc_input_generator.adsorbate_indices, [0, 1])
    assert_equal(mrcc_input_generator.slab_indices, [2, 3, 4, 5, 6, 7])
    assert_equal(
        mrcc_input_generator.ecp_region_indices,
        [8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24],
    )
    assert mrcc_input_generator.element_info == element_info
    assert mrcc_input_generator.include_cp is True
    assert mrcc_input_generator.multiplicities == {
        "adsorbate_slab": 3,
        "adsorbate": 1,
        "slab": 2,
    }

    # Check if error raise if quantum_cluster_indices and ecp_region_indices overlap

    with pytest.raises(
        ValueError, match="An atom in the quantum cluster is also in the ECP region."
    ):
        mrcc_input_generator = MRCCInputGenerator(
            adsorbate_slab_embedded_cluster=adsorbate_slab_embedded_cluster,
            quantum_cluster_indices=[0, 1, 2, 3, 4, 5, 6, 7],
            ecp_region_indices=[7, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24],
            element_info=element_info,
            include_cp=True,
        )


def test_MRCCInputGenerator_generate_input(mrcc_input_generator):
    mrcc_input_generator_nocp = deepcopy(mrcc_input_generator)

    mrcc_input_generator_nocp.include_cp = False
    input_dict = mrcc_input_generator_nocp.generate_input()

    # Check that the input_dictionary is correct
    assert (
        input_dict["adsorbate"]["geom"].split()[1],
        input_dict["slab"]["geom"].split()[1],
        input_dict["adsorbate_slab"]["geom"].split()[1],
    ) == ("2", "19", "21")

    mrcc_input_generator.generate_input()

    reference_block_collated = {
        "adsorbate_slab": {
            "float": [21.0, -2.0, 2.0, 2.0, 2.0, 0.1474277671],
            "string": ["basis_sm=atomtype", "def2/JK", "capECP"],
        },
        "adsorbate": {"float": [8.0], "string": ["basis_sm=atomtype", "C"]},
        "slab": {
            "float": [21.0, -2.0, 2.0, 2.0, 2.0, 0.1474277671],
            "string": ["basis_sm=atomtype", "def2/JK", "capECP"],
        },
    }

    reference_block_nocp_collated = {
        "adsorbate_slab": {
            "float": [21.0, -2.0, 2.0, 2.0, 2.0, 0.1474277671],
            "string": ["basis_sm=atomtype", "def2/JK", "capECP"],
        },
        "adsorbate": {"float": [2.0], "string": ["basis_sm=atomtype"]},
        "slab": {
            "float": [
                19.0,
                -4.22049352791,
                4.22049352791,
                4.22049352791,
                2.11024676395,
                -0.0,
            ],
            "string": ["basis_sm=atomtype", "no-basis-set", "charge=-8"],
        },
    }

    generated_block_collated = {
        system: {"float": [], "string": []}
        for system in ["adsorbate_slab", "adsorbate", "slab"]
    }
    generated_block_nocp_collated = {
        system: {"float": [], "string": []}
        for system in ["adsorbate_slab", "adsorbate", "slab"]
    }

    for system in ["adsorbate_slab", "adsorbate", "slab"]:
        generated_block_collated[system]["float"] = [
            float(x)
            for x in mrcc_input_generator.skzcam_input_str[system].split()
            if x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ][::300]
        generated_block_collated[system]["string"] = [
            x
            for x in mrcc_input_generator.skzcam_input_str[system].split()
            if not x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ][::50]

        generated_block_nocp_collated[system]["float"] = [
            float(x)
            for x in mrcc_input_generator_nocp.skzcam_input_str[system].split()
            if x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ][::300]
        generated_block_nocp_collated[system]["string"] = [
            x
            for x in mrcc_input_generator_nocp.skzcam_input_str[system].split()
            if not x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ][::50]

        assert_equal(
            generated_block_collated[system]["string"],
            reference_block_collated[system]["string"],
        )
        assert_allclose(
            generated_block_collated[system]["float"],
            reference_block_collated[system]["float"],
            rtol=1e-05,
            atol=1e-07,
        )

        assert_equal(
            generated_block_nocp_collated[system]["string"],
            reference_block_nocp_collated[system]["string"],
        )
        assert_allclose(
            generated_block_nocp_collated[system]["float"],
            reference_block_nocp_collated[system]["float"],
            rtol=1e-05,
            atol=1e-07,
        )


def test_MRCCInputGenerator_generate_basis_ecp_block(mrcc_input_generator):
    mrcc_input_generator_nocp = deepcopy(mrcc_input_generator)

    mrcc_input_generator_nocp.include_cp = False
    mrcc_input_generator_nocp._generate_basis_ecp_block()

    mrcc_input_generator._generate_basis_ecp_block()

    reference_mrcc_blocks_collated = {
        "adsorbate_slab": [
            "basis_sm=atomtype",
            "no-basis-set",
            "no-basis-set",
            "aug-cc-pVDZ",
            "no-basis-set",
            "def2/JK",
            "no-basis-set",
            "aug-cc-pVDZ/C",
            "no-basis-set",
            "none",
            "capECP",
        ],
        "slab": [
            "basis_sm=atomtype",
            "no-basis-set",
            "no-basis-set",
            "aug-cc-pVDZ",
            "no-basis-set",
            "def2/JK",
            "no-basis-set",
            "aug-cc-pVDZ/C",
            "no-basis-set",
            "none",
            "capECP",
        ],
        "adsorbate": [
            "basis_sm=atomtype",
            "aug-cc-pVDZ",
            "def2/JK",
            "cc-pVDZ/C",
            "none",
        ],
    }

    reference_mrcc_blocks_nocp_collated = {
        "adsorbate_slab": [
            "basis_sm=atomtype",
            "no-basis-set",
            "no-basis-set",
            "aug-cc-pVDZ",
            "no-basis-set",
            "def2/JK",
            "no-basis-set",
            "aug-cc-pVDZ/C",
            "no-basis-set",
            "none",
            "capECP",
        ],
        "slab": [
            "basis_sm=atomtype",
            "no-basis-set",
            "basis=atomtype",
            "no-basis-set",
            "dfbasis_scf=atomtype",
            "no-basis-set",
            "dfbasis_cor=atomtype",
            "no-basis-set",
            "ecp=atomtype",
            "capECP",
        ],
        "adsorbate": ["basis_sm=atomtype", "aug-cc-pVDZ/C"],
    }

    generated_mrcc_blocks_nocp_collated = {
        system: [] for system in ["adsorbate_slab", "slab", "adsorbate"]
    }
    generated_mrcc_blocks_collated = {
        system: [] for system in ["adsorbate_slab", "slab", "adsorbate"]
    }
    for system in ["adsorbate_slab", "adsorbate", "slab"]:
        generated_mrcc_blocks_collated[system] = mrcc_input_generator.skzcam_input_str[
            system
        ].split()[::10]
        generated_mrcc_blocks_nocp_collated[system] = (
            mrcc_input_generator_nocp.skzcam_input_str[system].split()[::10]
        )

        assert_equal(
            generated_mrcc_blocks_collated[system],
            reference_mrcc_blocks_collated[system],
        )
        assert_equal(
            generated_mrcc_blocks_nocp_collated[system],
            reference_mrcc_blocks_nocp_collated[system],
        )


def test_MRCCInputGenerator_create_atomtype_basis(mrcc_input_generator):
    generated_basis_block_without_ecp = mrcc_input_generator._create_atomtype_basis(
        quantum_region=mrcc_input_generator.adsorbate_slab_cluster,
        element_basis_info={
            element: mrcc_input_generator.element_info[element]["ri_cwft_basis"]
            for element in mrcc_input_generator.element_info
        },
    )
    generated_basis_block_with_ecp = mrcc_input_generator._create_atomtype_basis(
        quantum_region=mrcc_input_generator.adsorbate_slab_cluster,
        element_basis_info={
            element: mrcc_input_generator.element_info[element]["ri_cwft_basis"]
            for element in mrcc_input_generator.element_info
        },
        ecp_region=mrcc_input_generator.ecp_region,
    )

    reference_basis_block_without_ecp = "aug-cc-pVDZ/C\naug-cc-pVDZ/C\ncc-pVDZ/C\naug-cc-pVDZ/C\naug-cc-pVDZ/C\naug-cc-pVDZ/C\naug-cc-pVDZ/C\naug-cc-pVDZ/C\n"
    reference_basis_block_with_ecp = "aug-cc-pVDZ/C\naug-cc-pVDZ/C\ncc-pVDZ/C\naug-cc-pVDZ/C\naug-cc-pVDZ/C\naug-cc-pVDZ/C\naug-cc-pVDZ/C\naug-cc-pVDZ/C\nno-basis-set\nno-basis-set\nno-basis-set\nno-basis-set\nno-basis-set\nno-basis-set\nno-basis-set\nno-basis-set\nno-basis-set\nno-basis-set\nno-basis-set\nno-basis-set\nno-basis-set\n"

    assert generated_basis_block_without_ecp == reference_basis_block_without_ecp
    assert generated_basis_block_with_ecp == reference_basis_block_with_ecp


def test_MRCCInputGenerator_generate_coords_block(mrcc_input_generator):
    mrcc_input_generator_nocp = deepcopy(mrcc_input_generator)

    mrcc_input_generator_nocp.include_cp = False
    mrcc_input_generator_nocp._generate_coords_block()

    mrcc_input_generator._generate_coords_block()

    reference_block_collated = {
        "adsorbate_slab": {
            "float": [
                21.0,
                -2.12018425659,
                -2.12018425659,
                -0.04367284424,
                0.0,
                0.0,
                -0.04269731856,
            ],
            "string": ["charge=-8", "C", "O", "Mg", "Mg", "Mg"],
        },
        "adsorbate": {
            "float": [8.0, -2.12018425659, -2.12018425659],
            "string": ["charge=0", "C", "O"],
        },
        "slab": {
            "float": [
                21.0,
                -2.12018425659,
                -2.12018425659,
                -0.04367284424,
                0.0,
                0.0,
                -0.04269731856,
            ],
            "string": ["charge=-8", "C", "O", "Mg", "Mg", "Mg"],
        },
    }

    reference_block_nocp_collated = {
        "adsorbate_slab": {
            "float": [
                21.0,
                -2.12018425659,
                -2.12018425659,
                -0.04367284424,
                0.0,
                0.0,
                -0.04269731856,
            ],
            "string": ["charge=-8", "C", "O", "Mg", "Mg", "Mg"],
        },
        "adsorbate": {"float": [2.0], "string": ["charge=0", "C"]},
        "slab": {
            "float": [19.0, 2.12018425659, 2.11144262254, -0.04367284424, 0.0, 0.0],
            "string": ["charge=-8", "Mg", "O", "Mg", "Mg"],
        },
    }

    generated_block_collated = {
        system: {"float": [], "string": []}
        for system in ["adsorbate_slab", "adsorbate", "slab"]
    }
    generated_block_nocp_collated = {
        system: {"float": [], "string": []}
        for system in ["adsorbate_slab", "adsorbate", "slab"]
    }

    for system in ["adsorbate_slab", "adsorbate", "slab"]:
        generated_block_collated[system]["float"] = [
            float(x)
            for x in mrcc_input_generator.skzcam_input_str[system].split()
            if x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ][::10]
        generated_block_collated[system]["string"] = [
            x
            for x in mrcc_input_generator.skzcam_input_str[system].split()
            if not x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ][::5]

        generated_block_nocp_collated[system]["float"] = [
            float(x)
            for x in mrcc_input_generator_nocp.skzcam_input_str[system].split()
            if x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ][::10]
        generated_block_nocp_collated[system]["string"] = [
            x
            for x in mrcc_input_generator_nocp.skzcam_input_str[system].split()
            if not x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ][::5]

        assert_equal(
            generated_block_collated[system]["string"],
            reference_block_collated[system]["string"],
        )
        assert_allclose(
            generated_block_collated[system]["float"],
            reference_block_collated[system]["float"],
            rtol=1e-05,
            atol=1e-07,
        )

        assert_equal(
            generated_block_nocp_collated[system]["string"],
            reference_block_nocp_collated[system]["string"],
        )
        assert_allclose(
            generated_block_nocp_collated[system]["float"],
            reference_block_nocp_collated[system]["float"],
            rtol=1e-05,
            atol=1e-07,
        )


def test_MRCCInputGenerator_generate_point_charge_block(mrcc_input_generator):
    generated_point_charge_block = mrcc_input_generator._generate_point_charge_block()

    generated_point_charge_block_shortened = [
        float(x) for x in generated_point_charge_block.split()[5::180]
    ]

    reference_point_charge_block_shortened = [
        -0.04367284424,
        -0.03992370948,
        -2.14923989662,
        -6.37814204923,
        -2.1415520695,
        -4.26789528527,
        -2.1415520695,
        -0.03992370948,
        0.0,
    ]

    assert_allclose(
        generated_point_charge_block_shortened,
        reference_point_charge_block_shortened,
        rtol=1e-05,
        atol=1e-07,
    )


def test_ORCAInputGenerator_init(adsorbate_slab_embedded_cluster, element_info):
    pal_nprocs_block = {"nprocs": 1, "maxcore": 5000}

    method_block = {"Method": "hf", "RI": "on", "RunTyp": "Energy"}

    scf_block = {
        "HFTyp": "rhf",
        "Guess": "MORead",
        "MOInp": '"orca_svp_start.gbw"',
        "SCFMode": "Direct",
        "sthresh": "1e-6",
        "AutoTRAHIter": 60,
        "MaxIter": 1000,
    }

    ecp_info = {
        "Mg": """NewECP
N_core 0
lmax f
s 1
1      1.732000000   14.676000000 2
p 1
1      1.115000000    5.175700000 2
d 1
1      1.203000000   -1.816000000 2
f 1
1      1.000000000    0.000000000 2
end"""
    }
    orca_input_generator = ORCAInputGenerator(
        adsorbate_slab_embedded_cluster=adsorbate_slab_embedded_cluster,
        quantum_cluster_indices=[0, 1, 2, 3, 4, 5, 6, 7],
        ecp_region_indices=[8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24],
        element_info=element_info,
        include_cp=True,
        pal_nprocs_block=pal_nprocs_block,
        method_block=method_block,
        scf_block=scf_block,
        ecp_info=ecp_info,
    )

    # Check when multiplicities is not provided
    assert orca_input_generator.multiplicities == {
        "adsorbate_slab": 1,
        "adsorbate": 1,
        "slab": 1,
    }

    orca_input_generator = ORCAInputGenerator(
        adsorbate_slab_embedded_cluster=adsorbate_slab_embedded_cluster,
        quantum_cluster_indices=[0, 1, 2, 3, 4, 5, 6, 7],
        ecp_region_indices=[8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24],
        element_info=element_info,
        include_cp=True,
        multiplicities={"adsorbate_slab": 3, "adsorbate": 1, "slab": 2},
        pal_nprocs_block=pal_nprocs_block,
        method_block=method_block,
        scf_block=scf_block,
        ecp_info=ecp_info,
    )

    assert not compare_atoms(
        orca_input_generator.adsorbate_slab_embedded_cluster,
        adsorbate_slab_embedded_cluster,
    )
    assert_equal(orca_input_generator.quantum_cluster_indices, [0, 1, 2, 3, 4, 5, 6, 7])
    assert_equal(orca_input_generator.adsorbate_indices, [0, 1])
    assert_equal(orca_input_generator.slab_indices, [2, 3, 4, 5, 6, 7])
    assert_equal(
        orca_input_generator.ecp_region_indices,
        [8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24],
    )
    assert orca_input_generator.element_info == element_info
    assert orca_input_generator.include_cp is True
    assert orca_input_generator.multiplicities == {
        "adsorbate_slab": 3,
        "adsorbate": 1,
        "slab": 2,
    }

    assert orca_input_generator.pal_nprocs_block == pal_nprocs_block
    assert orca_input_generator.method_block == method_block
    assert orca_input_generator.scf_block == scf_block
    assert orca_input_generator.ecp_info == ecp_info

    # Check if error raise if quantum_cluster_indices and ecp_region_indices overlap

    with pytest.raises(
        ValueError, match="An atom in the quantum cluster is also in the ECP region."
    ):
        orca_input_generator = ORCAInputGenerator(
            adsorbate_slab_embedded_cluster=adsorbate_slab_embedded_cluster,
            quantum_cluster_indices=[0, 1, 2, 3, 4, 5, 6, 7],
            ecp_region_indices=[7, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24],
            element_info=element_info,
            include_cp=True,
            multiplicities={"adsorbate_slab": 3, "adsorbate": 1, "slab": 2},
            pal_nprocs_block=pal_nprocs_block,
            method_block=method_block,
            scf_block=scf_block,
            ecp_info=ecp_info,
        )


def test_ORCAInputGenerator_generate_input(orca_input_generator):
    orca_input_generator_nocp = deepcopy(orca_input_generator)

    orca_input_generator_nocp.include_cp = False
    orca_input_generator_nocp.generate_input()

    orca_input_generator.generate_input()

    reference_block_collated = {
        "adsorbate_slab": {
            "float": [1.0, 2.0, 1.0, 0.0, 2.0],
            "string": [
                "%pal",
                "NewNCore",
                "O",
                '"aug-cc-pVDZ/C"',
                '"orca_svp_start.gbw"',
                "O",
                "Mg>",
                "d",
                "f",
                "NewECP",
                "f",
                "s",
                "N_core",
                "end",
            ],
        },
        "adsorbate": {
            "float": [1.0],
            "string": ["%pal", "end", "end", "NewAuxCGTO", "Direct", "O:"],
        },
        "slab": {
            "float": [1.0, 2.0, 1.0, 0.0, 2.0],
            "string": [
                "%pal",
                "NewNCore",
                "O",
                '"aug-cc-pVDZ/C"',
                '"orca_svp_start.gbw"',
                "O:",
                "Mg>",
                "d",
                "f",
                "NewECP",
                "f",
                "s",
                "N_core",
                "end",
            ],
        },
    }

    reference_block_nocp_collated = {
        "adsorbate_slab": {
            "float": [1.0, 2.0, 1.0, 0.0, 2.0],
            "string": [
                "%pal",
                "NewNCore",
                "O",
                '"aug-cc-pVDZ/C"',
                '"orca_svp_start.gbw"',
                "O",
                "Mg>",
                "d",
                "f",
                "NewECP",
                "f",
                "s",
                "N_core",
                "end",
            ],
        },
        "adsorbate": {
            "float": [1.0],
            "string": ["%pal", "end", "end", "NewAuxCGTO", "Direct", "end"],
        },
        "slab": {
            "float": [1.0, 2.0, 2.10705287155, 0.0, 1.0],
            "string": [
                "%pal",
                "NewNCore",
                "O",
                '"aug-cc-pVDZ/C"',
                '"orca_svp_start.gbw"',
                "O",
                "N_core",
                "end",
                "p",
                "lmax",
                "Mg>",
                "d",
                "f",
                "end",
            ],
        },
    }

    generated_block_collated = {
        system: {"float": [], "string": []}
        for system in ["adsorbate_slab", "adsorbate", "slab"]
    }
    generated_block_nocp_collated = {
        system: {"float": [], "string": []}
        for system in ["adsorbate_slab", "adsorbate", "slab"]
    }

    for system in ["adsorbate_slab", "adsorbate", "slab"]:
        generated_block_collated[system]["float"] = [
            float(x)
            for x in orca_input_generator.orcablocks[system].split()
            if x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ][::77]
        generated_block_collated[system]["string"] = [
            x
            for x in orca_input_generator.orcablocks[system].split()
            if not x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ][::17]
        assert_equal(
            reference_block_collated[system]["string"],
            generated_block_collated[system]["string"],
        )
        assert_allclose(
            generated_block_collated[system]["float"],
            reference_block_collated[system]["float"],
            rtol=1e-05,
            atol=1e-07,
        )

        generated_block_nocp_collated[system]["float"] = [
            float(x)
            for x in orca_input_generator_nocp.orcablocks[system].split()
            if x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ][::77]
        generated_block_nocp_collated[system]["string"] = [
            x
            for x in orca_input_generator_nocp.orcablocks[system].split()
            if not x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ][::17]

        assert_equal(
            reference_block_nocp_collated[system]["string"],
            generated_block_nocp_collated[system]["string"],
        )
        assert_allclose(
            generated_block_nocp_collated[system]["float"],
            reference_block_nocp_collated[system]["float"],
            rtol=1e-05,
            atol=1e-07,
        )


def test_create_atom_coord_string(adsorbate_slab_embedded_cluster):
    atom = adsorbate_slab_embedded_cluster[0]

    # First let's try the case where it's a normal atom.
    atom_coord_string = create_atom_coord_string(atom=atom)

    with pytest.raises(
        ValueError, match="ECP info cannot be provided for ghost atoms."
    ):
        create_atom_coord_string(
            atom, atom_ecp_info="NewECP\nECP_info1\nECP_info2\n", is_ghost_atom=True
        )

    with pytest.raises(
        ValueError, match="Point charge value must be given for atoms with ECP info."
    ):
        create_atom_coord_string(atom, atom_ecp_info="NewECP\nECP_info1\nECP_info2\n")

    assert (
        atom_coord_string
        == "C                       0.00000000000    0.00000000000    2.00000000000\n"
    )

    # Let's now try the case where it is a ghost atom.
    atom_coord_string = create_atom_coord_string(atom=atom, is_ghost_atom=True)
    assert (
        atom_coord_string
        == "C:                      0.00000000000    0.00000000000    2.00000000000\n"
    )

    # Let's now try the case where it is an atom in the ECP region.
    atom_coord_string = create_atom_coord_string(
        atom=atom, atom_ecp_info="NewECP\nECP_info1\nECP_info2\n", pc_charge=2.0
    )
    assert (
        atom_coord_string
        == "C>     2.00000000000    0.00000000000    0.00000000000    2.00000000000\nNewECP\nECP_info1\nECP_info2\n"
    )


def test_ORCAInputGenerator_generate_coords_block(orca_input_generator):
    orca_input_generator_nocp = deepcopy(orca_input_generator)

    orca_input_generator_nocp.include_cp = False
    orca_input_generator_nocp._generate_coords_block()

    orca_input_generator._generate_coords_block()

    reference_block_collated = {
        "adsorbate_slab": {
            "float": [3.0, 1.0, 5.1757, 1.0, 0.0, 2.0, 1.0],
            "string": [
                "%coords",
                "coords",
                "O",
                "s",
                "N_core",
                "end",
                "p",
                "lmax",
                "Mg>",
                "d",
                "f",
                "NewECP",
                "f",
                "s",
                "N_core",
                "end",
                "p",
                "lmax",
                "Mg>",
                "d",
                "f",
                "end",
            ],
        },
        "adsorbate": {"float": [1.0], "string": ["%coords", "coords", "O:"]},
        "slab": {
            "float": [2.0, 1.0, 5.1757, 1.0, 0.0, 2.0, 1.0],
            "string": [
                "%coords",
                "coords",
                "O",
                "s",
                "N_core",
                "end",
                "p",
                "lmax",
                "Mg>",
                "d",
                "f",
                "NewECP",
                "f",
                "s",
                "N_core",
                "end",
                "p",
                "lmax",
                "Mg>",
                "d",
                "f",
                "end",
            ],
        },
    }

    reference_block_nocp_collated = {
        "adsorbate_slab": {
            "float": [3.0, 1.0, 5.1757, 1.0, 0.0, 2.0, 1.0],
            "string": [
                "%coords",
                "coords",
                "O",
                "s",
                "N_core",
                "end",
                "p",
                "lmax",
                "Mg>",
                "d",
                "f",
                "NewECP",
                "f",
                "s",
                "N_core",
                "end",
                "p",
                "lmax",
                "Mg>",
                "d",
                "f",
                "end",
            ],
        },
        "adsorbate": {"float": [1.0], "string": ["%coords", "coords"]},
        "slab": {
            "float": [2.0, 1.115, 2.0, 2.10705287155, 14.676, 1.0, 1.0],
            "string": [
                "%coords",
                "coords",
                "Mg>",
                "d",
                "f",
                "NewECP",
                "f",
                "s",
                "N_core",
                "end",
                "p",
                "lmax",
                "Mg>",
                "d",
                "f",
                "NewECP",
                "f",
                "s",
                "N_core",
                "end",
                "p",
            ],
        },
    }

    generated_block_collated = {
        system: {"float": [], "string": []}
        for system in ["adsorbate_slab", "adsorbate", "slab"]
    }
    generated_block_nocp_collated = {
        system: {"float": [], "string": []}
        for system in ["adsorbate_slab", "adsorbate", "slab"]
    }

    for system in ["adsorbate_slab", "adsorbate", "slab"]:
        generated_block_collated[system]["float"] = [
            float(x)
            for x in orca_input_generator.orcablocks[system].split()
            if x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ][::57]
        generated_block_collated[system]["string"] = [
            x
            for x in orca_input_generator.orcablocks[system].split()
            if not x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ][::7]

        assert_equal(
            reference_block_collated[system]["string"],
            generated_block_collated[system]["string"],
        )
        assert_allclose(
            generated_block_collated[system]["float"],
            reference_block_collated[system]["float"],
            rtol=1e-05,
            atol=1e-07,
        )

        generated_block_nocp_collated[system]["float"] = [
            float(x)
            for x in orca_input_generator_nocp.orcablocks[system].split()
            if x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ][::57]
        generated_block_nocp_collated[system]["string"] = [
            x
            for x in orca_input_generator_nocp.orcablocks[system].split()
            if not x.replace(".", "", 1).replace("-", "", 1).isdigit()
        ][::7]

        assert_equal(
            reference_block_nocp_collated[system]["string"],
            generated_block_nocp_collated[system]["string"],
        )
        assert_allclose(
            generated_block_nocp_collated[system]["float"],
            reference_block_nocp_collated[system]["float"],
            rtol=1e-05,
            atol=1e-07,
        )


def test_ORCAInputGenerator_format_ecp_info(orca_input_generator):
    with pytest.raises(
        ValueError, match="ECP info does not contain 'NewECP' or 'end' keyword."
    ):
        orca_input_generator._format_ecp_info(atom_ecp_info="dummy_info\nN_core0\nend")

    atom_ecp_info = """
NewECP
N_core 0
lmax s
s 1
1      1.732000000   14.676000000 2
end
"""
    formatted_atom_ecp_info = orca_input_generator._format_ecp_info(
        atom_ecp_info=atom_ecp_info
    )
    assert (
        formatted_atom_ecp_info
        == "NewECP\nN_core 0\nlmax s\ns 1\n1      1.732000000   14.676000000 2\nend\n"
    )


def test_ORCAInputGenerator_generate_preamble_block(orca_input_generator):
    # Make copy of orca_input_generator for further tests
    orca_input_generator_1 = deepcopy(orca_input_generator)
    orca_input_generator_2 = deepcopy(orca_input_generator)
    orca_input_generator_3 = deepcopy(orca_input_generator)

    # Generate the orca input preamble
    orca_input_generator_1._generate_preamble_block()

    assert (
        orca_input_generator_1.orcablocks["adsorbate_slab"]
        == '%pal nprocs 1 end\n%maxcore 5000 end\n%pointcharges "orca.pc"\n%method\nMethod hf\nRI on\nRunTyp Energy\nNewNCore C 2 end\nNewNCore Mg 2 end\nNewNCore O 2 end\nend\n%basis\nNewGTO C "aug-cc-pVDZ" end\nNewGTO Mg "cc-pVDZ" end\nNewGTO O "aug-cc-pVDZ" end\nNewAuxJGTO C "def2/J" end\nNewAuxJGTO Mg "def2/J" end\nNewAuxJGTO O "def2/JK" end\nNewAuxCGTO C "aug-cc-pVDZ/C" end\nNewAuxCGTO Mg "cc-pVDZ/C" end\nNewAuxCGTO O "aug-cc-pVDZ/C" end\nend\n%scf\nHFTyp rhf\nGuess MORead\nMOInp "orca_svp_start.gbw"\nSCFMode Direct\nsthresh 1e-6\nAutoTRAHIter 60\nMaxIter 1000\nend\n'
    )

    assert (
        orca_input_generator_1.orcablocks["adsorbate"]
        == '%pal nprocs 1 end\n%maxcore 5000 end\n%method\nMethod hf\nRI on\nRunTyp Energy\nNewNCore C 2 end\nNewNCore Mg 2 end\nNewNCore O 2 end\nend\n%basis\nNewGTO C "aug-cc-pVDZ" end\nNewGTO Mg "cc-pVDZ" end\nNewGTO O "aug-cc-pVDZ" end\nNewAuxJGTO C "def2/J" end\nNewAuxJGTO Mg "def2/J" end\nNewAuxJGTO O "def2/JK" end\nNewAuxCGTO C "aug-cc-pVDZ/C" end\nNewAuxCGTO Mg "cc-pVDZ/C" end\nNewAuxCGTO O "aug-cc-pVDZ/C" end\nend\n%scf\nHFTyp rhf\nGuess MORead\nMOInp "orca_svp_start.gbw"\nSCFMode Direct\nsthresh 1e-6\nAutoTRAHIter 60\nMaxIter 1000\nend\n'
    )
    assert (
        orca_input_generator_1.orcablocks["adsorbate_slab"]
        == orca_input_generator_1.orcablocks["slab"]
    )

    # Check the case if the element_info has all of the same values
    element_info = {
        "C": {
            "basis": "def2-SVP",
            "core": 2,
            "ri_scf_basis": "def2/J",
            "ri_cwft_basis": "def2-SVP/C",
        },
        "O": {
            "basis": "def2-SVP",
            "core": 2,
            "ri_scf_basis": "def2/J",
            "ri_cwft_basis": "def2-SVP/C",
        },
        "Mg": {
            "basis": "def2-SVP",
            "core": 2,
            "ri_scf_basis": "def2/J",
            "ri_cwft_basis": "def2-SVP/C",
        },
    }
    orca_input_generator_2.element_info = element_info
    orca_input_generator_2._generate_preamble_block()

    assert (
        orca_input_generator_2.orcablocks["adsorbate_slab"]
        == '%pal nprocs 1 end\n%maxcore 5000 end\n%pointcharges "orca.pc"\n%method\nMethod hf\nRI on\nRunTyp Energy\nNewNCore C 2 end\nNewNCore Mg 2 end\nNewNCore O 2 end\nend\n%basis\nBasis def2-SVP\nAux def2/J\nAuxC def2-SVP/C\nend\n%scf\nHFTyp rhf\nGuess MORead\nMOInp "orca_svp_start.gbw"\nSCFMode Direct\nsthresh 1e-6\nAutoTRAHIter 60\nMaxIter 1000\nend\n'
    )

    # Testing the case if we provide no blocks
    orca_input_generator_3.scf_block = None
    orca_input_generator_3.method_block = None
    orca_input_generator_3.pal_nprocs_block = None
    orca_input_generator_3.element_info = None
    orca_input_generator_3._generate_preamble_block()

    assert (
        orca_input_generator_3.orcablocks["adsorbate_slab"]
        == '%pointcharges "orca.pc"\n'
    )

    # Check whether error raised if not all element_info is provided
    element_info_error = {"C": element_info["C"]}
    orca_input_generator_3.element_info = element_info_error
    with pytest.raises(
        ValueError,
        match="Not all element symbols are provided in the element_info dictionary.",
    ):
        orca_input_generator_3._generate_preamble_block()


def test_ORCAInputGenerator_create_point_charge_file(orca_input_generator, tmp_path):
    # Create the point charge file
    orca_input_generator.create_point_charge_file(pc_file=tmp_path / "orca.pc")

    # Read the written file
    orca_pc_file = np.loadtxt(tmp_path / "orca.pc", skiprows=1)

    # Check that the contents of the file match the reference
    assert len(orca_pc_file) == 371

    assert_allclose(
        orca_pc_file[::30],
        np.array(
            [
                [-2.00000000e00, -2.11070451e00, 2.11070451e00, -2.14923990e00],
                [2.00000000e00, 2.11024676e00, -2.11024676e00, -4.26789529e00],
                [2.00000000e00, 6.32954443e00, 2.11144262e00, -4.36728442e-02],
                [-2.00000000e00, -4.22049353e00, 6.32889566e00, 7.72802266e-03],
                [2.00000000e00, -6.33074029e00, -2.11024676e00, -4.26789529e00],
                [-2.00000000e00, 4.22049353e00, -6.33074029e00, -4.26789529e00],
                [-2.00000000e00, 6.33074029e00, 2.11024676e00, -6.37814205e00],
                [-2.00000000e00, 2.11024676e00, -8.44098706e00, -4.26789529e00],
                [-2.00000000e00, -8.44098706e00, -6.32080280e00, 5.67209089e-03],
                [2.00000000e00, -2.11024676e00, 8.44098706e00, -6.37814205e00],
                [8.00000000e-01, -4.64254288e01, 3.79844418e01, -3.99237095e-02],
                [3.12302613e00, -0.00000000e00, -5.71441194e01, -2.36698692e01],
                [2.10472999e00, -2.36698692e01, 5.71441194e01, 2.59086514e01],
            ]
        ),
        rtol=1e-05,
        atol=1e-07,
    )


def test_CreateSKZCAMClusters_init():
    skzcam_clusters = CreateSKZCAMClusters(
        adsorbate_indices=[0, 1],
        slab_center_indices=[32],
        atom_oxi_states={"Mg": 2.0, "O": -2.0},
        adsorbate_slab_file=Path(FILE_DIR, "skzcam_files", "CO_MgO.poscar.gz"),
        pun_file="test.pun",
    )

    assert_equal(skzcam_clusters.adsorbate_indices, [0, 1])
    assert skzcam_clusters.slab_center_indices == [32]
    assert skzcam_clusters.atom_oxi_states == {"Mg": 2.0, "O": -2.0}
    assert skzcam_clusters.adsorbate_slab_file == Path(
        FILE_DIR, "skzcam_files", "CO_MgO.poscar.gz"
    )
    assert skzcam_clusters.pun_file == "test.pun"

    # Check if error raised if adsorbate_indices and slab_center_indices overlap
    with pytest.raises(
        ValueError, match="The adsorbate and slab center indices cannot be the same."
    ):
        skzcam_clusters = CreateSKZCAMClusters(
            adsorbate_indices=[0, 1],
            slab_center_indices=[0],
            atom_oxi_states={"Mg": 2.0, "O": -2.0},
            adsorbate_slab_file=Path(FILE_DIR, "skzcam_files", "CO_MgO.poscar.gz"),
            pun_file="test.pun",
        )

    # Check if error raised if both adsorbate_slab_file and pun_file are None
    with pytest.raises(
        ValueError, match="Either the adsorbate_slab_file or pun_file must be provided."
    ):
        skzcam_clusters = CreateSKZCAMClusters(
            adsorbate_indices=[0, 1],
            slab_center_indices=[32],
            atom_oxi_states={"Mg": 2.0, "O": -2.0},
            adsorbate_slab_file=None,
            pun_file=None,
        )


def test_CreateSKZCAMClusters_run_chemshell(skzcam_clusters, tmp_path):
    # Test if xyz file doesn't get written when write_xyz_file=False
    skzcam_clusters_nowrite = deepcopy(skzcam_clusters)
    skzcam_clusters_nowrite.convert_slab_to_atoms()
    skzcam_clusters_nowrite.run_chemshell(
        filepath=tmp_path / "ChemShell_Cluster.pun",
        chemsh_radius_active=5.0,
        chemsh_radius_cluster=10.0,
        write_xyz_file=False,
    )
    assert not os.path.isfile(tmp_path / "ChemShell_Cluster.xyz")

    skzcam_clusters.convert_slab_to_atoms()
    skzcam_clusters.run_chemshell(
        filepath=tmp_path / "ChemShell_Cluster.pun",
        chemsh_radius_active=5.0,
        chemsh_radius_cluster=10.0,
        write_xyz_file=True,
    )

    # Read the output .xyz file
    chemshell_embedded_cluster = read(tmp_path / "ChemShell_Cluster.xyz")

    # Check that the positions and atomic numbers match reference
    assert_allclose(
        chemshell_embedded_cluster.get_positions()[::100],
        np.array(
            [
                [0.00000000e00, 0.00000000e00, -7.72802046e-03],
                [-2.11024616e00, 2.11024616e00, -6.38586825e00],
                [6.33073849e00, -2.11024616e00, -6.38586825e00],
                [-1.09499282e01, -4.53560876e00, 4.95687508e00],
            ]
        ),
        rtol=1e-05,
        atol=1e-07,
    )

    assert_equal(
        chemshell_embedded_cluster.get_atomic_numbers()[::40].tolist(),
        [12, 12, 12, 8, 8, 8, 12, 9, 9],
    )


def test_CreateSKZCAMClusters_convert_pun_to_atoms(skzcam_clusters):
    slab_embedded_cluster = skzcam_clusters._convert_pun_to_atoms(
        pun_file=Path(FILE_DIR, "skzcam_files", "ChemShell_Cluster.pun.gz")
    )

    # Check that number of atoms matches our reference
    assert len(slab_embedded_cluster) == 390

    # Check that last 10 elements of the oxi_state match our reference
    assert_allclose(
        slab_embedded_cluster.get_array("oxi_states")[-10:],
        np.array(
            [
                -0.80812511,
                2.14427889,
                -0.96000248,
                2.14427887,
                -0.8081251,
                2.10472993,
                -0.89052904,
                2.10472993,
                -0.8081251,
                2.14427887,
            ]
        ),
        rtol=1e-05,
        atol=1e-07,
    )

    # Check that first 10 elements of atom_type array match our reference
    assert_equal(
        slab_embedded_cluster.get_array("atom_type")[:10],
        [
            "cation",
            "anion",
            "anion",
            "anion",
            "anion",
            "anion",
            "cation",
            "cation",
            "cation",
            "cation",
        ],
    )

    # Check that the positions of the atom matches
    assert_allclose(
        slab_embedded_cluster[200].position,
        np.array([6.33074029, -2.11024676, -6.37814205]),
        rtol=1e-05,
        atol=1e-07,
    )


def test_CreateSKZCAMClusters_convert_slab_to_atoms():
    # Test for CO on MgO example
    skzcam_clusters = CreateSKZCAMClusters(
        adsorbate_indices=[0, 1],
        slab_center_indices=[32],
        atom_oxi_states={"Mg": 2.0, "O": -2.0},
        adsorbate_slab_file=Path(FILE_DIR, "skzcam_files", "CO_MgO.poscar.gz"),
        pun_file=None,
    )
    skzcam_clusters.convert_slab_to_atoms()

    # Check adsorbate matches reference
    assert_allclose(
        skzcam_clusters.adsorbate.get_positions(),
        np.array([[0.0, 0.0, 2.44102236], [0.0, 0.0, 3.58784217]]),
        rtol=1e-05,
        atol=1e-07,
    )
    assert_equal(skzcam_clusters.adsorbate.get_atomic_numbers().tolist(), [6, 8])

    # Check slab matches reference
    assert_allclose(
        skzcam_clusters.slab.get_positions()[::10],
        np.array(
            [
                [0.00000000e00, 0.00000000e00, 0.00000000e00],
                [-2.11024616e00, 0.00000000e00, -6.37814023e00],
                [2.11024616e00, 2.11024616e00, -4.26789407e00],
                [2.10705227e00, 0.00000000e00, -2.14155146e00],
                [-4.22049233e00, -2.11024616e00, -4.26789407e00],
                [0.00000000e00, -4.22049233e00, -6.37814023e00],
                [0.00000000e00, -2.12018365e00, 5.67208927e-03],
            ]
        ),
        rtol=1e-05,
        atol=1e-07,
    )
    assert_equal(
        skzcam_clusters.slab.get_atomic_numbers().tolist()[::10],
        [12, 12, 12, 12, 8, 8, 8],
    )

    # Check center_position matches reference
    assert_allclose(
        skzcam_clusters.center_position,
        np.array([0.0, 0.0, 3.09607306]),
        rtol=1e-05,
        atol=1e-07,
    )

    # Check vector distance of adsorbate from first center atom (corresponding to first atom index) of slab matches reference
    assert_allclose(
        skzcam_clusters.adsorbate_vector_from_slab,
        np.array([0.0, 0.0, 2.44102236]),
        rtol=1e-05,
        atol=1e-07,
    )

    # Test for NO on MgO example
    skzcam_clusters = CreateSKZCAMClusters(
        adsorbate_indices=[0, 1],
        slab_center_indices=[32, 33],
        atom_oxi_states={"Mg": 2.0, "O": -2.0},
        adsorbate_slab_file=Path(FILE_DIR, "skzcam_files", "NO_MgO.poscar.gz"),
        pun_file=None,
    )
    skzcam_clusters.convert_slab_to_atoms()

    # Check adsorbate matches reference
    assert_allclose(
        skzcam_clusters.adsorbate.get_positions(),
        np.array(
            [[1.18932285, -0.14368533, 2.0777825], [1.68437633, 0.62999818, 2.83068075]]
        ),
        rtol=1e-05,
        atol=1e-07,
    )
    assert_equal(skzcam_clusters.adsorbate.get_atomic_numbers().tolist(), [7, 8])

    # Check slab matches reference
    assert_allclose(
        skzcam_clusters.slab.get_positions()[::10],
        np.array(
            [
                [0.0, 0.0, 0.0],
                [-4.2019821, -2.10867761, -6.39202884],
                [0.01851023, -4.21892378, -4.28178268],
                [0.01903204, -2.105465, -2.15224877],
                [-4.2019821, -2.10867761, -4.28178268],
                [0.01851023, -4.21892378, -6.39202884],
                [0.01900061, -2.11652633, -0.03021786],
            ]
        ),
        rtol=1e-05,
        atol=1e-07,
    )
    assert_equal(
        skzcam_clusters.slab.get_atomic_numbers().tolist()[::10],
        [12, 12, 12, 12, 8, 8, 8],
    )

    # Check center_position matches reference
    assert_allclose(
        skzcam_clusters.center_position,
        np.array([1.06307888, -1.06176564, 2.47922285]),
        rtol=1e-05,
        atol=1e-07,
    )

    # Check vector distance of adsorbate from first center atom (corresponding to first atom index) of slab matches reference
    assert_allclose(
        skzcam_clusters.adsorbate_vector_from_slab,
        np.array([1.18932285, -0.14368533, 2.0777825]),
        rtol=1e-05,
        atol=1e-07,
    )


def test_CreateSKZCAMClusters_find_cation_shells(
    skzcam_clusters, slab_embedded_cluster
):
    # Get distance of atoms from the center
    distances = _get_atom_distances(
        atoms=slab_embedded_cluster, center_position=[0, 0, 2]
    )

    # Find the cation shells from the distances
    cation_shells_distances, cation_shells_idx = skzcam_clusters._find_cation_shells(
        slab_embedded_cluster=slab_embedded_cluster,
        distances=distances,
        shell_width=0.005,
    )

    # As these list of lists do not have the same length, we flatten first 5 lists into a 1D list for comparison
    cation_shells_distances_flatten = [
        item for row in cation_shells_distances[:5] for item in row
    ]
    cation_shells_idx_flatten = [item for row in cation_shells_idx[:5] for item in row]

    # Check that these lists are correct
    assert_allclose(
        cation_shells_distances_flatten,
        np.array(
            [
                2.0,
                3.6184221134101624,
                3.6184221134101655,
                3.6184221134101655,
                3.6184221134101686,
                4.646732760541734,
                4.646732760541734,
                4.646732760541736,
                4.646732760541736,
                4.6888354582307805,
                4.6888354582307805,
                4.6888354582307805,
                4.6888354582307805,
                6.267895285274443,
            ]
        ),
        rtol=1e-05,
        atol=1e-07,
    )

    assert_equal(
        cation_shells_idx_flatten, [0, 9, 8, 6, 7, 11, 12, 10, 13, 19, 21, 18, 20, 22]
    )


def test_CreateSKZCAMClusters_get_anion_coordination(
    skzcam_clusters, slab_embedded_cluster, distance_matrix
):
    # Get the anions for the second SKZCAM shell
    anion_shell_idx = skzcam_clusters._get_anion_coordination(
        slab_embedded_cluster=slab_embedded_cluster,
        cation_shell_indices=[9, 8, 6, 7],
        dist_matrix=distance_matrix,
    )

    # Check anion indices match with reference
    assert_equal(
        anion_shell_idx, [1, 2, 3, 4, 14, 15, 16, 17, 23, 24, 25, 26, 27, 28, 29, 30]
    )


def test_CreateSKZCAMClusters_get_ecp_region(
    skzcam_clusters, slab_embedded_cluster, distance_matrix
):
    # Find the ECP region for the first cluster
    ecp_region_idx = skzcam_clusters._get_ecp_region(
        slab_embedded_cluster=slab_embedded_cluster,
        quantum_cluster_indices_set=[[0, 1, 2, 3, 4, 5]],
        dist_matrix=distance_matrix,
        ecp_dist=3,
    )

    # Check ECP region indices match with reference
    assert_equal(ecp_region_idx[0], [6, 7, 8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22])


def test_CreateSKZCAMClusters_create_adsorbate_slab_embedded_cluster(
    skzcam_clusters, slab_embedded_cluster
):
    skzcam_clusters.slab_embedded_cluster = slab_embedded_cluster
    skzcam_clusters.adsorbate = Atoms(
        "CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]], pbc=[False, False, False]
    )
    skzcam_clusters.adsorbate_vector_from_slab = [0.0, 0.0, 2.0]

    skzcam_clusters._create_adsorbate_slab_embedded_cluster(
        quantum_cluster_indices_set=[[0, 1, 3, 4], [5, 6, 7, 8]],
        ecp_region_indices_set=[[0, 1, 3, 4], [5, 6, 7, 8]],
    )

    # Check that the positions of the first 10 atoms of the embedded cluster matches the reference positions, oxi_states and atom_type
    assert_allclose(
        skzcam_clusters.adsorbate_slab_embedded_cluster.get_positions()[:10],
        np.array(
            [
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 3.128],
                [0.0, 0.0, 0.0],
                [-2.12018426, 0.0, 0.00567209],
                [0.0, 2.12018426, 0.00567209],
                [2.12018426, 0.0, 0.00567209],
                [0.0, -2.12018426, 0.00567209],
                [0.0, 0.0, -2.14129966],
                [-2.11144262, 2.11144262, -0.04367284],
                [2.11144262, 2.11144262, -0.04367284],
            ]
        ),
        rtol=1e-05,
        atol=1e-07,
    )

    assert_equal(
        skzcam_clusters.adsorbate_slab_embedded_cluster.get_chemical_symbols()[:10],
        ["C", "O", "Mg", "O", "O", "O", "O", "O", "Mg", "Mg"],
    )
    assert_allclose(
        skzcam_clusters.adsorbate_slab_embedded_cluster.get_array("oxi_states")[:10],
        np.array([0.0, 0.0, 2.0, -2.0, -2.0, -2.0, -2.0, -2.0, 2.0, 2.0]),
        rtol=1e-05,
        atol=1e-07,
    )
    assert_equal(
        skzcam_clusters.adsorbate_slab_embedded_cluster.get_array("atom_type")[:10],
        [
            "adsorbate",
            "adsorbate",
            "cation",
            "anion",
            "anion",
            "anion",
            "anion",
            "anion",
            "cation",
            "cation",
        ],
    )

    # Check that the quantum_idx and ecp_idx match the reference
    assert_equal(
        skzcam_clusters.quantum_cluster_indices_set,
        [[0, 1, 2, 3, 5, 6], [0, 1, 7, 8, 9, 10]],
    )
    assert_equal(skzcam_clusters.ecp_region_indices_set, [[2, 3, 5, 6], [7, 8, 9, 10]])


def test_CreateSKZCAMClusters_run_skzcam(skzcam_clusters, tmp_path):
    # Get quantum cluster and ECP region indices
    skzcam_clusters.center_position = [0, 0, 2]
    skzcam_clusters.pun_file = Path(
        FILE_DIR, "skzcam_files", "ChemShell_Cluster.pun.gz"
    )
    skzcam_clusters.adsorbate = Atoms(
        "CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]], pbc=[False, False, False]
    )
    skzcam_clusters.adsorbate_vector_from_slab = [0.0, 0.0, 2.0]

    skzcam_clusters.run_skzcam(
        shell_max=2,
        ecp_dist=3.0,
        shell_width=0.005,
        write_clusters=True,
        write_clusters_path=tmp_path,
    )

    # Check quantum cluster indices match with reference
    assert_equal(
        skzcam_clusters.quantum_cluster_indices_set[1],
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            16,
            17,
            18,
            19,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
        ],
    )

    # Check ECP region indices match with reference
    assert_equal(
        skzcam_clusters.ecp_region_indices_set[1],
        [
            12,
            13,
            14,
            15,
            20,
            21,
            22,
            23,
            24,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            78,
            79,
            80,
            81,
            82,
            83,
            84,
            85,
        ],
    )
    # Read the written output and check that it matches with the reference positions and atomic numbers
    skzcam_cluster_xyz = read(tmp_path / "SKZCAM_cluster_0.xyz")

    assert_allclose(
        skzcam_cluster_xyz.get_positions(),
        np.array(
            [
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 3.128],
                [0.0, 0.0, 0.0],
                [-2.12018426, 0.0, 0.00567209],
                [0.0, 2.12018426, 0.00567209],
                [2.12018426, 0.0, 0.00567209],
                [0.0, -2.12018426, 0.00567209],
                [0.0, 0.0, -2.14129966],
            ]
        ),
        rtol=1e-04,
        atol=1e-07,
    )

    assert_equal(
        skzcam_cluster_xyz.get_atomic_numbers().tolist(), [6, 8, 12, 8, 8, 8, 8, 8]
    )


def test_get_atom_distances():
    # Creating a H2 molecule as an Atoms object
    h2_molecule = Atoms("H2", positions=[(0, 0, 0), (0, 0, 2)])

    # Run _get_atom_distances function to get distance of h2 molecule atoms from a center position
    atom_distances = _get_atom_distances(atoms=h2_molecule, center_position=[2, 0, 0])

    assert_allclose(atom_distances, np.array([2.0, 2.82842712]), rtol=1e-05, atol=1e-07)
