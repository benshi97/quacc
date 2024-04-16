"""
A wrapper around ASE's ORCA calculator that makes it better suited for
high-throughput DFT with multi-level schemes.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.calculators.orca import ORCA as ORCA_
from ase.calculators.orca import OrcaTemplate
from ase.utils import writer


class ORCA(ORCA_):
    """
    This is a wrapper around the ASE ORCA calculator that allows retains all the
    functionality of the original ORCA calculator whilst adding the ability to
    make dangling bonds of pseudo-hydrogens if the atoms object contains the arrays
    "oxi_states" and "atom_type".

    Returns
    -------
    Atoms
        The ASE Atoms object with attached ORCA calculator.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        profile = kwargs.get("profile", None)
        directory = kwargs.get("directory", ".")
        parameters = kwargs.copy()
        parameters.pop("profile", None)
        parameters.pop("directory", None)

        def write_input(inner_self, directory, atoms, parameters, properties):
            parameters = dict(parameters)["parameters"]

            kw = {
                "charge": 0,
                "mult": 1,
                "orcasimpleinput": "B3LYP def2-TZVP",
                "orcablocks": "%pal nprocs 1 end",
            }
            kw.update(parameters)

            write_orca(directory / inner_self.input_file, atoms, kw)

        OrcaTemplate.write_input = write_input

        super().__init__(
            template=OrcaTemplate(),
            profile=profile,
            directory=directory,
            parameters=parameters,
        )


@writer
def write_orca(fd, atoms, params):
    """
    Function to write out the ORCA input file. This is a modified version of the
    write_input function in the ASE ORCA calculator. It allows for the use of
    pseudo-hydrogens to represent dangling bonds.

    Parameters
    ----------
    fd : file descriptor
        File descriptor to write to
    atoms : Atoms
        ASE Atoms object
    params : dict
        Dictionary of parameters to write out for ORCA. This can be charge, mult, orcasimpleinput, orcablocks.
    """

    # Write out the orca simple input and the orca blocks
    fd.write("! %s \n" % params["orcasimpleinput"])
    fd.write("%s \n" % params["orcablocks"])

    # If the atoms object has the arrays "oxi_states" and "atom_type" then we
    # need to add the point charges with the corresponding oxi_state value as
    # its charge and H> to create an embedding potential (with ghost basis)
    if "oxi_states" in atoms.arrays and "atom_type" in atoms.arrays:
        fd.write("%coords\n")
        fd.write("CTyp xyz\n")
        fd.write("Units angs\n")
        fd.write("coords\n")
        # Sum up cluster charge from the charges on the dangling bonds
        cluster_charge = 0
        oxi_states = atoms.get_array("oxi_states")
        atom_type = atoms.get_array("atom_type")
        for index, atom in enumerate(atoms):
            # If atom_type is 'pH', then it is a dangling bond
            if atom_type[index] == "pH":  # 71 is ascii G (Ghost)
                cluster_charge += oxi_states[index]
                symbol = atom.symbol + ">  "
                fd.write(
                    symbol
                    + "  "
                    + str(oxi_states[index])
                    + "  "
                    + str(atom.position[0])
                    + "  "
                    + str(atom.position[1])
                    + "  "
                    + str(atom.position[2])
                    + "\n"
                )
                # Add a fake ECP that does nothing
                fd.write("""NewECP
N_core 0
  lmax p
  s 0
end
""")
            # Standard approach to write out atom if not a dangling bond
            else:
                symbol = atom.symbol + "   "
                fd.write(
                    symbol
                    + "  "
                    + str(atom.position[0])
                    + "  "
                    + str(atom.position[1])
                    + "  "
                    + str(atom.position[2])
                    + "\n"
                )
        fd.write("end\n")
        fd.write("Charge %d \n" % np.rint(-cluster_charge))
        fd.write("Mult %d \n" % params["mult"])

    # Standard approach to write out atoms
    else:
        fd.write("*xyz")
        fd.write(" %d" % params["charge"])
        fd.write(" %d \n" % params["mult"])
        for atom in atoms:
            if atom.tag == 71:  # 71 is ascii G (Ghost)
                symbol = atom.symbol + " : "
            else:
                symbol = atom.symbol + "   "
            fd.write(
                symbol
                + "  "
                + str(atom.position[0])
                + "  "
                + str(atom.position[1])
                + "  "
                + str(atom.position[2])
                + "\n"
            )
    fd.write("end\n")


def cluster_list_to_atoms(cluster_list: list[list[str]]) -> Atoms:
    """
    Takes the list of atoms (+ their charge and position) and converts it to an ASE Atoms object appropriate to be written out

    Parameters
    ----------
    cluster_list : list[list[str]]
        List of atoms, their charge and position

    Returns
    -------
    Atoms
        ASE Atoms object
    """

    # Get the list of positions of the atoms
    positions = [[float(x) for x in atom[0].split()[2:]] for atom in cluster_list]

    # Get the charge and symbol of the atoms
    charges = [float(atom[0].split()[1]) for atom in cluster_list]
    # Tells us pH if it's a dangling bond
    true_symbols = [atom[0].split()[0] for atom in cluster_list]
    # Gives us H if it's a dangling bond
    symbols = [
        atom[0].split()[0] if atom[0].split()[0] != "pH" else "H"
        for atom in cluster_list
    ]

    # Create an ASE Atoms object with the symbols and positions
    cluster = Atoms(symbols=symbols, positions=positions)

    # Set an array "oxi_states" with the charges of the atoms
    cluster.set_array("oxi_states", np.array(charges))

    # Set an array "atom_type" with the true symbols of the atoms
    cluster.set_array("atom_type", np.array(true_symbols))

    return cluster
