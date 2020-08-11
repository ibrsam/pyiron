# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os
import numpy as np

from pyiron.base.generic.parameters import GenericParameters
from pyiron.atomistics.structure.atoms import Atoms
from pyiron.atomistics.job.atomistic import AtomisticGenericJob

from pyiron.base.settings.generic import Settings

__author__ = "Yury Lysogorskiy"
__copyright__ = "Copyright 2020, ICAMS-RUB "
__version__ = "1.0"
__maintainer__ = ""
__email__ = ""
__status__ = "trial"
__date__ = "Aug 11, 2020"

s = Settings()

class FHIAims(AtomisticGenericJob):
    def __init__(self, project, job_name):
        super(FHIAims, self).__init__(project, job_name)
        self.__name__ = "FHIaims"
        self._executable_activate(enforce=True) 
        self.input = FHIAimsInput()
    
    def write_input(self):
        # methods, called externally
        print("inside FHIAims write_inpute")
        self.input.write(structure=self.structure, working_directory=self.working_directory)


    def collect_output(self):
        output_dict = collect_output(output_file=os.path.join(self.working_directory, 'FHI.out'))
        with self.project_hdf5.open("output") as hdf5_output:
            with hdf5_output.open("generic") as hdf5_generic:
                for k, v in output_dict.items():
                    hdf5_generic[k] = v
        
    def to_hdf(self, hdf=None, group_name=None):
        super(FHIAims, self).to_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as hdf5_input:
            self.structure.to_hdf(hdf5_input)
            self.input.to_hdf(hdf5_input)

    def from_hdf(self, hdf=None, group_name=None):
        super(FHIAims, self).from_hdf(hdf=hdf, group_name=group_name)
        with self.project_hdf5.open("input") as hdf5_input:
            self.input.from_hdf(hdf5_input)
            self.structure = Atoms().from_hdf(hdf5_input)

class FHIAimsControlInput(GenericParameters):
    def __init__(self, input_file_name=None):
        super(FHIAimsControlInput, self).__init__(input_file_name=input_file_name,
                                                  table_name="control_in",
                                                  comment_char="#")

    def load_default(self):
        """
        Loading the default settings for the input file.
        """
        input_str = """
xc                 pbe
charge             0.
spin               none
relativistic       atomic_zora scalar
occupation_type    gaussian 0.10
mixer              pulay
n_max_pulay        10
charge_mix_param   0.05   
sc_iter_limit      500
sc_accuracy_rho  1E-5
sc_accuracy_eev  1E-3
sc_accuracy_etot 1E-7

compute_forces .true.
clean_forces sayvetz
sc_accuracy_forces 1E-4
final_forces_cleaned .true.
"""
        self.load_string(input_str)


class FHIAimsControlPotential(GenericParameters):
    def __init__(self, input_file_name=None):
        super(FHIAimsControlPotential, self).__init__(input_file_name=input_file_name,
                                                      table_name="control_potential",
                                                      comment_char="#")
        self._structure = None

    def load_default(self):
        """
        Loading the default settings for the input file.
        """
        input_str = """\
potential          tight  # Options: light, tight, really_tight
"""
        self.load_string(input_str)

    def set_structure(self, structure):
        self._structure = structure
        chem_symb = self._structure.get_chemical_symbols()
        atom_numb = self._structure.get_atomic_numbers()
        chem_symb_dict = {k: v for (k, v) in zip(chem_symb, atom_numb)}
        self._chem_symb_lst = sorted(chem_symb_dict.items())


    def _return_potential_file(self, file_name):
        for resource_path in s.resource_paths:
            resource_path_potcar = os.path.join(
                resource_path, "fhiaims", "potentials", self["potential"], file_name
            )
            if os.path.exists(resource_path_potcar):
                return resource_path_potcar
        return None

    def get_string_lst(self):
        settings = self["potential"]
        print("get_string_lst.settings=",settings)
        lines=[]
        for elem, atom_num in self._chem_symb_lst:
            file_name = "{atom_num}_{elem}_default".format(atom_num=atom_num, elem=elem)
            full_potential_file_name = self._return_potential_file(file_name)
            if full_potential_file_name is None:
                raise ValueError("Couldn't read file {} for settings '{}'".format(file_name, settings))
            with open(full_potential_file_name, "r") as f:
                lines += f.readlines()
        return lines


class FHIAimsInput:
    def __init__(self):
        self.control_input = FHIAimsControlInput()
        self.control_potential = FHIAimsControlPotential()

    def write(self, structure, working_directory):
        print("inside FHIAimsInput.write_input")
        print("Structure=")
        print(structure)
        print("working_directory=",working_directory)


        self.control_potential.set_structure(structure)
        control_in_filename = os.path.join(working_directory, "control.in")
        control_in_lst = self.control_input.get_string_lst()
        print("DEBUG: control_in_filename=",control_in_filename)
        print("DEBUG: control_input: \n", "".join(control_in_lst))
        control_in_lst += self.control_potential.get_string_lst()


        with open(control_in_filename, "w") as f:
            print("".join(control_in_lst), file=f)

        #TODO: write geometry.in

        pbc = structure.pbc
        is_periodic = np.all(pbc)
        if not is_periodic and not np.all(~pbc):
            raise ValueError("Structure for FHI-aims could be either fully periodic or fully non-periodic")

        lines=["# pyiron generated geometry.in"]
        if is_periodic:
            cell = structure.get_cell()
            for lattice_vec in cell:
                lines.append("lattice_vector {:.15f} {:.15f} {:.15f}".format(lattice_vec[0], lattice_vec[1], lattice_vec[2]))
        lines.append("")

        chem_symbs = structure.get_chemical_symbols()
        positions  = structure.get_positions()

        for symb, pos  in zip(chem_symbs, positions):
            lines.append("atom {:.15f} {:.15f} {:.15f}   {}".format(pos[0], pos[1], pos[2], symb))

        with open(os.path.join(working_directory, "geometry.in"), "w") as f:
            print("\n".join(lines), file=f)




    def to_hdf(self, hdf=None):
        with hdf.open("control_input") as hdf5_input:
            self.control_input.to_hdf(hdf5_input)

        with hdf.open("control_potential") as hdf5_input:
            self.control_potential.to_hdf(hdf5_input)

    def from_hdf(self, hdf=None):
        with hdf.open("control_input") as hdf5_input:
            self.control_input.from_hdf(hdf5_input)
        with hdf.open("control_potential") as hdf5_input:
            self.control_potential.from_hdf(hdf5_input)




    
def collect_output(output_file):
    free_energies_list = []
    energies_corrected_list = []
    forces_lst = []
    stresses_lst = []

    block_flag = False
    force_block_flag = False
    stress_block_flag = False

    stress_line_counter = 0
    current_forces = []
    current_stresses = []

    for line in open(output_file, 'r'):
        line = line.strip()
        if "Energy and forces in a compact form:" in line:
            block_flag = True

        if block_flag and "------------------------------------" in line:
            block_flag = False
            force_block_flag = False
            forces_lst.append(current_forces)

        if block_flag and 'Total energy corrected        :' in line:
            E0 = float(line.split()[5])
            energies_corrected_list.append(E0)
        elif block_flag and 'Electronic free energy        :' in line:
            F = float(line.split()[5])
            free_energies_list.append(F)

        if block_flag and "Total atomic forces" in line:
            force_block_flag = True
            current_forces = []

        if force_block_flag and line.strip().startswith("|"):
            current_forces.append([float(f) for f in line.split()[-3:]])

        if "|              Analytical stress tensor" in line or "Numerical stress tensor" in line:
            stress_block_flag = True
            current_stresses = []
            stress_line_counter = 0

        if stress_block_flag:
            stress_line_counter += 1

        if stress_block_flag and stress_line_counter in [6, 7, 8]:
            sline = [float(f) for f in line.split()[2:5]]
            current_stresses.append(sline)

        if stress_line_counter > 8:
            stress_block_flag = False
            stress_line_counter = 0
            stresses_lst.append(current_stresses)

    output_dict= {'energy_pot': free_energies_list, 'forces': forces_lst}

    return output_dict

