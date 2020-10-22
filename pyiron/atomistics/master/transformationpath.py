# coding: utf-8
# Distributed under the terms of the GPL3 License.

from collections import OrderedDict

import numpy as np
import spglib
from ase.build import bulk
from pyiron.atomistics.master.parallel import AtomisticParallelMaster
from pyiron.atomistics.structure.atoms import Atoms, ase_to_pyiron
from pyiron_base import JobGenerator

__author__ = "Yury Lysogorskiy"
__copyright__ = "Copyright 2020, ICAMS, RUB"
__version__ = "1.0"
__status__ = "development"
__date__ = "Sep 1, 2017"


def find_symmetry_group_number(struct):
    dataset = spglib.get_symmetry_dataset(struct)
    SGN = dataset["number"]
    return SGN


spgn_dict = {"fcc": 225, "bcc": 229, "diam": 227, "sc": 221}


def is_general_cubic(atoms):
    # either primitive/cubic fcc, bcc, diamond
    # or just cubic cell
    cell = np.array(atoms.get_cell())
    spgn = spglib.get_symmetry_dataset(atoms)['number']
    if spgn in [spgn_dict[sg_name] for sg_name in ["fcc", "bcc", "diam", "sc"]]:
        return True
    else:
        if cell[0, 0] == cell[1, 1] == cell[2, 2] and np.sum(np.abs(np.diag(cell))) == np.sum(np.abs(cell)):
            return True
        else:
            return False


def to_cubic_atoms(atoms):
    # either primitive/cubic fcc, bcc, diamond
    # or just cubic cell
    cell = np.array(atoms.get_cell())
    volume_per_atom = np.linalg.det(cell) / len(atoms)
    elem = atoms.get_chemical_symbols()[0]
    spgn = spglib.get_symmetry_dataset(atoms)['number']
    if spgn in [spgn_dict[sg_name] for sg_name in ["fcc", "bcc", "diam", "sc"]]:
        if spgn == spgn_dict["fcc"]:
            a0 = (volume_per_atom * 4.) ** (1. / 3.)
            new_atoms = bulk(elem, "fcc", a=a0, cubic=True)
        elif spgn == spgn_dict["bcc"]:
            a0 = (volume_per_atom * 2.) ** (1. / 3.)
            new_atoms = bulk(elem, "bcc", a=a0, cubic=True)
        elif spgn == spgn_dict["diam"]:
            a0 = (volume_per_atom * 8.) ** (1. / 3.)
            new_atoms = bulk(elem, "diamond", a=a0, cubic=True)
        elif spgn == spgn_dict["sc"]:
            a0 = (volume_per_atom * 1.) ** (1. / 3.)
            new_atoms = bulk(elem, "sc", a=a0, cubic=True)
        else:
            return None
        return ase_to_pyiron(new_atoms)
    else:
        if cell[0, 0] == cell[1, 1] == cell[2, 2] and np.sum(np.abs(np.diag(cell))) == np.sum(np.abs(cell)):
            return atoms
        else:
            return None


class TransformationPathGenerator(JobGenerator):
    HEXAGONAL = "hexagonal"
    TRIGONAL = "trigonal"
    ORTHOGONAL = "orthogonal"
    TETRAGONAL = "tetragonal"
    GENERAL_CUBIC_TETRAGONAL = "general_cubic_tetragonal"

    param_names = ["transformation_type", "num_of_point"]

    @staticmethod
    def job_name(parameter):
        # TODO: understand how this function is called?
        if isinstance(parameter[0], str):
            return parameter[0]
        else:
            return ("tp_{:.5f}".format(parameter[0])).replace('.', '_').replace("-", "m")

    def __init__(self, job, no_job_checks=False):
        super().__init__(job, no_job_checks)
        self._ref_job_prepared = False
        self._data = OrderedDict()
        self._structure_dict = OrderedDict()

    def prepare_ref_job(self):
        self.basis_ref = self._job.ref_job.structure.copy()
        sgn = find_symmetry_group_number(self.basis_ref)
        transformation_type = self._job.input["transformation_type"]
        atoms = self.basis_ref
        if transformation_type in [self.HEXAGONAL, self.TRIGONAL, self.ORTHOGONAL, self.TETRAGONAL] and \
                sgn not in [225, 229]:  # 225 - FCC, 229- BCC
            raise ValueError(
                "Only FCC(sg #225) or BCC (sg #229) structures are acceptable for {} transformation type, but you provide "
                "structure with space group #{}".format(transformation_type, sgn))
        elif transformation_type == self.GENERAL_CUBIC_TETRAGONAL:
            if not is_general_cubic(atoms):
                raise ValueError(
                    "Only FCC(sg #225), BCC (sg #229), SC (#221), DIAM(#227) or cubic-cell structures are acceptable for {} transformation type, but you provide "
                    "structure with space group #{} and cell {}".format(transformation_type, sgn, atoms.get_cell()))
        volume = self.basis_ref.get_volume() / len(self.basis_ref)
        a0 = (volume * 2.) ** (1. / 3.)  # lattice constant for BCC
        chem_symbols = list(set(self.basis_ref.get_chemical_symbols()))
        if len(chem_symbols) > 1:
            raise NotImplementedError(
                "Only unaries are supported, but your structure has {} elements".format(chem_symbols))
        elem = chem_symbols[0]
        self.type = self._job.input["transformation_type"]
        self.element = elem
        self.num_of_point = self._job.input["num_points"]
        self._ref_job_prepared = True

    def deformation_path(self):
        # TODO: ensure that high-symmetry points are also within the list
        if self.type == TransformationPathGenerator.TETRAGONAL:
            path_indices = np.linspace(0.8, 2, self.num_of_point)
        elif self.type == TransformationPathGenerator.ORTHOGONAL:
            path_indices = np.linspace(1., np.sqrt(2.), self.num_of_point)
        elif self.type == TransformationPathGenerator.TRIGONAL:
            path_indices = np.linspace(0.8, 5., self.num_of_point)
        elif self.type == TransformationPathGenerator.HEXAGONAL:
            path_indices = np.linspace(-0.5, 1.8, self.num_of_point)
        elif self.type == TransformationPathGenerator.GENERAL_CUBIC_TETRAGONAL:
            path_indices = np.linspace(0.2, 2.6, self.num_of_point)
        else:
            raise NotImplementedError("Transformation path <" + str(self.type) + "> is not implemented")
        return path_indices

    def generate_tetra_path(self, indices_only=False):
        def gen_tetr(a0, base_atoms, p):
            a = (a0 ** 3 / p) ** (1. / 3.)
            c = a0 ** 3 / a ** 2.
            atoms = base_atoms.copy()
            atoms.set_cell([(a, 0, 0), (0, a, 0), (0, 0, c)], scale_atoms=True)
            return atoms

        path_indices = self.deformation_path()
        if indices_only:
            return path_indices
        else:
            volume = self.basis_ref.get_volume() / len(self.basis_ref)
            a0 = (volume * 2.) ** (1. / 3.)  # lattice constant for BCC
            base_atoms = Atoms([self.element] * 2, scaled_positions=[(0, 0, 0), (1 / 2., 1. / 2, 1. / 2)],
                               cell=[(a0, 0, 0), (0, a0, 0), (0, 0, a0)], pbc=True)
            structures = []
            for p in path_indices:
                atoms = gen_tetr(a0, base_atoms, p)
                structures.append(atoms)
            atoms = gen_tetr(a0, base_atoms, p=1.0)
            self.base_structure = atoms
            return path_indices, structures

    def generate_general_tetra_path(self, indices_only=False):
        def gen_tetr(a0, base_atoms, p):

            a = (a0 ** 3 / p) ** (1. / 3.)
            c = a0 ** 3 / a ** 2.
            atoms = base_atoms.copy()
            atoms.set_cell([(a, 0, 0), (0, a, 0), (0, 0, c)], scale_atoms=True)
            return atoms

        path_indices = self.deformation_path()
        if indices_only:
            return path_indices
        else:
            base_atoms = to_cubic_atoms(self.basis_ref)
            a0 = base_atoms.cell[0, 0]

            structures = []
            for p in path_indices:
                atoms = gen_tetr(a0, base_atoms, p)
                structures.append(ase_to_pyiron(atoms))

            atoms = ase_to_pyiron(gen_tetr(a0, base_atoms, p=1.0))
            self.base_structure = atoms

            return path_indices, structures

    def generate_ortho_path(self, indices_only=False):
        def gen_orth(a0, p):
            a1 = a0 * np.array([np.sqrt(2.), 0., 0.])
            a2 = a0 * np.array([0., p, 0.])
            a3 = a0 * np.array([0., 0., np.sqrt(2.) / p])
            cell = np.array([a1, a2, a3])
            atoms = Atoms([self.element] * 4, scaled_positions=[(0, 0, 0), (.5, .5, .0), (.5, .0, .5), (.0, .5, .5)],
                          cell=cell, pbc=True)
            return atoms

        path_indices = self.deformation_path()
        if indices_only:
            return path_indices
        else:
            volume = self.basis_ref.get_volume() / len(self.basis_ref)
            a0 = (volume * 2.) ** (1. / 3.)  # lattice constant for BCC
            structures = []
            for p in path_indices:
                atoms = gen_orth(a0, p)
                structures.append(atoms)
            self.base_structure = gen_orth(a0, p=1.0)
            return path_indices, structures

    def generate_trigo_path(self, indices_only=False):
        def gen_tri(base_atoms, cell, p):
            trigo = np.array([[0., (-2. / np.sqrt(6.)) / ((np.power(p, 2. / 3.)) ** (1. / 2.)),
                               np.power(p, 2. / 3.) / np.sqrt(3.)],
                              [(np.sqrt(2.) / 2.) / ((np.power(p, 2. / 3.)) ** (1. / 2.)),
                               1. / np.sqrt(6.) / ((np.power(p, 2. / 3.)) ** (1. / 2.)),
                               np.power(p, 2. / 3.) / np.sqrt(3.)],
                              [(-np.sqrt(2.) / 2.) / ((np.power(p, 2. / 3.)) ** (1. / 2.)),
                               1. / np.sqrt(6.) / ((np.power(p, 2. / 3.)) ** (1. / 2.)),
                               np.power(p, 2. / 3.) / np.sqrt(3.)]])
            atoms = base_atoms.copy()
            atoms.set_cell(np.array(np.dot(cell, trigo)), scale_atoms=True)
            return atoms

        path_indices = self.deformation_path()
        if indices_only:
            return path_indices
        else:
            volume = self.basis_ref.get_volume() / len(self.basis_ref)
            a0 = (volume * 2.) ** (1. / 3.)  # lattice constant for BCC
            base_atoms = Atoms([self.element] * 2, scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)],
                               cell=[(a0, 0, 0), (0, a0, 0), (0, 0, a0)], pbc=True)
            cell = np.array(base_atoms.get_cell())
            structures = []
            for p in path_indices:
                atoms = gen_tri(base_atoms, cell, p)
                structures.append(atoms)
            self.base_structure = gen_tri(base_atoms, cell, p=1.0)
            return path_indices, structures

    def generate_hex_path(self, indices_only=False):
        def gen_hex(a0, p):
            brac1 = p * (2 * np.sqrt(3) - 3 * np.sqrt(2)) / 6. + np.sqrt(2) / 2.
            brac2 = p * (2 * np.sqrt(2) - 3.) / 3. + 1
            vv0 = np.sqrt(2) * brac1 * brac2
            a = a0 * np.sqrt(2) / (vv0 ** (1. / 3.))
            b = a * brac1
            c = a * brac2
            pos1 = (0.5 - p / 6., 0, 0.5)
            pos2 = (0, 0, 0)
            pos3 = (0.5, 0.5, 0)
            pos4 = (- p / 6., 0.5, 0.5)
            a1 = a * np.array([1, 0, 0])
            a2 = b * np.array([0, 1, 0])
            a3 = c * np.array([0, 0, 1])
            cell = np.array([a1, a2, a3])
            atoms = Atoms([self.element] * 4, scaled_positions=[pos1, pos2, pos3, pos4], cell=cell, pbc=True)
            cell *= ((a * b * c / atoms.get_volume()) ** (1 / 3.))
            atoms.set_cell(cell, scale_atoms=True)
            return atoms

        path_indices = self.deformation_path()
        if indices_only:
            return path_indices
        else:
            volume = self.basis_ref.get_volume() / len(self.basis_ref)
            a0 = (volume * 2.) ** (1. / 3.)  # lattice constant for BCC
            structures = []
            for p in path_indices:
                atoms = gen_hex(a0, p)
                structures.append(atoms)
            self.base_structure = gen_hex(a0, p=0.0)
            return path_indices, structures

    def generate_path(self, indices_only=False):
        if self.type == TransformationPathGenerator.TETRAGONAL:
            return self.generate_tetra_path(indices_only)
        elif self.type == TransformationPathGenerator.ORTHOGONAL:
            return self.generate_ortho_path(indices_only)
        elif self.type == TransformationPathGenerator.TRIGONAL:
            return self.generate_trigo_path(indices_only)
        elif self.type == TransformationPathGenerator.HEXAGONAL:
            return self.generate_hex_path(indices_only)
        elif self.type == TransformationPathGenerator.GENERAL_CUBIC_TETRAGONAL:
            return self.generate_general_tetra_path(indices_only)
        else:
            raise NotImplementedError("Transformation path <" + str(self.type) + "> is not implemented")

    @property
    def parameter_list(self):
        if not self._ref_job_prepared:
            self.prepare_ref_job()
        path_ind, structures = self.generate_path()
        self.indices = path_ind
        self._data["transformation_coordinates"] = self.indices

        parameter_lst = []
        for p, sc in zip(path_ind, structures):
            job_name = self.job_name((p, sc))
            parameter_lst.append([job_name, sc])
        return parameter_lst

    def modify_job(self, job, parameter):
        job.structure = parameter[1]
        job.calc_static()
        return job

    def analyse_structures(self, output_dict):
        if not self._ref_job_prepared:
            self.prepare_ref_job()
        path_ind = self.generate_path(indices_only=True)
        self.indices = path_ind
        energies_0 = []
        for p in path_ind:
            job_name = self.job_name((p,))
            energy_0 = output_dict[job_name]
            energies_0.append(energy_0)

        self._data['energies_0'] = np.array(energies_0)
        self._data["transformation_coordinates"] = self.indices


class TransformationPath(AtomisticParallelMaster):
    hdf_storage_group = "transformation_path"

    def __init__(self, project, job_name="transformation_path"):
        super(TransformationPath, self).__init__(project, job_name)
        self.__name__ = "TransformationPath"
        self.__version__ = '0.0.1'

        self.input['num_points'] = (50, 'number of sample points')

        self.input['transformation_type'] = ('tetragonal', '[tetragonal, trigonal, orthogonal, hexagonal]')

        self._job_generator = TransformationPathGenerator(self)
        self._data = OrderedDict()

    def list_structures(self):
        if self.structure is not None:
            return [struct for _, struct in self._job_generator.parameter_list]
        else:
            return []

    def collect_output(self):
        energies = {}
        self._data["id"] = []
        for job_id in self.child_ids:
            ham = self.project_hdf5.inspect(job_id)
            en = ham["output/generic/energy_tot"][0]
            energies[ham.job_name] = en
            self._data["id"].append(ham.job_id)

        self._job_generator.analyse_structures(energies)

        with self.project_hdf5.open("output") as hdf5_out:
            hdf5_out[self.hdf_storage_group] = self._job_generator._data
