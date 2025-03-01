# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os
import numpy as np
import unittest
import warnings
import scipy.constants
from pyiron.project import Project
from pyiron.atomistics.structure.periodic_table import PeriodicTable
from pyiron.atomistics.structure.atoms import Atoms
from pyiron.sphinx.base import Group

BOHR_TO_ANGSTROM = (
        scipy.constants.physical_constants["Bohr radius"][0] / scipy.constants.angstrom
)
HARTREE_TO_EV = scipy.constants.physical_constants["Hartree energy in eV"][0]
HARTREE_OVER_BOHR_TO_EV_OVER_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM


class TestSphinx(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.file_location = os.path.dirname(os.path.abspath(__file__))
        cls.project = Project(os.path.join(cls.file_location, "../static/sphinx"))
        pt = PeriodicTable()
        pt.add_element(parent_element="Fe", new_element="Fe_up", spin="0.5")
        Fe_up = pt.element("Fe_up")
        cls.basis = Atoms(
            elements=[Fe_up, Fe_up],
            scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            cell=2.6 * np.eye(3),
        )
        cls.sphinx = cls.project.create_job("Sphinx", "job_sphinx")
        cls.sphinx_band_structure = cls.project.create_job("Sphinx", "sphinx_test_bs")
        cls.sphinx_2_3 = cls.project.create_job("Sphinx", "sphinx_test_2_3")
        cls.sphinx_2_5 = cls.project.create_job("Sphinx", "sphinx_test_2_5")
        cls.sphinx_aborted = cls.project.create_job("Sphinx", "sphinx_test_aborted")
        cls.sphinx.structure = cls.basis
        cls.sphinx.fix_spin_constraint = True
        cls.sphinx_band_structure.structure = cls.project.create_structure("Fe", "bcc", 2.81)
        cls.sphinx_band_structure.structure = cls.sphinx_band_structure.structure.create_line_mode_structure()
        cls.sphinx_2_3.structure = Atoms(
            elements=["Fe", "Fe"],
            scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            cell=2.6 * np.eye(3),
        )
        cls.sphinx_2_5.structure = Atoms(
            elements=["Fe", "Ni"],
            scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
            cell=2.83 * np.eye(3),
        )
        cls.sphinx_aborted.structure = Atoms(
            elements=32 * ["Fe"],
            scaled_positions=np.arange(32 * 3).reshape(-1, 3) / (32 * 3),
            cell=3.5 * np.eye(3),
        )
        cls.sphinx_aborted.status.aborted = True
        cls.current_dir = os.path.abspath(os.getcwd())
        cls.sphinx._create_working_directory()
        cls.sphinx_2_3._create_working_directory()
        cls.sphinx.input["VaspPot"] = False
        cls.sphinx.structure.add_tag(selective_dynamics=(True, True, True))
        cls.sphinx.structure.selective_dynamics[1] = (False, False, False)
        cls.sphinx.load_default_groups()
        cls.sphinx.fix_symmetry = False
        cls.sphinx.write_input()
        cls.sphinx_2_3.to_hdf()
        cls.sphinx_2_3.decompress()
        cls.sphinx_2_5.decompress()

    @classmethod
    def tearDownClass(cls):
        cls.sphinx_2_3.decompress()
        cls.sphinx_2_5.decompress()
        cls.file_location = os.path.dirname(os.path.abspath(__file__))
        os.remove(
            os.path.join(
                cls.file_location,
                "../static/sphinx/job_sphinx_hdf5/job_sphinx/input.sx",
            )
        )
        os.remove(
            os.path.join(
                cls.file_location,
                "../static/sphinx/job_sphinx_hdf5/job_sphinx/spins.in",
            )
        )
        os.remove(
            os.path.join(
                cls.file_location,
                "../static/sphinx/job_sphinx_hdf5/job_sphinx/Fe_GGA.atomicdata",
            )
        )
        os.rmdir(
            os.path.join(
                cls.file_location, "../static/sphinx/job_sphinx_hdf5/job_sphinx"
            )
        )
        os.rmdir(os.path.join(cls.file_location, "../static/sphinx/job_sphinx_hdf5"))
        os.remove(
            os.path.join(cls.file_location, "../static/sphinx/sphinx_test_2_3.h5")
        )

    def test_id_pyi_to_spx(self):
        self.assertEqual(len(self.sphinx.id_pyi_to_spx), len(self.sphinx.structure))
        self.assertEqual(len(self.sphinx.id_spx_to_pyi), len(self.sphinx.structure))

    def test_potential(self):
        self.assertEqual([], self.sphinx.list_potentials())
        self.assertEqual(['Fe_GGA'], self.sphinx_2_3.list_potentials())
        self.assertEqual(['Fe_GGA'], self.sphinx_2_5.list_potentials())
        self.sphinx_2_3.potential.Fe = 'Fe_GGA'
        self.sphinx_2_5.potential["Fe"] = 'Fe_GGA'
        self.assertEqual('Fe_GGA', list(self.sphinx_2_3.potential.to_dict().values())[0])
        self.assertEqual('Fe_GGA', list(self.sphinx_2_5.potential.to_dict().values())[0])

    def test_write_input(self):

        file_content = [
            '//job_sphinx\n',
            '//SPHInX input file generated by pyiron\n',
            '\n',
            'format paw;\n',
            'include <parameters.sx>;\n',
            '\n',
            'pawPot {\n',
            '\tspecies {\n',
            '\t\tname = "Fe";\n',
            '\t\tpotType = "AtomPAW";\n',
            '\t\telement = "Fe";\n',
            '\t\tpotential = "Fe_GGA.atomicdata";\n',
            '\t}\n',
            '}\n',
            'structure {\n',
            '\tcell = [[4.913287924027003, 0.0, 0.0], [0.0, 4.913287924027003, 0.0], [0.0, 0.0, 4.913287924027003]];\n',
            '\tspecies {\n',
            '\t\telement = "Fe";\n',
            '\t\tatom {\n',
            '\t\t\tlabel = "spin_0.5";\n',
            '\t\t\tcoords = [0.0, 0.0, 0.0];\n',
            '\t\t\tmovable;\n',
            '\t\t}\n',
            '\t\tatom {\n',
            '\t\t\tlabel = "spin_0.5";\n',
            '\t\t\tcoords = [2.4566439620135014, 2.4566439620135014, 2.4566439620135014];\n',
            '\t\t}\n',
            '\t}\n',
            '\tsymmetry {\n',
            '\t\toperator {\n',
            '\t\t\tS = [[1,0,0],[0,1,0],[0,0,1]];\n',
            '\t\t}\n',
            '\t}\n',
            '}\n',
            'basis {\n',
            '\teCut = 24.989539079445393;\n',
            '\tkPoint {\n',
            '\t\tcoords = [0.5, 0.5, 0.5];\n',
            '\t\tweight = 1;\n',
            '\t\trelative;\n',
            '\t}\n',
            '\tfolding = [4, 4, 4];\n',
            '\tsaveMemory;\n',
            '}\n',
            'PAWHamiltonian {\n',
            '\tnEmptyStates = 6;\n',
            '\tekt = 0.007349864435130998;\n',
            '\txc = PBE;\n',
            '\tspinPolarized;\n',
            '}\n',
            'initialGuess {\n',
            '\twaves {\n',
            '\t\tlcao {}\n',
            '\t\tpawBasis;\n',
            '\t}\n',
            '\trho {\n',
            '\t\tatomicOrbitals;\n',
            '\t\tatomicSpin {\n',
            '\t\t\tlabel = "spin_0.5";\n',
            '\t\t\tspin = 0.5;\n',
            '\t\t}\n',
            '\t\tatomicSpin {\n',
            '\t\t\tlabel = "spin_0.5";\n',
            '\t\t\tspin = 0.5;\n',
            '\t\t}\n',
            '\t}\n',
            '\tnoWavesStorage = false;\n',
            '}\n',
            'main {\n',
            '\tscfDiag {\n',
            '\t\trhoMixing = 1.0;\n',
            '\t\tspinMixing = 1.0;\n',
            '\t\tdEnergy = 3.674932217565499e-06;\n',
            '\t\tmaxSteps = 100;\n',
            '\t\tblockCCG {}\n',
            '\t}\n',
            '\tevalForces {\n',
            '\t\tfile = "relaxHist.sx";\n',
            '\t}\n',
            '}\n',
            'spinConstraint {\n',
            '\tfile = "spins.in";\n',
            '}\n'
            ]
        file_name = os.path.join(
            self.file_location, "../static/sphinx/job_sphinx_hdf5/job_sphinx/input.sx"
        )
        with open(file_name) as input_sx:
            lines = input_sx.readlines()
        self.assertEqual(file_content, lines)


    def test_plane_wave_cutoff(self):
        with self.assertRaises(ValueError):
            self.sphinx.plane_wave_cutoff = -1

        with warnings.catch_warnings(record=True) as w:
            self.sphinx.plane_wave_cutoff = 25
            self.assertEqual(len(w), 1)

        self.sphinx.plane_wave_cutoff = 340
        self.assertEqual(self.sphinx.plane_wave_cutoff, 340)

    def test_set_kpoints(self):

        mesh = [2, 3, 4]
        center_shift = [0.1, 0.1, 0.1]

        trace = {"my_path": [("GAMMA", "H"), ("H", "N"), ("P", "H")]}

        kpoints_group = Group({
            'relative': True,
            'from': {
                'coords': np.array([0.0, 0.0, 0.0]),
                'label': '"GAMMA"'
            },
            'to': [
                {'coords': np.array([0.5, -0.5,  0.5]),
                 'nPoints': 20, 'label': '"H"'},
                {'coords': np.array([0.0,  0.0,  0.5]),
                 'nPoints': 20, 'label': '"N"'},
                {'coords': np.array([0.25, 0.25, 0.25]),
                 'nPoints': 0, 'label': '"P"'},
                {'coords': np.array([0.5, -0.5,  0.5]),
                 'nPoints': 20, 'label': '"H"'},
            ]
        })

        with self.assertRaises(ValueError):
            self.sphinx_band_structure.set_kpoints(symmetry_reduction="pyiron rules!")
        with self.assertRaises(ValueError):
            self.sphinx_band_structure.set_kpoints(scheme="no valid scheme")
        with self.assertRaises(ValueError):
            self.sphinx_band_structure.set_kpoints(scheme="Line", path_name="my_path")

        self.sphinx_band_structure.structure.add_high_symmetry_path(trace)
        with self.assertRaises(ValueError):
            self.sphinx_band_structure.set_kpoints(scheme="Line", n_path=20)
        with self.assertRaises(AssertionError):
            self.sphinx_band_structure.set_kpoints(scheme="Line", path_name="wrong name", n_path=20)

        self.sphinx_band_structure.set_kpoints(scheme="Line", path_name="my_path", n_path=20)
        self.assertTrue("kPoint" not in self.sphinx_band_structure.input.sphinx.basis)
        self.assertEqual(self.sphinx_band_structure.input.sphinx.to_sphinx(kpoints_group),
                         self.sphinx_band_structure.input.sphinx.basis.kPoints.to_sphinx())

        self.sphinx_band_structure.set_kpoints(scheme="MP", mesh=mesh, center_shift=center_shift)
        self.assertTrue("kPoints" not in self.sphinx_band_structure.input.sphinx.basis)
        self.assertEqual(self.sphinx_band_structure.input.KpointFolding, mesh)
        self.assertEqual(self.sphinx_band_structure.input.KpointCoords, center_shift)
        self.assertEqual(self.sphinx_band_structure.get_k_mesh_by_cell(2 * np.pi / 2.81), [1, 1, 1])

    def test_set_empty_states(self):
        with self.assertRaises(ValueError):
            self.sphinx.set_empty_states(-1)
        self.sphinx.set_empty_states(666)
        self.assertEqual(self.sphinx.input["EmptyStates"], 666)
        self.sphinx.set_empty_states()
        self.assertEqual(self.sphinx.input["EmptyStates"], "auto")

    def test_fix_spin_constraint(self):
        self.assertTrue(self.sphinx.fix_spin_constraint)
        with self.assertRaises(ValueError):
            self.sphinx.fix_spin_constraint = 3
        self.sphinx.fix_spin_constraint = False
        self.assertIsInstance(self.sphinx.fix_spin_constraint, bool)

    def test_calc_static(self):
        self.sphinx.calc_static(algorithm="wrong_algorithm")
        self.assertFalse(
            "keepRho"
            in self.sphinx.input.sphinx.main.to_sphinx()
        )
        self.assertTrue(
            "blockCCG"
            in self.sphinx.input.sphinx.main.to_sphinx()
        )
        self.sphinx.restart_file_list.append("randomfile")
        self.sphinx.calc_static(algorithm="ccg")
        self.assertTrue(
            "keepRho"
            in self.sphinx.input.sphinx.main.to_sphinx()
        )
        self.assertEqual(self.sphinx.input["Estep"], 100)
        self.assertTrue(
            "CCG"
            in self.sphinx.input.sphinx.main.to_sphinx()
        )

    def test_calc_minimize(self):
        self.sphinx.calc_minimize(electronic_steps=100, ionic_steps=50)
        self.assertEqual(self.sphinx.input["Estep"], 100)
        self.assertEqual(self.sphinx.input["Istep"], 50)
        self.assertEqual(self.sphinx.input.sphinx.main['ricQN']['maxSteps'], '50')

    def test_get_scf_group(self):
        with warnings.catch_warnings(record=True) as w:
            test_scf = self.sphinx_band_structure.get_scf_group(algorithm="wrong")
            self.assertEqual(len(w), 1)
            ref_scf = {
                'rhoMixing': '1.0',
                'spinMixing': '1.0',
                'dEnergy': 3.674932217565499e-06,
                'maxSteps': '100',
                'blockCCG': {}}
            self.assertEqual(test_scf, ref_scf)

        ref_scf = {
            'rhoMixing': '1.0',
            'spinMixing': '1.0',
            'nPulaySteps': '0',
            'dEnergy': 3.674932217565499e-06,
            'maxSteps': '100',
            'preconditioner': {
                'type': 0
                },
            'blockCCG': {
            'maxStepsCCG': 0,
            'blockSize': 0,
            'nSloppy': 0},
            'noWavesStorage': True
            }

        self.sphinx_band_structure.input["nPulaySteps"] = 0
        self.sphinx_band_structure.input["preconditioner"] = 0
        self.sphinx_band_structure.input["maxStepsCCG"] = 0
        self.sphinx_band_structure.input["blockSize"] = 0
        self.sphinx_band_structure.input["nSloppy"] = 0
        self.sphinx_band_structure.input["WriteWaves"] = False
        test_scf = self.sphinx_band_structure.get_scf_group()
        self.assertEqual(test_scf, ref_scf)

    def test_check_setup(self):
        self.assertFalse(self.sphinx.check_setup())

        self.sphinx_band_structure.load_default_groups()
        self.sphinx_band_structure.input.sphinx.basis.kPoint = {"coords": "0.5, 0.5, 0.5"}
        self.assertFalse(self.sphinx_band_structure.check_setup())

        self.sphinx_band_structure.load_default_groups()
        self.sphinx_band_structure.server.cores = 2000
        self.assertFalse(self.sphinx_band_structure.check_setup())

        self.sphinx_band_structure.input["EmptyStates"] = "auto"
        self.assertFalse(self.sphinx_band_structure.check_setup())
        self.sphinx_band_structure.structure.add_tag(spin=None)
        for i in range(len(self.sphinx_band_structure.structure)):
            self.sphinx_band_structure.structure.spin[i] = 4
        self.assertFalse(self.sphinx_band_structure.check_setup())

    def test_set_check_overlap(self):
        self.assertRaises(ValueError, self.sphinx_band_structure.set_check_overlap, 0)

    def test_set_occupancy_smearing(self):
        self.assertRaises(
            ValueError, self.sphinx_band_structure.set_occupancy_smearing, 0.1, 0.1
        )
        self.assertRaises(
            ValueError, self.sphinx_band_structure.set_occupancy_smearing, "fermi", -0.1
        )
        self.sphinx_band_structure.set_occupancy_smearing("fermi", 0.1)

    def test_load_default_groups(self):
        backup  = self.sphinx_band_structure.structure.copy()
        self.sphinx_band_structure.structure = None
        self.assertRaises(
            AssertionError, self.sphinx_band_structure.load_default_groups
        )
        self.sphinx_band_structure.structure = backup

    def test_validate_ready_to_run(self):

        backup = self.sphinx_band_structure.structure.copy()
        self.sphinx_band_structure.structure = None
        self.assertRaises(AssertionError, self.sphinx_band_structure.validate_ready_to_run)
        self.sphinx_band_structure.structure = backup

        self.sphinx_band_structure.input["THREADS"] = 20
        self.sphinx_band_structure.server.cores = 10
        self.assertRaises(AssertionError, self.sphinx_band_structure.validate_ready_to_run)

        self.sphinx_band_structure.input.sphinx.main.clear()
        self.assertRaises(AssertionError, self.sphinx_band_structure.validate_ready_to_run)

        backup = self.sphinx.input.sphinx.basis.eCut
        self.sphinx.input.sphinx.basis.eCut = 400
        self.assertFalse(self.sphinx.validate_ready_to_run())
        self.sphinx.input.sphinx.basis.eCut = backup

        backup = self.sphinx.input.sphinx.basis.kPoint.copy()
        self.sphinx.input.sphinx.basis.kPoint.clear()
        self.sphinx.input.sphinx.basis.kPoint.coords = [0.5, 0.5, 0.25]
        self.sphinx.input.sphinx.basis.kPoint.weight = 1
        self.assertFalse(self.sphinx.validate_ready_to_run())
        self.sphinx.input.sphinx.basis.kPoint = backup

        backup = self.sphinx.input.sphinx.PAWHamiltonian.ekt
        self.sphinx.input.sphinx.PAWHamiltonian.ekt = 0.0001
        self.assertFalse(self.sphinx.validate_ready_to_run())
        self.sphinx.input.sphinx.PAWHamiltonian.ekt = backup

        backup = self.sphinx.input.sphinx.PAWHamiltonian.xc
        self.sphinx.input.sphinx.PAWHamiltonian.xc = "Wrong"
        self.assertFalse(self.sphinx.validate_ready_to_run())
        self.sphinx.input.sphinx.PAWHamiltonian.xc = backup

        backup = self.sphinx.input.sphinx.PAWHamiltonian.xc
        self.sphinx.input.sphinx.PAWHamiltonian.xc = "Wrong"
        self.assertFalse(self.sphinx.validate_ready_to_run())
        self.sphinx.input.sphinx.PAWHamiltonian.xc = backup

        backup = self.sphinx.input.sphinx.PAWHamiltonian.nEmptyStates
        self.sphinx.input.sphinx.PAWHamiltonian.nEmptyStates = 100
        self.assertFalse(self.sphinx.validate_ready_to_run())
        self.sphinx.input.sphinx.PAWHamiltonian.nEmptyStates = backup

        backup = self.sphinx.input.sphinx.structure.copy()
        self.sphinx.input.sphinx.structure.cell = [[0,0,0],[0,0,0],[0,0,0]]
        self.assertFalse(self.sphinx.validate_ready_to_run())
        self.sphinx.input.sphinx.structure = backup

        self.assertTrue(self.sphinx.validate_ready_to_run())

    def test_set_mixing_parameters(self):
        self.assertRaises(
            AssertionError, self.sphinx.set_mixing_parameters, "LDA", 7, 1.0, 1.0
        )
        self.assertRaises(
            AssertionError, self.sphinx.set_mixing_parameters, "PULAY", 1.2, 1.0, 1.0
        )
        self.assertRaises(
            ValueError, self.sphinx.set_mixing_parameters, "PULAY", 7, -0.1, 1.0
        )
        self.assertRaises(
            ValueError, self.sphinx.set_mixing_parameters, "PULAY", 7, 1.0, 2.0
        )
        self.sphinx.set_mixing_parameters("PULAY", 7, 0.5, 0.2)
        self.assertEqual(self.sphinx.input["rhoMixing"], 0.5)
        self.assertEqual(self.sphinx.input["spinMixing"], 0.2)

    def test_exchange_correlation_functional(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.sphinx.exchange_correlation_functional = "llda"
            self.assertEqual(len(w), 1)
            self.assertIsInstance(w[-1].message, SyntaxWarning)
        self.sphinx.exchange_correlation_functional = "pbe"
        self.assertEqual(self.sphinx.exchange_correlation_functional, "PBE")

    def test_write_structure(self):
        cell = (self.sphinx.structure.cell / BOHR_TO_ANGSTROM).tolist()
        pos_2 = (self.sphinx.structure.positions[1] / BOHR_TO_ANGSTROM).tolist()

        file_content = [
            f'cell = {cell};\n',
            'species {\n',
            '\telement = "Fe";\n',
            '\tatom {\n',
            '\t\tlabel = "spin_0.5";\n',
            '\t\tcoords = [0.0, 0.0, 0.0];\n',
            '\t\tmovable;\n',
            '\t}\n',
            '\tatom {\n',
            '\t\tlabel = "spin_0.5";\n',
            '\t\tcoords = [2.4566439620135014, 2.4566439620135014, 2.4566439620135014];\n',
            '\t}\n',
            '}\n',
        ]
        self.assertEqual(''.join(file_content), self.sphinx.input.sphinx.structure.to_sphinx())

    def test_collect_aborted(self):
        with self.assertRaises(AssertionError):
            self.sphinx_aborted.collect_output()

    def test_collect_2_5(self):
        output = self.sphinx_2_5._output_parser
        output.collect(directory=self.sphinx_2_5.working_directory)
        self.assertTrue(
            all(
                (
                        output._parse_dict["scf_computation_time"][0]
                        - np.roll(output._parse_dict["scf_computation_time"][0], 1)
                )[1:]
                > 0
            )
        )
        self.assertTrue(
            all(
                np.array(output._parse_dict["scf_energy_free"][0])
                - np.array(output._parse_dict["scf_energy_int"][0])
                < 0
            )
        )
        self.assertTrue(
            all(
                np.array(output._parse_dict["scf_energy_free"][0])
                - np.array(output._parse_dict["scf_energy_zero"][0])
                < 0
            )
        )
        list_values = [
            "scf_energy_int",
            "scf_energy_zero",
            "scf_energy_free",
            "scf_convergence",
            "scf_electronic_entropy",
            "atom_scf_spins",
        ]
        for list_one in list_values:
            for list_two in list_values:
                self.assertEqual(
                    len(output._parse_dict[list_one]), len(output._parse_dict[list_two])
                )

        rho = self.sphinx_2_5._output_parser.charge_density
        vel = self.sphinx_2_5._output_parser.electrostatic_potential
        self.assertIsNotNone(rho.total_data)
        self.assertIsNotNone(vel.total_data)

    def test_check_band_occupancy(self):
        self.sphinx_2_5.collect_output()
        self.assertTrue(self.sphinx_2_5.output.check_band_occupancy())

    def test_collect_2_3(self):
        file_location = os.path.join(
            self.file_location, "../static/sphinx/sphinx_test_2_3_hdf5/sphinx_test_2_3/"
        )
        residue_lst = np.loadtxt(file_location + "residue.dat")[:, 1].reshape(1, -1)
        residue_lst = (residue_lst * HARTREE_TO_EV).tolist()
        energy_int_lst = np.loadtxt(file_location + "energy.dat")[:, 2].reshape(1, -1)
        energy_int_lst = (energy_int_lst * HARTREE_TO_EV).tolist()
        with open(file_location + "sphinx.log") as ffile:
            energy_free_lst = [[float(line.split('=')[-1]) * HARTREE_TO_EV for line in ffile if line.startswith('F(')]]
        energy_zero_lst = [(0.5 * (np.array(ff) + np.array(uu))).tolist() for ff, uu in
                           zip(energy_free_lst, energy_int_lst)]
        eig_lst = [np.loadtxt(file_location + "eps.dat")[:, 1:].tolist()]
        self.sphinx_2_3.collect_output()
        self.assertEqual(
            residue_lst, self.sphinx_2_3._output_parser._parse_dict["scf_residue"]
        )
        self.assertEqual(
            energy_int_lst, self.sphinx_2_3._output_parser._parse_dict["scf_energy_int"]
        )
        self.assertEqual(
            energy_zero_lst,
            self.sphinx_2_3._output_parser._parse_dict["scf_energy_zero"],
        )
        self.assertEqual(
            eig_lst,
            self.sphinx_2_3._output_parser._parse_dict["bands_eigen_values"].tolist(),
        )
        self.assertEqual(
            energy_free_lst,
            self.sphinx_2_3._output_parser._parse_dict["scf_energy_free"],
        )
        self.assertEqual(
            21.952 * BOHR_TO_ANGSTROM ** 3, self.sphinx_2_3._output_parser._parse_dict["volume"]
        )

    def test_structure_parsing(self):
        self.sphinx_2_3._output_parser.collect_relaxed_hist(
            file_name="relaxedHist_2.sx", cwd=self.sphinx_2_3.working_directory
        )


if __name__ == "__main__":
    unittest.main()
