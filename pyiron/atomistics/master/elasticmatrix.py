# coding: utf-8
# Distributed under the terms of the GPL3 License.

# Ref. Com. Phys. Comm. 184 (2013) 1861-1873
# and ElaStic source code
# for more details

from collections import OrderedDict

import numpy as np
import spglib

from pyiron.atomistics.master.parallel import AtomisticParallelMaster

__author__ = "Yury Lysogorskiy"
__copyright__ = "Copyright 2020, ICAMS, RUB"
__version__ = "1.0"
__status__ = "development"
__date__ = "Sep 1, 2017"


def find_symmetry_group_number(struct):
    dataset = spglib.get_symmetry_dataset(struct)
    SGN = dataset["number"]
    return SGN


Ls_Dic = {
    '01': [1., 1., 1., 0., 0., 0.],
    '02': [1., 0., 0., 0., 0., 0.],
    '03': [0., 1., 0., 0., 0., 0.],
    '04': [0., 0., 1., 0., 0., 0.],
    '05': [0., 0., 0., 2., 0., 0.],
    '06': [0., 0., 0., 0., 2., 0.],
    '07': [0., 0., 0., 0., 0., 2.],
    '08': [1., 1., 0., 0., 0., 0.],
    '09': [1., 0., 1., 0., 0., 0.],
    '10': [1., 0., 0., 2., 0., 0.],
    '11': [1., 0., 0., 0., 2., 0.],
    '12': [1., 0., 0., 0., 0., 2.],
    '13': [0., 1., 1., 0., 0., 0.],
    '14': [0., 1., 0., 2., 0., 0.],
    '15': [0., 1., 0., 0., 2., 0.],
    '16': [0., 1., 0., 0., 0., 2.],
    '17': [0., 0., 1., 2., 0., 0.],
    '18': [0., 0., 1., 0., 2., 0.],
    '19': [0., 0., 1., 0., 0., 2.],
    '20': [0., 0., 0., 2., 2., 0.],
    '21': [0., 0., 0., 2., 0., 2.],
    '22': [0., 0., 0., 0., 2., 2.],
    '23': [0., 0., 0., 2., 2., 2.],
    '24': [-1., .5, .5, 0., 0., 0.],
    '25': [.5, -1., .5, 0., 0., 0.],
    '26': [.5, .5, -1., 0., 0., 0.],
    '27': [1., -1., 0., 0., 0., 0.],
    '28': [1., -1., 0., 0., 0., 2.],
    '29': [0., 1., -1., 0., 0., 2.],
    '30': [.5, .5, -1., 0., 0., 2.],
    '31': [1., 0., 0., 2., 2., 0.],
    '32': [1., 1., -1., 0., 0., 0.],
    '33': [1., 1., 1., -2., -2., -2.],
    '34': [.5, .5, -1., 2., 2., 2.],
    '35': [0., 0., 0., 2., 2., 4.],
    '36': [1., 2., 3., 4., 5., 6.],
    '37': [-2., 1., 4., -3., 6., -5.],
    '38': [3., -5., -1., 6., 2., -4.],
    '39': [-4., -6., 5., 1., -3., 2.],
    '40': [5., 4., 6., -2., -1., -3.],
    '41': [-6., 3., -2., 5., -4., 1.]}


def get_symmetry_family_from_SGN(SGN):
    if 1 <= SGN <= 2:  # Triclinic
        LC = 'N'
    elif 3 <= SGN <= 15:  # Monoclinic
        LC = 'M'
    elif 16 <= SGN <= 74:  # Orthorhombic
        LC = 'O'
    elif 75 <= SGN <= 88:  # Tetragonal II
        LC = 'TII'
    elif 89 <= SGN <= 142:  # Tetragonal I
        LC = 'TI'
    elif 143 <= SGN <= 148:  # Rhombohedral II
        LC = 'RII'
    elif 149 <= SGN <= 167:  # Rhombohedral I
        LC = 'RI'
    elif 168 <= SGN <= 176:  # Hexagonal II
        LC = 'HII'
    elif 177 <= SGN <= 194:  # Hexagonal I
        LC = 'HI'
    elif 195 <= SGN <= 206:  # Cubic II
        LC = 'CII'
    elif 207 <= SGN <= 230:  # Cubic I
        LC = 'CI'
    else:
        raise ValueError('SGN should be 1 <= SGN <= 230')
    return LC


def get_LAG_Strain_List(LC):
    if LC == 'CI' or LC == 'CII':
        Lag_strain_list = ['01', '08', '23']
    elif LC == 'HI' or LC == 'HII':
        Lag_strain_list = ['01', '26', '04', '03', '17']
    elif LC == 'RI':
        Lag_strain_list = ['01', '08', '04', '02', '05', '10']
    elif LC == 'RII':
        Lag_strain_list = ['01', '08', '04', '02', '05', '10', '11']
    elif LC == 'TI':
        Lag_strain_list = ['01', '26', '27', '04', '05', '07']
    elif LC == 'TII':
        Lag_strain_list = ['01', '26', '27', '28', '04', '05', '07']
    elif LC == 'O':
        Lag_strain_list = ['01', '26', '25', '27', '03', '04', '05', '06', '07']
    elif LC == 'M':
        Lag_strain_list = ['01', '25', '24', '28', '29', '27', '20', '12', '03', '04', '05', '06', '07']
    else:  # (LC == 'N'):
        Lag_strain_list = ['02', '03', '04', '05', '06', '07', '08', '09', '10', '11',
                           '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22']
    return Lag_strain_list


def get_C_from_A2(A2, LC):
    C = np.zeros((6, 6))

    # %!%!%--- Cubic structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%
    if LC == 'CI' or LC == 'CII':
        C[0, 0] = -2. * (A2[0] - 3. * A2[1]) / 3.
        C[1, 1] = C[0, 0]
        C[2, 2] = C[0, 0]
        C[3, 3] = A2[2] / 6.
        C[4, 4] = C[3, 3]
        C[5, 5] = C[3, 3]
        C[0, 1] = (2. * A2[0] - 3. * A2[1]) / 3.
        C[0, 2] = C[0, 1]
        C[1, 2] = C[0, 1]
    # --------------------------------------------------------------------------------------------------

    # %!%!%--- Hexagonal structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%
    if LC == 'HI' or LC == 'HII':
        C[0, 0] = 2. * A2[3]
        C[0, 1] = 2. / 3. * A2[0] + 4. / 3. * A2[1] - 2. * A2[2] - 2. * A2[3]
        C[0, 2] = 1. / 6. * A2[0] - 2. / 3. * A2[1] + 0.5 * A2[2]
        C[1, 1] = C[0, 0]
        C[1, 2] = C[0, 2]
        C[2, 2] = 2. * A2[2]
        C[3, 3] = -0.5 * A2[2] + 0.5 * A2[4]
        C[4, 4] = C[3, 3]
        C[5, 5] = .5 * (C[0, 0] - C[0, 1])
    # --------------------------------------------------------------------------------------------------

    # %!%!%--- Rhombohedral I structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!
    if LC == 'RI':
        C[0, 0] = 2. * A2[3]
        C[0, 1] = A2[1] - 2. * A2[3]
        C[0, 2] = .5 * (A2[0] - A2[1] - A2[2])
        C[0, 3] = .5 * (-A2[3] - A2[4] + A2[5])
        C[1, 1] = C[0, 0]
        C[1, 2] = C[0, 2]
        C[1, 3] = -C[0, 3]
        C[2, 2] = 2. * A2[2]
        C[3, 3] = .5 * A2[4]
        C[4, 4] = C[3, 3]
        C[4, 5] = C[0, 3]
        C[5, 5] = .5 * (C[0, 0] - C[0, 1])
    # --------------------------------------------------------------------------------------------------

    # %!%!%--- Rhombohedral II structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%
    if LC == 'RII':
        C[0, 0] = 2. * A2[3]
        C[0, 1] = A2[1] - 2. * A2[3]
        C[0, 2] = .5 * (A2[0] - A2[1] - A2[2])
        C[0, 3] = .5 * (-A2[3] - A2[4] + A2[5])
        C[0, 4] = .5 * (-A2[3] - A2[4] + A2[6])
        C[1, 1] = C[0, 0]
        C[1, 2] = C[0, 2]
        C[1, 3] = -C[0, 3]
        C[1, 4] = -C[0, 4]
        C[2, 2] = 2. * A2[2]
        C[3, 3] = .5 * A2[4]
        C[3, 5] = -C[0, 4]
        C[4, 4] = C[3, 3]
        C[4, 5] = C[0, 3]
        C[5, 5] = .5 * (C[0, 0] - C[0, 1])
    # --------------------------------------------------------------------------------------------------

    # %!%!%--- Tetragonal I structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!
    if LC == 'TI':
        C[0, 0] = (A2[0] + 2. * A2[1]) / 3. + .5 * A2[2] - A2[3]
        C[0, 1] = (A2[0] + 2. * A2[1]) / 3. - .5 * A2[2] - A2[3]
        C[0, 2] = A2[0] / 6. - 2. * A2[1] / 3. + .5 * A2[3]
        C[1, 1] = C[0, 0]
        C[1, 2] = C[0, 2]
        C[2, 2] = 2. * A2[3]
        C[3, 3] = .5 * A2[4]
        C[4, 4] = C[3, 3]
        C[5, 5] = .5 * A2[5]
    # --------------------------------------------------------------------------------------------------

    # %!%!%--- Tetragonal II structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%
    if LC == 'TII':
        C[0, 0] = (A2[0] + 2. * A2[1]) / 3. + .5 * A2[2] - A2[4]
        C[1, 1] = C[0, 0]
        C[0, 1] = (A2[0] + 2. * A2[1]) / 3. - .5 * A2[2] - A2[4]
        C[0, 2] = A2[0] / 6. - (2. / 3.) * A2[1] + .5 * A2[4]
        C[0, 5] = (-A2[2] + A2[3] - A2[6]) / 4.
        C[1, 2] = C[0, 2]
        C[1, 5] = -C[0, 5]
        C[2, 2] = 2. * A2[4]
        C[3, 3] = .5 * A2[5]
        C[4, 4] = C[3, 3]
        C[5, 5] = .5 * A2[6]
    # --------------------------------------------------------------------------------------------------

    # %!%!%--- Orthorhombic structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!
    if LC == 'O':
        C[0, 0] = 2. * A2[0] / 3. + 4. * A2[1] / 3. + A2[3] - 2. * A2[4] - 2. * A2[5]
        C[0, 1] = 1. * A2[0] / 3. + 2. * A2[1] / 3. - .5 * A2[3] - A2[5]
        C[0, 2] = 1. * A2[0] / 3. - 2. * A2[1] / 3. + 4. * A2[2] / 3. - .5 * A2[3] - A2[4]
        C[1, 1] = 2. * A2[4]
        C[1, 2] = -2. * A2[1] / 3. - 4. * A2[2] / 3. + .5 * A2[3] + A2[4] + A2[5]
        C[2, 2] = 2. * A2[5]
        C[3, 3] = .5 * A2[6]
        C[4, 4] = .5 * A2[7]
        C[5, 5] = .5 * A2[8]
    # --------------------------------------------------------------------------------------------------

    # %!%!%--- Monoclinic structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!
    if LC == 'M':
        C[0, 0] = 2. * A2[0] / 3. + 8. * (A2[1] + A2[2]) / 3. - 2. * (A2[5] + A2[8] + A2[9])
        C[0, 1] = A2[0] / 3. + 4. * (A2[1] + A2[2]) / 3. - 2. * A2[5] - A2[9]
        C[0, 2] = (A2[0] - 4. * A2[2]) / 3. + A2[5] - A2[8]
        C[0, 5] = -1. * A2[0] / 6. - 2. * (A2[1] + A2[2]) / 3. + .5 * (A2[5] + A2[7] + A2[8] + A2[9] - A2[12])
        C[1, 1] = 2. * A2[8]
        C[1, 2] = -4. * (2. * A2[1] + A2[2]) / 3. + 2. * A2[5] + A2[8] + A2[9] + A2[12]
        C[1, 5] = -1. * A2[0] / 6. - 2. * (A2[1] + A2[2]) / 3. - .5 * A2[3] + A2[5] + .5 * (A2[7] + A2[8] + A2[9])
        C[2, 2] = 2. * A2[9]
        C[2, 5] = -1. * A2[0] / 6. + 2. * A2[1] / 3. - .5 * (A2[3] + A2[4] - A2[7] - A2[8] - A2[9] - A2[12])
        C[3, 3] = .5 * A2[10]
        C[3, 4] = .25 * (A2[6] - A2[10] - A2[11])
        C[4, 4] = .5 * A2[11]
        C[5, 5] = .5 * A2[12]
    # --------------------------------------------------------------------------------------------------

    # %!%!%--- Triclinic structures ---%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%!%
    if LC == 'N':
        C[0, 0] = 2. * A2[0]
        C[0, 1] = 1. * (-A2[0] - A2[1] + A2[6])
        C[0, 2] = 1. * (-A2[0] - A2[2] + A2[7])
        C[0, 3] = .5 * (-A2[0] - A2[3] + A2[8])
        C[0, 4] = .5 * (-A2[0] + A2[9] - A2[4])
        C[0, 5] = .5 * (-A2[0] + A2[10] - A2[5])
        C[1, 1] = 2. * A2[1]
        C[1, 2] = 1. * (A2[11] - A2[1] - A2[2])
        C[1, 3] = .5 * (A2[12] - A2[1] - A2[3])
        C[1, 4] = .5 * (A2[13] - A2[1] - A2[4])
        C[1, 5] = .5 * (A2[14] - A2[1] - A2[5])
        C[2, 2] = 2. * A2[2]
        C[2, 3] = .5 * (A2[15] - A2[2] - A2[3])
        C[2, 4] = .5 * (A2[16] - A2[2] - A2[4])
        C[2, 5] = .5 * (A2[17] - A2[2] - A2[5])
        C[3, 3] = .5 * A2[3]
        C[3, 4] = .25 * (A2[18] - A2[3] - A2[4])
        C[3, 5] = .25 * (A2[19] - A2[3] - A2[5])
        C[4, 4] = .5 * A2[4]
        C[4, 5] = .25 * (A2[20] - A2[4] - A2[5])
        C[5, 5] = .5 * A2[5]
    return C


class ElasticMatrixCalculator(object):
    zero_strain_job_name = "s_e_0"

    @staticmethod
    def subjob_name(i, eps):
        """

        Args:
            i: 
            eps: 

        Returns:

        """
        return ("s_%s_e_%.5f" % (i, eps)).replace(".", "_").replace("-", "m")

    def __init__(self, basis_ref, num_of_point=5, eps_range = 0.005, sqrt_eta = True, fit_order = 2):
        self.basis_ref = basis_ref.copy()
        self.num_of_point = num_of_point
        self.eps_range = eps_range
        self.sqrt_eta = sqrt_eta
        self.fit_order = fit_order
        self._data = OrderedDict()
        self._structure_dict = OrderedDict()


    def symmetry_analysis(self):
        """

        Returns:

        """
        self.SGN = find_symmetry_group_number(self.basis_ref)
        self._data["SGN"] = self.SGN
        self.v0 = self.basis_ref.get_volume()
        self._data["v0"] = self.v0
        self.LC= get_symmetry_family_from_SGN(self.SGN)
        self._data["LC"] = self.LC
        self.Lag_strain_list= get_LAG_Strain_List(self.LC)
        self._data["Lag_strain_list"] = self.Lag_strain_list
        self.epss = np.linspace(-self.eps_range, self.eps_range, self.num_of_point)
        self._data["epss"] = self.epss


    def generate_structures(self):
        """

        Returns:

        """
        self.symmetry_analysis()
        basis_ref = self.basis_ref
        Lag_strain_list = self.Lag_strain_list
        epss = self.epss

        if 0.0 in epss:
            self._structure_dict[self.zero_strain_job_name] = basis_ref.copy()

        for lag_strain in Lag_strain_list:
            Ls_list = Ls_Dic[lag_strain]
            for eps in epss:

                if eps == 0.0:
                    continue

                Ls = np.zeros(6)
                for ii in range(6):
                    Ls[ii] = Ls_list[ii]
                Lv = eps * Ls

                eta_matrix = np.zeros((3, 3))

                eta_matrix[0, 0] = Lv[0]
                eta_matrix[0, 1] = Lv[5] / 2.
                eta_matrix[0, 2] = Lv[4] / 2.

                eta_matrix[1, 0] = Lv[5] / 2.
                eta_matrix[1, 1] = Lv[1]
                eta_matrix[1, 2] = Lv[3] / 2.

                eta_matrix[2, 0] = Lv[4] / 2.
                eta_matrix[2, 1] = Lv[3] / 2.
                eta_matrix[2, 2] = Lv[2]

                norm = 1.0
                eps_matrix = eta_matrix
                if np.linalg.norm(eta_matrix) > 0.7:
                    raise Exception("Too large deformation %g" % eps)

                if self.sqrt_eta:
                    while norm > 1.e-10:
                        x = eta_matrix - np.dot(eps_matrix, eps_matrix) / 2.
                        norm = np.linalg.norm(x - eps_matrix)
                        eps_matrix = x

                # --- Calculating the M_new matrix ---------------------------------------------------------
                i_matrix = np.array([[1., 0., 0.],
                                     [0., 1., 0.],
                                     [0., 0., 1.]])
                def_matrix = i_matrix + eps_matrix
                scell = np.dot(basis_ref.get_cell(), def_matrix)
                nstruct = basis_ref.copy()
                nstruct.set_cell(scell, scale_atoms=True)

                jobname = self.subjob_name(lag_strain, eps)

                self._structure_dict[jobname] = nstruct

        return self._structure_dict


    def analyse_structures(self, output_dict):
        """

        Returns:

        """
        self.symmetry_analysis()
        epss = self.epss
        Lag_strain_list = self.Lag_strain_list

        ene0 = None
        if 0.0 in epss:
            ene0 = output_dict[self.zero_strain_job_name]
        self._data["e0"] = ene0
        strain_energy = []
        for lag_strain in Lag_strain_list:
            strain_energy.append([])
            for eps in epss:
                if not eps == 0.0:
                    jobname = self.subjob_name(lag_strain, eps)
                    ene = output_dict[jobname]
                else:
                    ene = ene0
                strain_energy[-1].append((eps, ene))
        self._data["strain_energy"] = strain_energy
        self.fit_elastic_matrix()

    def calculate_modulus(self):
        """
        
        Returns:

        """
        C = self._data['C']

        BV = (C[0, 0] + C[1, 1] + C[2, 2] + 2 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9
        GV = ((C[0, 0] + C[1, 1] + C[2, 2]) - (C[0, 1] + C[0, 2] + C[1, 2]) + 3 * (C[3, 3] + C[4, 4] + C[5, 5])) / 15
        EV = (9 * BV * GV) / (3 * BV + GV)
        nuV = (1.5 * BV - GV) / (3 * BV + GV)
        self._data["BV"] = BV
        self._data["GV"] = GV
        self._data["EV"] = EV
        self._data["nuV"] = nuV

        try:
            S = np.linalg.inv(C)

            BR = 1 / (S[0, 0] + S[1, 1] + S[2, 2] + 2 * (S[0, 1] + S[0, 2] + S[1, 2]))
            GR = 15 / (4 * (S[0, 0] + S[1, 1] + S[2, 2]) -
                       4 * (S[0, 1] + S[0, 2] + S[1, 2]) +
                       3 * (S[3, 3] + S[4, 4] + S[5, 5]))
            ER = (9 * BR * GR) / (3 * BR + GR)
            nuR = (1.5 * BR - GR) / (3 * BR + GR)

            BH = 0.50 * (BV + BR)
            GH = 0.50 * (GV + GR)
            EH = (9. * BH * GH) / (3. * BH + GH)
            nuH = (1.5 * BH - GH) / (3. * BH + GH)

            AVR = 100. * (GV - GR) / (GV + GR)
            self._data['S'] = S

            self._data['BR'] = BR
            self._data['GR'] = GR
            self._data['ER'] = ER
            self._data['nuR'] = nuR

            self._data['BH'] = BH
            self._data['GH'] = GH
            self._data['EH'] = EH
            self._data['nuH'] = nuH

            self._data['AVR'] = AVR
        except np.linalg.LinAlgError as e:
            print("LinAlgError:", e)

        eigval = np.linalg.eig(C)
        self._data['C_eigval'] = eigval


    def fit_elastic_matrix(self):
        """

        Returns:

        """
        strain_ene = self._data["strain_energy"]

        v0 = self._data["v0"]
        LC = self._data["LC"]
        A2 = []
        fit_order = int(self.fit_order)
        for s_e in strain_ene:
            ss = np.transpose(s_e)
            coeffs = np.polyfit(ss[0], ss[1]/v0, fit_order)
            A2.append(coeffs[fit_order - 2])


        A2 = np.array(A2)
        C = get_C_from_A2(A2, LC)

        for i in range(5):
            for j in range(i + 1, 6):
                C[j, i] = C[i, j]

        CONV = 160.21766208  # From eV/Ang^3 to GPa

        C *= CONV
        self._data["C"] = C
        self._data["A2"] = A2
        self.calculate_modulus()


class ElasticMatrix(AtomisticParallelMaster):

    hdf_storage_group = "elasticmatrix"

    def __init__(self, project, job_name="elasticmatrix"):
        super(ElasticMatrix, self).__init__(project, job_name)
        self.__name__ = "ElasticMatrix"
        self.__version__ = '0.0.1'
        self.input['num_of_points'] = (5, 'number of sample point per deformation directions')
        self.input['fit_order'] = (2, 'order of the fit polynom')
        self.input['eps_range'] = (0.005, 'strain variation')
        self.input['relax_atoms'] = (True, 'relax atoms in deformed structure')
        self.input['sqrt_eta'] = (True, 'calculate self-consistently sqrt of stress matrix eta')


    def create_calculator(self):
        ham = self.ref_job.copy()
        basis_ref = ham.structure.copy()
        self.property_calculator = ElasticMatrixCalculator(basis_ref,
                                                           num_of_point=self.input['num_of_points'],
                                                           eps_range=self.input['eps_range'],
                                                           sqrt_eta=self.input['sqrt_eta'],
                                                           fit_order=int(self.input['fit_order'])
                                                           )
        super(ElasticMatrix,self).create_calculator()

    def create_jobs(self):
        ham = self.ref_job.copy()
        basis_ref = ham.structure.copy()
        self.create_calculator()
        self.save_data_to_hdf()
        job_lst = []

        self.submission_status.submitted_jobs = 0
        for job_name, structure in self.structure_dict.items():
            if self.job_id and self.project.db.get_item_by_id(self.job_id)['status'] not in ['finished', 'aborted']:
                ham = self._create_child_job(job_name)
                if ham.server.run_mode.non_modal and self.get_child_cores() + ham.server.cores > self.server.cores:
                    break
                self._logger.debug('create job: %s %s', ham.job_info_str, ham.master_id)
                ham.structure = structure
                if self.input["relax_atoms"]:
                    ham.calc_minimize(pressure=None)
                else:
                    ham.calc_static()
                self._logger.info('ElasticMatrix: run job {}'.format(ham.job_name))
                self.submission_status.submit_next()
                if not ham.status.finished:
                    ham.run()
                self._logger.info('ElasticMatrix: finished job {}'.format(ham.job_name))
                self.ref_job.structure = basis_ref
                if ham.server.run_mode.thread:
                    job_lst.append(ham._process)
            else:
                self.refresh_job_status()
        self.structure = basis_ref
        process_lst = [process.communicate() for process in job_lst if process]
        self.status.suspended = True

    def collect_output(self):
        self.pre_collect_output()

        energies = {}
        self._data["id"]=[]
        for job_id in self.child_ids:
            ham = self.project_hdf5.inspect(job_id)
            en = ham["output/generic/energy_tot"][-1]
            energies[ham.job_name] = en
            self._data["id"].append(ham.job_id)

        self.analyse_structures(energies)