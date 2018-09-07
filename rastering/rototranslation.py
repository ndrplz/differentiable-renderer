import collections

from math import cos
from math import radians
from math import sin

import numpy as np
from numpy import matmul


Vector = collections.namedtuple('Vector', ['x', 'y', 'z'])


class RotoTranslation:
    def __init__(self, rotation: Vector, translation: Vector, angle_unit: str, notation: str='XYZ'):
        self.rotation    = rotation
        self.translation = translation

        self.angle_unit = angle_unit
        if self.angle_unit == 'degrees':
            self.rotation = Vector(*[radians(alpha) for alpha in rotation])

        self.R_x = None
        self.R_y = None
        self.R_z = None
        self.T   = None

        self.matrix = None
        self.notation = notation

        self._update_matrix()

    def _update_matrix(self):
        # Convention: counter-clockwise rotation
        self.R_x = np.array([[1.0,              0.0,               0.0,              0.0],
                             [0.0,              cos(self.rotation.x),  -sin(self.rotation.x),  0.0],
                             [0.0,              sin(self.rotation.x),   cos(self.rotation.x),  0.0],
                             [0.0,              0.0,               0.0,              1.0]])

        self.R_y = np.array([[cos(self.rotation.y),  0.0,               sin(self.rotation.y),  0.0],
                             [0.0,              1.0,               0.0,              0.0],
                             [-sin(self.rotation.y), 0.0,               cos(self.rotation.y),  0.0],
                             [0.0,              0.0,               0.0,              1.0]])

        self.R_z = np.array([[cos(self.rotation.z),  -sin(self.rotation.z),   0.0,              0.0],
                             [sin(self.rotation.z),   cos(self.rotation.z),   0.0,              0.0],
                             [0.0,               0.0,               1.0,              0.0],
                             [0.0,               0.0,               0.0,              1.0]])

        self.T   = np.array([[1.0,               0.0,               0.0,              self.translation.x],
                             [0.0,               1.0,               0.0,              self.translation.y],
                             [0.0,               0.0,               1.0,              self.translation.z],
                             [0.0,               0.0,               0.0,              1.0]])

        if self.notation == 'XYZ':
            self.R = matmul(self.R_z, matmul(self.R_y, self.R_x))
        if self.notation == 'YXZ':
            self.R = matmul(self.R_z, matmul(self.R_x, self.R_y))
        if self.notation == 'YZX':
            self.R = matmul(self.R_x, matmul(self.R_z, self.R_y))
        if self.notation == 'XZY':
            self.R = matmul(self.R_y, matmul(self.R_z, self.R_x))
        if self.notation == 'ZYX':
            self.R = matmul(self.R_x, matmul(self.R_y, self.R_z))
        if self.notation == 'ZXY':
            self.R = matmul(self.R_y, matmul(self.R_x, self.R_z))

        self.matrix = matmul(self.T, self.R)

    def __str__(self):
        return RotoTranslation.pretty_string(self.matrix)

    @staticmethod
    def pretty_string(matrix, border=False, border_len=40):
        pretty_str = '\n'.join([''.join(['{:9.3f}'.format(item) for item in row]) for row in matrix])
        if border:
            pretty_str = border_len * '*' + '\n' + pretty_str + '\n' + border_len * '*'
        return pretty_str

    @property
    def alpha_x(self):
        alpha_x = self.rotation.x
        if self.angle_unit == 'degrees':
            alpha_x = np.degrees(alpha_x)
        return alpha_x

    @property
    def alpha_y(self):
        alpha_y = self.rotation.y
        if self.angle_unit == 'degrees':
            alpha_y = np.degrees(alpha_y)
        return alpha_y

    @property
    def alpha_z(self):
        alpha_z = self.rotation.z
        if self.angle_unit == 'degrees':
            alpha_z = np.degrees(alpha_z)
        return alpha_z

    @property
    def t_x(self):
        return self.translation.x

    @property
    def t_y(self):
        return self.translation.y

    @property
    def t_z(self):
        return self.translation.z

    @alpha_x.setter
    def alpha_x(self, value):

        if self.angle_unit == 'degrees':
            value = np.radians(value)

        self.rotation = Vector(value, self.rotation.y, self.rotation.z)
        self._update_matrix()

    @alpha_y.setter
    def alpha_y(self, value):
        if self.angle_unit == 'degrees':
            value = np.radians(value)

        self.rotation = Vector(self.rotation.x, value, self.rotation.z)
        self._update_matrix()

    @alpha_z.setter
    def alpha_z(self, value):
        if self.angle_unit == 'degrees':
            value = np.radians(value)

        self.rotation = Vector(self.rotation.x, self.rotation.y, value)
        self._update_matrix()

    @t_x.setter
    def t_x(self, value):
        self.translation = Vector(value, self.translation.y, self.translation.z)
        self._update_matrix()

    @t_y.setter
    def t_y(self, value):
        self.translation = Vector(self.translation.x, value, self.translation.z)
        self._update_matrix()

    @t_z.setter
    def t_z(self, value):
        self.translation = Vector(self.translation.x, self.translation.y, value)
        self._update_matrix()
