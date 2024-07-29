# -----------------------------------------------------------------------
# Copyright: 2010-2022, imec Vision Lab, University of Antwerp
#            2013-2022, CWI, Amsterdam
#
# Contact: astra@astra-toolbox.com
# Website: http://www.astra-toolbox.com/
#
# This file is part of the ASTRA Toolbox.
#
#
# The ASTRA Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The ASTRA Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.
#
# -----------------------------------------------------------------------


import astra
import numpy as np

class SIRTPlugin(astra.plugin.base):
    """SIRT.

    Options:

    'Relaxation': relaxation factor (optional)
    'MinConstraint': constrain values to at least this (optional)
    'MaxConstraint': constrain values to at most this (optional)
    """

    astra_name = "SIRT-PLUGIN"

    def initialize(self,cfg, Relaxation = 1, MinConstraint = None, MaxConstraint = None):
        self.W = astra.OpTomo(cfg['ProjectorId'])
        self.vid = cfg['ReconstructionDataId']
        self.sid = cfg['ProjectionDataId']
        self.min_constraint = MinConstraint
        self.max_constraint = MaxConstraint

        try:
            v = astra.data2d.get_shared(self.vid)
            s = astra.data2d.get_shared(self.sid)
            self.data_mod = astra.data2d
        except Exception:
            v = astra.data3d.get_shared(self.vid)
            s = astra.data3d.get_shared(self.sid)
            self.data_mod = astra.data3d

        self.R = self.W * np.ones(v.shape,dtype=np.float32).ravel();
        self.R[self.R < 0.000001] = np.inf
        self.R = 1 / self.R
        self.R = self.R.reshape(s.shape)

        self.mrC = self.W.T * np.ones(s.shape,dtype=np.float32).ravel();
        self.mrC[self.mrC < 0.000001] = np.inf
        self.mrC = -Relaxation / self.mrC
        self.mrC = self.mrC.reshape(v.shape)
        

    def run(self, its):
        v = self.data_mod.get_shared(self.vid)
        s = self.data_mod.get_shared(self.sid)
        tv = np.zeros(v.shape, dtype=np.float32)
        ts = np.zeros(s.shape, dtype=np.float32)
        W = self.W
        mrC = self.mrC
        R = self.R
        for i in range(its):
            W.FP(v,out=ts)
            ts -= s
            ts *= R # ts = R * (W*v - s)

            W.BP(ts,out=tv)
            tv *= mrC

            v += tv # v = v - rel * C * W' * ts

            if self.min_constraint is not None or self.max_constraint is not None:
                v.clip(min=self.min_constraint, max=self.max_constraint, out=v)

