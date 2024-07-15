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

class CGLSPlugin(astra.plugin.base):
    """CGLS."""

    astra_name = "CGLS-PLUGIN"

    def initialize(self,cfg):
        self.W = astra.OpTomo(cfg['ProjectorId'])
        self.vid = cfg['ReconstructionDataId']
        self.sid = cfg['ProjectionDataId']

        try:
            v = astra.data2d.get_shared(self.vid)
            s = astra.data2d.get_shared(self.sid)
            self.data_mod = astra.data2d
        except Exception:
            v = astra.data3d.get_shared(self.vid)
            s = astra.data3d.get_shared(self.sid)
            self.data_mod = astra.data3d

    def run(self, its):
        v = self.data_mod.get_shared(self.vid)
        s = self.data_mod.get_shared(self.sid)
        z = np.zeros(v.shape, dtype=np.float32)
        p = np.zeros(v.shape, dtype=np.float32)
        r = np.zeros(s.shape, dtype=np.float32)
        w = np.zeros(s.shape, dtype=np.float32)
        W = self.W

        # r = s - W*v
        W.FP(v, out=w)
        r[:] = s
        r -= w

        # p = W'*r
        W.BP(r, out=p)

        # gamma = <p,p>
        gamma = np.dot(p.ravel(), p.ravel())

        for i in range(its):
            # w = W * p
            W.FP(p, out=w)

            # alpha = gamma / <w,w>
            alpha = gamma / np.dot(w.ravel(), w.ravel())

            # v += alpha * p
            z[:] = p
            z *= alpha
            v += z

            # r -= alpha * w
            w *= -alpha;
            r += w

            # z = W' * r
            W.BP(r, out=z)

            # beta = <z,z> / gamma
            newgamma = np.dot(z.ravel(), z.ravel())
            beta = newgamma / gamma

            # gamma = <z,z>
            gamma = newgamma

            # p = z + beta * p
            p *= beta
            p += z

