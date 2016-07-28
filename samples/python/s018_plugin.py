#-----------------------------------------------------------------------
#Copyright 2015 Centrum Wiskunde & Informatica, Amsterdam
#
#Author: Daniel M. Pelt
#Contact: D.M.Pelt@cwi.nl
#Website: http://dmpelt.github.io/pyastratoolbox/
#
#
#This file is part of the Python interface to the
#All Scale Tomographic Reconstruction Antwerp Toolbox ("ASTRA Toolbox").
#
#The Python interface to the ASTRA Toolbox is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#The Python interface to the ASTRA Toolbox is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with the Python interface to the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.
#
#-----------------------------------------------------------------------

import astra
import numpy as np
import six

# Define the plugin class (has to subclass astra.plugin.base)
# Note that usually, these will be defined in a separate package/module
class LandweberPlugin(astra.plugin.base):
    """Example of an ASTRA plugin class, implementing a simple 2D Landweber algorithm.

    Options:

    'Relaxation': relaxation factor (optional)
    """

    # The astra_name variable defines the name to use to
    # call the plugin from ASTRA
    astra_name = "LANDWEBER-PLUGIN"

    def initialize(self,cfg, Relaxation = 1):
        self.W = astra.OpTomo(cfg['ProjectorId'])
        self.vid = cfg['ReconstructionDataId']
        self.sid = cfg['ProjectionDataId']
        self.rel = Relaxation

    def run(self, its):
        v = astra.data2d.get_shared(self.vid)
        s = astra.data2d.get_shared(self.sid)
        tv = np.zeros(v.shape, dtype=np.float32)
        ts = np.zeros(s.shape, dtype=np.float32)
        W = self.W
        for i in range(its):
            W.FP(v,out=ts)
            ts -= s # ts = W*v - s

            W.BP(ts,out=tv)
            tv *= self.rel / s.size

            v -= tv # v = v - rel * W'*(W*v-s) / s.size

if __name__=='__main__':

    vol_geom = astra.create_vol_geom(256, 256)
    proj_geom = astra.create_proj_geom('parallel', 1.0, 384, np.linspace(0,np.pi,180,False))

    # As before, create a sinogram from a phantom
    import scipy.io
    P = scipy.io.loadmat('phantom.mat')['phantom256']
    proj_id = astra.create_projector('cuda',proj_geom,vol_geom)

    # construct the OpTomo object
    W = astra.OpTomo(proj_id)

    sinogram = W * P
    sinogram = sinogram.reshape([180, 384])

    # Register the plugin with ASTRA
    # First we import the package that contains the plugin
    import s018_plugin
    # Then, we register the plugin class with ASTRA
    astra.plugin.register(s018_plugin.LandweberPlugin)

    # Get a list of registered plugins
    six.print_(astra.plugin.get_registered())

    # To get help on a registered plugin, use get_help
    six.print_(astra.plugin.get_help('LANDWEBER-PLUGIN'))

    # Create data structures
    sid = astra.data2d.create('-sino', proj_geom, sinogram)
    vid = astra.data2d.create('-vol', vol_geom)

    # Create config using plugin name
    cfg = astra.astra_dict('LANDWEBER-PLUGIN')
    cfg['ProjectorId'] = proj_id
    cfg['ProjectionDataId'] = sid
    cfg['ReconstructionDataId'] = vid

    # Create algorithm object
    alg_id = astra.algorithm.create(cfg)

    # Run algorithm for 100 iterations
    astra.algorithm.run(alg_id, 100)

    # Get reconstruction
    rec = astra.data2d.get(vid)

    # Options for the plugin go in cfg['option']
    cfg = astra.astra_dict('LANDWEBER-PLUGIN')
    cfg['ProjectorId'] = proj_id
    cfg['ProjectionDataId'] = sid
    cfg['ReconstructionDataId'] = vid
    cfg['option'] = {}
    cfg['option']['Relaxation'] = 1.5
    alg_id_rel = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id_rel, 100)
    rec_rel = astra.data2d.get(vid)

    # We can also use OpTomo to call the plugin
    rec_op = W.reconstruct('LANDWEBER-PLUGIN', sinogram, 100, extraOptions={'Relaxation':1.5})

    import pylab as pl
    pl.gray()
    pl.figure(1)
    pl.imshow(rec,vmin=0,vmax=1)
    pl.figure(2)
    pl.imshow(rec_rel,vmin=0,vmax=1)
    pl.figure(3)
    pl.imshow(rec_op,vmin=0,vmax=1)
    pl.show()

    # Clean up.
    astra.projector.delete(proj_id)
    astra.algorithm.delete([alg_id, alg_id_rel])
    astra.data2d.delete([vid, sid])
