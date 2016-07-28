#-----------------------------------------------------------------------
#Copyright 2013 Centrum Wiskunde & Informatica, Amsterdam
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

from . import plugin_c as p
from . import log
from . import data2d
from . import data2d_c
from . import data3d
from . import projector
import inspect
import traceback

class base(object):

    def astra_init(self, cfg):
        args, varargs, varkw, defaults = inspect.getargspec(self.initialize)
        if not defaults is None:
            nopt = len(defaults)
        else:
            nopt = 0
        if nopt>0:
            req = args[2:-nopt]
            opt = args[-nopt:]
        else:
            req = args[2:]
            opt = []

        try:
            optDict = cfg['options']
        except KeyError:
            optDict = {}

        cfgKeys = set(optDict.keys())
        reqKeys = set(req)
        optKeys = set(opt)

        if not reqKeys.issubset(cfgKeys):
            for key in reqKeys.difference(cfgKeys):
                log.error("Required option '" + key + "' for plugin '" + self.__class__.__name__ + "' not specified")
            raise ValueError("Missing required options")

        if not cfgKeys.issubset(reqKeys | optKeys):
            log.warn(self.__class__.__name__ + ": unused configuration option: " + str(list(cfgKeys.difference(reqKeys | optKeys))))

        args = [optDict[k] for k in req]
        kwargs = dict((k,optDict[k]) for k in opt if k in optDict)
        self.initialize(cfg, *args, **kwargs)

class ReconstructionAlgorithm2D(base):

    def astra_init(self, cfg):
        self.pid = cfg['ProjectorId']
        self.s = data2d.get_shared(cfg['ProjectionDataId'])
        self.v = data2d.get_shared(cfg['ReconstructionDataId'])
        self.vg = projector.volume_geometry(self.pid)
        self.pg = projector.projection_geometry(self.pid)
        if not data2d_c.check_compatible(cfg['ProjectionDataId'], self.pid):
            raise ValueError("Projection data and projector not compatible")
        if not data2d_c.check_compatible(cfg['ReconstructionDataId'], self.pid):
            raise ValueError("Reconstruction data and projector not compatible")
        super(ReconstructionAlgorithm2D,self).astra_init(cfg)

class ReconstructionAlgorithm3D(base):

    def astra_init(self, cfg):
        self.pid = cfg['ProjectorId']
        self.s = data3d.get_shared(cfg['ProjectionDataId'])
        self.v = data3d.get_shared(cfg['ReconstructionDataId'])
        self.vg = data3d.get_geometry(cfg['ReconstructionDataId'])
        self.pg = data3d.get_geometry(cfg['ProjectionDataId'])
        super(ReconstructionAlgorithm3D,self).astra_init(cfg)

def register(className):
    """Register plugin with ASTRA.
    
    :param className: Class name or class object to register
    :type className: :class:`str` or :class:`class`
    
    """
    p.register(className)

def get_registered():
    """Get dictionary of registered plugins.
    
    :returns: :class:`dict` -- Registered plugins.
    
    """
    return p.get_registered()

def get_help(name):
    """Get help for registered plugin.
    
    :param name: Plugin name to get help for
    :type name: :class:`str`
    :returns: :class:`str` -- Help string (docstring).
    
    """
    return p.get_help(name)