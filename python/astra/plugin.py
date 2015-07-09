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
import inspect

class base(object):

    def astra_init(self, cfg):
        try:
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
        except Exception as e:
            log.error(str(e))
            raise

    def astra_run(self, its):
        try:
            self.run(its)
        except Exception as e:
            log.error(str(e))
            raise

def register(name, className):
    """Register plugin with ASTRA.
    
    :param name: Plugin name to register
    :type name: :class:`str`
    :param className: Class name or class object to register
    :type className: :class:`str` or :class:`class`
    
    """
    p.register(name,className)

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