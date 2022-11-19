from __future__ import absolute_import
import os
import sys

from .osutils import mkdir_if_missing


class Logger():
    """
    Class to modify the 'write' function of sys.stdout object.
    print is just a thin wrapper that formats the inputs (modifiable, but by default with a space between args and newline at the end) 
    and calls the write function of a given object. By default this object is sys.stdout (print (something) == sys.stdout.write(something))
    redefine the sys.stdout with object from this class gives the ability to print to file if path is provided during initializtion.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg) # sys.stdout.write()
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

