'''
Created on Oct 16, 2015

@author: kuby
'''

import ctypes as cts
import numpy as np

class Region(cts.Structure):
    '''
    classdocs
    '''
    
    _fields_ = [
        ("x", cts.POINTER(cts.c_float)),
        ("y", cts.POINTER(cts.c_float)),
        ("count", cts.c_int)]

class Box():
    region = None
    
    def __init__(self, region=None):
        self.region = region
        
    def setRegion(self, box):
        x = box[0::2]
        y = box[1::2]
        count = len(x)
            
        x = (cts.c_float * count)(*x)
        y = (cts.c_float * count)(*y)
        
        self.region = cts.pointer(Region(x, y, cts.c_int(count)))
        

    def toList(self):
        
        x = np.fromiter(self.region.contents.x, dtype=np.float, count=self.region.contents.count)
        y = np.fromiter(self.region.contents.y, dtype=np.float, count=self.region.contents.count)
        
        box = [None] * (len(x)+len(y))
        box[0::2] = x
        box[1::2] = y
        
        return box
        
        
class TraxClient():
    
    def __init__(self):
        '''
        Constructor
        '''
        self.lib = cts.cdll.LoadLibrary("/home/fhdiaze/TrackingAgent/Benchmarks/vot-toolkit/tracker/examples/native/libvot.so")
        self.getFrame = self.lib.vot_frame
        self.getFrame.restype = cts.c_char_p
        
        self.initialize = self.lib.vot_initialize
        self.initialize.restype = cts.POINTER(Region)
        
        self.regionRelease = self.lib.vot_region_release
        self.regionRelease.argtypes = [cts.POINTER(cts.POINTER(Region))]
        
        self.report = self.lib.vot_report
        self.report.argtypes = [cts.POINTER(Region)]
    
    def getInitialRegion(self):
        return Box(self.initialize())
    
    def nextFrame(self):
        path = self.getFrame()
        
        if (path):
            path = str(path)   
        else:
            path = ""
        
        return path
    
    def reportRegion(self, box):
        self.report(box.region)
        
    def releaseRegion(self, box):
        self.regionRelease(cts.byref(box.region))
        
    def quit(self):
        self.lib.vot_quit()
        