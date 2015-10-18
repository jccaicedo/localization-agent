'''
Created on Oct 16, 2015

@author: kuby
'''

import ctypes as cts

class Region(cts.Structure):
    '''
    classdocs
    '''
    
    _fields_ = [
        ("x", cts.POINTER(cts.c_float)),
        ("y", cts.POINTER(cts.c_float)),
        ("count", cts.c_int)]
        
        
class TraxClient(object):
    
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
    
    
    def initialize(self):
        return self.initialize()
    
    def nextFramePath(self):
        path = self.getFrame()
        
        if (path):
            path = str(path, encoding='utf-8')   
        else:
            path = ""
        
        return path
    
    def reportRegion(self, region):
        self.report(region)
        
    def releaseRegion(self, region):
        self.regionRelease(cts.byref(region))
        
    def quit(self):
        self.lib.vot_quit()