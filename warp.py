import os,sys

class Warping():
    def __init__(self):
        self.sx = 1.0
        self.sy = 1.0
        self.tx = 0.0
        self.ty = 0.0
        
    def prepare(self,origin,target):
        wo = origin[2]-origin[0]
        ho = origin[3]-origin[1]
        wt = target[2]-target[0]
        ht = target[3]-target[1]
        self.sx = float(wt)/float(wo)
        self.sy = float(ht)/float(ho)
        self.tx = target[0]-origin[0]
        self.ty = target[1]-origin[1]
        
    def transform(self,box):
        return [self.sx*box[0]+self.tx, self.sy*box[1]+self.ty, self.sx*box[2]+self.tx, self.sy*box[3]+self.ty]
    
    def scale(self,box):
        return [self.sx*box[0],self.sy*box[1],self.sx*box[2],self.sy*box[3]]
    
    def translate(self,box):
        return [box[0]+self.tx, box[1]+self.ty, box[2]+self.tx, box[3]+self.ty]
    
    def inverseTransform(self,box):
        return [box[0]/self.sx-self.tx, box[1]/self.sy-self.ty, box[2]/self.sx-self.tx, box[3]/self.sy-self.ty]
    
    