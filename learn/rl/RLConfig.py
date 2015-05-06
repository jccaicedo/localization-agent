import os,sys

configValues = {}

def readConfiguration(inputFile):
  data = [x.split() for x in open(inputFile) if not x.startswith('#')]
  for d in data:
    configValues[d[0]] = d[1]

def get(key, type=lambda x:x):
  try: 
    val = type(configValues[key])
  except: 
    val = None
    print 'Configuration variable:',key,'does not exist'
  return val

def geti(key):
  return get(key,int)

def getf(key):
  return get(key,float)


