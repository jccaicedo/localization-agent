import runpy
import sys
import learn.rl.RLConfig

def listRunners():
  options = ['BoxSearch','Tracker','SetupTracker','TrackerParameterSearch']
  print 'Available options are:'
  for o in options:
    print '\t',o

if len(sys.argv) > 1:
  module = sys.argv.pop(1)
  if module == 'BoxSearch':
    runpy.run_module('detection.boxsearch.BoxSearchRunner', run_name='__main__', alter_sys=True)
  elif module == 'Tracker':
    runpy.run_module('tracking.TrackerRunner', run_name='__main__', alter_sys=True)
  elif module == 'SetupTracker':
    runpy.run_module('tracking.setupTrackingExperiment', run_name='__main__', alter_sys=True)
  elif module == 'TrackerParameterSearch':
    runpy.run_module('tracking.parameterSearch', run_name='__main__', alter_sys=True)
  else:
    print 'Unknown module'
    listRunners()
else:
  print 'No module given.'
  listRunners()
