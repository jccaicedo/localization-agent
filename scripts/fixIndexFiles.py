import os,sys

if len(sys.argv) < 2:
  print 'fixIndexFiles.py dir'
  sys.exit()

indir = sys.argv[1]
outdir = indir + '/fixed'

if not os.path.exists(outdir):
  os.makedirs(outdir)

for f in os.listdir(indir):
  if not f.endswith('.idx'):
    continue
  data = [x for x in open(indir + '/' + f)]
  name = f.replace('.idx','')
  lines = data[0].split(name)
  out = open(outdir + '/' + f, 'w')
  for l in lines:
    if l != '':
      out.write(name + l + '\n')
  out.close()
