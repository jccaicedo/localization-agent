import os,sys
import xml.etree.ElementTree as ET

def findObjects(xmlFile):
  boxes = []
  tree = ET.parse(xmlFile)
  for child in tree.getroot().findall('object'):
    if child.find('difficult').text != '1':
      bn = child.find('bndbox')
      box = map(float, [bn.find('xmin').text, bn.find('ymin').text, bn.find('xmax').text, bn.find('ymax').text])
      area = (box[2]-box[0])*(box[3]-box[1])
      # Skip small objects
      if area >= 400.0:
        boxes.append( box + [child.find('name').text])
  return boxes

## MAIN PROGRAM
def mainProgram():
  if len(sys.argv) < 4:
    print 'Use: extractImageNetBoxes.py fileList xmlDir bboxOutput'
    sys.exit()

  imageList = [x.replace('\n','') for x in open(sys.argv[1])]
  xmlDir = sys.argv[2]
  outputFile = sys.argv[3]

  out = open(outputFile,'w')
  allBoxes = dict()
  for img in imageList:
    allBoxes[img] = []
    boxes = findObjects(xmlDir+'/'+img)
    for box in boxes:
      out.write(img.replace('.xml','').replace('./','')+' '+' '.join(map(str,map(int,box[0:4]))) + '\n')
  out.close()

if __name__ == "__main__":
  mainProgram()

