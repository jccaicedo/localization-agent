import os,sys
import xml.etree.ElementTree as ET

def findObjects(xmlFile, category):
  boxes = []
  tree = ET.parse(xmlFile)
  for child in tree.getroot().findall('object'):
    if child.find('name').text == category and child.find('difficult').text != '1':
      bn = child.find('bndbox')
      box = map(float, [bn.find('xmin').text, bn.find('ymin').text, bn.find('xmax').text, bn.find('ymax').text])
      area = (box[2]-box[0])*(box[3]-box[1])
      # Skip small objects
      if area >= 400.0:
        boxes.append( box )
  return boxes

## MAIN PROGRAM
def mainProgram():
  if len(sys.argv) < 5:
    print 'Use: extractPascalBoxes.py trainvalList category xmlDir bboxOutput'
    sys.exit()

  category = sys.argv[2]
  xmlDir = sys.argv[3]
  outputFile = sys.argv[4]
  imageList = [x.split() for x in open(sys.argv[1])]

  out = open(outputFile,'w')
  allBoxes = dict()
  for img in imageList:
    if img[1] == "1":
      allBoxes[img[0]] = []
      boxes = findObjects(xmlDir+'/'+img[0]+'.xml', category)
      for box in boxes:
	out.write(img[0]+' '+' '.join(map(str,map(int,box)))+'\n')
  out.close()

if __name__ == "__main__":
  mainProgram()

