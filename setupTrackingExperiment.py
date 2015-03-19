import os
import random
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: python', sys.argv[0], 'sequenceDir'
        sys.exit(0)
    baseDir = sys.argv[1]
    images = sorted([name.strip().replace('.jpg', '') for name in os.listdir(os.path.join(baseDir, 'img')) if name.endswith('.jpg')])
    gt = [map(int, line.strip().replace(',', '\t').split()) for line in open(os.path.join(baseDir, 'groundtruth_rect.txt'))]
    size = len(images)
    if not size == len(gt):
        print 'Images: {} GtTs: {}'.format(size, len(gt))
        sys.exit(1)
    ids = range(size)
    random.shuffle(ids)
    trainIds = ids[:size/2]
    testIds = ids[size/2:]
    all = open(os.path.join(baseDir, 'allImagesList.txt'), 'w')
    train = open(os.path.join(baseDir, 'train.txt'), 'w')
    test = open(os.path.join(baseDir, 'test.txt'), 'w')
    trainGt = open(os.path.join(baseDir, 'train_gt.txt'), 'w')
    testGt = open(os.path.join(baseDir, 'test_gt.txt'), 'w')
    for i in range(size):
        all.write(str(images[i])+'\n')
    for i in range(len(trainIds)):
        train.write(str(images[trainIds[i]])+'\n')
        try:
            bbox = gt[trainIds[i]]
        except IndexError:
            print i, trainIds[i], len(gt)
            sys.exit(1)
        trainGt.write('{} {} {} {} {}'.format(images[trainIds[i]], bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3])+'\n')
    for i in range(len(testIds)):
        test.write(str(images[testIds[i]])+'\n')
        try:
            bbox = gt[testIds[i]]
        except IndexError:
            print i, testIds[i], len(gt)
            sys.exit(1)
        testGt.write('{} {} {} {} {}'.format(images[testIds[i]], bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3])+'\n')
    all.close()
    train.close()
    test.close()
    trainGt.close()
    testGt.close()
