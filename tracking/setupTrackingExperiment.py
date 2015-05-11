import os
import random
import sys
import random

import tracking.benchmarkUtils as benchutils
import tracking.sequence


def write_gt(baseDir, name, images, ids, gt):
    gtFile = open(os.path.join(baseDir, name), 'w')
    for i in range(len(ids)):
        bbox = map(int, gt[ids[i]])
        gtFile.write('{} {} {} {} {}'.format(images[ids[i]], bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3])+'\n')
    gtFile.close()

def write_database(baseDir, name, images, ids):
    dbFile = open(os.path.join(baseDir, name), 'w')
    for i in range(len(ids)):
        dbFile.write(str(images[ids[i]])+'\n')
    dbFile.close()

def sampleSequences(seqDir, trainPath, testPath, trainProp=0.02, trainSeq=40, sampleThreshold=5, excludes=['Football1', 'David', 'Freeman3', 'Freeman4', 'Jogging']):
    seqDirs = [aSeqDir for aSeqDir in os.listdir(seqDir) if aSeqDir not in excludes]
    random.shuffle(seqDirs)
    trainSequences = seqDirs[:trainSeq]
    testSequences = seqDirs[trainSeq:]
    #Test on full sequences
    testFile = open(testPath, 'w')
    for testSequence in testSequences:
        testFile.write(testSequence + '\n')
    testFile.close()
    trainFile = open(trainPath, 'w')
    for trainSequence in trainSequences:
        aSequence = tracking.sequence.fromdir(os.path.join(seqDir, trainSequence, 'img'), os.path.join(seqDir, trainSequence, 'groundtruth_rect.txt'))
        sampleSize = trainProp*len(aSequence.frames)
        if sampleSize < sampleThreshold:
            sampleSize = sampleThreshold
        sampleSize = int(sampleSize)
        #step of 2 to avoid consecutive sampling
        sampleFrames = random.sample(xrange(2, len(aSequence.frames), 2), sampleSize)
        for sampleFrame in sampleFrames:
            trainFile.write(trainSequence + '[{}:{}]'.format(sampleFrame, sampleFrame+1) + '\n')
    trainFile.close()

def sequential_link(seqDir, textDir, outputDir, excludes=['Football1', 'David', 'Freeman3', 'Freeman4', 'Jogging'], suffix='.jpg', proportion=0.5, pattern='groundtruth_rect.txt'):
    sequences = sorted([sequence for sequence in os.listdir(seqDir) if sequence not in excludes])
    trainSize = int(len(sequences)*proportion)
    randomSequences = list(sequences)
    random.shuffle(randomSequences)
    trainSequences = randomSequences[:trainSize]
    testSequences = randomSequences[trainSize:]
    trainDatabase = []
    testDatabase = []
    databaseGt = []
    index = 0
    omittedIndexes = []
    for sequence in sequences:
        if sequence in excludes:
            continue
        seqPath = os.path.join(seqDir, sequence)
        gt = benchutils.parse_gt(os.path.join(seqPath, pattern))
        print seqPath
        frames = [aFrame for aFrame in sorted(os.listdir(os.path.join(seqPath, 'img'))) if aFrame.endswith(suffix)]
        if not len(frames) == len(gt):
            raise Exception('Mismatching frame and gt length ({} vs. {})'.format(len(frames), len(gt)))
        gtIndex = 0
        if sequence in trainSequences:
            editDatabase = trainDatabase
        elif sequence in testSequences:
            editDatabase = testDatabase
        for frame in frames:
            os.symlink(os.path.join(seqDir, sequence, 'img', frame), os.path.join(outputDir, str(index)+suffix))
            if not int(frame.replace(suffix, '')) == 1:
                editDatabase.append(index)
            else:
                print 'Omitted frame {} for sequence {} at index {}'.format(frame, sequence, index)
                omittedIndexes.append(index)
            databaseGt.append(gt[gtIndex])
            gtIndex += 1
            index += 1
    linkNames = sorted([int(link.replace('.jpg', '')) for link in os.listdir(outputDir)])
    print 'Links: {} train: {} test: {} gt: {}'.format(len(linkNames), len(trainDatabase), len(testDatabase), len(databaseGt))
    write_database(textDir, 'train.txt', linkNames, trainDatabase)
    write_gt(textDir, 'train_gt.txt', linkNames, trainDatabase + omittedIndexes, databaseGt)
    write_database(textDir, 'test.txt', linkNames, testDatabase)
    write_gt(textDir, 'test_gt.txt', linkNames, testDatabase + omittedIndexes, databaseGt)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: python', sys.argv[0], 'sequenceDir'
        sys.exit(0)
    baseDir = sys.argv[1]
    images = sorted([name.strip().replace('.jpg', '') for name in os.listdir(os.path.join(baseDir, 'img')) if name.endswith('.jpg')])
    gt = benchutils.parse_gt(os.path.join(baseDir, 'groundtruth_rect.txt'))
    size = len(images)
    #30 frames ~= 1 seg
    trainSamples = 30
    if not size == len(gt):
        print 'Images: {} GtTs: {}'.format(size, len(gt))
        sys.exit(1)
    ids = range(size)
    random.shuffle(ids)
    trainIds = ids[:trainSamples]
    testIds = ids[trainSamples:]
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
