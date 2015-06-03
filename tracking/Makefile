#Path to save sequences
sequenceZipDir:=/tmp

#The page hosted in google sites points to another one, when using -r (among other options) i couldnt download the sequences, maybe by the nofollow attribute, so i take this workaround
#Get the page, pipe to stdout, omit wgets output, filter sequence URL patterns, iterate download
downloadBenchmarkSequences:
	for aSequenceUrl in $$(wget --output-document=- --quiet https://sites.google.com/site/trackerbenchmark/benchmarks/v10 | grep -o -e 'http:\/\/cvlab.hanyang.ac.kr\/tracker_benchmark\/seq\/[A-Z].*\.zip'); do wget --directory-prefix=$(sequenceZipDir) $$aSequenceUrl; done

sequencesUnzippedDir:=/tmp/tracker_benchmark
videosDir:=/tmp/tracker_benchmark/videos

generateVideos:
	for aSequenceDir in $$(ls $(sequencesUnzippedDir)); do if [ -d $(sequencesUnzippedDir)/$$aSequenceDir/img ]; then time avconv -f image2 -r 30 -i $(sequencesUnzippedDir)/$$aSequenceDir/img/%04d.jpg -r 30 $(videosDir)/$$aSequenceDir.mp4; else echo "img dir $(sequencesUnzippedDir)/$$aSequenceDir/img not found"; fi; done
