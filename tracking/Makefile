#Path to save sequences
sequenceDir:=/tmp

#The page hosted in google sites points to another one, when using -r (among other options) i couldnt download the sequences, maybe by the nofollow attribute, so i take this workaround
#Get the page, pipe to stdout, omit wgets output, filter sequence URL patterns, iterate download
downloadBenchmarkSequences:
	for aSequenceUrl in $$(wget --output-document=- --quiet https://sites.google.com/site/trackerbenchmark/benchmarks/v10 | grep -o -e 'http:\/\/cvlab.hanyang.ac.kr\/tracker_benchmark\/seq\/[A-Z].*\.zip'); do wget --directory-prefix=$(sequenceDir) $$aSequenceUrl; done
