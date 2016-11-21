#!/usr/bin/python
import argparse
import glob
import os
import hw3_corpus_tool as corpus_reader
from os import path
import re

def getlabelsfromOutput(filename):
    classified = {}
    file_text = ""
    with open(filename,"r") as fh:
        file_text = fh.read()
    file_text.rstrip()
    fblocks = file_text.split("\n\n")
    fblocks.pop()
    for block in fblocks:
        temp = block.split("\n")
        filename = re.match("Filename=\"(.*)\"",temp[0]).group(1)
        if filename != None:
            classified[filename] = temp[1:]
    return classified

def __extract_filename(filepath):
    return os.path.basename(filepath)

def getFileLabels(utterances):
    labels = []
    prev_speaker = None  
    for utterance in utterances:
        labels.append(utterance.act_tag)
    return labels

def compareLabels(dirname, outfile):
    classified = {}
    actual = {}

    # Get Actual Labels for a file
    dialog_filenames = sorted(glob.glob(os.path.join(dirname, "*.csv")))
    for dialog_filename in dialog_filenames:
        filename = __extract_filename(dialog_filename)
        utteranceList =  corpus_reader.get_utterances_from_filename(dialog_filename)
        labels = getFileLabels(utteranceList)
        actual[filename] = labels

    # Get classified labels for a file
    classified = getlabelsfromOutput(outfile)
    # compare results
    wrong = 0
    correct = 0
    for test_file in classified:
        for x,y in zip(classified[test_file],actual[test_file]):
            if x != y:
                wrong += 1
            else:
                correct += 1
    print("Correctly Labelled :"+str(correct))
    print("Wrongly Labelled :"+str(wrong))
    print("of Total :"+str(wrong+correct))
    print("Percentage Accuracy:"+str((correct*100)/(wrong+correct)))

# Main Function
def main():
    parser = argparse.ArgumentParser(usage="python baseline_crf.py <TESTDIR> <OUTPUTFILE>", description="Use CRF model to predict utterance act")
    parser.add_argument('testdir', help="testdir help")
    parser.add_argument('outfile', help="outputfile help")
    args = parser.parse_args()
    compareLabels(args.testdir, args.outfile)

# Entry point
if __name__ == "__main__":
    main()