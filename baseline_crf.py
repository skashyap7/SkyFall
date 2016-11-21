#!/usr/bin/python
import argparse
import pycrfsuite as crf
import glob
import hw3_corpus_tool as corpus_reader
import os
from os import path
import random

stopwords = [',']
trainer = crf.Trainer(verbose=False)
trainer.set_params({
        'c1': 2.5,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 75,  # stop earlier
        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })

#Extract feature from an utterance
def extractFeatures(utterance, prev_speaker):
    feature = []
    speaker_change = 0 # by default it is false
    is_beginning = 1 if prev_speaker == None else 0 # If no previous speaker, then beginning of seq
    # Get the current speaker    
    speaker = utterance.speaker
    if prev_speaker != utterance.speaker and not is_beginning:
        speaker_change = 1
    feature.append("speaker="+str(speaker_change))
    feature.append("BOD="+str(is_beginning))

    if utterance.pos != None:
        for token in utterance.pos:
            #if token.token not in stopwords:
            feature.append("token=TOKEN_"+token.token)
        for token in utterance.pos:
            #if token.token not in stopwords:
            feature.append("pos=POS_"+token.pos)

    else:
        feature.append("token=TOKEN_None")
        feature.append("pos=POS_None")
    return ([feature],speaker)

def generateFeatures(corpus_iter):
    #for dlg_file in corpus_iter:
    features = []
    labels = []
    prev_speaker = None  
    for utterance in corpus_iter:
        feat,prev_speaker = extractFeatures(utterance, prev_speaker)
        features.append(feat)
        labels.append([utterance.act_tag])
    #print(features,labels)
    return features,labels


def trainCRF(dirname):
    dialog_filenames = sorted(glob.glob(os.path.join(dirname, "*.csv")))
    for dialog_filename in dialog_filenames:
        utteranceList =  corpus_reader.get_utterances_from_filename(dialog_filename)
        features,labels = generateFeatures(utteranceList)
        #trainer.append(features,labels)
        for xseq,yseq in zip(features,labels):
            trainer.append(xseq,yseq)
    trainer.train('seq_labeling.crfsuite')

def tagCRF(dirname):
    tag_info = {}
    tagger = crf.Tagger()
    tagger.open('seq_labeling.crfsuite')
    dialog_filenames = sorted(glob.glob(os.path.join(dirname, "*.csv")))
    #random.shuffle(dialog_filenames)
    for dialog_filename in dialog_filenames:
        utteranceList =  corpus_reader.get_utterances_from_filename(dialog_filename)
        features,labels = generateFeatures(utteranceList)
        tag_info[dialog_filename] = []
        for f in features:
            label = tagger.tag(f)
            tag_info[dialog_filename].append(label)
    return tag_info

def __extract_filename(filepath):
    return os.path.basename(filepath)

def print_results(outfile, tagged_data):
    text_str = ""
    for f in tagged_data:
        file_name = __extract_filename(f)
        labels = "\n"
        for x in tagged_data[f]:
            labels += x[0]+"\n"
        text_str += "Filename=\""+file_name+"\""
        text_str += labels+"\n"
    with open(outfile,"w") as result:
        result.write(text_str)
        result.close()

# Main Function
def main():
    parser = argparse.ArgumentParser(usage="python baseline_crf.py <INPUTDIR> <TESTDIR> <OUTPUTFILE>", description="Use CRF model to predict utterance act")
    parser.add_argument('inputdir', help="inputdir help")
    parser.add_argument('testdir', help="testdir help")
    parser.add_argument('outfile', help="outputfile help")
    args = parser.parse_args()
    trainCRF(args.inputdir)
    tagged_data = tagCRF(args.testdir)
    print_results(args.outfile,tagged_data)

# Entry point
if __name__ == "__main__":
    main()