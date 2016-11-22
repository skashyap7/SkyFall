#!/usr/bin/python
import argparse
import pycrfsuite as crf
import glob
import hw3_corpus_tool as corpus_reader
import os
from os import path
import random
import itertools

f1_enable = False
trainer = crf.Trainer(verbose=False)
trainer.set_params({
        'c1': 2.5,   # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 120,  # stop earlier
        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def getpairwise(iterable):
    p = pairwise(iterable)
    return [ x+","+y for x,y in p]    

def addPairFeature(token_list,feature):
    idx1 = 0
    idx2 = 1
    for f_add in getpairwise(token_list):
        ft_tag = "token["+str(idx1)+":"+str(idx2)+"]="
        feature.append(ft_tag+f_add)
        idx1 += 1
        idx2 += 1

#Extract feature from an utterance
def extractFeatures(utterance, prev_speaker, prev_pos):
    feature = []
    speaker_change = False # by default it is false
    is_beginning = True if prev_speaker == None else False # If no previous speaker, then beginning of seq
    # Get the current speaker    
    speaker = utterance.speaker
    if prev_speaker != utterance.speaker and not is_beginning:
        speaker_change = True
    feature.append("speaker_change="+str(speaker_change))
    feature.append("BOD="+str(is_beginning))
    if utterance.pos != None:
        prev_token = None
        for idx,token in enumerate(utterance.pos):
            feature.append("token=TOKEN_"+token.token)
            # Additional Feature
            if prev_token:
                feature.append("token[-1]=TOKEN_"+prev_token)
            if idx != len(utterance.pos)-1:
                feature.append("token[1]=TOKEN_"+utterance.pos[idx+1].token)
            prev_token = token.token

        for token in utterance.pos:
            feature.append("pos=POS_"+token.pos)
        # Adding new feature that takes combinations of words in
        # utterances as a feature
        #if f1_enable:
        addPairFeature([u.token for u in utterance.pos],feature)
    if prev_pos:
        for x in prev_pos:
            feature.append("token[-1]=TOKEN_"+x.token+"=True")
    return (feature,speaker, utterance.pos)

def generateFeatures(corpus_iter):
    #for dlg_file in corpus_iter:
    features = []
    labels = []
    prev_speaker = None  
    prev_pos = []
    for utterance in corpus_iter:
        feat,prev_speaker,prev_pos = extractFeatures(utterance, prev_speaker, prev_pos)
        features.append(feat)
        labels.append(utterance.act_tag)
    return features,labels


def trainCRF(dirname):
    feature_list = []
    label_list = []
    dialog_filenames = sorted(glob.glob(os.path.join(dirname, "*.csv")))
    for dialog_filename in dialog_filenames:
        utteranceList =  corpus_reader.get_utterances_from_filename(dialog_filename)
        features,labels = generateFeatures(utteranceList)
        feature_list.extend(features)
        label_list.extend(labels)
        #for xseq,yseq in zip(features,labels):
        #    trainer.append(xseq,yseq)
    trainer.append(feature_list,label_list)
    trainer.train('seq_labeling.crfsuite')
    print(len(trainer.logparser.iterations), trainer.logparser.iterations[-1])

def tagCRF(dirname):
    tag_info = {}
    tagger = crf.Tagger()
    tagger.open('seq_labeling.crfsuite')
    dialog_filenames = sorted(glob.glob(os.path.join(dirname, "*.csv")))
    for dialog_filename in dialog_filenames:
        utteranceList =  corpus_reader.get_utterances_from_filename(dialog_filename)
        features,labels = generateFeatures(utteranceList)
        tag_info[dialog_filename] = []
        #for f in features:
        label = tagger.tag(features)
        tag_info[dialog_filename] = label
    return tag_info

def __extract_filename(filepath):
    return os.path.basename(filepath)

def print_results(outfile, tagged_data):
    text_str = ""
    for f in tagged_data:
        file_name = __extract_filename(f)
        labels = "\n"
        for x in tagged_data[f]:
            labels += x+"\n"
        text_str += "Filename=\""+file_name+"\""
        text_str += labels+"\n"
    with open(outfile,"w") as result:
        result.write(text_str)
        result.close()

# Main Function
def main():
    parser = argparse.ArgumentParser(usage="python advanced_crf.py <INPUTDIR> <TESTDIR> <OUTPUTFILE>", description="Use CRF model to predict utterance act")
    parser.add_argument('-p', help="pairwise help")
    parser.add_argument('inputdir', help="inputdir help")
    parser.add_argument('testdir', help="testdir help")
    parser.add_argument('outfile', help="outputfile help")
    args = parser.parse_args()
    if args.p:
        f1_enable = True
    trainCRF(args.inputdir)
    tagged_data = tagCRF(args.testdir)
    print_results(args.outfile,tagged_data)

# Entry point
if __name__ == "__main__":
    main()