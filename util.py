# generate training and test data
import os
import random
from os import path
import shutil

def genTrainingData(dirname,percent):
    filelist = []
    trainList = []
    devList = []
    for root, dirs, files in os.walk(dirname):
        for sfile in files:
            filelist.append((os.path.join(root,sfile)))
    if filelist:
        # random shuffle , to ensure no bias towards ham and spam
        random.shuffle(filelist)
        new_len = (len(filelist)*percent)//100
        trainList = filelist[0:new_len+1]
        devList = filelist[new_len+1:]
    return trainList,devList


def copyFiles(trainList,devList):
    # if not already present, create train dir
    train_dir = os.path.join(os.getcwd(),"train")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    # copy files from trainList into the train dir
    for fname in trainList:
        shutil.copy2(fname,train_dir)

    # if not already present, create train dev
    dev_dir = os.path.join(os.getcwd(),"dev")
    if not os.path.exists(dev_dir):
        os.makedirs(dev_dir)
        # copy files from trainList into the train dir
    for fname in devList:
        shutil.copy2(fname,dev_dir)
    # copy files from trainList into the train dir
            

tlist,dlist = genTrainingData("D:\\NLP-544\\Assignment3\\labeled data\\labeled data",75)
copyFiles(tlist,dlist)