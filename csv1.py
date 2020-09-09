import os
import csv
import tensorflow as tf
import numpy as np
import keras
np.random.seed(0)
ampmax=1540690000.0
freqmax =44
#   function to shuffle the dataset and training_labels
def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return a,b
#   function to get the names of all files in the folder
def getlist(filepath):
    list= os.listdir(filepath)
    list.sort()
    return list
#   function to convert list to float
def convert_to_float(eg):
    for i in range(len(eg)):
        for j in range(len(eg[i])):
            eg[i][j]=np.float(eg[i][j].replace(",","."))
    return eg


def optdata(folder):
    totallen = len(getlist(folder[0]))+len(getlist(folder[1]))+len(getlist(folder[2]))
    count=0
    #initializing X_train and Y_train
    train_label = []
    X=np.array([])
    fulldata = open('bacon7.csv', 'w', newline="")
    outputWriter = csv.writer(fulldata)
    for i in range(0,len(folder)):
        file_list = getlist(folder[i])
        for filename in file_list:
            File = open(folder[i]+"/"+filename)
            filedata = list(csv.reader(File, delimiter=";"))
            File.close()
            ex = np.array(convert_to_float(filedata[15:][:]))
            list1 = list(ex[:,1])+[folder[i]]
            outputWriter.writerow(list1)
    fulldata.close()
    return X


folder_train =["3DES_new","AES256_new","AES128_new"]
X_train=optdata(folder_train)
