from __future__ import print_function

import os
import sys
import tensorflow as tf

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.activations import softmax
from keras.optimizers import RMSprop
import keras
from keras.layers import *
from keras.callbacks import *
from keras.regularizers import *
from keras.models import Sequential
from keras.models import load_model
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
np.set_printoptions(precision=10, suppress=True)
# start


output_file = sys.argv[1] #will be where the intensity  output fule
RNA_complete = sys.argv[2] # the file where  the rna squenses we need to check the intensity
RBP_files = sys.argv[3:] # the diffrent files where the protin is connected in diffrent nno molar
# start

kernel_window=8 #will the the length of the rna/dna seq that the NN will get as input
kernel_rna_slides=39-kernel_window #will be the number of substring of the rna seq sized kernel window
kernel_rbn_slieds=20 - kernel_window #will be the number of substring of the dna seq sized kernel window

#will return a list of substrings from index start to start +kernel_window  in every string in the string list.
# for example stringlist=["abc,"hty","yil"]
#and the start is 0 and kernel window is 2 so the function return ["ab","ht","yi"]
def get_substrings(strings_list, start, length=20):
    return [s[start:start + kernel_window] for s in strings_list]

# BLOOK 1

#will get all the rna string in the second  argument the max lenght is 39 so  the code pad string smaller then that with N.
rnaseq = []
with open(RNA_complete, 'r') as file:
    lines = file.readlines()
    # Strip newline characters from each line
    rnaseq = [line.strip().ljust(38, 'N') for line in lines]
#It will be an arrays of substings each index i will be a collaction of all rna seq substring from i to i_kernal window.
#It was made for faster predicts
rnasubseq = []
rnasubseq = [get_substrings(rnaseq, i) for i in range(kernel_rna_slides)]
# BLOOK 1
# BLOOK 2
n = 40000
#It will be an array of content each index i from the array will contain  all the sub dna seqs
# at the size of kernel window of argument file 3+i all untill line 40000 in the file.
all_content = []
for RBP in RBP_files:
    content = []
    with open(RBP, 'r') as file:
        for _ in range(n):
            line = file.readline()
            if not line:
                break  # Break the loop if the end of the file is reached
            st=line.split("\t")[0].strip()
            start1=0
            end=kernel_window
            input_len = len(st)
            while end <= input_len:
              substring = st[start1:end]
              
              start1 += 1
              end += 1
              content.append(substring)# Print or process the line as needed
    all_content.append(content)

# will get a dna string  with the size of kernel window and return one hot matrix of the string
#RNA will be the same
def dna_to_one_hot(dna_sequence):
    matrix = np.zeros((kernel_window, 4), dtype=float)
    for i, letter in enumerate(dna_sequence):
        if letter == 'A':
            matrix[i, 0] = 1
            continue
        if letter == 'T':
            matrix[i, 1] = 1
            continue
        if letter == 'C':
            matrix[i, 2] = 1
            continue
        if letter == 'G':
            matrix[i, 3] = 1
            continue
        if letter == 'N':
            matrix[i, :] = 0.25
    return matrix
def rna_to_one_hot(dna_sequence):
    matrix = np.zeros((kernel_window, 4), dtype=float)
    for i, letter in enumerate(dna_sequence):
        if letter == 'A':
            matrix[i, 0] = 1
            continue
        if letter == 'U':
            matrix[i, 1] = 1
            continue
        if letter == 'C':
            matrix[i, 2] = 1
            continue
        if letter == 'G':
            matrix[i, 3] = 1
            continue
        if letter == 'N':
            matrix[i, :] = 0.25
    return matrix

# BLOOK 3

#The loop is to  make a tupple list of  one hot matrix that we will send to the NN and the output it shoul give.
#It will loop troug the all_content array. index i in the array will contain the sub dna of the file argument 3+i.
#The output of the seqs nedd to be the vector  ei.
num_of_files = len(RBP_files)
RNAlist = []
result = np.zeros(num_of_files)
for i, content in enumerate(all_content):
    result[i] = 1
    r = result.copy()
    for c in content:
        RNAlist.append((dna_to_one_hot(c), r))
    result[i] = 0


#This will be The NN at firs we flatter the matrix that its going to get.
#Then the nex layers are 64 32 32 using rellu then softmax at size of the number of files of the protin.
#The NN will return the propebilies for the seq it will get the probabilities for being in each cycle.


model_1 = Sequential()

# Flatten the 20x4 input matrix

model_1.add(Conv2D(8, kernel_size=(4, 4), input_shape=(kernel_window, 4, 1), activation='relu'))

# Add three fully connected layers with specified sizes and activation functions
model_1.add(Flatten())
model_1.add(Dense(64))
model_1.add(Activation('relu'))
model_1.add(Dense(32))
model_1.add(Activation('relu'))
model_1.add(Dense(16))
model_1.add(Activation('relu'))

# Add a softmax layer with size 6 for classification
model_1.add(Dense(num_of_files))
model_1.add(Activation('softmax'))

# Compile the model
model_1.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Print the model summary
model_1.summary()


batch_size = 64


x_val = np.array([tup[0] for tup in RNAlist])
y_val = np.array([tup[1] for tup in RNAlist])

print ("starting training the NN model\n")
model_1.fit(x_val, y_val,
              batch_size=batch_size,
              epochs=10,
              shuffle=True)
print ("The training ended\n")			  

#This is the part of the prediction predlist will containt array of preadiction. 
#Index i is the array of the prediction of sub rna seq from i to i+ kernel window
# each prediction contain array of probabilities of softmax			  
print ("starting with the prediction:")
predlist=[]
for i in range(kernel_rna_slides):
   
    test=[]
    for j in rnasubseq[i]:
        test.append(rna_to_one_hot(j))
    test= np.array(test)
    pred = model_1.predict(test)
    predlist.append(pred)			  
print ("The prediction ended\n")			  
#Object  of the  class will represt the probrbilies of all the sub rna seq at the size of kernel window.
#And it will contain an array of list and in list I their will be an array of probltis of each sub seq that is clasifies to cycle.
class rnapred:
    def __init__(self,n):
       self.cyclepred=[]
       for i in range(n):
           self.cyclepred.append([])
           
    def addtocycle(self,pred,cyclenum):
        self.cyclepred[cyclenum].append(pred)
    def maxfromcycle(self,n):
        return max (self.cyclepred[n])
    def minfromcycle(self,n):
        return min(self.cyclepred[n])
			  
rnanum= len(predlist[0])
rnapredlist=[]
for i in range(rnanum):
    rnapredlist.append(rnapred(num_of_files))
for j in range(rnanum):
    for  t in range(kernel_rna_slides):
     for s in range(num_of_files):
         rnapredlist[j].addtocycle(predlist[t][j][s],s)

#will get the intensity
rnaintenselist = []
for i in range(rnanum):
    intense = max(rnapredlist[i].cyclepred[num_of_files - 1]) + max(rnapredlist[i].cyclepred[num_of_files - 2]) + max(
        rnapredlist[i].cyclepred[num_of_files - 3]) - min(rnapredlist[i].cyclepred[0]) - min(rnapredlist[i].cyclepred[1])
    rnaintenselist.append(intense)
    t = 5

with open(output_file, 'w') as f:
    for value in rnaintenselist:
        f.write(str(value) + '\n')

print ("The intensity is saved at the file: "+ output_file)

