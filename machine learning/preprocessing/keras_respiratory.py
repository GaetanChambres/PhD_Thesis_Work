import sys
import numpy as np
import pandas as pd
import csv
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import autokeras as ak

def compute_res_matrix(data,pred):
    mc = 3*[0]
    for i in range(len(mc)): mc[i] =3*[0]
    t=0
    cc = tc = cn = tn = 0
    for i in range(0,len(data)):
        vr = int(data[i])
        vp = int(pred[i])

        mc[vr][vp]+=1
        mc[vr][2]+=1
        mc[2][vp]+=1
        t+=1

        if vr == 1:
            tc+=1
            if vp == 1:
                cc+=1
        if vr == 0:
            tn+=1
            if vp == 0:
                cn+=1
    mc[2][2] = t
    verif=(mc[2][0]+mc[2][1]+mc[2][2])-(mc[0][2]+mc[1][2]+mc[2][2])
    # print("verif = %d" % verif)
    if(verif != 0):
        print("error during matrix computation")
    return mc,cc,tc,cn,tn

def csv_nb_cols(fname,delimiter):
    line = fname.readline()
    data = line.split(delimiter)
    nb_col = len(data)
    return nb_col

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

arguments = sys.argv

input_folder = arguments[1]

input_train = input_folder+"challenge/train_lowlevel.csv"
input_info_train = input_folder+"challenge/train_info.csv"
input_test = input_folder+"challenge/test_lowlevel.csv"
input_info_test = input_folder+"challenge/test_info.csv"

print(input_train)
print()
print(input_test)

#############################
with open(input_train) as f1:
    line1 = f1.readline()
    data1= line1.split(',')
    # print(data1)
    nbcols_train = len(data1)
    print(nbcols_train)

with open(input_test) as f2:
    line2 = f2.readline()
    data2 = line2.split(',')
    # print(data2)
    nbcols_test = len(data2)
    print(nbcols_test)
#############################

print("Opening data as Numpy array")
print("Loading data")
print("LF -- Loading train files")
train_dataset = np.loadtxt(input_train, delimiter=",", skiprows = 1, usecols=range(1,nbcols_train))
train_info = np.loadtxt(input_info_train,delimiter = ',',skiprows = 0,usecols=range(8,10))
# print(train_dataset)
# print(train_info)
print("LF -- Reading data in train files")
classification_train_crackle = train_info[:,0:1]
classification_train_wheeze = train_info[:,1:2]
features_train = train_dataset[:,1:]
# print(classification_train_crackle)
# print()
# print(classification_train_wheeze)
# print()
# print(features_train)

print("LF -- Loading test files")
test_dataset = np.loadtxt(input_test, delimiter=",", skiprows = 1, usecols=range(1,nbcols_test))
test_info = np.loadtxt(input_info_test,delimiter = ',', skiprows = 0, usecols=range(8,10))
print("LF -- Reading data in test files")
classification_test_crackle = test_info[:,0:1]
classification_test_wheeze = test_info[:,1:2]
features_test = test_dataset[:,1:]
# print(test_dataset)
# print(test_info)

nb_lines = len(test_dataset)
# # create model
# model = Sequential()
# model.add(Dense(2048, input_dim=len(features_train[0]), activation='relu'))
# model.add(Dense(1024, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1, activation='softmax'))
#
# # Compile model
# opt = SGD(lr=0.01)
# model.compile(loss='binary_crossentropy', optimizer=(opt), metrics=['accuracy'])

X = features_train
Y = classification_train_crackle

XT = features_test
YT = classification_test_crackle

# Fit the model
# model.fit(X, Y, epochs=200, batch_size=16)

clf = ak.DeepSupervised()
clf.fit(X, Y)
results = clf.predict(XT)

# evaluate the model
scores = model.evaluate(X, Y)
print(scores)
print("********************************************************")
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predicted = model.predict_classes(XT);
print(predicted)
# crackle_predictions = [round(value) for value in predicted]
# print("CM -- Crackle -- Predictions Extracted")
confusion_crackles,pred_crackles,total_crackles,pred_normal,total_normal = compute_res_matrix(YT,predicted)
print(confusion_crackles)
