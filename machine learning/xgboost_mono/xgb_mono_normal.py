#!/usr/bin/env python3

import pandas as pd
import numpy as np
import xgboost as xgb
import csv as csv
# from sklearn.model_selection import train_test_split
import sklearn.metrics as sklm

import warnings
warnings.filterwarnings("ignore")

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


print("LF -- LOADING FILES")
input_train = "./data/csv/challenge/train_lowlevel.csv"
info_train = "./data/csv/challenge/train_info.csv"

input_test = "./data/csv/challenge/test_lowlevel.csv"
info_test = "./data/csv/challenge/test_info.csv"

# tmp_feat  = "./data/csv/challenge/tmp_feat.csv"
# tmp_info  = "./data/csv/challenge/tmp_info.csv"

with open(input_train) as f1:
    nbcols_train = csv_nb_cols(f1,delimiter = ",")
    # print(nbcols_train)
with open(input_test) as f2:
    nbcols_test = csv_nb_cols(f2,delimiter = ",")
    # print(nbcols_test)
# with open(info_train) as f3:
#     nbcols_info_train = csv_nb_cols(f3,delimiter = ",")
#     # print(nbcols_info_train)
# with open(info_test) as f4:
#     nbcols_info_test = csv_nb_cols(f4,delimiter = ",")
#     # print(nbcols_info_test)

print("LF -- Loading train files")
train_dataset = np.loadtxt(input_train, delimiter=",", skiprows = 1, usecols=range(1,nbcols_train))
train_info = np.loadtxt(info_train,delimiter = ',',skiprows = 0,usecols=range(7,9))
print("LF -- Reading data in train files")
classification_train_crackle = train_info[:,0:1]
classification_train_wheeze = train_info[:,1:2]
features_train = train_dataset[:,1:]

print("LF -- Loading test files")
test_dataset = np.loadtxt(input_test, delimiter=",", skiprows = 1, usecols=range(1,nbcols_test))
test_info = np.loadtxt(info_test,delimiter = ',', skiprows = 0, usecols=range(7,9))
print("LF -- Reading data in test files")
classification_test_crackle = test_info[:,0:1]
classification_test_wheeze = test_info[:,1:2]
features_test = test_dataset[:,1:]
final_res = test_dataset[:,0:1]

nb_lines = len(test_dataset)

print("CM -- CLASSIFICATION MODEL")
model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None, silent=True, subsample=1)

print("CM -- Normal -- Working on normal")
print("CM -- Normal -- Train the model")
model.fit(features_train, classification_train_crackle)
print("CM -- Normal -- Model trained")

print("CM -- Normal -- Predict the model")
crackle_pred = model.predict(features_test)
print("CM -- Normal -- Model Predicted")
print("CM -- Normal -- Extract predictions")
crackle_predictions = [round(value) for value in crackle_pred]
print("CM -- Normal -- Predictions Extracted")

# confusion_crackles,pred_crackles,total_crackles,pred_normal,total_normal = compute_res_matrix(classification_test_crackle,crackle_predictions)
# print(len(crackle_pred))
# print(len(crackle_predictions))
# print(confusion_crackles)


saved_prediction = len(crackle_predictions)*[0]

for cpt in range(0,len(crackle_predictions)):
    if(crackle_predictions[cpt] == 0):
        saved_prediction[cpt] = 1
    else:
        saved_prediction[cpt] = 0
print(saved_prediction)

print("CM -- Crackle -- Working on crackles")
print("CM -- Crackle -- Train the model")
model.fit(features_train, classification_train_crackle)
print("CM -- Crackle -- Model trained")

print("CM -- Crackle -- Predict the model")
crackle_pred = model.predict(features_test)
print("CM -- Crackle -- Model Predicted")
print("CM -- Crackle -- Extract predictions")
crackle_predictions = [round(value) for value in crackle_pred]
print("CM -- Crackle -- Predictions Extracted")
print(saved_prediction)
# confusion_wheezes,pred_wheezes,total_wheezes,pred_normal,total_normal = compute_res_matrix(classification_test_wheeze,wheeze_predictions)
# print(len(wheeze_pred))
# print(len(wheeze_predictions))
# print(confusion_wheezes)

for cpt in range(0,len(crackle_predictions)):
    if(saved_prediction[cpt] == 0): # already crackle
        if(crackle_predictions[cpt] == 1): # predicted as wheeze
            saved_prediction[cpt] = 2 # classified in both
print("CM -- Wheeze -- Working on wheeze")
print("CM -- Wheeze -- Train the model")
model.fit(features_train, classification_train_wheeze)
print("CM -- Wheeze -- Model trained")

print("CM -- Wheeze -- Predict the model")
wheeze_pred = model.predict(features_test)
print("CM -- Wheeze -- Model Predicted")
print("CM -- Wheeze -- Extract predictions")
wheeze_predictions = [round(value) for value in wheeze_pred]
print("CM -- Wheeze -- Predictions Extracted")

# confusion_wheezes,pred_wheezes,total_wheezes,pred_normal,total_normal = compute_res_matrix(classification_test_wheeze,wheeze_predictions)
# print(len(wheeze_pred))
# print(len(wheeze_predictions))
# print(confusion_wheezes)

for cpt in range(0,len(wheeze_predictions)):
    if(saved_prediction[cpt] == 0): # already crackle
        if(wheeze_predictions[cpt] == 1): # predicted as wheeze
            saved_prediction[cpt] = 3 # classified in both
    elif(saved_prediction[cpt] == 2): # not crackle
        if(wheeze_predictions[cpt] == 1): # predicted as wheeze
            saved_prediction[cpt] = 4 #classified in wheeze
print(saved_prediction)
mc = 5*[0]
for i in range(len(mc)): mc[i] =5*[0]
t=0
cc = tc = cw = tw = cb = tb = cn = tn = 0
for i in range(0,len(saved_prediction)):
    vr = int(final_res[i,0])
    vp = int(saved_prediction[i])

    mc[vr-1][vp-1]+=1
    mc[vr-1][4]+=1
    mc[4][vp-1]+=1
    t+=1

    if vr == 2:
        tc+=1
        if vp == 2:
            cc+=1
    if vr == 3:
        tw+=1
        if vp == 3:
            cw+=1
    if vr == 4:
        tb+=1
        if vp == 4:
            cb+=1
    if vr == 1:
        tn+=1
        if vp == 1:
            cn+=1
mc[4][4] = t
print("final !")
print("cc = %d, tc = %d, cw = %d, tw = %d, cb = %d, tb = %d, cn = %d, tn = %d" % (cc,tc,cw,tw,cb,tb,cn,tn))
verif=(mc[4][0]+mc[4][1]+mc[4][2]+mc[4][3])-(mc[0][4]+mc[1][4]+mc[2][4]+mc[3][4])
print("voici la matrice : ")
print(mc)
SEc = cc/tc
SEw = cw/tw
SEb = cb/tb
print("Sensitivity crackles (cc/tc) : %.2f" % SEc)
print("Sensitivity wheezes (cw/tw) : %.2f" % SEw)
print("Sensitivity both (cb/tb) : %.2f" % SEb)
sensitivity_num = cc+cw+cb
sensitivity_denum = tc+tw+tb
sensitivity_global = sensitivity_num/sensitivity_denum
specificity_global = cn/tn
accuracy = (sensitivity_global + specificity_global)/2
print("sensitivity global : %.2f" % sensitivity_global)
print("specificity (cn/tn)) : %.2f" % specificity_global)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print('*********************')
