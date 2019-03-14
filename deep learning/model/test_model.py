
#  ██  ██      ███████  ██████  ██    ██ ██████   ██████ ███████
# ████████     ██      ██    ██ ██    ██ ██   ██ ██      ██
#  ██  ██      ███████ ██    ██ ██    ██ ██████  ██      █████
# ████████          ██ ██    ██ ██    ██ ██   ██ ██      ██
#  ██  ██      ███████  ██████   ██████  ██   ██  ██████ ███████
###############################################################################
# https://github.com/PacktPublishing/Python-Artificial-Intelligence-Projects-for-Beginners/blob/master/Chapter04/GenreIdentifier.py
###############################################################################
###############################################################################
#  ██  ██      ██ ███    ███ ██████   ██████  ██████  ████████
# ████████     ██ ████  ████ ██   ██ ██    ██ ██   ██    ██
#  ██  ██      ██ ██ ████ ██ ██████  ██    ██ ██████     ██
# ████████     ██ ██  ██  ██ ██      ██    ██ ██   ██    ██
#  ██  ██      ██ ██      ██ ██       ██████  ██   ██    ██
###############################################################################
###############################################################################
import librosa
import librosa.feature
import librosa.display
import glob
import numpy as np
import scipy
import csv
import sys
import os
import shutil
#import matplotlib.pyplot as plt
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical
from progressbar import Percentage, ProgressBar,Bar,ETA
###############################################################################
###############################################################################
#  ██  ██      ███████ ██    ██ ███    ██  ██████ ████████ ██  ██████  ███    ██ ███████
# ████████     ██      ██    ██ ████   ██ ██         ██    ██ ██    ██ ████   ██ ██
#  ██  ██      █████   ██    ██ ██ ██  ██ ██         ██    ██ ██    ██ ██ ██  ██ ███████
# ████████     ██      ██    ██ ██  ██ ██ ██         ██    ██ ██    ██ ██  ██ ██      ██
#  ██  ██      ██       ██████  ██   ████  ██████    ██    ██  ██████  ██   ████ ███████
###############################################################################
###############################################################################
def stack_uneven(arrays, fill_value=0.):
    '''
    Fits arrays into a single numpy array, even if they are
    different sizes. `fill_value` is the default value.

    Args:
            arrays: list of np arrays of various sizes
                (must be same rank, but not necessarily same size)
            fill_value (float, optional):

    Returns:
            np.ndarray
    '''
# https://stackoverflow.com/questions/44951624/numpy-stack-with-unequal-shapes
    sizes = [a.shape for a in arrays]
    max_sizes = np.max(list(zip(*sizes)), -1)
    # The resultant array has stacked on the first dimension
    result = np.full((len(arrays),) + tuple(max_sizes), fill_value)
    for i, a in enumerate(arrays):
      # The shape of this array `a`, turned into slices
      slices = tuple(slice(0,s) for s in sizes[i])
      # Overwrite a block slice of `result` with this array `a`
      result[i][slices] = a
    return result
###############################################################################
###############################################################################
#def display_mfcc(song):
#    y, _ = librosa.load(song)
#    mfcc = librosa.feature.mfcc(y)
#
#    plt.figure(figsize=(10, 4))
#    librosa.display.specshow(mfcc, x_axis='time', y_axis='mel')
#    plt.colorbar()
#    plt.title(song)
#    plt.tight_layout()
#    plt.show()

#display_mfcc('genres/disco/disco.00035.au')
###############################################################################
###############################################################################
def extract_features_song(f):
    y, _ = librosa.load(f)

    # get Mel-frequency cepstral coefficients
    mfcc = librosa.feature.mfcc(y)
    # normalize values between -1,1 (divide by max)
    mfcc /= np.amax(np.absolute(mfcc))
    # gives mfcc in 1 dim array
    array = np.ndarray.flatten(mfcc)
    # Zero padding step
    result = np.zeros(20000)
    index = 0
    for i in range(0,array.shape[0]-1):
        result[i] = array[i]
    return result
###############################################################################
###############################################################################
def generate_features_and_labels(folder_index):
    all_features = []
    all_labels = []

    if(folder_index == 0):
        folder = "train/"
    elif(folder_index == 1):
        folder = "test/"
    else:
        print("ERROR")
        print("Wrong folder index")
        sys.exit()

    genres = ['crackles','wheezes','both','normal']
    for genre in genres:
        N=300
        pgb = ProgressBar(widgets=[Bar('=', '[', ']'), ' ', Percentage(), ' ', ETA()],
                           maxval=N)
        sound_files = glob.glob('data/'+folder+genre+'/*.wav')
        print()
        print('Processing %d songs in %s genre...' % (len(sound_files), genre))
        for f in pgb(sound_files):
            features = extract_features_song(f)
            all_features.append(features)
            all_labels.append(genre)

    # convert labels to one-hot encoding
    label_uniq_ids, label_row_ids = np.unique(all_labels, return_inverse=True)
    label_row_ids = label_row_ids.astype(np.int32, copy=False)
    onehot_labels = to_categorical(label_row_ids, len(label_uniq_ids))
    print(len(all_features))
    return stack_uneven(all_features), onehot_labels
###############################################################################
###############################################################################
def generate_folders(path):
    crackles = path+"crackles/"
    os.makedirs(crackles, exist_ok=True)
    wheezes = path+"wheezes/"
    os.makedirs(wheezes, exist_ok=True)
    both = path+"both/"
    os.makedirs(both, exist_ok=True)
    normal = path+"normal/"
    os.makedirs(normal, exist_ok=True)
    return crackles,wheezes,both,normal
###############################################################################
###############################################################################
def move_files(path,crackles,wheezes,both,normal):
    cr = wh = bo = no = 0
    old = ""
    with open(path+"info.csv", newline='') as csvfile:
        data = list(csv.reader(csvfile))
    cpt=0
    for i in range(0,len(data)):
        record = data[i][0]

        if(cpt == 0):
            old = record
            cpt+=1
            file_index = str(cpt).zfill(2)
        elif(record != old):
            old = record
            cpt = 1
            file_index = str(cpt).zfill(2)
        else:
            cpt+=1
            file_index = str(cpt).zfill(2)
        filename = record+"_"+file_index+".wav"
        label = int(data[i][3])
        dest_folder = ''

        if label == 0:
            dest_folder = normal
            no+=1
        elif label == 1:
            dest_folder = crackles
            cr+=1
        elif label == 2:
            dest_folder = wheezes
            wh+=1
        elif label == 3:
            dest_folder = both
            bo+=1
        shutil.move(path+filename,dest_folder)
    return cr,wh,bo,no
###############################################################################
###############################################################################
#  ██  ██      ███    ███  █████  ██ ███    ██
# ████████     ████  ████ ██   ██ ██ ████   ██
#  ██  ██      ██ ████ ██ ███████ ██ ██ ██  ██
# ████████     ██  ██  ██ ██   ██ ██ ██  ██ ██
#  ██  ██      ██      ██ ██   ██ ██ ██   ████
###############################################################################
###############################################################################

train_folder = './data/train/'
# trainCr,trainWh,trainBo,trainNo = generate_folders(train_folder)
# Tcr,Twh,Tbo,Tno = move_files(train_folder,trainCr,trainWh,trainBo,trainNo)
# print("In TRAIN folder,")
# print(str(Tcr)+" files in crackles folder")
# print(str(Twh)+" files in wheezes folder")
# print(str(Tbo)+" files in both folder")
# print(str(Tno)+" files in normal folder")

test_folder = './data/test/'
# testCr,testWh,testBo,testNo = generate_folders(test_folder)
# tcr,twh,tbo,tno = move_files(test_folder,testCr,testWh,testBo,testNo)
# print("In TEST folder,")
# print(str(tcr)+" files in crackles folder")
# print(str(twh)+" files in wheezes folder")
# print(str(tbo)+" files in both folder")
# print(str(tno)+" files in normal folder")

features_train, labels_train = generate_features_and_labels(0)
features_test, labels_test = generate_features_and_labels(1)

print(np.shape(features_train))
print(np.shape(labels_train))

print(np.shape(features_test))
print(np.shape(labels_test))


training_split = 0.8

# last column has genre, turn it into unique ids
train_data = np.column_stack((features_train, labels_train))
test_data = np.column_stack((features_test, labels_test))
train, test = train_data, test_data

print(np.shape(train))
print(np.shape(test))

train_input = train[:,:-4]
train_labels = train[:,-4:]

test_input = test[:,:-4]
test_labels = test[:,-4:]

print(np.shape(train_input))
print(np.shape(train_labels))


model = Sequential([
    Dense(100, input_dim=np.shape(train_input)[1]),
    Activation('relu'),
    Dense(4),
    Activation('softmax'),
    ])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())

model.fit(train_input, train_labels, epochs=100, batch_size=32,
          validation_split=0.2)

loss, acc = model.evaluate(test_input, test_labels, batch_size=16)

print("Done!")
print("Loss: %.4f, accuracy: %.4f" % (loss, acc))
y_pred = model.predict(test_input)
y_pred = np.argmax(y_pred, axis=1)

print("Making redictions")
print(y_pred.shape)
final = test_labels.shape[0]*[0]
for i in range(0,test_labels.shape[0]):
    if test_labels[i, 0] == 1:
        final[i] = 0
    elif test_labels[i, 1] == 1:
        final[i] = 1
    elif test_labels[i, 2] == 1:
        final[i] = 2
    elif test_labels[i, 3] == 1:
        final[i] = 3

print("Confusion matrix between expectations and predictions")
matrix = metrics.confusion_matrix(final,y_pred)
print(matrix)
