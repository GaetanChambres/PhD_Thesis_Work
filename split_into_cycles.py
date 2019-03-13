
#  ██  ██      ███████ ██████  ██      ██ ████████ ████████ ██ ███    ██  ██████
# ████████     ██      ██   ██ ██      ██    ██       ██    ██ ████   ██ ██
#  ██  ██      ███████ ██████  ██      ██    ██       ██    ██ ██ ██  ██ ██   ███
# ████████          ██ ██      ██      ██    ██       ██    ██ ██  ██ ██ ██    ██
#  ██  ██      ███████ ██      ███████ ██    ██       ██    ██ ██   ████  ██████

###########################################################################
# When using this script, you will take all the resp. records in a given directory (input argument)
# 1) - parse all the annotation files and create a csv named "info.csv" that contains,
#       for each cycles of each records of the directory :
#       filename | start of cycle | and of cycle | label
# 2) - Using the previous csv file, split all the records into audio samples for each resp cycles
#
# So in the input folder, this script will add :
# -- a csv file named "info.csv"
# -- a folder named "splitted_into_cycles"
###########################################################################

import sys
import os
import csv
import pydub
from progressbar import ProgressBar

#  ██  ██      ██    ██ ████████ ██ ██      ██ ████████ ██ ███████ ███████
# ████████     ██    ██    ██    ██ ██      ██    ██    ██ ██      ██
#  ██  ██      ██    ██    ██    ██ ██      ██    ██    ██ █████   ███████
# ████████     ██    ██    ██    ██ ██      ██    ██    ██ ██           ██
#  ██  ██       ██████     ██    ██ ███████ ██    ██    ██ ███████ ███████


###########################################################################
# Function that returns the number of lines from a given file
###########################################################################
def nb_lines(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

###########################################################################
# Function that returns the number of files from a given directory
###########################################################################
def nb_files (directory):
    return(len(os.listdir(directory)))

##########################################################################
# Function that return an alphabetically ordered list
# of the files included in a given directory
###########################################################################
def ordered_files_list(directory):
    return(sorted(os.listdir(directory)))

###########################################################################
# Function that parses all the annotation files from ICBHI Challenge Data
# in a given directory and create a csv file including the following data :
# Filename | start of resp. cycle | end of resp. cycle | label of cycle
# for each cycle of each record in the given directory
###########################################################################
def mining_symptoms_info (directory,csv_file) :
    nb_files_in_dir = nb_files(directory)
    ordered_list = ordered_files_list(directory)
    for filename in ordered_list:
        if(filename.endswith('.txt')):
            record_name=filename[:-4]
            input = directory + filename
            input_file = open(input,"r")
            content = input_file.readline()
            while content :
                start_time,end_time,crackle,wheeze = content.split('\t')
                crackle = int(crackle)
                wheeze=int(wheeze)
                if crackle == 1 and wheeze == 0:
                    label = 1
                elif crackle == 0 and wheeze == 1:
                    label = 2
                elif crackle == 1 and wheeze == 1:
                    label = 3
                else:
                    label = 0
                csv_file.write(record_name+","+start_time+","+end_time+","+str(label)+"\n")
                content = input_file.readline()

###########################################################################
# Function that splits full resp. records into resp. cycles
# It takes a input directory, a csv file and an output directory
# (csv should be as the one given by the function #mining_info()#)
# All the sub-samples are saved in the given output directory
###########################################################################
def split_record_in_cycle(dir,file_csv,output_dir) :
    pbar = ProgressBar()
    lines = nb_lines(file_csv)
    with open(file_csv, newline='') as csvfile:
        data = list(csv.reader(csvfile))
    # print(data)
    input_dir = ordered_files_list(dir)
    # print(input_dir)
    i=0
    for filename in pbar(input_dir):
        if(filename.endswith('.wav')):
            cpt=1
            save_file_name = filename[:-4]
            print()
            while data[i][0] == save_file_name:
                print("Processed record = "+data[i][0]+" nb cycle = "+str(cpt))
                myaudio = pydub.AudioSegment.from_wav(dir+data[i][0]+".wav")
                chunk_data = myaudio[int(float(data[i][1])*1000):int(float(data[i][2])*1000)]
                saved_file = (output_dir+save_file_name+"_"+f"{cpt:02d}"+".wav")
                # print("saved cycle name = "+saved_file)
                chunk_data.export(saved_file, format="wav")
                i+=1
                cpt+=1
                if i == lines:
                    break
    return i

###########################################################################

#  ██  ██      ███    ███  █████  ██ ███    ██
# ████████     ████  ████ ██   ██ ██ ████   ██
#  ██  ██      ██ ████ ██ ███████ ██ ██ ██  ██
# ████████     ██  ██  ██ ██   ██ ██ ██  ██ ██
#  ██  ██      ██      ██ ██   ██ ██ ██   ████

###########################################################################

arguments = sys.argv
if(len(sys.argv) != 2):
    print("ERROR parsing arguments")
    print("Give as input the directory containing the data you want to preprocess")
    sys.exit()
directory = arguments[1]
sample_save_place = directory+"splitted_into_cycles/"
os.makedirs(sample_save_place, exist_ok=True)
csv_output = sample_save_place+"info.csv"
csv_info = open(csv_output, "w")
mining_symptoms_info(directory,csv_info)
csv_info.close()


total = split_record_in_cycle(directory,csv_output,sample_save_place)
print()
print("All files splitted :")
print(str(total)+" samples saved in "+sample_save_place )

###########################################################################
