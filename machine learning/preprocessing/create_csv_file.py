import os
import csv
import re


input_directory = './data/database/debug/train/'
output_directory = './data/csv/debug/'
output_file = 'train_info.csv'
output = output_directory + output_file
out_file = open(output, "w")

cpt = 0
nb_files = (len(os.listdir(input_directory)))/2

ordered_files = sorted(os.listdir(input_directory))

info_pathologies = "./data/diag.txt"
with open(info_pathologies) as fp:
    diag = fp.read().splitlines()

for filename in ordered_files:
    if(filename.endswith('.txt')):
        cpt = cpt+1
        input = input_directory + filename

        input_file = open(input,"r")

        content = input_file.readline()
        tmp = filename[:-4]
        patient_number,record_index,body_area,channel,record_tool = tmp.split("_")
        patho = 0
        for i in range(0,len(diag)-1):
            tmp_patient, tmp_patho = diag[i].split("\t")
            if patient_number == tmp_patient:

                if(tmp_patho == "Asthma"):
                    patho = 1
                if(tmp_patho == "LRTI"):
                    patho = 2
                if(tmp_patho == "Pneumonia"):
                    patho = 3
                if(tmp_patho == "Bronchiectasis"):
                    patho = 4
                if(tmp_patho == "Bronchiolitis"):
                    patho = 5
                if(tmp_patho == "URTI"):
                    patho = 6
                if(tmp_patho == "COPD"):
                    patho = 7
                if(tmp_patho == "Healthy"):
                    patho = 8

        # print(filename)
        # print(patient_number)
        # print(patho)
        # print(record_index)
        # print(body_area)
        # print(channel)
        # print(record_tool)

        while content:
            start_time,end_time,crackle,wheezle = content.split('\t')
            # print "start_time = ",start_time
            # print "end_time = ",end_time
            # print "crackles = ",crackle
            # print "wheezle = ",wheezle

            # print(patient_number + "," + record_index + "," + body_area + "," + record_tool + "," + channel + "," + start_time + "," + end_time + "," + crackle + "," + wheezle)
            out_file.write(patient_number + "," + str(patho) + "," + record_index + "," + body_area + "," + record_tool + "," + channel + "," + start_time + "," + end_time + "," + crackle + "," + wheezle)

            content = input_file.readline()
        print "processed ",cpt,"over ",nb_files
