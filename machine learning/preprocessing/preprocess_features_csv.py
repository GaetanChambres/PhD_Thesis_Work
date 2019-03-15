import csv
import sys

def file_len(fname):
    with open(fname) as f:
        nb = 0
        while(f.readline()):
            nb+=1
    return nb

input1 = './data/csv/challenge/test_lowlevel.csv'
input2 = './data/csv/challenge/test_info.csv'
output = './data/csv/challenge/test_pathologies.csv'

feat_len = file_len(input1)-1
info_len = file_len(input2)

if((feat_len)!=info_len):
    print("ERROR ! Invalid files")
    sys.exit()

features = open(input1,'r')
info = open(input2,'r')
out = open(output,'w')


header = features.readline()
file = header[:8]
rest = header[9:]

feat = features.readline()
infos = info.readline()
cpt=0
while(feat and infos):

    values = feat.split(',')
    name = values[0]
    feats = ""
    for i in range(1,len(values)):
        feats += (str(values[i])+",")

    patient_number,patho,record_index,body_area,record_tool,channel,start_time,end_time,c,w = infos.split(",")
    feat = features.readline()
    infos = info.readline()
    if(cpt==0):
        out.write(file+","+"pathologie"+","+rest)
        cpt = 1
    else:
        out.write(name+","+patho+","+feats[:-1])
