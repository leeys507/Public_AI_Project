import os
import glob
import random
from fingerSnapUtil import *

# path.txt 파일 안에 60만 txt파일들이 있는 디렉토리를 써놔요
# ex) path.txt << C:/Users/me/Desktop/txtfiles
pathFile = open("./path.txt", "r", encoding="utf8")
path = pathFile.readline()
pathFile.close()

txtFiles = []
while len(txtFiles) < 1:
    count = str(random.randrange(0, 623))
    # count = 000 ~ 622
    txtFiles = glob.glob(os.path.join(path, "KsponSpeech_"+count+"*.txt"))

dataFile = open("./data.csv", "a", encoding="utf8")

for txtFile in txtFiles:
    fp = open(txtFile, "r", encoding="euc-kr")
    for line in fp.readlines():
        line = strip(line)
        validLabels = ["0", "1", "2", "3", "4", "n"]
        label = "999"
        while not(label in validLabels):
            print(line)
            label = input("0:hello 1:sorry 2:thank 3:emergency 4:weather, n:none. which? ")
            if label == "q":
                exit()
        if label == "n":
            continue
        lineOut = "\"" + line + "\"," + label + "\n"
        lineOut = lineOut.encode('utf-8')
        lineOut = lineOut.decode('utf-8')
        dataFile.write(lineOut)
    fp.close()
    os.remove(txtFile)

dataFile.close()
