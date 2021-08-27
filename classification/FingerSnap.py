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
count = ""
while len(txtFiles) < 1:
    count = str(random.randrange(200, 401))
    # count = 000 ~ 622
    txtFiles = glob.glob(os.path.join(path, "KsponSpeech_"+count+"*.txt"))

for txtFile in txtFiles:
    fp = open(txtFile, "r", encoding="euc-kr")
    for line in fp.readlines():
        line = strip(line)
        validLabels = ["0", "1", "2",   # "인사", "미안", "고마워",
                       "3", "4", "5",   # "위급", "날씨", "부탁",
                       "6", "7", "8",   # "구입", "거절", "계절",
                       "n"]             # nope?
        label = "999"
        while not(label in validLabels):
            print("\n>>>" + line)
            label = input("0:인사 1:죄송 2:감사 3:비상 4:날씨,\n"
                          "5:부탁 6:구입 7:거절 8:계절 n:none. which? ")
            if label == "q":
                exit()
        if label == "n":
            label = "9"
        dataFile = open("./data.csv", "a", encoding="utf8")
        lineOut = "\"" + line + "\"," + label + "\n"
        lineOut = lineOut.encode('utf-8')
        lineOut = lineOut.decode('utf-8')
        dataFile.write(lineOut)
        dataFile.close()
    fp.close()
    os.remove(txtFile)

print(count + " done")

