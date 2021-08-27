import os
import glob

# path.txt 파일 안에 60만 txt파일들이 있는 디렉토리를 써놔요
# ex) path.txt << C:/Users/me/Desktop/txtfiles
pathFile = open("./path.txt", "r", encoding="utf8")
path = pathFile.readline()
pathFile.close()

count = "000"
# count = 000 ~ 622

txtFiles = glob.glob(os.path.join(path, "KsponSpeech_"+count+"*.txt"))

dataFile = open("./data.csv", "a", encoding="utf8")

for txtFile in txtFiles:
    txtFile = open(txtFile, "r", encoding="euc-kr")
    for line in txtFile.readlines():
        validLabels = ["0", "1", "2", "3", "4", "n"]
        label = "999"
        while not(label in validLabels):
            print(line)
            label = input("0:hello 1:sorry 2:thank 3:emergency 4:weather, n:none. which? ")
            if label == "q":
                exit()
        if label == "n":
            continue
        lineOut = "\"" + line[:-1] + "\"," + label + "\n"
        lineOut = lineOut.encode('utf-8')
        lineOut = lineOut.decode('utf-8')
        print(lineOut)
        exit()
        dataFile.write(lineOut)
    txtFile.close()

dataFile.close()
