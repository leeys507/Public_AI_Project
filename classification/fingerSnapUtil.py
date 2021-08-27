# b/ : 숨소리
# n/ : 주변잡음
# l/ : 웃음
# o/ : 다른 사람과 겹침
# aaa(aaa) : 이중 전사
# 응/ : 간투어
# +나 *는 지워도 됨

def strip(line):
    badChars = ["b/", "n/", "l/", "o/", "u/", "/", "+", "*", "\n"]
    for c in badChars:
        line = line.replace(c, "")

    temp = line.replace("  ", " ")
    while line != temp:
        line = temp.replace("  ", " ")
        temp = line.replace("  ", " ")
        print("fingerSnapUtil.py >> strip loop")

    return line
