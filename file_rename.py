import os
file_path = 'C:/Users/me1/Desktop/ai_data\Car_Data/train'
file_names = os.listdir(file_path)

i = 1
for name in file_names:
    src = os.path.join(file_path, name)
    dst = name.replace(" ", "")
    dst = os.path.join(file_path, dst)
    os.rename(src, dst)
    i += 1