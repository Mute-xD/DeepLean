import os
import shutil

'''

raw -> train/cat, train/dog

'''
TRAIN_SIZE = 4000
TEST_SIZE = 1000

picList = [file for file in os.listdir('./dataSet/raw')]
catList, dogList = [], []
for name in picList:
    if name.split('.')[0] == 'cat':
        catList.append(name)
    else:
        dogList.append(name)
print(catList[:10], '\n\n', dogList[:10])
os.makedirs('./dataSet/train/cat')
os.makedirs('./dataSet/train/dog/')
os.makedirs('./dataSet/test/cat')
os.makedirs('./dataSet/test/dog')
for name in catList[:int(TRAIN_SIZE/2)]:
    shutil.copy('./dataSet/raw/'+name, './dataSet/train/cat/'+name)
for name in dogList[:int(TRAIN_SIZE/2)]:
    shutil.copy('./dataSet/raw/'+name, './dataSet/train/dog/'+name)

for name in catList[int(TRAIN_SIZE/2):int(TRAIN_SIZE/2+TEST_SIZE/2)]:
    shutil.copy('./dataSet/raw/'+name, './dataSet/test/cat/'+name)
for name in dogList[int(TRAIN_SIZE/2):int(TRAIN_SIZE/2+TEST_SIZE/2)]:
    shutil.copy('./dataSet/raw/'+name, './dataSet/test/dog/'+name)
print('Done!')
