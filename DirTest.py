import os

import numpy

cnt=0
path='C:/Users/Owner/PycharmProjects/FA/LastVGG/Train'
subfolders = [ os.path.basename(f) for f in os.scandir(path) if f.is_dir() ]
print(subfolders)


boxes=[]
x=2
y=4
w=3
h=8
idx=1
x1=2
y1=8
w1=3
h1=8
idx1=2
boxes.append([x,y,w,h,idx])
boxes.append([x1,y1,w1,h1,idx1])
print(boxes)
list = []
##use append to add items to the list.

list.append({'A':0,'C':0,'G':0,'T':0})
list.append({'A':1,'C':1,'G':1,'T':1})
print(numpy.shape(list))
#if u need to add n no of items to the list, use range with append:
for i in range(len(list)):
    list.append({'A':0,'C':0,'G':0,'T':0})

print (list)

class_names = {
    "0": "Bill",
    "1": "Elon",
    "2": "Jihyo",
    "3": "Sana",
    "4": "Suji"
}
class_names=[]
class_name={}
for idx in range(len(subfolders)):
    class_name={idx: subfolders[idx]}
    class_names.append(class_name)

print(class_names)
print(class_names)

result ='[ {} ]'.format(class_names[1])
print(result)