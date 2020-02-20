import xml.etree.ElementTree as ET
import os
from os import getcwd

classes = ['Car','SUV','Ute','Van','Bus','Motorcycle','Truck/Trailer','Other']
count=[0,0,0,0,0,0,0,0]

def convert_annotation(filename, list_file):
    in_file = open('model_data/pic/%s'%(filename))
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        count[cls_id]=count[cls_id]+1

        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
    print(count)

filenames = os.listdir("model_data/pic")
list_file = open('model_data/train.txt', 'w')
for filename in filenames:
    # only care xml
    if filename.endswith(".jpg"):
        continue

    # write
    list_file.write('model_data/pic/%s.jpg'%(filename[:-4]))
    convert_annotation(filename, list_file)
    list_file.write('\n')
list_file.close()

