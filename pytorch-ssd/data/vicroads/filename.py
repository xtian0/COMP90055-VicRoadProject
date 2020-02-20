import os

names = os.listdir('500/xml/')
i=0
file = open('testval.txt','w')
for name in names:
    index = name.rfind('.')
    name = name[:index]
    file.write(name+'\n')
    i=i+1
print(i)

