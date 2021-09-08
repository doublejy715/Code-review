"""
Black_Hair : 9 (1:black ~ 0:birght hair?)
Eyeglasses : 16(0:no / 1:yes )
Heavy_Makeup : 19(0:no / 1:yes)
Male : 21(0:female / 1:male)
Smiling : 32(0:no / 1:yes)
Young : 40(0:old / 1:young)
"""
import json

result = []
label = open('dataset/list_attr_celeba.txt','r')
result = open('dataset/list_attr_celeba_3.txt','w')
for line in label.readlines():
    line = line.split(' ')
    line = [item for item in line if item != '']
    line = [line[9]+' '+line[16]+' '+line[19]+' '+ line[21]+' '+line[32]+' '+line[40]]
    result.write(*line)
