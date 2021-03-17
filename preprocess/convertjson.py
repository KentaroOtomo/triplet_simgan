import sys
import numpy as np
#mport cv2
import os
import json
import pdb
#import pprint
#import copy

json_num = 1
num = 101624
#num = len(os.listdir("/home/silvias15/simulated-unsupervised-tensorflow/data/gaze/UnityEyes"))
#num = len(os.listdir("/media/silvias15/40e69c69-43ae-4816-b6c5-c46f45bf6469/simgan-torch/data/new_unityeyes_3class/left/"))
#print(file_num)
value_x = []
value_y = []
value_z = []
#for experiment none and insert

while int(json_num) <= num:
    with open(os.path.join("/media/silvias15/40e69c69-43ae-4816-b6c5-c46f45bf6469/triplet_simgan/data/cgimg_9direction/", "{}.json".format(json_num))) as json_file:

        alldata = json.load(json_file)
        keylist = alldata.keys()
        #print(alldata)
        for k in keylist:
            if k == 'interior_margin_2d':
                #print(k)
                datadict = alldata[k]
                datadict2 = ",".join(datadict)
                datadict2_new = datadict2.replace('"','')
                datadict2_list_new = datadict2_new.split(",")
                #print(datadict2_new)
                #print(type(datadict))
                #print(datadict[1])
                
                for i , name in enumerate(datadict):
                    value = datadict[i]
                    #print(i)
                    #print(datadict[i])
                    #print(type(datadict))
                    value = value.split(",")
                    value[0] = value[0].replace("(" , "")
                    value[0] = value[0].replace("．" , ".")
                    value[1] = value[1].replace(" ", "")
                    value[1] = value[1].replace("．" , ".")
                    #value[2] = value[2].replace(")", "")
                    #print(type(value[0]))
                    #print(float(value[0]))
                    value[1] = float(value[1])
                    value[0] = float(value[0])
                    #value[2] = float(value[2])
                    value_x.append(value[0])
                    value_y.append(value[1])
                    value_z.append(value[2])
                #print(max(value_x))
                #print(min(value_x))
                #print('value_y = ' + str(len(value_y)))
                x_min, x_max = min(value_x), max(value_x)
                y_min, y_max = min(value_y), max(value_y)
                #print("x_min = " + str(x_min))
                #print("x_max = " + str(x_max))
                #print("y_min = " + str(y_min))
                #print("y_max = " + str(y_max))
                #x_min, x_max = int(min(json_data[key_eyelid][:,0])), int(max(json_data[key_eyelid][:,0]))
                #y_min, y_max = int(min(json_data[key_eyelid][:,1])), int(max(json_data[key_eyelid][:,1]))
                x_center = int((x_min + x_max)/2)
                x_cen2 = x_center
                y_center = int((y_min + y_max)/2)
                y_cen2 = y_center
                #print("x_cen = " + str(x_center))
                #print("y_cen = " + str(y_center))
                for number in range(0, len(value_x)):
                    #print("num = " + str(num))
                    #print("value_y[num] = " + str(value_y[number]))
                    value_x[number] = (value_x[number] - (x_cen2-120)) * (2/3)
                    value_y[number] = ((value_y[number] - (y_cen2-72)) * (2/3))
                    #print("after value_y[num] = " + str(value_y[number]))
                        #json_data[key_caruncle][:,0] = (json_data[key_caruncle][:,0] - (x_center-135)) * (1/3)
                        #json_data[key_caruncle][:,1] = (json_data[key_caruncle][:,1] - (y_center-81)) * (1/3)
                        #json_data[key_iris][:,0] = (json_data[key_iris][:,0] - (x_center-135)) * (1/3)
                        #json_data[key_iris][:,1] = (json_data[key_iris][:,1] - (y_center-81)) * (1/3)
                        #center_x = (max(value_x)+min(value_x)) / 2    
                        #edge_x = center_x - 110
                        #center_y = (max(value_y)+min(value_y)) / 2
                        #edge_y = center_y - 70
                        #print(len(value_y))
                #for number in range(0, len(value_x)):
                    #value_x[number] = round(((value_x[number] - edge_x) / 4), 4)
                    #value_y[number] = round(((value_y[number] - edge_y) / 4), 4)
                    
                
                alldata[k] = {}
                listbox = []
                for i in range(len(value_x)):
                #    if i == 0:
                    value = str("(") + str(value_x[i]) + str(",") + str(" ") + str(value_y[i]) + str(",") + value_z[i]
                #        datadict[i] = value
                #        alldata[k] = value
                #    else:
                #        value += str("(") + str(value_x[i]) + str(",") + str(" ") + str(value_y[i]) + str(",") + value_z[i] + str(",")
                #        datadict[i] = value
                    listbox.append(value)
            # print(value)
                    
                #print(datadict[i])
                alldata[k] = listbox

                value_x = []
                value_y = []
                value_z = []
                #print('1st finish.')  


            if k == 'caruncle_2d':
                #print(k)
                #print('test')
                datadict = alldata[k]
                datadict2 = ",".join(datadict)
                datadict2_new = datadict2.replace('"','')
                datadict2_list_new = datadict2_new.split(",")
                #print(datadict2_new)
                #print(type(datadict))
                #print(datadict[1])
                
                for i , name in enumerate(datadict):
                    value = datadict[i]
                    #print(datadict[i])
                    #print(type(datadict))
                    value = value.split(",")
                    value[0] = value[0].replace("(" , "")
                    value[0] = value[0].replace("．" , ".")
                    value[1] = value[1].replace(" ", "")
                    value[1] = value[1].replace("．" , ".")
                    #value[2] = value[2].replace(")", "")
                    #print(type(value[0]))
                    #print(float(value[0]))
                    value[1] = float(value[1])
                    value[0] = float(value[0])
                    #value[2] = float(value[2])
                    value_x.append(value[0])
                    value_y.append(value[1])
                    value_z.append(value[2])
                """
                #print(max(value_x))
                #print(min(value_x))
                #print(value_x)
                center_x = (max(value_x)+min(value_x)) / 2    
                edge_x = center_x - 110
                center_y = (max(value_y)+min(value_y)) / 2
                edge_y = center_y - 70
                #print(len(value_y))
                for number in range(0, len(value_x)):
                    value_x[number] = round(((value_x[number] - edge_x) / 4), 4)
                    value_y[number] = round(((value_y[number] - edge_y) / 4), 4)
                """
                x_min, x_max = min(value_x), max(value_x)
                y_min, y_max = min(value_y), max(value_y)
                #print("x_min = " + str(x_min))
                #print("x_max = " + str(x_max))
                #print("y_min = " + str(y_min))
                #print("y_max = " + str(y_max))
                #x_min, x_max = int(min(json_data[key_eyelid][:,0])), int(max(json_data[key_eyelid][:,0]))
                #y_min, y_max = int(min(json_data[key_eyelid][:,1])), int(max(json_data[key_eyelid][:,1]))
                x_center, y_center = int((x_min + x_max)/2), int((y_min + y_max)/2)
                for number in range(0, len(value_x)):
                    value_x[number] = (value_x[number] - (x_cen2-120)) * (2/3)
                    value_y[number] = ((value_y[number] - (y_cen2-72)) * (2/3))
                
                alldata[k] = {}
                listbox = []
                for i in range(len(value_x)):
                #    if i == 0:
                    value = str("(") + str(value_x[i]) + str(",") + str(" ") + str(value_y[i]) + str(",") + value_z[i]
                #        datadict[i] = value
                #        alldata[k] = value
                #    else:
                #        value += str("(") + str(value_x[i]) + str(",") + str(" ") + str(value_y[i]) + str(",") + value_z[i] + str(",")
                #        datadict[i] = value
                    listbox.append(value)
            # print(value)
                    
                #print(datadict[i])
                alldata[k] = listbox

            value_x = []
            value_y = []
            value_z = []
            #print('2nd finish.')  
 
            
            if k == 'iris_2d':
                #print(k)
                datadict = alldata[k]
                datadict2 = ",".join(datadict)
                datadict2_new = datadict2.replace('"','')
                datadict2_list_new = datadict2_new.split(",")
                #print(datadict2_new)
                #print(type(datadict))
                #print(datadict[1])
                
                for i , name in enumerate(datadict):
                    value = datadict[i]
                    #print(datadict[i])
                    #print(type(datadict))
                    value = value.split(",")
                    value[0] = value[0].replace("(" , "")
                    value[0] = value[0].replace("．" , ".")
                    value[1] = value[1].replace(" ", "")
                    value[1] = value[1].replace("．" , ".")
                    #value[2] = value[2].replace(")", "")
                    #print(type(value[0]))
                    #print(float(value[0]))
                    value[1] = float(value[1])
                    value[0] = float(value[0])
                    #value[2] = float(value[2])
                    value_x.append(value[0])
                    value_y.append(value[1])
                    value_z.append(value[2])
                #print(max(value_x))
                #print(min(value_x))
                #print(value_x)
                """
                center_x = (max(value_x)+min(value_x)) / 2    
                edge_x = center_x - 110
                center_y = (max(value_y)+min(value_y)) / 2
                edge_y = center_y - 70
                #print(len(value_y))
                for number in range(0, len(value_x)):
                    value_x[number] = round(((value_x[number] - edge_x) / 4), 4)
                    value_y[number] = round(((value_y[number] - edge_y) / 4), 4)
                """
                x_min, x_max = min(value_x), max(value_x)
                y_min, y_max = min(value_y), max(value_y)
                #print("x_min = " + str(x_min))
                #print("x_max = " + str(x_max))
                #print("y_min = " + str(y_min))
                #print("y_max = " + str(y_max))
                #x_min, x_max = int(min(json_data[key_eyelid][:,0])), int(max(json_data[key_eyelid][:,0]))
                #y_min, y_max = int(min(json_data[key_eyelid][:,1])), int(max(json_data[key_eyelid][:,1]))
                x_center, y_center = int((x_min + x_max)/2), int((y_min + y_max)/2)
                for number in range(0, len(value_x)):
                    value_x[number] = (value_x[number] - (x_cen2-120)) * (2/3)
                    value_y[number] = ((value_y[number] - (y_cen2-72)) * (2/3))
                    
                alldata[k] = {}
                listbox = []
                for i in range(len(value_x)):
                #    if i == 0:
                    value = str("(") + str(value_x[i]) + str(",") + str(" ") + str(value_y[i]) + str(",") + value_z[i]
                #        datadict[i] = value
                #        alldata[k] = value
                #    else:
                #        value += str("(") + str(value_x[i]) + str(",") + str(" ") + str(value_y[i]) + str(",") + value_z[i] + str(",")
                #        datadict[i] = value
                    listbox.append(value)
            # print(value)
                    
                #print(datadict[i])
                alldata[k] = listbox

            value_x = []
            value_y = []
            value_z = []
            #print('all finish.')  

    f = open(os.path.join("/media/silvias15/40e69c69-43ae-4816-b6c5-c46f45bf6469/triplet_simgan/data/unityeyes_mpiigaze_json/", "{}.json".format(json_num)), 'w')
    json.dump(alldata, f, indent=2)
    json_num = json_num + 1
