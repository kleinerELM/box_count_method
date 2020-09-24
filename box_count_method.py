# -*- coding: utf-8 -*-
import csv
import os, sys, getopt
import tifffile as tiff
import tkinter as tk
import math
import cv2
import numpy as np
from tkinter import filedialog
import random
import time
import multiprocessing
import pandas as pd
import statistics 
import random

import matplotlib
import matplotlib.pyplot as plt

def programInfo():
    print("#########################################################")
    print("# Automatically stich a random image from otherwise     #")
    print("# generated TIFF tiles.                                 #")
    print("#                                                       #")
    print("# © 2020 Florian Kleiner                                #")
    print("#   Bauhaus-Universität Weimar                          #")
    print("#   Finger-Institut für Baustoffkunde                   #")
    print("#                                                       #")
    print("#########################################################")
    print()

#### process given command line arguments
def processArguments( settings ):
    argv = sys.argv[1:]
    usage = sys.argv[0] + " [-h] [-x] [-y] [-d]"
    try:
        opts, args = getopt.getopt(argv,"hcx:y:d",[])
    except getopt.GetoptError:
        print( usage )
    for opt, arg in opts:
        if opt == '-h':
            print( 'usage: ' + usage )
            print( '-h,                  : show this help' )
            print( '-o,                  : setting output directory name [{}]'.format(settings["outputDirName"]) )
            print( '-c                   : creating subfolders for each image [./{}/FILENAME/]'.format(settings["outputDirName"]) )
            print( '-d                   : show debug output' )
            print( '' )
            sys.exit()
        elif opt in ("-o"):
            settings["outputDirName"] = arg
            print( 'changed output directory to ' + settings["outputDirName"] )
        elif opt in ("-c"):
            settings["createFolderPerImage"] = True
            print( 'creating subfolders for images' )
        elif opt in ("-d"):
            print( 'show debugging output' )
            settings["showDebuggingOutput"] = True
    print( '' )
    return settings

class phaseContent():
    pass

red = np.array((255,118,198), dtype = "uint8")
green = np.array((130,255,79), dtype = "uint8")
blue = np.array((0,0,255), dtype = "uint8")

phase_list = [red,green,blue]

def read_dataset( settings, file, resultDF, showMasks = False ):

    selected_img = cv2.imread( settings["workingDirectory"] + os.sep + file, cv2.IMREAD_COLOR )
    height, width = selected_img.shape[:2]
    if ( height > 0 and width > 0 ):
        #if ( settings["min_height"] > height or settings["min_height"] == 0 ) : settings["min_height"] = height
        #if ( settings["min_width"] > width or settings["min_width"] == 0 ) : settings["min_width"] = width
        
        new_row = {'filename':file, 'height':height, 'width':width }
        
        mask_list = []

        plot_count = len(phase_list) + 1
        plot_column_count = math.ceil(plot_count/2)
        plot_pos = 1
        mask_index = 0
        if showMasks: plt.subplot(2,plot_column_count,plot_pos),plt.imshow(selected_img)
        for phase in phase_list:
            plot_pos += 1
            mask_list.append( cv2.inRange(selected_img, phase, phase) )

            if showMasks: plt.subplot(2,plot_column_count,plot_pos),plt.imshow(mask_list[mask_index])
            
            new_row['phase_{:02d}'.format(mask_index)] = cv2.countNonZero( mask_list[mask_index] )
            new_row['phase_{:02d}_percent'.format(mask_index)] = new_row['phase_{:02d}'.format(mask_index)]/(height*width)*100
            mask_index +=1

        if showMasks:
            plt.show()

        resultDF = resultDF.append(new_row, ignore_index=True)
        return resultDF

    else:
        print('image dimensions are strange! ({} x {} px)'.format(height, width))

def getStdDevrow( experiment_list ):
    stdDevRow = []
    len(experiment_list)
    for i in range( len(experiment_list) ):
        if i > 1:
            stdDevRow.append( statistics.pstdev(experiment_list[:i]) )
            #statistics.pstdev( experiment_list[:10] )
            #population stddev (keine Stichprobe, also nicht n-1)
            #print('i={}, ±{:.2f} %'.format(i, statistics.pstdev(experiment_list[:i])))
    return stdDevRow

### actual program start
if __name__ == '__main__':
    #remove root windows
    root = tk.Tk()
    root.withdraw()
    coreCount = multiprocessing.cpu_count()
    settings = {
        "showDebuggingOutput" : False,
        "delete_interim_results" : True,
        "col_count" : 2,
        "row_count" : 2,
        "home_dir" : os.path.dirname(os.path.realpath(__file__)),
        "workingDirectory" : "",
        "targetDirectory"  : "resulting_sets",
        "referenceFilePath" : "",
        "outputDirName" : "corrected",
        "count" : 0,
        "coreCount" : multiprocessing.cpu_count(),
        "processCount" : (coreCount - 1) if coreCount > 1 else 1,
        "min_height" : 0,
        "min_width" : 0,
        "image_count": 0
    }

    programInfo()
    settings = processArguments( settings )
    print( "Please select the directory with the source image tiles.", end="\r" )
    settings["workingDirectory"] = filedialog.askdirectory(title='Please select the image / working directory')
    print( "                                                        ", end="\r" )

    if ( settings["showDebuggingOutput"] ) : 
        print( 'Found ' + str( settings["coreCount"] ) + ' CPU cores. Using max. ' + str( settings["processCount"] ) + ' processes at once.' )
        print( "I am living in '" + settings["home_dir"] + "'" )
        print( "Selected working directory: " + settings["workingDirectory"], end='\n\n' )
        print( "Selected reference file: " + os.path.basename( settings["referenceFilePath"] ) )

    ## count files
    fileList = []
    if os.path.isdir( settings["workingDirectory"] ):
        settings["targetDirectory"] = settings["workingDirectory"] + os.sep + settings["outputDirName"] + os.sep
        if not os.path.exists( settings["targetDirectory"] ):
            os.makedirs( settings["targetDirectory"] )

        column_list = ['filename' , 'height', 'width' ]
        mask_index = 0
        for phase in phase_list:
            column_list.append( 'phase_{:02d}'.format(mask_index) )
            column_list.append( 'phase_{:02d}_percent'.format(mask_index) )
            mask_index +=1


        resultDF = pd.DataFrame(columns = column_list)

        print('Checking files in the directory')
        for file in os.listdir(settings["workingDirectory"]):
            if ( file.endswith( ".tif" ) ):
                fileList.append( file )
                if settings["count"] > 0 and settings["count"] % 10 == 0: 
                    print('processing file #{:02d}'.format( settings["count"] ))#, end='')
                resultDF = read_dataset( settings, file, resultDF )
                
                settings["count"] += 1
        print('Found {} images'.format(settings["count"]))
        
        
        if ( settings["count"] > 1 ):
            # calculate stabw
            keyName = 'phase_{:02d}_percent'
            experimentCount = 100
            sampleFraction = 10/len(resultDF)
            resultList = {}
            stDevDFList = {}
            stDevcols = list(range(experimentCount-2))
            i = 0
            for phase in phase_list:
                key = keyName.format(i)
                resultList[key] = []
                stDevDFList[key] = pd.DataFrame(columns = stDevcols )
                i +=1
            
            for c in range(0, experimentCount):
                sampleDF = resultDF.sample(frac=sampleFraction)
                i = 0
                for phase in phase_list:
                    key = keyName.format(i)
                    resultList[key].append( sampleDF[key].mean(skipna = True) )
                    i +=1

            randomTries = 100
            i = 0
            for phase in phase_list:
                key = keyName.format(i)
                print('mean std deviation of ' + key)
                for j in range( randomTries ):
                    random.shuffle( resultList[key] )
                    #print(  '{}: {:.2f}'.format(key, statistics.pstdev( resultList[key] ) ) )
                    row = getStdDevrow( resultList[key] )
                    stDevDFList[key] = stDevDFList[key].append( pd.Series(row, index = stDevcols) , ignore_index=True)  
                    #print(resultList[key])
                
                for col in stDevcols:
                    stdev = stDevDFList[key][col].values.std(ddof=1)
                    mean = stDevDFList[key][col].mean()
                    print( ' - {:02d} images: {:.2f}±{:.2f} % '.format(col+2, mean, stdev) )

                i +=1
        else:
            print( "not enough images found!" )
    
    print( "Script DONE!" )