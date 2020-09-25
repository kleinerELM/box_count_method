# -*- coding: utf-8 -*-
import csv
import os, sys, getopt
import tifffile as tiff
import tkinter as tk
from tkinter import filedialog
import math
import cv2
import numpy as np
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
    # colors used in the example file
    red = np.array((255,118,198), dtype = "uint8")
    green = np.array((130,255,79), dtype = "uint8")
    blue = np.array((0,0,255), dtype = "uint8")

    # list of phases
    phase_list = [red,green,blue]
    phase_count = 3

    # list of columns within the main dataframe excluding phase contents
    column_list = ['filename' , 'height', 'width' ]

    image_count = 0
    tile_with = 0
    tile_height = 0

    #
    image_shuffle_count = 100
    stdev_shuffle_count = 75
    
    stDev_mean = {}
    stDev_stDev = {}
    
    # dataframes
    phase_content_DF = None

    def init_phase_list(self, phase_list):
        if phase_list is not None: 
            self.phase_list = phase_list
            self.phase_count = len(self.phase_list)

    def init_column_list(self):
        i = 0
        for phase in self.phase_list:
            self.column_list.append( 'phase_{:02d}'.format(i) )
            self.column_list.append( 'phase_{:02d}_percent'.format(i) )
            i +=1

    def show_masked_phases(self, img, mask_list):
        plot_count = len(self.phase_list) + 1
        plot_column_count = math.ceil(plot_count/2)
        plot_pos = 1
        mask_index = 0
        plt.subplot(2,plot_column_count,plot_pos),plt.imshow(img)
        for phase in self.phase_list:
            plot_pos += 1

            plt.subplot(2,plot_column_count,plot_pos),plt.imshow(mask_list[mask_index])
            mask_index +=1

        plt.show()

    def check_tile_dimension(self, height, width):
        if self.tile_with + self.tile_height == 0:
            self.tile_with = width
            self.tile_height = height
            return True
        else:
            if self.tile_with == width and self.tile_height == height:
                return True
            else:
                print( 'Tile sizes do not match! (w: {} != {} | h: {} != {})'.format(width, self.tile_with, height, self.tile_height) )
                sys.exit()
        return False

    def getStdDevrow( self, experiment_list ):
        stdDevRow = []
        for i in range( len(experiment_list) ):
            if i > 1:
                stdDevRow.append( statistics.pstdev(experiment_list[:i]) )
                #statistics.pstdev( experiment_list[:10] )
                #population stddev (keine Stichprobe, also nicht n-1)
                #print('i={}, ±{:.2f} %'.format(i, statistics.pstdev(experiment_list[:i])))
            #if i > 96:
            #    print(i, len(experiment_list[:i]))
        return stdDevRow

    def read_dataset( self, img, file, showMasks = False ):
        height, width = img.shape[:2]
        if self.check_tile_dimension(height, width):            
            new_row = {'filename':file, 'height':height, 'width':width }
            
            mask_list = []

            mask_index = 0
            for phase in self.phase_list:
                mask_list.append( cv2.inRange(img, phase, phase) )

                new_row['phase_{:02d}'.format(mask_index)] = cv2.countNonZero( mask_list[mask_index] )
                new_row['phase_{:02d}_percent'.format(mask_index)] = new_row['phase_{:02d}'.format(mask_index)]/(height*width)*100
                mask_index +=1

            self.phase_content_DF = self.phase_content_DF.append(new_row, ignore_index=True)
            if showMasks: self.show_masked_phases(img, mask_list)

    def load_files(self, folder):
        print('Checking files in the directory')
        for file in os.listdir(folder):
            if ( file.endswith( ".tif" ) ):
                if self.image_count > 0 and self.image_count % 10 == 0: 
                    print('processing file #{:02d}'.format( self.image_count ))#, end='')
                    
                img = cv2.imread( folder + os.sep + file, cv2.IMREAD_COLOR )
                self.read_dataset( img, file )
                
                self.image_count += 1
        print('Found {} images'.format(self.image_count))

    
    def __init__(self, folder, phase_list=None ):
        self.folder = folder
        self.target_folder = os.path.abspath(os.path.join(self.folder, os.pardir))
        self.init_phase_list(phase_list)
        self.init_column_list()

        self.phase_content_DF = pd.DataFrame(columns = self.column_list)

        self.load_files(folder)

        # make shure the image_shuffle_count does not exceed the image count!
        if self.image_shuffle_count > self.image_count:
            self.image_shuffle_count = self.image_count

        # main process 
        if ( self.image_count > 1 ):
            # calculate stabw
            keyName = 'phase_{:02d}_percent'
            sampleFraction = 10/self.image_count
            self.resultList = {}
            self.stDevDFList = {}
            self.stDevcols = list(range(self.image_shuffle_count-2))
            self.phase_mean = {}

            i = 0
            # create headers for each phase
            for phase in self.phase_list:
                key = keyName.format(i)
                self.resultList[key] = []
                self.stDevDFList[key] = pd.DataFrame(columns = self.stDevcols)
                self.stDev_mean[key] = []
                self.stDev_stDev[key] = []
                i +=1
            
            # randomly select images (different image amount from 0 to image_shuffle_count)
            # and process the mean phase area for each phase
            for c in range(0, self.image_shuffle_count):
                sampleDF = self.phase_content_DF.sample(frac=sampleFraction)
                i = 0
                for phase in self.phase_list:
                    key = keyName.format(i)
                    self.phase_mean[key] = sampleDF[key].mean(skipna = True)
                    self.resultList[key].append( self.phase_mean[key] )
                    i +=1

            # get the standard deviation of the standard deviation
            i = 0
            for phase in self.phase_list:
                key = keyName.format(i)
                print('mean std deviation of ' + key)
                for j in range( self.stdev_shuffle_count ):
                    random.shuffle( self.resultList[key] )
                    row = self.getStdDevrow( self.resultList[key] )
                    self.stDevDFList[key] = self.stDevDFList[key].append( pd.Series(row, index = self.stDevcols) , ignore_index=True)
                
                for col in self.stDevcols:
                    self.stDev_mean[key].append( self.stDevDFList[key][col].mean() )
                    self.stDev_stDev[key].append( self.stDevDFList[key][col].values.std(ddof=1) )
                    #print( ' - {:02d} images: {:.2f}±{:.2f} % '.format(col+2, mean, stdev) )

                i +=1
        else:
            print( "not enough images found!" )


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
        "processCount" : (coreCount - 1) if coreCount > 1 else 1
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

    phaseContent = phaseContent(settings["workingDirectory"])
    
    print( "Script DONE!" )