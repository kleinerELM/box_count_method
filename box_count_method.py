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

#remove root windows
root = tk.Tk()
root.withdraw()

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

home_dir = os.path.dirname(os.path.realpath(__file__))

rsb_file = 'image_slicer'
rsb_path = os.path.dirname( home_dir ) + os.sep + 'image_slicer' + os.sep
if ( os.path.isdir( rsb_path ) and os.path.isfile( rsb_path +rsb_file + '.py' ) or os.path.isfile( home_dir + rsb_file + '.py' ) ):
    if ( os.path.isdir( rsb_path ) ): sys.path.insert( 1, rsb_path )
    import image_slicer
else:
    programInfo()
    print( 'missing ' + rsb_path + rsb_file + '.py!' )
    print( 'download from https://github.com/kleinerELM/image_slicer' )
    sys.exit()
    
ts_path = os.path.dirname( home_dir ) + os.sep + 'tiff_scaling' + os.sep
ts_file = 'set_tiff_scaling'
if ( os.path.isdir( ts_path ) and os.path.isfile( ts_path + ts_file + '.py' ) or os.path.isfile( home_dir + ts_file + '.py' ) ):
    if ( os.path.isdir( ts_path ) ): sys.path.insert( 1, ts_path )
    import extract_tiff_scaling as es
else:
    programInfo()
    print( 'missing ' + ts_path + ts_file + '.py!' )
    print( 'download from https://github.com/kleinerELM/tiff_scaling' )
    sys.exit()

def getBaseSettings():
    settings = image_slicer.getBaseSettings()
    settings['slice_image'] = False
    settings["col_count"] = 10
    settings["row_count"] = 10
    settings["createFolderPerImage"] = True
    settings["outputDirectory"] = ''
    return settings

#### process given command line arguments
def processArguments():
    settings = getBaseSettings()
    col_changed = False
    row_changed = False
    argv = sys.argv[1:]
    usage = sys.argv[0] + " [-h] [-x] [-y] [-s] [-c] [-d]"
    try:
        opts, args = getopt.getopt(argv,"hcx:y:sd",[])
    except getopt.GetoptError:
        print( usage )
    for opt, arg in opts:
        if opt == '-h':
            print( 'usage: ' + usage )
            print( '-h,                  : show this help' )
            print( '-s                   : slice the image [OFF]' )
            print( '-x,                  : amount of slices in x direction [{}]'.format(settings["col_count"]) )
            print( '-y,                  : amount of slices in y direction [{}]'.format(settings["row_count"]) )
            print( '-c                   : creating subfolders for each image [./{}/FILENAME/]'.format(settings["outputDirectory"]) )
            print( '-d                   : show debug output' )
            print( '' )
            sys.exit()
        elif opt in ("-c"):
            settings["createFolderPerImage"] = True
            print( 'creating subfolders for images' )
        elif opt in ("-s"):
            settings["slice_image"] = True
            print( 'The image will be sliced' )
        elif opt in ("-x"):
            settings["col_count"] = int( arg )
            col_changed = True
            print( 'changed amount of slices in x direction to {}'.format(settings["col_count"]) )
        elif opt in ("-y"):
            settings["row_count"] = int( arg )
            row_changed = True
        elif opt in ("-d"):
            print( 'show debugging output' )
            settings["showDebuggingOutput"] = True
    if settings["slice_image"]:
        if col_changed and not row_changed:
            settings["row_count"] = settings["col_count"]
            print( 'changed amount of slices in y direction also to {}'.format(settings["row_count"]) )
        elif row_changed and not col_changed:
            settings["col_count"] = settings["row_count"]
            print( 'changed amount of slices in x direction also to {}'.format(settings["col_count"]) )
    print( '' )
    return settings

def getFolderScaling(directory):
    scaling = image_slicer.getEmptyScaling()
    for filename in os.listdir( directory ):
        if ( filename.endswith( ".tif" ) ):
            scaling = es.autodetectScaling( filename, directory )
            break
    return scaling

def singeFileProcess(settings=None, x=10, y=10, verbose=False):
    if settings == None: 
        settings = getBaseSettings()
        settings["col_count"] = x
        settings["row_count"] = y
    print( "Please select the directory with the source image tiles.", end="\r" )
    filepath = filedialog.askopenfilename( title='Please select the reference image', filetypes=[("Tiff images", "*.tif;*.tiff")] )
    settings["workingDirectory"] = os.path.dirname( filepath )
    file_name, file_extension = os.path.splitext( filepath )
    file_name = os.path.basename(file_name)

    scaling = image_slicer.sliceImage( settings, file_name, file_extension )#, verbose=settings["showDebuggingOutput"] )

    settings["workingDirectory"] = settings["workingDirectory"] + os.sep + file_name
    return phaseContent(settings["workingDirectory"], scaling=scaling, verbose=verbose)

def folderProcess(verbose=False):
    print( "Please select the directory with the source image tiles.", end="\r" )
    working_directory = filedialog.askdirectory(title='Please select the working directory')
    print( " "*60, end="\r" )
    
    # try to get the scaling from the fist file in the folder
    scaling = getFolderScaling( working_directory )

    print('start processing Files in "{}"...'.format(working_directory))

    return phaseContent(working_directory, scaling=scaling, verbose=verbose)

class phaseContent():
    # colors used in the example file
    red = np.array((255,118,198), dtype = "uint8")
    green = np.array((130,255,79), dtype = "uint8")
    blue = np.array((0,0,255), dtype = "uint8")

    # list of phases
    phase_list = [red,green,blue]
    phase_names = ['resin/pores', 'C-S-H', 'C₃S']
    phase_count = 3

    # list of columns within the main dataframe excluding phase contents
    column_list = ['filename' , 'height', 'width' ]

    # file name of the CSV
    CSV_appendix = '_box_count_intermediate.csv'

    image_count = 0
    tile_width = 0
    tile_height = 0

    # maximum image count in the sample
    image_shuffle_count = 100

    # standard experiment repeats
    repeat_sampling = 75
    
    stDev_mean = {}
    stDev_stDev = {}
    
    # dataframes
    phase_content_DF = None

    scaling = es.getEmptyScaling()

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
        if self.tile_width + self.tile_height == 0:
            self.tile_width = width
            self.tile_height = height
            return True
        else:
            if self.tile_width == width and self.tile_height == height:
                return True
            else:
                print( 'Tile sizes do not match! (w: {} != {} | h: {} != {})'.format(width, self.tile_width, height, self.tile_height) )
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

    def load_files(self, folder, verbose=False):
        CSVfilepath = folder + self.CSV_appendix
        if not os.path.isfile( CSVfilepath ):
            if verbose: print('Checking files in the directory')
            # count files
            for file in os.listdir(folder):
                if ( file.endswith( ".tif" ) ):
                    self.image_count += 1
            pos = 0
            for file in os.listdir(folder):
                if ( file.endswith( ".tif" ) ):
                    pos += 1
                    if pos % 10 == 0: 
                        if verbose: print('processing file #{:02d} of {:02d}'.format( pos, self.image_count ))#, end='')
                        
                    img = cv2.imread( folder + os.sep + file, cv2.IMREAD_COLOR )
                    self.read_dataset( img, file )
            
            self.saveToCSV( CSVfilepath, verbose=verbose )
        else:
            self.loadCSV( CSVfilepath, verbose=verbose )
            self.image_count = len(self.phase_content_DF)
        if verbose: print('Loaded {} images'.format(self.image_count))

    def saveToCSV(self, CSVfilepath, verbose=False):
        self.phase_content_DF.to_csv( CSVfilepath, index=False )
        if verbose: print('Phase analysis saved to {}'.format(CSVfilepath))

    def loadCSV(self, filepath=None, verbose=False):
        if verbose: print('reading CSV "{}.csv"'.format(os.path.splitext(os.path.basename(filepath))[0]))
        self.phase_content_DF = pd.read_csv(filepath, encoding='utf-8')
        self.phase_content_DF.fillna(0, inplace=True)
        self.check_tile_dimension(self.phase_content_DF['height'][1], self.phase_content_DF['width'][1])

    def reprocess_mean_and_stdev(self, repeat_sampling=None, verbose=True):
        if repeat_sampling == None: repeat_sampling = self.repeat_sampling
        # main process 
        if ( self.image_count > 1 ):
            # calculate stabw
            keyName = 'phase_{:02d}_percent'
            sampleFraction = 0.5# 10/self.image_count
            self.meanAfterNExperiments = {}
            self.stDevDFList = {}
            self.stDevcols = list(range(self.image_shuffle_count-2))
            self.phase_mean = {}

            if verbose: print( 'processing {} experiments'.format(repeat_sampling))

            # randomly select images (different image amount from 0 to image_shuffle_count)
            # and process the mean phase area for each phase
            
            # für jede Phase
            for i in range(self.phase_count):
                key = keyName.format(i)
                # create headers for each phase
                self.meanAfterNExperiments[key] = []
                self.stDevDFList[key] = pd.DataFrame(columns = self.stDevcols)
                self.stDev_mean[key] = []
                self.stDev_stDev[key] = []

                if verbose: print( 'mean & std deviation of {}'.format(self.phase_names[i]) )
                self.meanAfterNExperiments[key] = {}
                # für eine Samplegröße von 0 bis self.image_shuffle_count (100) Bildern
                for c in range(1, self.image_shuffle_count+1):
                    
                    #if c > 0 and c % 10 == 0: 
                    #    print('{:02d} / {:02d}'.format( c-1, self.image_shuffle_count ))

                    self.meanAfterNExperiments[key][c] = []
                    sampleFraction = c/self.image_count
                    # wiederholungen
                    for j in range( repeat_sampling ):
                        sampleDF = self.phase_content_DF.sample(frac=sampleFraction)
                        self.meanAfterNExperiments[key][c].append( sampleDF[key].mean(skipna = True) )
                    
                    self.stDev_mean[key].append( statistics.mean( self.meanAfterNExperiments[key][c] ) )
                    self.stDev_stDev[key].append( statistics.stdev( self.meanAfterNExperiments[key][c] ) )
            
            if verbose: print( '-'*20 )
        else:
            print( "not enough images found!" )

    def __init__(self, folder, phase_list=None, phase_names=None, scaling=None, verbose=False ):
        self.folder = folder
        self.target_folder = os.path.abspath(os.path.join(self.folder, os.pardir))
        self.init_phase_list(phase_list)
        self.init_column_list()
        if not scaling == None:
            self.scaling = scaling
        self.phase_content_DF = pd.DataFrame(columns = self.column_list)

        self.load_files(folder,verbose=verbose)

        # make shure the image_shuffle_count does not exceed the image count!
        if self.image_shuffle_count > self.image_count:
            self.image_shuffle_count = self.image_count

        self.reprocess_mean_and_stdev(verbose=verbose)


### actual program start
if __name__ == '__main__':
    #remove root windows
    root = tk.Tk()
    root.withdraw()

    coreCount = multiprocessing.cpu_count()
    mc_settings = {
        "coreCount" : multiprocessing.cpu_count(),
        "processCount" : (coreCount - 1) if coreCount > 1 else 1
    }

    programInfo()
    settings = processArguments()
    if settings["showDebuggingOutput"]: 
        print( 'Found {} CPU cores. Using max. {} processes at once.'.format(mc_settings["coreCount"], mc_settings["processCount"]) )
        print( "I am living in '" + settings["home_dir"] + "'" )

    if settings["slice_image"]:
        phaseContent = singeFileProcess(settings, verbose=settings["showDebuggingOutput"])
    else:
        phaseContent = folderProcess(verbose=settings["showDebuggingOutput"])
    
    print( "Script DONE!" )