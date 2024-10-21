# -*- coding: utf-8 -*-
import os, sys, getopt, math,cv2, multiprocessing, statistics, random, time
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import exposure

random.seed(time.time())

#remove root windows
root = tk.Tk()
root.withdraw()

def programInfo():
    print("#########################################################")
    print("# get statistical information if an image is large      #")
    print("# enough to measure the phase content.                  #")
    print("#                                                       #")
    print("# © 2024 Florian Kleiner                                #")
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
ts_file = 'extract_tiff_scaling'
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



def sliceImage( filepath, file_name, image, row_cnt, col_cnt, overwrite_existing = False, show_result = False, cmap = None ):
    if cmap is None: cmap = 'gist_rainbow'
    height = image.shape[0]
    width = image.shape[1]

    #cropping width / height
    crop_height = int(height/row_cnt)
    crop_width  = int( width/col_cnt)

    slice_name = "{}_{:04d}_{:04d}.tif"
    targetDirectory = filepath + os.path.sep + file_name
    if not os.path.isdir(targetDirectory): os.mkdir(targetDirectory)
    
    slices_already_exists = True
    if not overwrite_existing:
        for i in range(row_cnt): # start at i = 0 to row_count-1
            for j in range(col_cnt): # start at j = 0 to col_count-1
                fileij = slice_name.format(file_name, i, j)
                if not os.path.isfile( targetDirectory + fileij ):
                    slices_already_exists = False
    else:
        slices_already_exists = False

    if slices_already_exists:
        print("  The expected sliced images already exist! Doing nothing...")
    else:
        for i in range(row_cnt): # start at i = 0 to row_count-1
            print( "   - " + fileij )
            for j in range(col_cnt): # start at j = 0 to col_count-1
                fileij = slice_name.format(file_name, i,j)
                cropped_filename = targetDirectory + os.path.sep + fileij

                image[i*crop_height: (i+1)*crop_height,
                      j*crop_width : (j+1)*crop_width  ]
                cv2.imwrite( cropped_filename, image[i*crop_height: (i+1)*crop_height, j*crop_width : (j+1)*crop_width ], params=(cv2.IMWRITE_TIFF_COMPRESSION, 5) )
                
    if show_result:
        fig, ax = plt.subplots( 1, 1, figsize = ( 9, 9 ) )

        img = ax.imshow( image,	 cmap=cmap )
        ax.set_axis_off()
        for pos in range(col_cnt):
            ax.axvline(pos*crop_height, 0, 1, color='white')
        for pos in range(row_cnt):
            ax.axhline(pos*crop_width, 0, 1, color='white')
        plt.show()
        plt.savefig('box_count_method.svg')
        
    return targetDirectory


def measure_phase_composition( image, phases=None, verbose = True ):
    #counts, _  = np.histogram( image.ravel(), bins=range(len(phases)+1), density=True)
    counts = np.bincount(image.ravel()) / len(image.ravel()) # much faster (like 3x) than: colors, counts = np.unique(img.ravel(),
    if verbose and not phases is None:
        print("\nArea proportions:")
        for i, c in enumerate(counts):
            print('  {}: {:.2f}%'.format(phases[i], c*100))
    return counts
    
def growImageArea( image, phases, seed_window_f = 5, step_w=5, show_result = False, cmap=None ):
    if cmap is None: cmap = 'gist_rainbow' 
    height = image.shape[0]
    width = image.shape[1]
    
    #seed_window_f = 5 # Anteil des Bildes im Zentrum, in dem sich die Seed-position bewegen darf
    f_x = random.random()
    f_y = random.random()
    seed_pos_x = int( (width - f_x * width / seed_window_f)/2 )
    seed_pos_y = int( (height - f_y * height / seed_window_f)/2 )
    #print(f_x, f_y, seed_pos_x, seed_pos_y)
    if show_result: 
        fig, ax = plt.subplots( 1, 1, figsize = ( 9, 9 ) )
        ax.imshow( image,	 cmap=cmap )
        ax.set_axis_off()
        circ = patches.Circle((seed_pos_x,seed_pos_y),width*.01,edgecolor='white', facecolor='black')
        ax.add_patch(circ)
        
    areas = []
    results = []
    for f in range(step_w,100,step_w):
        size = (
            int( width*f/100 ),
            int( height*f/100 )
        )
        start_pos = (
            int(seed_pos_x - size[0]/2),
            int(seed_pos_y - size[1]/2)
        )
        # check if rectangle is outside the image area
        if  ( (start_pos[0]+size[0]) < width and start_pos[0] >= 0) and ( (start_pos[1]+size[1]) < height and start_pos[1] >= 0):
            results.append(
                measure_phase_composition( image[start_pos[1] : start_pos[1]+size[1], start_pos[0] : start_pos[0]+size[0]], phases, verbose = (show_result and f%25==0) )
            )
            areas.append(size[0]*size[1])
            
            if show_result and f%10==0:
                rect = patches.Rectangle(start_pos, size[0], size[1], linewidth=1, edgecolor='white', facecolor='none')
                ax.add_patch(rect)
        else: # the rectangle is outside of the image area -> stop
            break
            
    if show_result: 
        plt.show()
        plt.savefig('box_growth_method.svg')
           
    return pd.DataFrame(results, columns = phases), areas

def make_area_readable( value, unit, decimal = 0):
    if unit != 'px':
        u = es.unit()
        return u.make_area_readable(value, unit, decimal)
    else:
        return value, unit
          
class image_loader():
    image = {
        'raw' : None,
        'enh' : None,
        'nlm' : None,
        'seg' : None
    }
    filename = {
        'raw' : '{}',
        'enh' : '{}_enh',
        'nlm' : '{}_nlm',
        'seg' : '{}_seg'
    }
    scaling = es.getEmptyScaling()
    empty_scaling = scaling
    
    def enhance_image( self ):
        if self.image['enh'] is None:
            if self.verbose: print( "  Enhancing image contrast" )
            self.image['enh'] = exposure.adjust_log( self.image['raw'] )
            ## it the histogram enhacement is fast and the saved file is usually larger than the raw image - there is not really a point saving that intermediary image
            #if self.save_intermediates: 
            #    cv2.imwrite( self.path + self.filename['enh'] + '.tif', self.image['enh'], params=(cv2.IMWRITE_TIFF_COMPRESSION, 5) )
    
    def denoise_NLMCV2( self, h=15, templateWindowSize=7, searchWindowSize=23 ):
        if self.image['nlm'] is None:
            if self.verbose: print( "  Denoising image using Non Local Means" )
            self.image['nlm'] = np.zeros(self.image['enh'].shape, np.uint8) # empty image
            cv2.fastNlMeansDenoising( self.image['enh'], self.image['nlm'], float( h ), templateWindowSize, searchWindowSize )
            if self.save_intermediates: 
                cv2.imwrite( self.path + self.filename['nlm'] + '.tif', self.image['nlm'], params=(cv2.IMWRITE_TIFF_COMPRESSION, 5) )
        
    def threshold_segmentation( self, cmap = None ):
        """
        segments the image

        Parameters
        ----------
        thresh_values : dict
            dictionary containing threshold values at which a phase starts, eg: {  0: 'pores', 125: 'CSH', 167: 'CH', 198: 'alite' }
        show_result : boolean
            toggle to output the histogram and the segmentation result
        save : boolean
            toggle to save the resulting mask
        
        Returns
        -------
        np.typing.NDArray[np.uint8]
            The resulting mask image, where each phase is represented by an integer value from 0 to n (n=phase count -1)
        """
        if cmap is None: cmap = 'gist_rainbow'
        if self.image['seg'] is None:
            self.image['seg'] = np.zeros((self.image['nlm'].shape[0],self.image['nlm'].shape[1]), np.uint8)
            i = 0
            pixel_count = self.image['nlm'].shape[0]*self.image['nlm'].shape[1]
            for thresh_value, label in self.thresh_values.items():
                if self.verbose: print('  {}; range: {}-{}, value: {}'.format(label, thresh_value, self.t_keys[i+1], i))
                if thresh_value > 0:
                    _, phase_mask = cv2.threshold(self.image['nlm'], thresh_value, 1, cv2.THRESH_BINARY)
                    self.image['seg'] += phase_mask
                i+=1

        self.phase_composition = measure_phase_composition(self.image['seg'], self.phase_labels)

        if self.verbose:
            fig, ax = plt.subplots( 1, 3, figsize = ( 18, 5 ) )
            if not self.image['nlm'] is False and not self.image['nlm'] is None:
                i = 0
                ax[i].imshow( self.image['nlm'],	 cmap='grey' )
                
                i = 1

                for thresh_value in self.thresh_values.keys():
                    ax[i].axvline(thresh_value, 0, pixel_count, color='red')
                ax[i].hist(self.image['nlm'].ravel(),255,[0,255])
                ax[i].set_xlim([0,255])
                ax[i].set_yticks( [], [] )
                ax[i].set_xlabel( "grey value" )
                ax[i].set_ylabel( "frequency" )

            i = 2
            img = ax[i].imshow( self.image['seg'],	 cmap = cmap )

            colors = [ img.cmap(img.norm(i)) for i in range(len(self.phase_labels))]
            # create a patch (proxy artist) for every color 
            patches = [ patches.Patch(color=colors[i], label=self.phase_labels[i] ) for i in range(len(self.phase_labels)) ]
            # put those patched as legend-handles into the legend
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
            plt.show()

        if self.save_intermediates:
            cv2.imwrite( self.path + self.filename['seg'] + '.tif', self.image['seg'], params=(cv2.IMWRITE_TIFF_COMPRESSION, 5) )
    
    def load_image( self, force_reprocess = False ):
        image_loaded = False
        # populate filename dict
        for key, fn in self.filename.items():
            self.filename[key] = fn.format(self.file_name )
        #load images
        for key, fn in reversed(list(self.filename.items())):
            file_path = self.path + self.filename[key] + self.file_extension
            if image_loaded:
                self.image[key] = False
                print('avoid loading {}'.format(self.filename[key] + self.file_extension))
            elif os.path.exists( file_path ):
                print('loading {}'.format(self.filename[key] + self.file_extension))
                self.image[key] = cv2.imread( file_path, cv2.IMREAD_GRAYSCALE )
                self.scaling = es.autodetectScaling( self.filename['raw'] + self.file_extension, self.path )
                image_loaded = True
        
        return image_loaded
        
    
    def __init__(self, filepath, thresh_values, save_intermediates = True, force_reprocess = False, verbose = False, cmap = None ):
        """
        Initiate class image_loader

        Parameters
        ----------
        filepath : string
            full path to the raw image file
        scaling : dict
            dictionary containing the scaling
        force_reprocess : boolean
            force reprocess every step
        
        Returns
        -------
        image_loader object
        """
        self.thresh_values = thresh_values
        self.phase_labels = list( thresh_values.values() )
        self.t_keys = list( thresh_values.keys() )
        self.t_keys.append(255)
        self.save_intermediates = save_intermediates
        self.verbose = verbose
        
        self.path = os.path.dirname( filepath ) + os.path.sep
        print(filepath, self.path)
        self.file_name, self.file_extension = os.path.splitext( filepath )
        self.file_name = os.path.basename(self.file_name)
        if self.load_image( force_reprocess ):
            self.enhance_image()
            self.denoise_NLMCV2( h=18 )
            self.threshold_segmentation( cmap = cmap )
        
        self.res_x, self.res_y = self.image['seg'].shape
        area = (self.scaling['x'] * self.scaling['y'])*(self.res_x * self.res_y)
        area, unit = make_area_readable(area, self.scaling['unit'], 2)
        print("\nImage meta data")
        print("  image area: {:.3f} {}".format(area, unit))
        print("  resolution: {} x {} px".format(self.res_x, self.res_y))
        print("  scale:      {:.3f} {}/px".format(self.scaling['x'], self.scaling['unit']))
        
class chordLengthDistribution():
    pass

class phaseContent():
    # colors used in the example file
    #red   = np.array((255,118,198), dtype = "uint8")
    #green = np.array((130,255, 79), dtype = "uint8")
    #blue  = np.array((  0,  0,255), dtype = "uint8")

    # list of phases
    phase_colors = [0,1,2]
    phase_names  = ['resin/pores', 'C-S-H', 'C₃S']
    phase_count  = len(phase_names)

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
    experiments = 75
    
    stDev_mean = {}
    stDev_stDev = {}
    
    # dataframes
    phase_content_DF = None

    scaling = es.getEmptyScaling()

    def get_tile_area(self, readable=True, in_unit='auto', decimal=0):
        area = self.tile_width * self.tile_height * self.scaling['x']**2
        unit = self.scaling['unit']
        if in_unit == 'auto':
            if readable: area, unit = make_area_readable(area, unit, decimal)
        else:
            u = es.unit()
            area = u.get_area_in_unit( area, self.scaling['unit'], in_unit )
            unit = in_unit
        return area, unit

    def get_dataset_area(self, decimal = 0):
        area, unit = self.get_tile_area( False )
        area = area * self.image_count
        area, unit = make_area_readable(area, unit, decimal)
        return area, unit

    def init_phase_list(self, phase_names ):
        if phase_names is not None:
            self.phase_names  = phase_names
            self.phase_count  = len(phase_names)
            self.phase_colors = range(self.phase_count)

    def init_column_list(self):
        for i in range(len(self.phase_names)):
            self.column_list.append( 'phase_{:02d}'.format(i) )
            self.column_list.append( 'phase_{:02d}_percent'.format(i) )

    def show_masked_phases(self, img, mask_list):
        plot_count = len(self.phase_list) + 1
        plot_column_count = math.ceil(plot_count/2)
        plot_pos = 1
        plt.subplot(2,plot_column_count,plot_pos),plt.imshow(img)
        for mask_index in range(self.phase_count):
            plot_pos += 1

            plt.subplot(2,plot_column_count,plot_pos),plt.imshow(mask_list[mask_index])

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

    def read_dataset( self, img, file ):
        height, width = img.shape[:2]
        px_cnt = height*width
        if self.check_tile_dimension(height, width):
            new_row = {'filename':file, 'height':height, 'width':width }
            
            phase_comp = measure_phase_composition( img )
            counts = phase_comp * px_cnt
            percentages = phase_comp * 100
            
            for color, percentage in enumerate(percentages):
                label = 'phase_{:02d}'.format(color)

                new_row[label] = counts[color]
                new_row[label + '_percent'] = percentage

            self.phase_content_DF = pd.concat([self.phase_content_DF, pd.DataFrame([new_row])], ignore_index=True)

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
                        
                    img = cv2.imread( folder + os.sep + file, cv2.IMREAD_GRAYSCALE )
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

    def box_count_method( self, experiments=None, verbose=True ):
        if experiments == None: experiments = self.experiments
        # main process 
        if ( self.image_count > 1 ):
            # calculate stabw
            keyName = 'phase_{:02d}_percent'
            sampleFraction = 1#0.5# 10/self.image_count
            self.meanAfterNExperiments = {}
            self.stDevDFList = {}
            self.stDevcols = list(range(self.image_shuffle_count-2))
            self.phase_mean = {}

            if verbose: print( 'processing {} experiments'.format(experiments))

            # randomly select images (different image amount from 0 to image_shuffle_count)
            # and process the mean phase area for each phase
            
            # for every phase
            for i in range(self.phase_count):
                key = keyName.format(i)
                # create headers for each phase
                self.meanAfterNExperiments[key] = []
                self.stDevDFList[key] = pd.DataFrame(columns = self.stDevcols)
                self.stDev_mean[key] = []
                self.stDev_stDev[key] = []

                if verbose: print( 'mean & std deviation of {}'.format(self.phase_names[i]) )
                self.meanAfterNExperiments[key] = {}
                # for a sample size from 0 to self.image_shuffle_count (100) images
                for c in range(1, self.image_shuffle_count+1):
                    
                    #if c > 0 and c % 10 == 0: 
                    #    print('{:02d} / {:02d}'.format( c-1, self.image_shuffle_count ))

                    self.meanAfterNExperiments[key][c] = []
                    sampleFraction = c/self.image_count
                    # repeat the experiments 'experiments' times
                    for j in range( experiments ):
                        sampleDF = self.phase_content_DF.sample(frac=sampleFraction)
                        self.meanAfterNExperiments[key][c].append( sampleDF[key].mean(skipna = True) )
                    
                    #print( self.meanAfterNExperiments[key][c] )
                    self.stDev_mean[key].append( statistics.mean( self.meanAfterNExperiments[key][c] ) )
                    self.stDev_stDev[key].append( statistics.stdev( self.meanAfterNExperiments[key][c] ) )
            
            if verbose: print( '-'*20 )
        else:
            print( "not enough images found!" )

    def area_growth_method( self, segmented_image, seed_window_f=5, step_w=5, experiments=None, verbose=True, cmap=None ):
        if experiments == None: experiments = self.experiments
        self.agm_result_list=[]
        min_area_rows = -1
        areas = []
        for i in range(experiments):
            result_df, areas = growImageArea( segmented_image, self.phase_names, seed_window_f = seed_window_f, step_w=step_w, show_result = (verbose and (i==0)), cmap = cmap )
            self.agm_result_list.append( result_df )
            area_rows = len(self.agm_result_list[-1])
            if min_area_rows < 0 or min_area_rows > area_rows: min_area_rows = area_rows
            if verbose and (i==1 or i%10==0): print("{}% done...".format(i))


        # reduce the data size to the lowest common row count
        result_list_cropped = []
        for df in self.agm_result_list:
            result_list_cropped.append( df.head(min_area_rows) )

        # convert to xarray data array for stats
        da = xr.DataArray(result_list_cropped, dims=["experiment", "area_size", "phase"])
        return da, areas


    def __init__(self, folder, phase_names, scaling=None, verbose=False ):
        """
        Initiate class phaseContent

        Parameters
        ----------
        folder : string
            directory in which the images can be found.
        phase_names : list
            eg: [ 'pores', 'CSH', 'CH', 'alite' ]
        scaling : dict
            dictionary containing the scaling
        verbose : boolean
            endable some more debugging message
        
        Returns
        -------
        phaseContent object
        """
        self.folder = folder
        self.target_folder = os.path.abspath(os.path.join(self.folder, os.pardir))
        self.init_phase_list( phase_names )
        self.init_column_list()
        if not scaling == None:
            self.scaling = scaling
            
        self.phase_content_DF = pd.DataFrame(columns = self.column_list)

        self.load_files(folder,verbose=verbose)

        # make sure the image_shuffle_count does not exceed the image count!
        if self.image_shuffle_count > self.image_count:
            self.image_shuffle_count = self.image_count

        self.box_count_method(verbose=verbose)

### actual program start
if __name__ == '__main__':
    coreCount = multiprocessing.cpu_count()
    mc_settings = {
        "coreCount" : multiprocessing.cpu_count(),
        "processCount" : (coreCount - 1) if coreCount > 1 else 1
    }

    programInfo()
    settings = processArguments()
    if settings["showDebuggingOutput"]: 
        print( 'Found {} CPU cores. Using max. {} processes at once.'.format(mc_settings["coreCount"], mc_settings["processCount"]) )
        print( "I am living in '{}'".format(settings["home_dir"]) )

    # if settings["slice_image"]:
    #     phaseContent = singeFileProcess(settings, verbose=settings["showDebuggingOutput"])
    # else:
    #     phaseContent = folderProcess(verbose=settings["showDebuggingOutput"])
    
    print( "Script DONE!" )