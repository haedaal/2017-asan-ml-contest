import numpy as np

#from glob import glob
#import os

import matplotlib
import matplotlib.pyplot as plt

import math
from sys import stdout

from skimage import measure, morphology
from scipy import ndimage

## Tag pixels
NUM_TAGS = 8
# EXTERNAL_AIR = OMITTED
BODY = 0
ROI = 1
# CAVITY = 1 # BODY.filled_space - BODY
SOLITARY_VOLUME = 2 # CAVITY > threshold1
INTERNAL_AIR = 3 # CAVITY < threshold2
INCLUDED_VOLUME = 4
NODULE = 5 # either benign or malign
BENIGN = 6
MALIGN = 7

def getRestoreRegionFunction(originalSize):
    def rs(region):
        image = region.image
        bbox = region.bbox
        
        width = bbox[3] - bbox[1]
        depth = bbox[2] - bbox[0]

        right = np.zeros((depth, bbox[1]))
        left = np.zeros((depth, originalSize[1] - bbox[3]))
        anterior = np.zeros((bbox[0], originalSize[1]))
        posterior = np.zeros((originalSize[0] - bbox[2], originalSize[1]))

        fitWidth = np.hstack([right, image, left])
        fit = np.vstack([anterior, fitWidth, posterior])
        return fit
    return rs

def getRestoreSizeFunction(bbox, originalSize):
    def rs(image):
        width = bbox[3] - bbox[1]
        depth = bbox[2] - bbox[0]

        right = np.zeros((depth, bbox[1]))
        left = np.zeros((depth, originalSize[1] - bbox[3]))
        anterior = np.zeros((bbox[0], originalSize[1]))
        posterior = np.zeros((originalSize[0] - bbox[2], originalSize[1]))

        fitWidth = np.hstack([right, image, left])
        fit = np.vstack([anterior, fitWidth, posterior])
        return fit
    return rs

## It returns mask of 0 or 1 data type
## Currently not able to exclude intestinal air correctly
## But it's not difficult to exclude by 'labeling and blackhat separately' -- (1)
def segmentation_2D(image, mask, isBenign):
    
    tag = np.zeros((512, 512, NUM_TAGS), dtype=np.bool)
    
    labeled = measure.label((image > -500))
    region = measure.regionprops(labeled)
    
    region.sort(key = lambda r : -r.area)
    body = region[0]
    bbox = body.bbox
    
    ### Interal air and noise, lung candidate
    lung = body.filled_image ^ body.image
    ### Decaying noise, became useless cause Blackhat does all?
    decayed = ndimage.binary_dilation(ndimage.binary_erosion(lung, iterations=2), iterations=2)

    ## (1) Currently take all parts into one area and draw outline and perform blackhat
    ## But if you want to prevent 'merging intestinal air and lung air while blackhat breaks thin wall'
    ## Should perform measure.label once again, and perform blackhat separately
    
    outline = ndimage.morphological_gradient(decayed, size=(3,3))
    ### Currently redundant, cuase outline is already array of Boolean
    #outline = outline.astype(bool)

    #Performing Black-Tophat Morphology for reinclusion
    #Creation of the disk-kernel and increasing its size a bit
    blackhat_struct = [[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]]
    blackhat_struct = ndimage.iterate_structure(blackhat_struct, 8)

    #Perform the Black-Hat
    #blackhat = ndimage.black_tophat(outline, structure=blackhat_struct)
    ## shortcut!
    blackhat = ndimage.binary_erosion(ndimage.binary_dilation(decayed, structure=blackhat_struct), structure=blackhat_struct)
    inclusion = outline + blackhat
    
    #plt.imshow((decayed * 1) + ((blackhat & (decayed == 0)) * 2), plt.cm.bone)
    #plt.show()

    roi = decayed | inclusion
    
    ### Fit ROI to 512 * 512
    orig_size = (512, 512)
    restore = getRestoreSizeFunction(bbox, orig_size)
    
    roi = restore(roi).astype(bool)
    
    tag[:,:,BODY] = restore(body.image > 0)
    tag[:,:,ROI] = roi > 0
    tag[:,:,SOLITARY_VOLUME] = restore(decayed).astype(bool) & (image > -500)
    tag[:,:,INTERNAL_AIR] = restore(decayed).astype(bool) & (image < -500)
    tag[:,:,INCLUDED_VOLUME] = restore(blackhat ^ decayed)
    tag[:,:,NODULE] = mask > 0
    tag[:,:,BENIGN] = (mask > 0) & isBenign
    tag[:,:,MALIGN] = (mask > 0) & (not isBenign)

    return tag

### Take ndarray as input
def segmentation_3D(patient, isBenign):
    image = patient['image']
    mask = patient['mask']
    num_slice = image.shape[0]
    segment_tag = np.array([segmentation_2D(image[i], mask[i], isBenign) for i in range(num_slice)])
    return segment_tag