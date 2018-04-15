import os
import dicom
import numpy as np
import scipy
import matplotlib.pyplot as plt
from glob import glob
import re
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utils.preprocess import ROI

MIN_BOUND = -1000.0
MAX_BOUND = 300.0
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return zero_center(image)

PIXEL_MEAN = 0.25

def zero_center(image):
    image = image - PIXEL_MEAN
    return image

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    
    return image, new_spacing

def load_patient(baseDir, hu=True):
    files = glob(os.path.join(baseDir, '*', '*'))
    mask_keyword = re.compile('ROI_MASK')
    isMask = lambda fileName: len(mask_keyword.findall(fileName)) != 0
    slicefs = list(filter(lambda f: not isMask(f), files))
    maskfs = list(filter(isMask, files))
    
    slices = [dicom.read_file(f) for f in slicefs]
    masks = [dicom.read_file(f) for f in maskfs]
    
    slices.sort(key = lambda x: int(x.InstanceNumber))
    masks.sort(key = lambda x: int(x.InstanceNumber))
    
    _masks = []
    for s in slices:
        _m = None
        for m in masks:
            if s.InstanceNumber == m.InstanceNumber:
                _m = m.pixel_array
                break
        if _m == None:
            _m = np.zeros(s.pixel_array.shape)
        _masks.append(_m)
    
    masks = _masks
    
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    
    return {
        'slice_thickness': slice_thickness,
        'image' : get_pixels_hu(slices) if hu else np.array([s.pixel_array for s in slices]),
        'mask' : np.array(masks)
    }

def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    p = p[:,:,::-1]
    
    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()
    
    
class Patient:
    def __init__(self, patient):
        self.patient = patient
        self.benign = patient['benign']
        self.image = patient['image']
        self.num_slice = self.image.shape[0]
        self.mask = patient['mask']
        self.background = patient['roi']
        ### Do some infitialization
        ### like extract lesion coords and save
        ###
        
        ### get nodules(either benign or malign) and save

        self.lesions = []
        self.lesionArea = 0
        for _idx, mask_layer in enumerate(self.mask[1:-1]):
            idx = _idx + 1 ## first layer is stripped
            regions = measure.regionprops(measure.label(self.mask[idx]))
            if len(regions) == 0 or regions[0].area == 54:
                pass
            else:
                region = regions[0]
                bbox = region.bbox
                centroid = region.centroid
                ## save lesion's centroid
                x, y = int(centroid[0]), int(centroid[1])
                self.lesions.append((idx, x, y, region.area))
                self.lesionArea += region.area
                ## remove region around lesion from roi(==background)
                self.background[idx-1:idx+2][:, bbox[0]-48:bbox[2]+48, bbox[1]-48:bbox[3]+48] = 0
                ## TODO
        
        self.background[0,:,:] = 0
        self.background[-1,:,:] = 0
        
        self.regions = measure.regionprops(measure.label(self.background))
        self.totalRoiArea = np.sum(list(map(lambda r: r.area, self.regions)))                
        ### prepare non_nodule region(but how? efficiently?)
        ### np.random.choice((roi - lesionArea).coords) ???
        
    ### recommended shift = 32
    def getRandomLesion(self, shift=0, hflip=False, vflip=False):
        idx, x, y, area = self.lesions[np.random.randint(len(self.lesions))]
        #print(idx, x, y)
        xdev, ydev = (0,0) if shift == 0 else np.random.randint(shift * 2, size=2) - shift
        return self.image[idx][x+xdev-48:x+xdev+48, y+ydev-48:y+ydev+48]

    def getRandomLesion3D(self, shift=0, hflip=False, vflip=False):
        while True:
            rand = np.random.randint(self.lesionArea)
            idx, x, y = 0, None, None
            for lesion in self.lesions:
                if rand < lesion[3]:
                    idx, x, y, area = lesion
                else:
                    rand -= lesion[3]

            if x < 48 + shift or x > 512 - 48 - shift or y < 48 + shift or y > 512 - 48 - shift:
                continue

            #print(idx, x, y)
            xdev, ydev = (0,0) if shift == 0 else np.random.randint(shift * 2, size=2) - shift
            ret = self.image[idx-1:idx+2][:,x+xdev-48:x+xdev+48, y+ydev-48:y+ydev+48]
            if hflip and np.random.randint(1):
                ret = ret[::-1,:]
            if vflip and np.random.randint(1):
                ret = ret[:,::-1]
            return normalize(ret.reshape((3,96,96,1)))
    
    def isBenign(self):
        return self.benign
    
    ## DO USE 'tag' of patient object to rip off unneccessary region
    ## Exclude EXTERNAL_AIR area, inclu
    def getRandomBackground3D(self):
        #roi = self.roi[np.random.randint(self.num_slice - 2) + 1,:,:]
        rand = np.random.randint(self.totalRoiArea)
        for region in self.regions:
            if rand < region.area:
                coords = region.coords
                idx, x, y = coords[np.random.randint(coords.shape[0])]
                if x < 48:
                    x = 48
                if x > 512 - 48:
                    x = 512 - 48
                if y < 48:
                    y = 48
                if y > 512 - 48:
                    y = 512 - 48
                ret = self.image[idx-1:idx+2][:,x-48:x+48, y-48:y+48]
                if ret.shape != (3, 96, 96):
                    print(idx, x, y)
                return normalize(ret.reshape((3,96,96,1)))
            else:
                rand -= region.area
                pass

                
    
    ## To stress on specific regions
    def memorizeCoord(self, coord):
        pass

get_fname = lambda p: p['filename']
load_function = lambda fname: {'filename': fname, 'data': Patient(np.load(fname).all())}
    
def data_generator(files, nodule_ratio = 0.5, get_size=1, get_num=16, hand_size = 16, turnover = 1):
    pool = files[:]
    hand = []
    while True:
        np.random.shuffle(hand)
        expire = hand[:turnover]
        expired = list(map(get_fname, expire))
        hand = hand[turnover:]
        for elem in expire:
            del elem
        del expire
        
        num_fill = hand_size - len(hand)
        np.random.shuffle(pool)
        fill = pool[:num_fill]
        
        for p in list(map(load_function, fill)):
            hand.append(p)
            
        ## return expired files to pool
        pool = pool + expired
        
        for i in range(get_num):
            X = []
            y = []
            for p in np.array(hand)[np.random.choice(range(len(hand)), get_size)]:
                if np.random.rand() < nodule_ratio:
                    X.append(p['data'].getRandomLesion3D(shift=10, hflip=True, vflip=True))
                    if p['data'].isBenign():
                        y.append([0,1,0,1])
                    else:
                        y.append([0,0,1,1])
                else:
                    X.append(p['data'].getRandomBackground3D())
                    y.append([1,0,0,0])
            yield np.stack(X), np.vstack(y)
        

def symmetric_data_generator(xList, yList, batch_num, sym=False, generate_all=False):
    num_category = len(xList)
    while True:
        w = np.maximum((np.random.randn(num_category) + 1.5), 0) + 0.001
        nums = np.floor(w / w.sum() * batch_num)
        nums[np.random.randint(num_category)] += (batch_num - nums.sum())
        X = []
        y = []
        for i in range(num_category):
            X.append(xList[i][np.random.randint(0,xList[i].shape[0], int(nums[i]))])
            y.append(np.tile(yList[i], (int(nums[i]), 1)))
        X = np.vstack(X)
        y = np.vstack(y)
        yield (X, y)
    