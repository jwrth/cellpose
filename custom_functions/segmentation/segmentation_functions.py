import numpy as np
import cv2


def set_histogram(data, lower=0, upper=None, bit_type=np.uint8, clip=True):
    if lower is None:
        lower = np.min(data)
        
    if upper is None:
        upper = np.max(data)
        
    if bit_type is np.uint8:
        max_int = 255
    elif bit_type is np.uint16:
        max_int = 65536
    else:
        print("Unknown bit type.")
        return
    
    norm = ((data - lower) / (upper - lower))
        
    if clip:
        norm = np.clip(norm, a_min=0,  a_max=1)
    
    norm *= max_int
            
    return norm.astype(bit_type)

def multi_grayscale_to_rgb(r=None, g=None, b=None, bit_type="8bit", lowers = [None]*3, uppers=[None]*3):
    '''
    Function to transform multiple grayscale images into a rgb image.
    '''
    if bit_type is "8bit":
        bit_type = np.uint8
    elif bit_type is "16bit":
        bit_type = np.uint16
    else:
        print("Unknown bit type.")
        return
    
    if r is not None:
        shape = r.shape
    elif g is not None:
        shape = g.shape
    elif b is not None:
        shape = b.shape
    else:
        print("All channels empty.")
        return
        
    if r is None:
        r = np.zeros(shape, dtype=bit_type)
    else:
        r = set_histogram(r, bit_type=bit_type, lower=lowers[0], upper=uppers[0])
        
    if g is None:
        g = np.zeros(shape, dtype=bit_type)
    else:
        g = set_histogram(g, bit_type=bit_type, lower=lowers[1], upper=uppers[1])
        
    if b is None:
        b = np.zeros(shape, dtype=bit_type)
    else:
        b = set_histogram(b, bit_type=bit_type, lower=lowers[2], upper=uppers[2])
    
    
    rgb = cv2.merge((r, g, b))
    
    return rgb