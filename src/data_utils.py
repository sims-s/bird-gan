import numpy as np
from PIL import Image
import imageio
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def preview_img(img, x_bb=None, y_bb=None, width_bb=None, height_bb=None, title=None):
    ax = plt.gca()
    if title is not None:
        plt.title(title)
    plt.imshow(img)
    if x_bb:
        rect = patches.Rectangle((x_bb, y_bb),width_bb, height_bb, linewidth=2, edgecolor='r',facecolor='none')
        ax.add_patch(rect)

    plt.show()


def downsize(img, target_width, target_height):
    curr_height = img.shape[1]
    curr_width = img.shape[0]
    
    if curr_height > curr_width:
        new_height = target_height
        new_width  = int(new_height * curr_width / curr_height)
    else:
        new_width = target_width
        new_height = int(new_width * curr_height / curr_width)
        
    img = Image.fromarray(img)
    img = img.resize((new_height, new_width), Image.ANTIALIAS)
    img = np.array(img)
    
    first_axis_pad = target_width - img.shape[0]
    second_axis_pad = target_height - img.shape[1]
    
    if first_axis_pad % 2:
        top_pad = int(first_axis_pad / 2) + 1
        bottom_pad = int(first_axis_pad / 2)
    else:
        top_pad = int(first_axis_pad / 2)
        bottom_pad = int(first_axis_pad / 2)
    
    
    if second_axis_pad % 2:
        l_pad = int(second_axis_pad / 2) + 1
        r_pad = int(second_axis_pad / 2)
    else:
        l_pad = int(second_axis_pad / 2)
        r_pad = int(second_axis_pad / 2)
    
    padding = [(top_pad, bottom_pad), (l_pad, r_pad), (0,0)]
    img = np.pad(img, padding, mode='constant', constant_values=0)

    return img

# Make the images as square as possible
def process_boundingbox(x_bb, y_bb, width, height):
    new_height = max(width, height)
    new_width = max(width, height)

    width_delta = new_width - width
    height_delta = new_height - height
    
    new_x_bb = int(max(x_bb - width_delta/2,1))
    new_y_bb = int(max(y_bb - height_delta/2,1))
    
    return int(new_x_bb), int(new_y_bb), int(new_width), int(new_height)
    

def preprocess_img(img_row, data_path, target_width, target_height, preview=True):
    img = imageio.imread(data_path + img_row['fname'])
    # Smallest index should be 1 not zero to make the boubnding box appear on the image
    img_row['x_bb'] = max(img_row['x_bb'], 1)
    img_row['y_bb'] = max(img_row['y_bb'], 1)
    
    if preview:
        print('Filename: ', img_row['fname'])
        print('Original size: ', img.shape)
        print('Image & Bounding box')
        preview_img(img, img_row['x_bb'], img_row['y_bb'], img_row['width_bb'], img_row['height_bb'], None)
    # First, widen the bounding box
    new_x_bb, new_y_bb, new_width, new_height = process_boundingbox(img_row['x_bb'], img_row['y_bb'], img_row['width_bb'], img_row['height_bb'])
    if preview:
        print('Squarified bounding box')
        preview_img(img, new_x_bb, new_y_bb, new_width, new_height, None)

    img = img[new_y_bb:new_y_bb + new_height,new_x_bb:new_x_bb+new_width]
    if preview:
        print('Cropped: ', img.shape)
        preview_img(img, None, None, None, None, None)
        
    img = downsize(img, target_width, target_height)
    if preview:
        print('Downsize to %dx%d'%(img.shape[0], img.shape[1]))
        preview_img(img, None, None, None, None, None)
        
    return img


def save_img(row, save_path, data_path, target_width, target_height, preview=False):
    if os.path.exists(save_path + 'images/%s.jpg'%row['id']): 
        return
    try:
        img = preprocess_img(row, data_path, target_width, target_height, preview=False)
    except ValueError:
        print('skip: ', row['id'])
        return 'FAILED'
    img = Image.fromarray(img[:,:,:3])
    img.save(save_path + '%s.jpg'%row['id'])
    return 