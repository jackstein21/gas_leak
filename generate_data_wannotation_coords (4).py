import os
import argparse
from joblib import Parallel, delayed

import random
import numpy as np
import cv2 as cv
import albumentations as A
from pascal_voc_writer import Writer
import pandas as pd

def get_backgrounds(path):
    backgrounds = sorted(os.listdir(path))
    backgrounds = [os.path.join(path, f) for f in backgrounds]
    return backgrounds

def get_smokes(path):
    smokes = sorted(os.listdir(path))
    smokes = [os.path.join(path, f) for f in smokes]
    return smokes

def get_coordinates(path):
  df = pd.read_csv(path)
  return df

def sample_background(backgrounds):
    background = random.sample(backgrounds, 1)[0]
    background = cv.imread(background)
    return background

def sample_smoke(smokes, coordinate_df):
    smoke = random.sample(smokes, 1)[0]
    smoke_file = os.path.basename(smoke)
    emitter_coordinate = coordinate_df.loc[coordinate_df['Filename'] == smoke_file, 'Emitter Coordinates'].values[0]
    emitter_coordinate = tuple(map(int, emitter_coordinate.strip('()').split(', ')))
    smoke = cv.imread(smoke, cv.IMREAD_UNCHANGED)

    return smoke, emitter_coordinate

def get_random_parameters(H, W):
    #scale_factor = random.uniform(0.5, 1)
    scale_factor = 1
    size = int(min([W, H]) * scale_factor)
    x = random.randint(0, H-size)
    y = random.randint(0, W-size)
    beta = random.uniform(0.95, 1)
    return size, x, y, beta

def resize_coordinates(emitter_coordinate, smoke, W, H):
  original_shape = smoke.shape
  
  scale_x = W / original_shape[0]
  scale_y = H / original_shape[1]
  
  original_x = emitter_coordinate[0]
  original_y = emitter_coordinate[1]

  transformed_x = int(original_x * scale_x)
  transformed_y = int(original_y * scale_y)

  emitter_coordinate = (transformed_x, transformed_y)

  return emitter_coordinate 

def normalize(array, radious=50):
    array_mean = int(np.mean(array))
    upper_lim = min(255, array_mean + radious)
    lower_lim = max(0, array_mean - radious)
    array_max = np.amax(array)
    array_min = np.amin(array)
    array = lower_lim \
        + (upper_lim-lower_lim)/(array_max-array_min) * (array-array_min)
    return array
 
def get_background_augmentator():
    augmentator = A.Emboss(alpha=(0.2, 0.3))
    return augmentator

def generate_frame(background, smoke, size, x, y, beta):
    alpha = cv.cvtColor(smoke[:,:,3], cv.COLOR_GRAY2RGB) / 255.
    smoke_rgb = cv.cvtColor(smoke[:,:,0:3], cv.COLOR_BGR2GRAY)
    #smoke_rgb = normalize(smoke_rgb)
    smoke_rgb = cv.cvtColor(smoke_rgb.astype('uint8'), cv.COLOR_GRAY2RGB)
    frame = background.copy()
    frame = (1 - beta*alpha) * frame + (beta*alpha) * smoke_rgb
    return frame.astype('uint8')

def generate_mask(background, smoke, size, H, W, x, y, beta, threshold=50):
    alpha = cv.cvtColor(smoke[:,:,3], cv.COLOR_GRAY2RGB) / 255.
    smoke_rgb = cv.cvtColor(smoke[:,:,0:3], cv.COLOR_BGR2GRAY)
    #smoke_rgb = normalize(smoke_rgb)
    smoke_rgb = cv.cvtColor(smoke_rgb.astype('uint8'), cv.COLOR_GRAY2RGB)
    mask = np.zeros((H, W))
    mask = beta * smoke[:,:,3]
    mask = mask > threshold
    #delta = np.abs((beta*alpha) * smoke_rgb \
    #    - (1 - beta*alpha) * background[x:x+size, y:y+size, :])
    #delta = cv.cvtColor(delta.astype('uint8'), cv.COLOR_BGR2GRAY)
    #mask[x:x+size, y:y+size] = mask[x:x+size, y:y+size] * (delta >= 0*threshold)
    mask = (mask * 255).astype('uint8')
    #kernel = np.ones((3,3), np.uint8)
    #mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    return mask

def draw_square(img,center):
  start_point = [x-40 for x in center]
  end_point = [x+40 for x in center]
  image = cv.rectangle(img, start_point, end_point,(255,0,0))
  return image

# def find_bounding_box(image):
#     """
#     Finds the bounding box around the mask.
#     Args:
#         image: The image that contains the mask.
#     Returns:
#         The bounding box around the mask.
#     """
#     # Find the left and right most white pixels.
#     left = image.shape[1]
#     right = 0
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             if (image[i][j] >= 245).any():
#                 pixel_is_white = True
#             else:
#                 pixel_is_white = False

#             if pixel_is_white and j < left:
#                 left = j
#             elif pixel_is_white and j > right:
#                 right = j

#     # Find the top and bottom most white pixels.
#     top = image.shape[0]
#     bottom = 0
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             if (image[i][j] >= 245).any():
#                 pixel_is_white = True
#             else:
#                 pixel_is_white = False

#             if pixel_is_white and i < top:
#                 top = i
#             elif pixel_is_white and i > bottom:
#                 bottom = i

#     # Find the center of the bounding box.
#     # center_x = (left + right) // 2
#     # center_y = (top + bottom) // 2

#     return (left,right,top,bottom)


def generate_annotation(image_frame, image_path, emitter_coordinate):
  # draw box
  annotated_image = draw_square(image_frame, emitter_coordinate)

  # write xml
  writer = Writer(image_path, annotated_image.shape[0], annotated_image.shape[1])
  start_point = [x-40 for x in emitter_coordinate]
  end_point = [x+40 for x in emitter_coordinate]
  writer.addObject('source', start_point[0], start_point[1], end_point[0], end_point[1])

  # save xml to same folder as the frame
  writer.save(image_path[:-4]+'.xml') 
  
  # return image
  return annotated_image

def make_dirs(output_dir):
    output_frames = os.path.join(output_dir, 'frames')
    try:
        os.makedirs(output_frames)
    except:
        pass
    output_masks = os.path.join(output_dir, 'masks')
    try:
        os.makedirs(output_masks)
    except:
        pass
    output_annotated = os.path.join(output_dir, 'annotated')
    try:
        os.makedirs(output_annotated)
    except:
        pass
    
    return output_frames, output_masks, output_annotated

def get_evaluation_data(
    output_frames, output_masks, output_frames_eval, output_masks_eval, output_annotated_eval):
    frames = [f for f in sorted(os.listdir(output_frames)) if 'frame' in f]
    backgrounds = [f for f in sorted(os.listdir(output_frames)) if 'background' in f]
    masks = sorted(os.listdir(output_masks))
    n_eval = int(len(frames) / 3)
    random.seed(123)
    frames = random.sample(frames, n_eval)
    random.seed(123)
    backgrounds = random.sample(backgrounds, n_eval)
    random.seed(123)
    masks = random.sample(masks, n_eval)
    for frame, background, mask in zip(frames, backgrounds, masks):
        src_frame = os.path.join(output_frames, frame)
        dst_frame = os.path.join(output_frames_eval, frame)
        os.replace(src_frame, dst_frame)
        src_background = os.path.join(output_frames, background)
        dst_backfround = os.path.join(output_frames_eval, background)
        os.replace(src_background, dst_backfround)
        src_mask = os.path.join(output_masks, mask)
        dst_mask = os.path.join(output_masks_eval, mask)
        os.replace(src_mask, dst_mask)

def pipeline(backgrounds, smokes, coordinate_df, i, output_frames, output_masks, output_annotated, zero_trail):
    background = sample_background(backgrounds)
    background = cv.cvtColor(background, cv.COLOR_BGR2GRAY)
    background = cv.cvtColor(background, cv.COLOR_GRAY2BGR)
    smoke, emitter_coordinate = sample_smoke(smokes, coordinate_df)
    H, W = background.shape[:2]
    size, x, y, beta = get_random_parameters(H, W)
    #print(emitter_coordinate)
    emitter_coordinate = resize_coordinates(emitter_coordinate, smoke, W, H)
    #print(emitter_coordinate,"/n")
    #print(smoke.shape)
    smoke = cv.resize(smoke, (W, H))
    #print(smoke.shape)
    frame = generate_frame(background, smoke, size, x, y, beta)

    augmentator = get_background_augmentator()
    background = augmentator(image=background)['image']

    path_frame = os.path.join(output_frames, 'frame{0:0>{zeros}}.jpg'.format(i, zeros=max(zero_trail, 3)))
    cv.imwrite(path_frame, frame)  # Save frame file
    path_background = os.path.join(output_frames, 'background{0:0>{zeros}}.jpg'.format(i, zeros=max(zero_trail, 3)))
    cv.imwrite(path_background, background)  # Save background file

    mask = generate_mask(
        background, smoke, size, H, W, x, y, beta, threshold=50)

    path_mask = os.path.join(output_masks, 'mask_{0:0>{zeros}}.jpg'.format(i, zeros=max(zero_trail, 3)))
    cv.imwrite(path_mask, mask) # Save mask file

    #generate annoataion
    emitter_coordinate = (emitter_coordinate[0]-x,emitter_coordinate[1]-y)
    annotated_img = generate_annotation(frame, path_frame, emitter_coordinate)
    path_annotation = os.path.join(output_annotated, 'annotated_{0:0>{zeros}}.jpg'.format(i, zeros=max(zero_trail, 3)))
    cv.imwrite(path_annotation, annotated_img) # Save annotated image
    


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-b", "--backgrounds", required=True,
        help="path to the directory containing background images")
    ap.add_argument("-s", "--smokes", required=True,
        help="path to the directory containing smoke (RGBA) images")
    ap.add_argument("-o", "--output", required=True,
        help="output directory")
    ap.add_argument("-e", "--eval", default=None, required=False,
        help="output directory for evaluation data")
    ap.add_argument("-n", "--number", required=True,
        help="number of images to generate")
    ap.add_argument("-c", "--coordinates", required=True,
        help="path to the coordinates file")
    args = vars(ap.parse_args())  # By default, it takes arguments from sys.argv

    # Gathers all the data
    output_frames, output_masks, output_annotated = make_dirs(args['output'])
    backgrounds = get_backgrounds(args['backgrounds'])
    smokes = get_smokes(args['smokes'])
    coordinate_df =  get_coordinates(args['coordinates'])

    # Determines how many images to create
    if args['eval'] != None:
        output_frames_eval, output_masks_eval, output_annotated_eval = make_dirs(args['eval'])
        n_iterations = int(int(args['number']) * 3/2)
    else:
        n_iterations = int(args['number'])

    random.seed(42)

    # Pipeline runs for every image
    print('Generating {} frames/masks...'.format(n_iterations))
    # print(coordinate_df)
    for i in range(n_iterations):
        pipeline(backgrounds, smokes, coordinate_df, i, output_frames, output_masks, output_annotated, zero_trail=len(str(n_iterations)))

    if args['eval'] != None:
        get_evaluation_data(
            output_frames, output_masks,
            output_frames_eval, output_masks_eval, output_annotated_eval)
