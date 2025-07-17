# This demo verifys the data legitimacy

import os
import numpy as np
import cv2
import gzip
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from PIL import Image
# from .utils import draw_mask, draw_keypoints, draw_bbox, mask_to_bbox

for subset in ['inside', 'outside']:
    path = f'datasets/progait/videos/{subset}/'
    path_ann = f'/home/eric/projects/ProGait/datasets/progait/annotations/{subset}/'
    
    f_list = sorted(os.listdir(path))

    pbar = tqdm(f_list)
    for filename in pbar:
        if not filename.endswith('mp4'):
            continue
        pbar.set_description(f'Verifying {filename}')
        trial_id = os.path.splitext(filename)[0]
        
        with gzip.open(f'{path_ann}{trial_id}_masks.npy.gz', 'rb') as f:
            mask = np.load(f)
        with gzip.open(f'{path_ann}{trial_id}_keypoints.npy.gz', 'rb') as f:
            keypoint = np.load(f)
        
        cap = cv2.VideoCapture(path + filename)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frame_count != mask.shape[0]:
            raise ValueError(f'Frame count mismatch for {filename}: {frame_count} vs {mask.shape[0]}')
        if frame_count != keypoint.shape[0]:
            raise ValueError(f'Frame count mismatch for {filename}: {frame_count} vs {keypoint.shape[0]}')
        
        cap.release()

print('Complete')
