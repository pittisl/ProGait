import torch
import numpy as np
import cv2
from copy import copy
import xml.etree.ElementTree as ET
from datetime import datetime
from tqdm.auto import tqdm
from torchvision.ops import masks_to_boxes
from pycocotools import mask as mask_utils
from pytorchvideo.data import LabeledVideoDataset

import matplotlib.pyplot as plt
from matplotlib import animation

from scipy.interpolate import make_smoothing_spline

def video_to_numpy(video_path, stride=1):
    out = []
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        success, frame = cap.read() # Extract frame
        if not success:
            break
        if idx % stride == 0:
            frame = frame[...,::-1] # Convert from BGR to RGB
            out += [frame]
        idx += 1
    out = np.array(out)
    print(out.shape)
    cap.release()
    return out
    
def show_video(video):
    plt.rcParams["animation.html"] = "jshtml"
    fig, ax = plt.subplots(1, figsize=(8,8))
    im = ax.imshow(video[0,...])
    plt.axis('off')
    
    def update(frame):
        im.set_array(video[frame,...])
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=video.shape[0], interval=200)
    display(ani)
    ani.save('animation.gif', writer='pillow', fps=10)
    plt.close(fig)

def show_masked_video(video, mask):
    color = np.array([0, 0.8, 0, 0.6])
    mask = mask * color.reshape(1,1,1,-1)
    plt.rcParams["animation.html"] = "jshtml"
    fig, ax = plt.subplots(1, figsize=(8,8))
    im = ax.imshow(video[0,...])
    im_msk = ax.imshow(mask[0,...])
    plt.axis('off')
    
    def update(frame):
        im.set_array(video[frame,...])
        im_msk.set_array(mask[frame,...])
    
    ani = animation.FuncAnimation(fig=fig, func=update, frames=video.shape[0], interval=200)
    display(ani)
    ani.save('animation.gif', writer='pillow', fps=10)
    plt.close(fig)

def draw_keypoints(frame, keypoints, keypoints_visible, threshold=0.7, show_scores=True, show_invisible=False):
    frame = frame.copy()
    connections = [
        (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),
        (3, 5), (4, 6), (5, 6), (5, 7), (6, 8),
        (7, 9), (8, 10), (5, 11), (6, 12), (11, 12),
        (11, 13), (12, 14), (13, 15), (14, 16),
        (15, 17), (15, 18), (15, 19),
        (16, 20), (16, 21), (16, 22)
    ]
    colors = [
        (51, 153, 255), (0, 255, 255), (255, 0, 0), (0, 255, 255), (255, 0, 0),
        (0, 255, 0), (255, 128, 0), (0, 255, 0), (255, 128, 0), (0, 255, 0),
        (255, 128, 0), (0, 255, 0), (255, 128, 0), (0, 255, 0), (255, 128, 0),
        (0, 255, 0), (255, 128, 0), (255, 255, 0), (255, 255, 0), (255, 255, 0),
        (255, 0, 255), (255, 0, 255), (255, 0, 255)
    ]
    assert keypoints.shape[0] == 23
    assert keypoints_visible.shape[0] == 23
    
    for conn in connections:
        x1 = round(keypoints[conn[0], 0])
        y1 = round(keypoints[conn[0], 1])
        x2 = round(keypoints[conn[1], 0])
        y2 = round(keypoints[conn[1], 1])
        if keypoints_visible[conn[0]] > threshold and keypoints_visible[conn[1]] > threshold:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
        elif show_invisible:
            cv2.line(frame, (x1, y1), (x2, y2), (128, 128, 128), 3)
        
    for idx in range(23):
        x = round(keypoints[idx, 0])
        y = round(keypoints[idx, 1])
        score = keypoints_visible[idx]
        color = colors[idx]
        color_dim = [x//2 for x in color]
        if score > threshold:
            cv2.circle(frame, (x, y), 15, color, -1)
            if show_scores:
                org = (x, y)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 255, 255)
                thickness = 2
                cv2.putText(frame, '%.2f'%score, org, font, fontScale, color, thickness, cv2.LINE_AA)
        elif show_invisible:
            cv2.circle(frame, (x, y), 15, color_dim, -1)
            if show_scores:
                org = (x, y)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 255, 255)
                thickness = 2
                cv2.putText(frame, '%.2f'%score, org, font, fontScale, color, thickness, cv2.LINE_AA)
            
    return frame

def draw_mask(frame, mask, alpha=0.5):
    frame = frame.copy()
    # color = (51, 153, 255) # Blue
    color = (0, 255, 0)
    assert frame.ndim == 3
    assert mask.ndim == 3
    assert frame.shape[0:2]==mask.shape[0:2]
    
    background = frame * (1-mask) + frame * mask * alpha
    foreground = mask * color * (1-alpha)

    frame = (background + foreground).astype(np.uint8)
    
    return frame

def draw_bbox(frame, bbox, text):
    color = (51, 153, 255)
    assert frame.ndim == 3
    assert len(bbox) == 4
    
    x1, y1, x2, y2 = round(bbox[0]), round(bbox[1]), round(bbox[2]), round(bbox[3])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    org = (x1, y1-10)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 2
    color = (0, 255, 0)
    thickness = 3
    cv2.putText(frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    
    return frame

def smooth_keypoints(arr, outlier_threshold=0.7, lam=10):
    arr = arr.copy()  # Avoid modifying original array 
    t = np.array(range(arr.shape[0]))
    for i in range(arr.shape[1]):
        spline = make_smoothing_spline(t, arr[:,i,0], lam=lam)
        arr[:,i,0] = spline(t)
        outlier = arr[:,i,2] < outlier_threshold
        for j in [0, 1, 2]:
            spline = make_smoothing_spline(t, arr[:,i,j], w = (1-outlier*(1-1e-8)), lam=lam)
            arr[:,i,j] = spline(t)
    return arr

def numpy_to_coco(masks, image_ids, category_id=1):
    """
    Convert a list of NumPy masks into COCO JSON format.

    masks: List of NumPy arrays, each being a binary mask (H x W).
    image_ids: List of image IDs corresponding to each mask.
    category_id: COCO category ID (default: 1 for a single class).

    Returns:
        COCO-formatted dictionary.
    """

    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [{"id": category_id, "name": "object", "supercategory": "none"}]
    }
    
    annotation_id = 1

    for i, (mask, image_id) in enumerate(zip(masks, image_ids)):
        height, width = mask.shape

        # Add image metadata
        coco_output["images"].append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": f"{image_id}.jpg"
        })

        # Get binary mask as RLE (Run-Length Encoding)
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")  # Convert to string for JSON serialization

        # # Convert mask to polygon format (alternative)
        # contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segmentation = []
        # for contour in contours:
        #     if contour.size >= 6:  # Needs at least 3 points
        #         segmentation.append(contour.flatten().tolist())

        # Add annotation
        coco_output["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": segmentation if segmentation else rle,  # Use polygons if available
            "area": int(np.sum(mask)),  # Count nonzero pixels as area
            "bbox": mask_to_bbox(mask),
            "iscrowd": 1
        })
        
        annotation_id += 1

    return coco_output


def mask_to_bbox(mask):
    """Convert binary mask to bounding box [x, y, width, height]."""
    pos = np.where(mask > 0)
    if len(pos[0]) == 0:
        return [0, 0, 0, 0]
    x_min, y_min = np.min(pos[1]), np.min(pos[0])
    x_max, y_max = np.max(pos[1]), np.max(pos[0])
    return [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]

    
class CvatExporter():
    def __init__(self):
        self.version = "1.1"
        self.date = datetime.today().strftime('%Y-%m-%d')

    def create_xml(self, id_str, masks, keypoints, file_path):
        root = ET.Element("annotations")
        version = ET.SubElement(root, "version")
        version.text = self.version

        meta = ET.SubElement(root, "meta")
        task = ET.SubElement(meta, "task")
        
        ET.SubElement(task, "id").text = id_str
        ET.SubElement(task, "size").text = str(masks.shape[0])
        ET.SubElement(task, "mode").text = "interpolation"
        ET.SubElement(task, "overlap").text = "5"
        ET.SubElement(task, "bugtracker").text = ""
        ET.SubElement(task, "created").text = self.date
        ET.SubElement(task, "updated").text = self.date

        
        labels = ET.SubElement(task, "labels")
        
        label = ET.SubElement(labels, "label")
        ET.SubElement(label, "name").text = "a human with prosthetic leg."
        ET.SubElement(label, "type").text = "mask"
        ET.SubElement(label, "attributes").text = ""

        label = ET.SubElement(labels, "label")
        ET.SubElement(label, "name").text = "body-foot"
        ET.SubElement(label, "type").text = "skeleton"
        ET.SubElement(label, "attributes").text = ""
        for idx in range(23):
            label = ET.SubElement(labels, "label")
            ET.SubElement(label, "name").text = str(idx+1)
            ET.SubElement(label, "type").text = "points"
            ET.SubElement(label, "attributes").text = ""
            ET.SubElement(label, "parent").text = "body-foot"

        segments = ET.SubElement(task, "segments")
        segment = ET.SubElement(segments, "segment")
        ET.SubElement(segment, "id").text = id_str
        ET.SubElement(segment, "start").text = "0"
        ET.SubElement(segment, "stop").text = str(masks.shape[0]-1)

        original_size = ET.SubElement(task, "original_size")
        ET.SubElement(original_size, "width").text = str(masks.shape[2])
        ET.SubElement(original_size, "height").text = str(masks.shape[1])

        # print("Writing pose keypoints...")
        track = ET.SubElement(
            root, "track", 
            id="0", 
            label="body-foot", 
            source="auto"
        )
        for idx in range(keypoints.shape[0]):
            skeleton = ET.SubElement(
                track, "skeleton", 
                frame=str(idx), 
                keyframe="1",  
                z_order="0"
            )
            for p_idx in range(keypoints.shape[1]):
                x, y, conf = tuple(keypoints[idx, p_idx, :])
                if conf > 0.7:
                    points = ET.SubElement(
                        skeleton, "points", 
                        label=str(p_idx+1),
                        keyframe="1",
                        outside="0",
                        occluded="0",
                        points="%3.2f,%3.2f" % (x, y)
                    )
                else:
                    points = ET.SubElement(
                        skeleton, "points", 
                        label=str(p_idx+1),
                        keyframe="1",
                        outside="1",
                        occluded="0",
                        points="%3.2f,%3.2f" % (x, y)
                    )
        
        # print("Writing masks...")
        for idx in range(masks.shape[0]):
            msk = masks[idx,...]
            box = self._mask_to_bbox(msk)
            if box is None:
                continue
            x, y, w, h = box[0], box[1], box[2]-box[0]+1, box[3]-box[1]+1
            msk = msk[y:y+h, x:x+w]
            rle = self._binary_mask_to_rle(msk)
            track = ET.SubElement(
                root, "track", 
                id=str(idx+1), 
                label="a human with prosthetic leg.", 
                source="auto"
            )                
            mask = ET.SubElement(
                track, "mask", 
                frame=str(idx), 
                keyframe="1", 
                outside="0", 
                occluded="0", 
                rle=str(rle)[1:-1],
                left=str(x),
                top=str(y),
                width=str(w),
                height=str(h),
                z_order="0"
            )
        
        tree = ET.ElementTree(root)
        tree.write(file_path, encoding="utf-8", xml_declaration=True)

    def _mask_to_bbox(self, mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
    
        if not np.any(rows) or not np.any(cols):
            return None  # No bounding box found
    
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
    
        return (x_min, y_min, x_max, y_max)
        
    def _binary_mask_to_rle(self, mask):
        mask = mask.flatten()
        rle = []
        last_elem = 0
        cnt = 0
        for elem in mask:
            if elem == last_elem:
                cnt += 1
            else:
                last_elem = elem
                rle += [cnt]
                cnt = 1
        return rle
    