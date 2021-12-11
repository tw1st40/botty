import time
import random
import ctypes
import numpy as np
from logger import Logger
import cv2
from typing import List, Tuple
import requests
import os
from version import __version__


def send_discord(msg, url: str, color_it: bool = False):
    if not url:
        return
    msg = f"{msg} (v{__version__})"
    if color_it:
        if " magic_" in msg:
            msg = f"```ini\n[ {msg} \n```"
        elif " set_" in msg:
            msg = f"```diff\n+ {msg} \n```"
        elif " rune_" in msg:
            msg = f"```css\n[ {msg} ]\n```"
        elif " uniq_" in msg or "rare" in msg:
            # TODO: It is more gold than yellow, find a better yellow highlight
            msg = f"```fix\n- {msg} \n```"
        elif " eth_" in msg:
            msg = f"```python\n# {msg} \n```"
        else:
            msg = f"```\n {msg} \n```"
    requests.post(url, json={"content": msg})

def wait(min_seconds, max_seconds = None):
    if max_seconds is None:
        max_seconds = min_seconds
    time.sleep(random.uniform(min_seconds, max_seconds))
    return

def kill_thread(thread):
    thread_id = thread.ident
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, ctypes.py_object(SystemExit))
    if res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
        Logger.error('Exception raise failure')

def cut_roi(img, roi):
    x, y, width, height = roi
    return img[y:y+height, x:x+width]

def is_in_roi(roi: List[float], pos: Tuple[float, float]):
    x, y, w, h = roi
    is_in_x_range = x < pos[0] < x + w
    is_in_y_range = y < pos[1] < y + h
    return is_in_x_range and is_in_y_range

def color_filter(img, color_range):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv_img, color_range[0], color_range[1])
    filtered_img = cv2.bitwise_and(img, img, mask=color_mask)
    return color_mask, filtered_img

def hms(seconds: int):
    seconds = int(seconds)
    h = seconds // 3600
    m = seconds % 3600 // 60
    s = seconds % 3600 % 60
    return '{:02d}:{:02d}:{:02d}'.format(h, m, s)

def load_template(path, scale_factor: float = 1.0, alpha: bool = False):
    if os.path.isfile(path):
        template_img = cv2.imread(path, cv2.IMREAD_UNCHANGED) if alpha else cv2.imread(path)
        template_img = cv2.resize(template_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
        return template_img
    return None

def alpha_to_mask(img: np.ndarray):
    # create a mask from template where alpha == 0
    if img.shape[2] == 4:
        if np.min(img[:, :, 3]) == 0:
            _, mask = cv2.threshold(img[:,:,3], 1, 255, cv2.THRESH_BINARY)
            return mask
    return None

def list_files_in_folder(path: str):
    r = []
    for root, _, files in os.walk(path):
        for name in files:
            r.append(os.path.join(root, name))
    return r

def img_to_bytes(image: np.ndarray, color: str = 'BGR'):
    """ Sets an OpenCV-style image for recognition.
    https://github.com/sirfz/tesserocr/issues/198

    'image' is a numpy ndarray in color, grayscale, or binary (boolean)
        format.
    'color' is a string representing the current color of the image,
        for conversion using OpenCV into an RGB array image. By default
        color images in OpenCV use BGR, but any valid channel
        specification can be used (e.g. 'BGRA', 'XYZ', 'YCrCb', 'HSV', 'HLS',
        'Lab', 'Luv', 'BayerBG', 'BayerGB', 'BayerRG', 'BayerGR').
        Conversion only occurs if the third dimension of the array is
        not 1, else 'color' is ignored.

    """
    bytes_per_pixel = image.shape[2] if len(image.shape) == 3 else 1
    height, width   = image.shape[:2]
    bytes_per_line  = bytes_per_pixel * width

    if bytes_per_pixel != 1 and color != 'RGB':
        # non-RGB color image -> convert to RGB
        image = cv2.cvtColor(image, getattr(cv2, f'COLOR_{color}2RGB'))
    elif bytes_per_pixel == 1 and image.dtype == bool:
        # binary image -> convert to bitstream
        image = np.packbits(image, axis=1)
        bytes_per_line  = image.shape[1]
        width = bytes_per_line * 8
        bytes_per_pixel = 0
    # else image already RGB or grayscale

    return image.tobytes(), width, height, bytes_per_pixel, bytes_per_line