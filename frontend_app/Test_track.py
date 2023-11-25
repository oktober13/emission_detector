import os
import cv2
import numpy as np

import ultralytics

ultralytics.checks()

from ultralytics import YOLO

MODEL = "yolov8x.pt"
model = YOLO(MODEL)
# model.to('cuda')

# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names
# class_ids of interest - car, motorcycle, bus and truck
CLASS_ID = [2, 3, 5, 7]

from PIL import Image, ImageDraw


def resize(polygon, w, h):
    pts = np.array(polygon)
    pts[:, 0] *= w
    pts[:, 1] *= h
    return pts


def get_true_zone(zone):
    tz = [zone[0][0], zone[0][3], zone[1][3], zone[1][0]]
    return [(p[0], p[1]) for p in tz]


def make_zone_longer(zone):
    z = np.array(zone)
    left_vector = zone[3] - zone[0]
    right_vector = zone[2] - zone[1]
    return [
        zone[0] - left_vector / 2.0,
        zone[1] - right_vector / 2.0,
        zone[2] + right_vector / 2.0,
        zone[3] + left_vector / 2.0,
    ]


def make_zone_wider(zone):
    z = np.array(zone)
    near_vector = zone[2] - zone[3]
    far_vector = zone[1] - zone[0]
    return [
        zone[0] - far_vector / 2.0,
        zone[1] + far_vector / 2.0,
        zone[2] + near_vector / 2.0,
        zone[3] - near_vector / 2.0,
    ]


def get_base_height(conus_near, conus_far):
    xn, yn = conus_near
    xf, yf = conus_far
    r = np.sqrt((xn - xf) * (xn - xf) + (yn - yf) * (yn - yf))  # 20 meters
    return int(r / 2.0)


def make_vpoly_across(bottom_near, bottom_far, base_h):
    xn, yn = bottom_near
    xf, yf = bottom_far
    return [(xn, yn), (xf, yf), (xf, yf - int(base_h * 0.85)), (xn, yn - int(base_h))]


def make_vpoly_along(bottom_near, bottom_far, base_h):
    xn, yn = bottom_near
    xf, yf = bottom_far
    return [(xn, yn), (xf, yf), (xf, yf - int(base_h * 0.5)), (xn, yn - int(base_h))]


def add_box_to_mask(mask, box_poly):
    ''' box format: lt, rt, rb, lb'''
    poly = [(p[0], p[1]) for p in box_poly]
    draw = ImageDraw.Draw(mask)
    draw.polygon(poly, fill="white", outline="white")
    return mask


def create_mask(boxes, img_width, img_height):
    mask = Image.new("RGB", [img_width, img_height], "black")
    for b in boxes:
        mask = add_box_to_mask(mask, b)
    mask = np.array(mask)
    return mask


def get_masked_image(image, mask):
    masked_numpy_img = np.array(image)
    idx = (mask == 0)
    masked_numpy_img[idx] = mask[idx]
    # masked_img = Image.fromarray(masked_numpy_img.astype('uint8'), 'RGB')
    return masked_numpy_img


from typing import List
import numpy as np


# Ray tracing
def ray_tracing_on_box(box, poly):
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


import json
import cv2
import os
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
from collections import defaultdict

# answer dataframe
df_mean_speeds = pd.DataFrame(columns=[
    'car', 'quantity_car', 'average_speed_car',
    'van', 'quantity_van', 'average_speed_van',
    'bus', 'quantity_bus', 'average_speed_bus'
])
df_mean_speeds.index.name = 'file_name'

video_dir = r"C:\Users\Danie\OneDrive\Рабочий стол\Новая папка (2)"
jsons_dir = r"C:\Users\Danie\OneDrive\Рабочий стол\Sasha Back\imoscow_hackathon\json"

DET_FREQ = 6


def get_result(video_list, json_list):
    result_list = []
    for source_video, json_file in zip(video_list, json_list):
        fname = os.path.basename(source_video)

        videoname = fname[:-4]

        with open(json_file, 'r') as f:
            data = json.load(f)

        cap = cv2.VideoCapture(source_video)
        if cap.isOpened():
            v_width = int(cap.get(3))
            v_height = int(cap.get(4))
            v_fps = int(cap.get(5))

        # блок с вычислением маски
        big_zone = resize(get_true_zone(data['zones']), v_width, v_height)
        base_h = get_base_height(big_zone[3], big_zone[0])

        boxes = [
            big_zone,
            make_vpoly_across(big_zone[0], big_zone[1], base_h * 0.5),  # back
            make_vpoly_along(big_zone[2], big_zone[1], base_h * 0.85),  # right
            make_vpoly_across(big_zone[3], big_zone[2], base_h),  # front
            make_vpoly_along(big_zone[3], big_zone[0], base_h),  # left
            make_zone_wider(big_zone),
            make_zone_longer(big_zone)
        ]

        # генерация маски
        mask = create_mask(boxes, v_width, v_height)

        # инициализация счетчиков траекторий
        tracking_zone = make_zone_wider(big_zone)
        tracker_ids = set()
        state = {}
        max_conf = {}
        obj_class = {}

        frame_start = {}
        frame_finish = {}

        # прогон видеофайла детекция + трекинг + счетчики зоны контроля
        counter = 0
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                break
            masked_img = get_masked_image(img, mask)
            # model prediction on single frame and conversion to supervision Detections
            if counter % DET_FREQ == 0:
                result = model.track(
                    masked_img, persist=True,
                    tracker="bytetrack.yaml",
                    show=False,
                    verbose=False
                )[0]
                is_track = result.boxes.is_track  # ???
                if is_track:
                    boxes = result.boxes
                    for xyxy_t, conf_t, class_id_t, tracker_id_t in zip(
                            boxes.xyxy, boxes.conf, boxes.cls, boxes.id
                    ):
                        class_id = class_id_t.item()
                        if class_id not in [2, 5, 7]:
                            continue
                        xyxy = xyxy_t.cpu().tolist()
                        current_state = ray_tracing_on_box(xyxy, tracking_zone)

                        confidence = conf_t.item()
                        tracker_id = tracker_id_t.item()
                        if tracker_id not in state:
                            tracker_ids.add(tracker_id)
                            max_conf[tracker_id] = confidence
                            obj_class[tracker_id] = class_id
                            state[tracker_id] = False
                            continue

                        if state[tracker_id] != current_state:
                            if current_state is True:
                                frame_start[tracker_id] = counter
                            else:
                                frame_finish[tracker_id] = counter
                            state[tracker_id] = current_state
                        if confidence > max_conf[tracker_id]:
                            obj_class[tracker_id] = class_id
                            max_conf[tracker_id] = confidence
            counter += 1
            if counter % 1000 == 0:
                # if counter > 500:
                print(f'{counter} frames processed')
            #     break

        cap.release()

        # расчет скоростей движения всех тс по id
        speed = {}

        for tracker_id in tracker_ids:
            if tracker_id in frame_start and tracker_id not in frame_finish and counter - frame_start[tracker_id] < 200:
                frame_finish[tracker_id] = counter

            if tracker_id in frame_finish and tracker_id not in frame_start and frame_finish[tracker_id] < 200:
                frame_start[tracker_id] = 0

        for tracker_id in (set(frame_start.keys()) & set(frame_finish.keys())):
            # еще есть такие, кто все время был внутри
            frames_in_zone = float(frame_finish[tracker_id] - frame_start[tracker_id])
            seconds_in_zone = frames_in_zone / v_fps
            meters_over_seconds = 20 / (seconds_in_zone + 0.001)
            km_over_hour = meters_over_seconds * 3.6
            speed[tracker_id] = km_over_hour

        # формирование строки ответа по видео
        speeds_cls_cls = {tmp: [] for tmp in [2, 7, 5]}
        class_count = defaultdict(float)

        for tracker_id in (set(frame_start.keys()) & set(frame_finish.keys())):
            c = obj_class[tracker_id]
            class_count[c] += 1
            speeds_cls_cls[c].append(speed[tracker_id])

        quantity = {}
        average_speed = {}

        for class_name, class_id in zip(['car', 'van', 'bus'], [2, 7, 5]):
            quantity[class_name] = class_count[class_id]
            speed_arra_for_class = np.array(speeds_cls_cls[class_id])
            mask_speed = (speed_arra_for_class > 1) & (speed_arra_for_class < 110)
            if class_count[class_id]:
                average_speed[class_name] = np.mean(speed_arra_for_class[mask_speed])
            else:
                average_speed[class_name] = 0.0

        answer_line = [
            'car', quantity['car'], average_speed['car'],
            'van', quantity['van'], average_speed['van'],
            'bus', quantity['bus'], average_speed['bus'],
        ]

        result_list.append([fname] + answer_line)
    return result_list
