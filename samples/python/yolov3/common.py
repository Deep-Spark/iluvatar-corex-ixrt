# Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.
#

import re

import cv2
import numpy as np
from tqdm import tqdm


def letterbox(
    im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32
):
    # Resize and pad image while meeting stride-multiple constraints

    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))
    dw, dh = new_shape[1] - new_unpad[1], new_shape[0] - new_unpad[0]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape != new_unpad:  # resize
        im = cv2.resize(im, new_unpad[::-1], interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im1 = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border

    return im1, r, (dw, dh)


def scale_boxes(net_shape, boxes, ori_shape, use_letterbox=False):
    # Rescale boxes (xyxy) from net_shape to ori_shape

    if use_letterbox:

        gain = min(
            net_shape[0] / ori_shape[0], net_shape[1] / ori_shape[1]
        )  # gain  = new / old
        pad = (net_shape[1] - ori_shape[1] * gain) / 2, (
            net_shape[0] - ori_shape[0] * gain
        ) / 2.0

        boxes[:, [0, 2]] -= pad[0]  # x padding
        boxes[:, [1, 3]] -= pad[1]  # y padding
        boxes[:, :4] /= gain
    else:
        x_scale, y_scale = net_shape[1] / ori_shape[1], net_shape[0] / ori_shape[0]

        boxes[:, 0] /= x_scale
        boxes[:, 1] /= y_scale
        boxes[:, 2] /= x_scale
        boxes[:, 3] /= y_scale

    clip_boxes(boxes, ori_shape)
    return boxes


def clip_boxes(boxes, shape):

    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def post_process_withoutNMS(
    ori_img_shape,
    imgsz,
    output_io_buffers,
    sample_num,
    use_letterbox=False,
    n_max_keep=1000,
):

    all_box = []
    data_offset = 0

    box_datas = output_io_buffers[0].flatten()
    box_nums = output_io_buffers[1].flatten()

    for i in range(sample_num):
        box_num = box_nums[i]
        if box_num == 0:
            boxes = None
        else:
            cur_box = box_datas[data_offset : data_offset + box_num * 6].reshape(-1, 6)
            boxes = scale_boxes(
                (imgsz[0], imgsz[1]), cur_box, ori_img_shape[i], use_letterbox
            )
            # xyxy2xywh
            boxes[:, 2] -= boxes[:, 0]
            boxes[:, 3] -= boxes[:, 1]

        all_box.append(boxes)
        data_offset += n_max_keep * 6

    return all_box


def save2json(batch_img_id, pred_boxes, json_result, class_trans):
    for i, boxes in enumerate(pred_boxes):
        if boxes is not None:
            image_id = int(batch_img_id[i])
            # print(image_id)
            for x, y, w, h, c, p in boxes:
                x, y, w, h, p = float(x), float(y), float(w), float(h), float(p)
                c = int(c)
                json_result.append(
                    {
                        "image_id": image_id,
                        "category_id": class_trans[c - 1],
                        "bbox": [x, y, w, h],
                        "score": p,
                    }
                )


def pre_process(img_file, imgsz, use_letterbox=False):
    # HWC  BGR
    org_img = cv2.imread(img_file)
    image = org_img.copy()
    if use_letterbox:
        image, ratio, dwdh = letterbox(image, new_shape=imgsz, auto=False, scaleup=True)
    else:
        image = cv2.resize(image, (imgsz[0], imgsz[1]))

    image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    image = np.ascontiguousarray(image).astype(np.float32) / 255.0
    return image, org_img


def get_batch_input(img_list, config):
    input_io_buffer, batch_imgs_shape, batch_img_id = [], [], []
    for img_path in img_list:
        img, org_img = pre_process(
            img_path, [config.imgsz, config.imgsz], config.use_letterbox
        )
        batch_imgs_shape.append(org_img.shape)
        batch_img_id.append(re.findall(r"[1-9]\d*.jpg", img_path)[0][:-4])
        input_io_buffer.append(np.expand_dims(img, 0))

    return np.concatenate(input_io_buffer, 0), batch_imgs_shape, batch_img_id


def precess_batch_input(img_list, config, batch_size):
    input_io_batch_buffer, all_img_shape, all_img_id = [], [], []
    single_input, single_img_shape, single_img_id = [], [], []

    for img_path in tqdm(img_list, desc="Load All Image."):
        img, org_img = pre_process(
            img_path, [config.imgsz, config.imgsz], config.use_letterbox
        )
        single_img_shape.append(org_img.shape)
        single_input.append(np.expand_dims(img, 0))

        if len(single_input) == batch_size:
            input_io_batch_buffer.append(np.concatenate(single_input, 0))
            all_img_shape.append(single_img_shape)
            single_input, single_img_shape = [], []

    if len(single_input) > 0:
        input_io_batch_buffer.append(np.concatenate(single_input, 0))
        all_img_shape.append(single_img_shape)

    return input_io_batch_buffer, all_img_shape
