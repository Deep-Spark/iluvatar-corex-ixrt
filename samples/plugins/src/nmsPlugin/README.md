# NMSPlugin

**Table Of Contents**
- [Description](#description)
    * [Structure](#structure)
- [Parameters](#parameters)
- [Algorithms](#algorithms)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

The `NMSPlugin` implements a non-maximum suppression (NMS) step over boxes for object detection networks.

Non-maximum suppression is typically the universal step in object detection inference. This plugin is used after you’ve processed the bounding box prediction and object classification to get the final bounding boxes for objects.

With this plugin, you can incorporate the non-maximum suppression step during IxRT inference. During inference, the neural network generates a fixed number of bounding boxes with box coordinates, identified class and confidence levels. Not all bounding boxes, but the most representative ones, have to be drawn on the original image.

Non-maximum suppression is the way to eliminate the boxes which have low confidence or do not have object in and keep the most representative ones. For example, the objects within an image might be covered by many boxes with different levels of confidence. The goal of the non-maximum suppression step is to find the most confident box for the object and remove all the less confident ones.


### Structure

The `NMSPlugin` takes two inputs, boxes input and scores input.

- `Boxes input`
The boxes input are of shape `[batch_size, number_box_parameters, number_classes, number_boxes]`. The box location usually consists of four parameters such as `[x1, y1, x2, y2]` where (x1, y1) and (x2, y2) are the coordinates of any diagonal pair of box corners. For example, if your model outputs `8400` bounding boxes given one image, there are `100` candidate classes, the shape of boxes input will be `[1, 4, 100, 8400]`. if the boxes are shared across all classes, the shape of boxes input will be `[1, 4, 1, 8400]`.

- `Scores input`
The scores input are of shape `[batch_size, number_classes, number_boxes]`. Each box has an array of probability for each candidate class. For example, if your model outputs `8400` bounding boxes given one image, there are `100` candidate classes, the shape of scores input will be `[1, 100, 8400]`.

The boxes input and scores input generates the following four outputs:

- `num_detections`
The `num_detections` input are of shape `[batch_size, 1]`. The last dimension of size 1 is an INT32 scalar indicating the number of valid detections per batch item. It can be less than `keepTopK`. Only the top `num_detections[i]` entries in `nmsed_boxes[i]`, `nmsed_scores[i]` and `nmsed_classes[i]` are valid.

- `nmsed_boxes`
A `[batch_size, keepTopK, 4]` float32 tensor containing the coordinates of non-max suppressed boxes.

- `nmsed_scores`
A `[batch_size, keepTopK]` float32 tensor containing the scores for the boxes.

- `nmsed_classes`
A `[batch_size, keepTopK]` float32 tensor containing the classes for the boxes.


## Parameters

| Type     | Parameter                | Description
|----------|--------------------------|--------------------------------------------------------
|`bool`    |`share_location`          |If set to `true`, the boxes input are shared across all classes. If set to `false`, the boxes input should account for per-class box data.
|`int`     |`background_class`        |The label ID for the background class. If there is no background class, set it to `-1`.
|`int`     |`max_output_boxes`        |The number of total bounding boxes to be kept per-image after the NMS step.
|`float`   |`score_threshold`         |The scalar threshold for score (low scoring boxes are removed).
|`float`   |`iou_threshold`           |The scalar threshold for IOU (new boxes that have high IOU overlap with previously selected boxes are removed).

## Algorithms

The NMS algorithm used in this particular plugin first sorts the bounding boxes indices by the score for each class, then sorts the bounding boxes by the updated scores, and finally collects the desired number of bounding boxes with the highest scores.

It is mainly accelerated using the `TODO nmsInference` kernel defined in the `NMSInference.cu` file.

Specifically, the NMS algorithm:
- Sorts the bounding box indices by the score for each class. Before sorting, the bounding boxes with a score less than `score_threshold` are discarded by setting their indices to `-1` and their scores to `0`. This is using the `sortScoresPerClass` kernel defined in the `sortScoresPerClass.cu` file.

- Finds the most confident box for the object and removes all the less confident ones using the iterative non-maximum suppression step step for each class. Starting from the bounding box with the highest score in each class, the bounding boxes that has overlap higher than `iou_threshold` is suppressed by setting their indices to `-1` and their scores to `0`. Then all the less confident bounding boxes were suppressed for each class. This is using the `allClassNMS` kernel defined in the `allClassNMS.cu` file.

- Sorts the bounding boxes per image using the updated scores. At this time, all the classes were mixed before sort. Discarded and suppressed bounding boxes will go to the end of the sorted array since their score is `0`. This is using the `sortScoresPerImage` kernel defined in the `sortScoresPerImage.cu` file.

- Collects the desired number, `max_output_boxes`, of bounding box indices with the highest scores from the top of the sorted array, their bounding box coordinates, and their object classification information. This is using the `gatherNMSOutputs` kernel defined in the `gatherNMSOutputs.cu` file.


## Additional resources

The following resources provide a deeper understanding of the `NMSPlugin` plugin:

**Networks**

- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
- [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)
- [YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/abs/2107.08430)


**Documentation**

- [NMS algorithm](https://www.coursera.org/lecture/convolutional-neural-networks/non-max-suppression-dvrjH)


## License

TODO


## Changelog

June 2023
This is the first release of this `README.md` file.
