/*
 * Copyright (c) 2024, Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
 * All Rights Reserved.
 *
 *   Licensed under the Apache License, Version 2.0 (the "License"); you may
 *   not use this file except in compliance with the License. You may obtain
 *   a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *   WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *   License for the specific language governing permissions and limitations
 *   under the License.
 */



#include "postprocess_utils.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include <ostream>
#include <sstream>

#include "coco_labels.h"
#include "imagenet_labels.h"
using namespace nvinfer1;
const int32_t kYoloV3CellSize = 6;
const int32_t kRetinafaceLdmSize = 10;

void GetClassificationResult(const float* scores, const int32_t size, int32_t top_k,
                             std::vector<ClassificationResult>* output) {
    std::vector<std::pair<float, int32_t>> pairs(size);
    for (int32_t i = 0; i < size; i++) {
        pairs[i] = std::make_pair(scores[i], i);
    }

    auto cmp_func = [](const std::pair<float, int32_t>& p0, const std::pair<float, int32_t>& p1) -> bool {
        return p0.first > p1.first;
    };

    std::nth_element(pairs.begin(), pairs.begin() + top_k, pairs.end(),
                     cmp_func);  // get top K results & sort
    std::sort(pairs.begin(), pairs.begin() + top_k, cmp_func);
    if (output) {
        output->clear();
        ClassificationResult single_result;
        for (int32_t i = 0; i < top_k; ++i) {
            single_result.prob.emplace_back(pairs[i].first);
            single_result.idx.emplace_back(pairs[i].second);
            single_result.names.emplace_back(imagenet_labels_tab[pairs[i].second]);
        }
        output->emplace_back(single_result);
    } else {
        printf("top %d results:\n", top_k);
        for (int32_t i = 0; i < top_k; ++i) {
            printf("%dth: %-10f %-10d %s\n", i + 1, pairs[i].first, pairs[i].second,
                   imagenet_labels_tab[pairs[i].second]);
        }
    }
}

void ShowClassificationResults(const std::vector<ClassificationResult>& results) {
    std::cout << "Classification result: " << std::endl;
    const auto& res = results.front();
    int32_t top_k = res.prob.size();
    printf("top %d results:\n", top_k);
    for (int32_t i = 0; i < top_k; ++i) {
        printf("%dth: %-10f %-10d %s\n", i + 1, res.prob.at(i), res.idx.at(i), res.names.at(i).c_str());
    }
}

void correct_yolo_boxes(DetectionResult* dets, int n, int w, int h, int netw, int neth, int relative, int letter) {
    int i;
    int new_w = 0;
    int new_h = 0;
    if (letter) {
        if (((float)netw / w) < ((float)neth / h)) {
            new_w = netw;
            new_h = (h * netw) / w;
        } else {
            new_h = neth;
            new_w = (w * neth) / h;
        }
    } else {
        new_w = netw;
        new_h = neth;
    }
    for (i = 0; i < n; ++i) {
        Box2D b = dets[i].bbox;
        b.x = (b.x - (netw - new_w) / 2. / netw) / ((float)new_w / netw);
        b.y = (b.y - (neth - new_h) / 2. / neth) / ((float)new_h / neth);
        b.w *= (float)netw / new_w;
        b.h *= (float)neth / new_h;
        if (!relative) {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

int get_yolo_decode_detections(float* input_data, int32_t num_anchor, int32_t decoder_dim,
                               std::vector<DetectionResult>* dets, int32_t num_class, int w, int h, int netw, int neth,
                               float thresh, int* map, int relative, int letter, int sample_ind = 0) {
    float* predictions = input_data;

    int32_t anchor_cell_size = num_anchor * decoder_dim;
    int32_t batch_cell_size = w * h * anchor_cell_size;
    int32_t space_size = w * h;
    int32_t space_idx;
    for (auto i = 0; i < space_size; ++i) {
        space_idx = sample_ind * batch_cell_size + i * anchor_cell_size;
        for (auto n = 0; n < num_anchor; ++n) {
            /*
                6  : left_top_x, left_top_y, right_bottom_x, right_bottom_y, class_id, final_conf(max_prob * conf)
                85 : center_x, center_y, w, h, conf, prob0, prob1, ..., prob_79
            */
            int obj_index = space_idx + n * decoder_dim;
            int class_index;
            float center_x, center_y, w, h, objectness, max_prob;

            if (decoder_dim == 85) {
                center_x = predictions[obj_index];
                center_y = predictions[obj_index + 1];
                w = predictions[obj_index + 2];
                h = predictions[obj_index + 3];

                class_index = 0;
                max_prob = predictions[obj_index + 5];
                for (int k = 1; k < decoder_dim - 5; k++) {
                    if (predictions[obj_index + 5 + k] > max_prob) {
                        max_prob = predictions[obj_index + 5 + k];
                        class_index = k;
                    }
                }
                objectness = max_prob * predictions[obj_index + 4];
            } else if (decoder_dim == 6) {
                center_x = (predictions[obj_index] + predictions[obj_index + 2]) / 2;
                center_y = (predictions[obj_index + 1] + predictions[obj_index + 3]) / 2;
                w = (predictions[obj_index + 2] - predictions[obj_index]);
                h = (predictions[obj_index + 3] - predictions[obj_index + 1]);
                class_index = (int)(predictions[obj_index + 4]) - 1;
                objectness = predictions[obj_index + 5];
            } else {
                printf("Error decoder_dim:%d, should be 6/85\n", decoder_dim);
            }
            // if (objectness <= thresh) continue;   // incorrect behavior for
            // Nan values
            if (objectness > thresh) {
                DetectionResult cache;
                cache.bbox.x = center_x * 1.0 / netw;
                cache.bbox.y = center_y * 1.0 / neth;
                cache.bbox.w = w * 1.0 / netw;
                cache.bbox.h = h * 1.0 / neth;
                cache.objectness = objectness;
                cache.classes = num_class;
                // for (auto j = 0; j < num_class; ++j) {
                //     cache.prob[j] = 0;
                // }
                cache.prob.resize(num_class, 0.f);
                cache.class_idx = class_index;
                cache.prob[class_index] = objectness;
                // ++count;
                dets->emplace_back(cache);
            }
        }
    }

    correct_yolo_boxes(dets->data(), dets->size(), w, h, netw, neth, relative, letter);
    return dets->size();
}

int get_retinaface_decode_detections(float* input_data, float* ldm_data, int32_t num_anchor, int32_t decoder_dim,
                                     int32_t ldm_dim, std::vector<DetectionWithLandmark>* dets, int32_t num_class,
                                     int w, int h, int netw, int neth, float thresh, int* map, int relative, int letter,
                                     int sample_ind = 0) {
    float* predictions = input_data;
    float* landmark = ldm_data;

    int32_t anchor_cell_size = num_anchor * decoder_dim;
    int32_t batch_cell_size = w * h * anchor_cell_size;
    int32_t space_size = w * h;
    int32_t space_idx, ldm_space_idx;
    for (auto i = 0; i < space_size; ++i) {
        space_idx = sample_ind * batch_cell_size + i * anchor_cell_size;
        ldm_space_idx = sample_ind * w * h * num_anchor * ldm_dim + i * num_anchor * ldm_dim;
        for (auto n = 0; n < num_anchor; ++n) {
            int obj_index = space_idx + n * decoder_dim;
            int ldm_index = ldm_space_idx + n * ldm_dim;
            float objectness = predictions[obj_index + 5];
            // if (objectness <= thresh) continue;   // incorrect behavior for
            // Nan values
            if (objectness > thresh) {
                DetectionWithLandmark cache;
                cache.bbox.x = (predictions[obj_index] + predictions[obj_index + 2]) / 2 * 1.0 / netw;
                cache.bbox.y = (predictions[obj_index + 1] + predictions[obj_index + 3]) / 2 * 1.0 / neth;
                cache.bbox.w = (predictions[obj_index + 2] - predictions[obj_index]) * 1.0 / netw;
                cache.bbox.h = (predictions[obj_index + 3] - predictions[obj_index + 1]) * 1.0 / neth;
                cache.objectness = objectness;
                cache.classes = num_class;
                // for (auto j = 0; j < num_class; ++j) {
                //     cache.prob[j] = 0;
                // }
                cache.prob.resize(num_class, 0.f);
                int class_index = (int)(predictions[obj_index + 4]) - 1;
                cache.class_idx = class_index;
                cache.prob[class_index] = objectness;

                cache.pts.resize(ldm_dim, 0.f);
                for (int m = 0; m < ldm_dim; m++) {
                    cache.pts[m] = landmark[ldm_index + m];
                }
                // ++count;
                dets->emplace_back(cache);
            }
        }
    }

    return dets->size();
}

float overlap(float x1, float w1, float x2, float w2) {
    float l1 = x1 - w1 / 2;
    float l2 = x2 - w2 / 2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1 / 2;
    float r2 = x2 + w2 / 2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(Box2D a, Box2D b) {
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0) return 0;
    float area = w * h;
    return area;
}

float box_union(Box2D a, Box2D b) {
    float i = box_intersection(a, b);
    float u = a.w * a.h + b.w * b.h - i;
    return u;
}

float box_iou(Box2D a, Box2D b) { return box_intersection(a, b) / box_union(a, b); }

int nms_comparator(const void* pa, const void* pb) {
    DetectionResult a = *(DetectionResult*)pa;
    DetectionResult b = *(DetectionResult*)pb;
    float diff = 0;
    if (b.sort_class >= 0) {
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    } else {
        diff = a.objectness - b.objectness;
    }
    if (diff < 0)
        return 1;
    else if (diff > 0)
        return -1;
    return 0;
}

int nms_ldm_comparator(const void* pa, const void* pb) {
    DetectionWithLandmark a = *(DetectionWithLandmark*)pa;
    DetectionWithLandmark b = *(DetectionWithLandmark*)pb;
    float diff = 0;
    if (b.sort_class >= 0) {
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    } else {
        diff = a.objectness - b.objectness;
    }
    if (diff < 0)
        return 1;
    else if (diff > 0)
        return -1;
    return 0;
}

void do_nms_sort(DetectionResult* dets, int total, int classes, float thresh) {
    int i, j, k;
    k = total - 1;
    for (i = 0; i <= k; ++i) {
        if (dets[i].objectness == 0) {
            DetectionResult swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k + 1;

    for (k = 0; k < classes; ++k) {
        for (i = 0; i < total; ++i) {
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(DetectionResult), nms_comparator);
        for (i = 0; i < total; ++i) {
            // printf("  k = %d, \t i = %d \n", k, i);
            if (dets[i].prob[k] == 0) continue;
            Box2D a = dets[i].bbox;
            for (j = i + 1; j < total; ++j) {
                Box2D b = dets[j].bbox;
                if (box_iou(a, b) > thresh) {
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}

void do_nms_ldm_sort(DetectionWithLandmark* dets, int total, int classes, float thresh) {
    int i, j, k;
    k = total - 1;
    for (i = 0; i <= k; ++i) {
        if (dets[i].objectness == 0) {
            DetectionWithLandmark swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k + 1;

    for (k = 0; k < classes; ++k) {
        for (i = 0; i < total; ++i) {
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(DetectionWithLandmark), nms_comparator);
        for (i = 0; i < total; ++i) {
            // printf("  k = %d, \t i = %d \n", k, i);
            if (dets[i].prob[k] == 0) continue;
            Box2D a = dets[i].bbox;
            for (j = i + 1; j < total; ++j) {
                Box2D b = dets[j].bbox;
                if (box_iou(a, b) > thresh) {
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}

void PrintDetectionResult(const std::vector<DetectionResult>& det_outputs) {
    int32_t test_h{576};
    int32_t test_w{768};
    PrintDetectionResult(det_outputs, test_h, test_w);
}

void YoloGetResults(const IOBuffers& buffers, std::vector<DetectionResult>* output, int32_t num_class,
                    const std::vector<int32_t>& num_anchor, int32_t ih, int32_t iw, int32_t batch_idx, float nms_thre,
                    float select_thre) {
    std::vector<DetectionResult> pre_output;
    for (auto i = 0; i < buffers.size(); ++i) {
        auto& buffer = buffers.at(i);
        std::vector<DetectionResult> results;

        auto h = buffer.shape.dims.at(2) + buffer.shape.padding.at(2);
        auto w = buffer.shape.dims.at(3) + buffer.shape.padding.at(3);
        auto c = buffer.shape.dims.at(1) + buffer.shape.padding.at(1);
        auto batch_offset = h * w * c;
        float* data = reinterpret_cast<float*>(buffer.data) + batch_idx * batch_offset;

        get_yolo_decode_detections(data, num_anchor.at(i), kYoloV3CellSize, &results, num_class, w, h, iw, ih,
                                   select_thre, nullptr, 1, 0);
        pre_output.insert(pre_output.end(), results.begin(), results.end());
    }

    if (nms_thre > 0.f) {
        do_nms_sort(pre_output.data(), pre_output.size(), num_class, nms_thre);
    }
    for (const auto& det : pre_output) {
        if (det.prob[det.class_idx] > select_thre) {
            output->emplace_back(det);
        }
    }
}
void YoloGetResults(float* buffer, std::vector<DetectionResult>* output, int32_t num_class, const int32_t num_anchor,
                    int32_t w, int32_t h, int32_t c, int32_t ih, int32_t iw, int32_t batch_idx, float nms_thre,
                    float select_thre) {
    std::vector<DetectionResult> pre_output;
    int batch_offset = h * w * c;
    float* data = reinterpret_cast<float*>(buffer) + batch_idx * batch_offset;
    get_yolo_decode_detections(data, num_anchor, c / num_anchor, &pre_output, num_class, w, h, iw, ih, select_thre,
                               nullptr, 1, 0);

    if (nms_thre > 0.f) {
        do_nms_sort(pre_output.data(), pre_output.size(), num_class, nms_thre);
    }
    for (const auto& det : pre_output) {
        if (det.prob[det.class_idx] > select_thre) {
            output->emplace_back(det);
        }
    }
}

void RetinafaceGetResults(const IOBuffers& buffers, const IOBuffers& ldm_buffers,
                          std::vector<DetectionWithLandmark>* output, int32_t num_class,
                          const std::vector<int32_t>& num_anchor, int32_t ih, int32_t iw, int32_t batch_idx,
                          float nms_thre, float select_thre) {
    std::vector<DetectionWithLandmark> pre_output;
    for (auto i = 0; i < buffers.size(); ++i) {
        auto& buffer = buffers.at(i);
        auto& ldm_buffer = ldm_buffers.at(i);
        std::vector<DetectionWithLandmark> results;

        auto h = buffer.shape.dims.at(2) + buffer.shape.padding.at(2);
        auto w = buffer.shape.dims.at(3) + buffer.shape.padding.at(3);
        auto c = buffer.shape.dims.at(1) + buffer.shape.padding.at(1);
        auto batch_offset = h * w * c;
        float* data = reinterpret_cast<float*>(buffer.data) + batch_idx * batch_offset;

        auto ldm_h = ldm_buffer.shape.dims.at(2) + ldm_buffer.shape.padding.at(2);
        auto ldm_w = ldm_buffer.shape.dims.at(3) + ldm_buffer.shape.padding.at(3);
        auto ldm_c = ldm_buffer.shape.dims.at(1) + ldm_buffer.shape.padding.at(1);
        auto ldm_batch_offset = ldm_h * ldm_w * ldm_c;
        float* ldm_data = reinterpret_cast<float*>(ldm_buffer.data) + batch_idx * ldm_batch_offset;

        get_retinaface_decode_detections(data, ldm_data, num_anchor.at(i), kYoloV3CellSize, kRetinafaceLdmSize,
                                         &results, num_class, w, h, iw, ih, select_thre, nullptr, 1, 0);
        pre_output.insert(pre_output.end(), results.begin(), results.end());
    }

    if (nms_thre > 0.f) {
        do_nms_ldm_sort(pre_output.data(), pre_output.size(), num_class, nms_thre);
    }
    for (const auto& det : pre_output) {
        if (det.prob[det.class_idx] > select_thre) {
            output->emplace_back(det);
        }
    }
}

void YoloV3BatchCellPostProcess(IOBuffers* buffers, void* outputs, int32_t batch_idx, int32_t num_class,
                                std::vector<int32_t> num_anchor, int32_t ih, int32_t iw, float nms_thre,
                                float select_thre) {
    if ((not buffers) or (not outputs)) {
        return;
    }
    auto output_ptr = reinterpret_cast<BatchDetectionResults*>(outputs);
    auto& cur_ptr = output_ptr->at(batch_idx);

    cur_ptr->clear();
    std::vector<DetectionResult> pre_output;
    for (auto i = 0; i < buffers->size(); ++i) {
        auto& buffer = buffers->at(i);
        std::vector<DetectionResult> results;

        auto h = buffer.shape.dims.at(1) + buffer.shape.padding.at(1);
        auto w = buffer.shape.dims.at(2) + buffer.shape.padding.at(2);
        auto c = buffer.shape.dims.at(3) + buffer.shape.padding.at(3);
        auto batch_offset = h * w * c;
        float* data = reinterpret_cast<float*>(buffer.data) + batch_idx * batch_offset;

        get_yolo_decode_detections(data, num_anchor.at(i), kYoloV3CellSize, &results, num_class, w, h, iw, ih,
                                   select_thre, nullptr, 1, 0);
        pre_output.insert(pre_output.end(), results.begin(), results.end());
    }

    if (nms_thre > 0.f) {
        do_nms_sort(pre_output.data(), pre_output.size(), num_class, nms_thre);
    }

    for (const auto& det : pre_output) {
        if (det.prob[det.class_idx] > select_thre) {
            cur_ptr->emplace_back(det);
        }
    }
}

void PrintDetectionResult(const std::vector<DetectionResult>& det_outputs, int32_t test_h, int32_t test_w) {
    std::cout << "Total detection output size: " << det_outputs.size() << std::endl;
    for (const auto& det : det_outputs) {
        printf("%s: %.0f%%", coco_labels_tab[det.class_idx], det.prob[det.class_idx] * 100);
        printf("\t(left_x: %4.0f   top_y: %4.0f   width: %4.0f   height: %4.0f)\n",
               round((det.bbox.x - det.bbox.w / 2) * test_w), round((det.bbox.y - det.bbox.h / 2) * test_h),
               round(det.bbox.w * test_w), round(det.bbox.h * test_h));
    }
}

void PrintRetinafaceDetectionResult(const std::vector<DetectionWithLandmark>& det_outputs, int32_t test_h,
                                    int32_t test_w) {
    std::cout << "Total detection output size: " << det_outputs.size() << std::endl;
    for (const auto& det : det_outputs) {
        printf("face: %.0f%%", det.prob[det.class_idx] * 100);
        printf("\t(left_x: %4.0f   top_y: %4.0f   width: %4.0f   height: %4.0f)\n",
               round((det.bbox.x - det.bbox.w / 2) * test_w), round((det.bbox.y - det.bbox.h / 2) * test_h),
               round(det.bbox.w * test_w), round(det.bbox.h * test_h));
    }
}

void SaveRetinaFaceDetectionResult(const std::string output_dir, const std::string new_file,
                                   const std::string file_name, const std::vector<DetectionWithLandmark>& det_outputs,
                                   int net_h, int net_w, float scale) {
    std::string det_file = output_dir + new_file;
    std::ofstream ofs(det_file);
    if (not ofs.is_open()) {
        return;
    }

    int det_num = static_cast<int>(det_outputs.size());

    ofs << file_name.c_str() << std::endl;
    ofs << det_num << std::endl;
    for (auto det : det_outputs) {
        auto score = det.prob[det.class_idx];
        auto lt_x = (det.bbox.x - det.bbox.w / 2) * net_w / scale;
        auto lt_y = (det.bbox.y - det.bbox.h / 2) * net_h / scale;
        auto w = det.bbox.w * net_w / scale;
        auto h = det.bbox.h * net_h / scale;

        ofs << lt_x << " " << lt_y << " " << w << " " << h << " " << score << std::endl;
    }
}
