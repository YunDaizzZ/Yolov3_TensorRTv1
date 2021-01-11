#ifndef _YOLO_LAYER_H
#define _YOLO_LAYER_H

#include <iostream>
#include <vector>
#include "NvInfer.h"

namespace Yolo
{
    static constexpr int CHECK_COUNT = 3;
    static constexpr float IGNOR_THRESH = 0.1f;
    static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
    static constexpr int CLASS_NUM = 80;
    static constexpr int BATCH_SIZE = 1;
    static constexpr int INPUT_H = 416;
    static constexpr int INPUT_W = 416;

    struct YoloKernel
    {
        int width;
        int height;
        float anchors[CHECK_COUNT*2];
    };

    static constexpr YoloKernel yolo1 = {
        INPUT_W / 32,
        INPUT_H / 32,
        {116,90,  156,198,  373,326}
    };
    static constexpr YoloKernel yolo2 = {
        INPUT_W / 16,
        INPUT_H / 16,
        {30,61,  62,45,  59,119}
    };
    static constexpr YoloKernel yolo3 = {
        INPUT_W / 8,
        INPUT_H / 8,
        {10,13,  16,30,  33,23}
    };

    static constexpr int LOCATIONS = 4;
    struct alignas(float) Detection{
        // x y w h
        float bbox[LOCATIONS];
        float det_confidence;
        float class_id;
        float class_confidence;
    };
}

#endif