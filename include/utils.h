#ifndef __TRT_UTILS_H_
#define __TRT_UTILS_H_

#include <iostream>
#include <vector>
#include <algorithm>
#include <cudnn.h>
#include <dirent.h>
#include <opencv2/opencv.hpp>

#ifndef CUDA_CHECK

#define CUDA_CHECK(callstr)                                                                    \
    {                                                                                          \
        cudaError_t error_code = callstr;                                                      \
        if (error_code != cudaSuccess) {                                                       \
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
            assert(0);                                                                         \
        }                                                                                      \
    }

#endif

static inline cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h) {
    int iw, ih, nw, nh;
    float scale, r_w, r_h;
    iw = img.size().width;
    ih = img.size().height;
    r_w = input_w / (iw * 1.0);
    r_h = input_h / (ih * 1.0);
    scale = std::min(r_w, r_h);
    nw = int(iw * scale);
    nh = int(ih * scale);

    cv::Mat new_img(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Mat image;

    cv::resize(img, image, cv::Size(nw, nh), cv::INTER_LINEAR);
    image.copyTo(new_img(cv::Range(floor((input_h - nh) / 2), floor((input_h - nh) / 2) + nh), cv::Range(floor((input_w - nw) / 2), floor((input_w - nw) / 2) + nw)));

    return new_img;
}

#endif
