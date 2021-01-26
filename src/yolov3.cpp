#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <assert.h>
#include <cstdlib>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "cuda_runtime_api.h"
#include "utils.h"
#include "logging.h"
#include "yololayer.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define CONF_THRESH 0.5

using namespace nvinfer1;
using namespace std;
using namespace std::chrono;
using namespace cv;

string labelFile = "/home/bhap/Desktop/yolov3/model_data/classescoco.txt";
string onnxFile = "/home/bhap/Desktop/yolov3/model_data/yolov3.onnx";
string engineFile = "/home/bhap/Desktop/yolov3/model_data/yolov3.trt";
string videoFile = "/home/bhap/Documents/Video/test8.MP4";

static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int outSize1 = Yolo::BATCH_SIZE * INPUT_H / 32 * INPUT_W / 32 * (CLASS_NUM + 5) * 3;
static const int outSize2 = Yolo::BATCH_SIZE * INPUT_H / 16 * INPUT_W / 16 * (CLASS_NUM + 5) * 3;
static const int outSize3 = Yolo::BATCH_SIZE * INPUT_H / 8 * INPUT_W / 8 * (CLASS_NUM + 5) * 3;

static Logger gLogger;

void correct_box(Mat& img, float bbox[4], vector<int>& tlbr) {
    int l, r, t, b;
    float r_w = INPUT_W / (img.cols * 1.0);
    float r_h = INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (INPUT_H - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f - (INPUT_H - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2] / 2.f - (INPUT_W - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f - (INPUT_W - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    tlbr.push_back(max(0, t));
    tlbr.push_back(max(0, l));
    tlbr.push_back(min(img.rows, b));
    tlbr.push_back(min(img.cols, r));
}

bool cmp(const Yolo::Detection& a, const Yolo::Detection& b) {
    return a.det_confidence > b.det_confidence;
}

float sigmoid(float in) {
    return 1.0f / (1.0f + expf(-in));
}

float* merge(float* out1, float* out2, float* out3, int bsize_out1, int bsize_out2, int bsize_out3) {
    float* out_total = new float[bsize_out1 + bsize_out2 + bsize_out3];

    for (int j = 0; j < bsize_out1; ++j) {
        int index = j;
        out_total[index] = out1[j];
    }
    for (int j = 0; j < bsize_out2; ++j) {
        int index = j + bsize_out1;
        out_total[index] = out2[j];
    }
    for (int j = 0; j < bsize_out3; ++j) {
        int index = j + bsize_out1 + bsize_out2;
        out_total[index] = out3[j];
    }

    return out_total;
}

float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        std::max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
        std::min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
        std::max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
        std::min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
    };

    if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);

    return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

void nms(vector<Yolo::Detection> detections, vector<Yolo::Detection>& res, float nms_thresh = NMS_THRESH) {
    std::map<float, std::vector<Yolo::Detection>> m;
    int length = detections.size();
    for (int i = 0; i < length; ++i) {
        if (detections[i].det_confidence <= CONF_THRESH)
            continue;
        Yolo::Detection det;
        memcpy(&det, &detections[i], 7 * sizeof(float));
        if (m.count(det.class_id) == 0) 
        {
            m.emplace(det.class_id, vector<Yolo::Detection>());
        }
            
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        auto& dets = it->second;
        sort(dets.begin(), dets.end(), cmp);
        for (size_t k = 0; k < dets.size(); ++k) {
            auto& item = dets[k];
            res.push_back(item);
            for (size_t n = k + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin() + n);
                    --n;
                }
            }
        }
    }
}

vector<Yolo::Detection> postProcess(float* output, int img_width, int img_height) {
    vector<Yolo::Detection> detections;
    vector<Yolo::Detection> res_tmp;
    vector<Yolo::Detection> res;
    int offset = 0; 

    vector<Yolo::YoloKernel> anchors;
    anchors.push_back(Yolo::yolo1);
    anchors.push_back(Yolo::yolo2);
    anchors.push_back(Yolo::yolo3);
    
    vector<vector<int>> output_shape;
    vector<vector<int>> shapes;
    for (int i = 0; i < 3; ++i) {
        output_shape.push_back({1, 3 * (CLASS_NUM + 5), anchors[i].height, anchors[i].width});
        shapes.push_back({anchors[i].height, anchors[i].width, 3, (CLASS_NUM + 5)});
    }

    // 维度转换 (转换前同一个目标的15个预测信息不连续)
    float * transposed_output = new float[outSize1 + outSize2 + outSize3];
	float * transposed_output_t = transposed_output;
	for (int i = 0; i < 3; i++) {
		auto shape = output_shape[i];
		int chw = shape[1] * shape[2] * shape[3];
		int hw = shape[2] * shape[3];
		for (int n = 0; n < shape[0]; n++) {
			int offset_n = offset + n * chw;
			for (int h = 0; h < shape[2]; h++) {
				for (int w = 0; w < shape[3]; w++) {
					int h_w = h * shape[3] + w;
					for (int c = 0; c < shape[1]; c++) {
						int offset_c = offset_n + hw * c + h_w;
						*transposed_output_t++ = output[offset_c];
					}
				}
			}
		}
		offset += shape[0] * chw;
	}

    offset = 0;
    // 解码
    for (int i = 0; i < 3; ++i) {
        auto masks = anchors[i].anchors;
        auto shape = shapes[i];

        for (int h = 0; h < shape[0]; ++h) {
            int offset_h = offset + h * shape[1] * shape[2] * shape[3];
            for (int w = 0; w < shape[1]; ++w) {
                int offset_w = offset_h + w * shape[2] * shape[3];
                for (int c = 0; c < shape[2]; ++c) {
                    int offset_c = offset_w + c * shape[3];
                    float* ptr = transposed_output + offset_c;

                    int class_id = 0;
                    float max_cls_prob = 0.0;
                    for (int n = 5; n < shape[3]; ++n) {
                        float p = sigmoid(ptr[n]);
                        if (p > max_cls_prob) {
                            max_cls_prob = p;
                            class_id = n - 5;
                        }
                    }

                    float obj_prob = sigmoid(ptr[4]);
                    if (max_cls_prob < Yolo::IGNOR_THRESH || obj_prob < Yolo::IGNOR_THRESH)
                        continue;

                    ptr[0] = (w + sigmoid(ptr[0])) * INPUT_W / shape[0];
                    ptr[1] = (h + sigmoid(ptr[1])) * INPUT_H / shape[1];
                    ptr[2] = expf(ptr[2]) * masks[2 * c];
                    ptr[3] = expf(ptr[3]) * masks[2 * c + 1];

                    Yolo::Detection det;
                    det.bbox[0] = ptr[0];
                    det.bbox[1] = ptr[1];
                    det.bbox[2] = ptr[2];
                    det.bbox[3] = ptr[3];
                    det.det_confidence = obj_prob;
                    det.class_id = class_id;
                    det.class_confidence = max_cls_prob;
                    detections.push_back(det);
                }
            }
        }
        offset += shape[0] * shape[1] * shape[2] * shape[3];
    }
    delete[]transposed_output;

    // 非极大抑制
    nms(detections, res_tmp);
    for (size_t i = 0; i < res_tmp.size(); ++i) {
        if (res_tmp[i].class_confidence * res_tmp[i].det_confidence > CONF_THRESH) {
            res.push_back(res_tmp[i]);
        }
    }

    return res;
}

bool readTrtFile(const std::string& engineFile, IHostMemory*& trtModelStream) {
    std::fstream file;
    cout << "loading filename from: " << engineFile << endl;
    file.open(engineFile, ios::binary | ios::in);
    file.seekg(0, ios::end);
    int length = file.tellg();
    file.seekg(0, ios::beg);
    std::unique_ptr<char[]> data(new char[length]);
    file.read(data.get(), length);
	file.close();
    cout << "load engine done" << endl;
    std::cout << "deserializing" << endl;
    nvinfer1::IRuntime* trtRuntime = createInferRuntime(gLogger.getTRTLogger());
    ICudaEngine* engine = trtRuntime->deserializeCudaEngine(data.get(), length);
    cout << "deserialize done" << endl;
	trtModelStream = engine->serialize();

    return true;
}

bool onnxToTRTModel(const std::string& modelFile, const std::string& filename, IHostMemory*& trtModelStream) {
    // modelFile: onnx文件名
    // filename: TensorRT引擎名
    // trtModelStream: output buffer for the TensorRT model
    int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;

    // ！！！在TensorRT7.0中要确定batchsize（网上说的大概是这种意思吧，也没太看明白），网上代码大多用6.0写的不一样
    // 参考7.0下的sample里的xxxonnxmnist.cpp，且下面使用的是createNetworkV2()函数
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    // 创建builder
    IBuilder* builder = createInferBuilder(gLogger);  // 创建构建器(即指向Ibuilder类型对象的指针)
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);  
    // 等价于bulider.createNetwork(),通过Ibulider定义的名为creatNetwork()方法，创建INetworkDefinition的对象，ntework这个指针指向这个对象

    // 创建解析器解析onnx模型
    auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());
    
    //判断是否成功解析ONNX模型
	if (!parser->parseFromFile(modelFile.c_str(), static_cast<int>(gLogger.getReportableSeverity())))
	{
		cout << "Failure while parsing ONNX file" << endl;
        return false;
	}

    // 建立推理引擎
    builder->setMaxBatchSize(Yolo::BATCH_SIZE);
    builder->setMaxWorkspaceSize(1 << 30);
    builder->setFp16Mode(true);

    cout << "start building engine" << endl;
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    cout << "build engine done" << endl;
    assert(engine);

    // 销毁模型解释器
    parser->destroy();

    // 序列化引擎
    trtModelStream = engine->serialize();

    // 保存引擎
    nvinfer1::IHostMemory* data = engine->serialize();
    std::ofstream file;
    file.open(filename, std::ios::binary | std::ios::out);
    cout << "writing engine file..." << endl;
    file.write((const char*)data->data(), data->size());
    cout << "save engine file done" << endl;
    file.close();

    // 销毁所有相关的东西
    engine->destroy();
    network->destroy();
    builder->destroy();

    return true;
}

void doInference(IExecutionContext& context, float* input, float* out1, float* out2, float* out3, int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 4);
    void* buffers[4];

    int inputSize = batchSize * 3 * INPUT_H * INPUT_W;

    CUDA_CHECK(cudaMalloc(&buffers[0], inputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[1], outSize1 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[2], outSize2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[3], outSize3 * sizeof(float)));

    // 创建CUDA流以执行此推断
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, inputSize * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(out1, buffers[1], outSize1 * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK(cudaMemcpyAsync(out2, buffers[2], outSize2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(out3, buffers[3], outSize3 * sizeof(float), cudaMemcpyDeviceToHost, stream));

    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
	CUDA_CHECK(cudaFree(buffers[0]));
	CUDA_CHECK(cudaFree(buffers[1]));
	CUDA_CHECK(cudaFree(buffers[2]));
    CUDA_CHECK(cudaFree(buffers[3]));
}

int main(int argc, char** argv) {
    srand(111);
    vector<Scalar> colors;
    vector<string> class_names;
    for (int i = 0; i < CLASS_NUM; ++i) 
    {
        colors.push_back(Scalar(rand()%256, rand()%256, rand()%256));
    }
    string s;
    ifstream inf(labelFile);
    while (getline(inf, s)) {
        class_names.push_back(s);
    }

    cudaSetDevice(DEVICE);

    IHostMemory* trtModelStream{nullptr};
    std::fstream existEngine;
    existEngine.open(engineFile, std::ios::in);
    if (existEngine)
    {
        readTrtFile(engineFile, trtModelStream);
        assert(trtModelStream != nullptr);
    }
    else
    {
        onnxToTRTModel(onnxFile, engineFile, trtModelStream);
        assert(trtModelStream != nullptr);
    }

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size());
    assert(engine != nullptr);

    // 创建推理引擎
    trtModelStream->destroy();
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    float data[Yolo::BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    float output1[outSize1];
    float output2[outSize2];
    float output3[outSize3];
    float *prob = NULL;

    double fps = 0.0;

    VideoCapture capture(0);
    capture.set(CAP_PROP_FRAME_HEIGHT, 960);
    capture.set(CAP_PROP_FRAME_WIDTH, 720);
    
    // VideoCapture capture(videoFile);

    while (1) {
        auto start = system_clock::now();

        Mat img;
        capture >> img;
        Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H);

        for (int i = 0; i < INPUT_H * INPUT_W; ++i) {
            data[i] = pr_img.at<Vec3b>(i)[2] / 255.0;
            data[i + INPUT_H * INPUT_W] = pr_img.at<Vec3b>(i)[1] / 255.0;
            data[i + 2 * INPUT_H * INPUT_W] = pr_img.at<Vec3b>(i)[0] / 255.0;
        }  //opencv里面有个BGR和RGB的转化

        // Run inference
        doInference(*context, data, output1, output2, output3, Yolo::BATCH_SIZE);
        prob = merge(output1, output2, output3, outSize1, outSize2, outSize3);

        vector<Yolo::Detection> res;
        // res里bbox为xywh
        res = postProcess(prob, img.cols, img.rows);
	delete[] prob;

        string predicted_class, label, score;
        for (size_t j = 0; j < res.size(); ++j) {
            vector<int> tlbr;
            predicted_class = class_names[(int)res[j].class_id];
            score = format("%.2f", (res[j].det_confidence * res[j].class_confidence));
            label = predicted_class + " " + score;

            correct_box(img, res[j].bbox, tlbr);
            rectangle(img, Point(tlbr[1], tlbr[0]), Point(tlbr[3], tlbr[2]), colors[(int)res[j].class_id], 2, 8);
            int l_w = label.length() * 7 + 30;
            int l_h = 11;
            rectangle(img, Point(tlbr[1], tlbr[0] - l_h), Point(tlbr[1] + l_w, tlbr[0]), colors[(int)res[j].class_id], -1, 8);
            putText(img, label, Point(tlbr[1], tlbr[0] - 1), FONT_HERSHEY_TRIPLEX, 0.5, Scalar(0, 0, 0), 1, 8);
        }
        string label_fps;
        auto duration = duration_cast<microseconds>(system_clock::now() - start);
        double duration_s = (double)(duration.count()) * microseconds::period::num / microseconds::period::den;
        fps = (fps + 1. / duration_s) / 2;
        label_fps = "Fps= " + format("%.2f", fps);
        putText(img, label_fps, Point(10, 40), FONT_HERSHEY_TRIPLEX, 1, Scalar(0, 255, 0), 1, 8);

        // cv::resize(img, img, cv::Size(1280, 960), cv::INTER_LINEAR);

        imshow("yolov3", img);
        waitKey(10);
    }

    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}
