#include <sys/time.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <assert.h>
#include <string.h>

#include "libai_core.hpp"

using namespace std;
using namespace ucloud;

typedef struct APIConfig{
    AlgoAPIName apiName;
    std::string modelName;
    APIConfig(AlgoAPIName _apiName, std::string _modelName):apiName(_apiName),modelName(_modelName){};
}APIConfig;

#ifdef  MLU220
vector<APIConfig> config = {
  APIConfig(AlgoAPIName::FACE_DETECTOR, "retinaface_736x416_mlu220.cambricon"),
  APIConfig(AlgoAPIName::FACE_EXTRACTOR, "resnet101_112x112_mlu220_fp16.cambricon"),
  APIConfig(AlgoAPIName::GENERAL_DETECTOR, "yolov5s-conv-9_736x416_mlu220_bs1c1_fp16.cambricon"),
  APIConfig(AlgoAPIName::SKELETON_DETECTOR, "pose_resnet_50_256x192_mlu220_bs1c1_fp16.cambricon")
};
#else//mlu270
vector<APIConfig> config = {
  APIConfig(AlgoAPIName::FACE_DETECTOR, "/project/workspace/samples/mlu_videofacerec/weights/face_det/retinaface_736x416_mlu270.cambricon"),
  APIConfig(AlgoAPIName::FACE_EXTRACTOR, "/project/workspace/samples/mlu_videofacerec/weights/face_rec/resnet101_mlu270.cambricon"),
  APIConfig(AlgoAPIName::GENERAL_DETECTOR, "/project/workspace/samples/yolov5/mlu270/yolov5s-conv-9_736x416_mlu270_bs1c1_fp16.cambricon"),
  APIConfig(AlgoAPIName::SKELETON_DETECTOR, "/project/workspace/samples/deep-high-resolution-net/mlu270/pose_resnet_50_256x192_mlu270_bs1c1_fp16.cambricon")
};
#endif

/**
 * ./test_factory_face_recognition {image name} {fake yuv if given} 
 **/
int main(int argc, char* argv[]) {
    bool use_fake_yuv = false;
    ucloud::RET_CODE ret;
    string imgname = "test.jpg";
    if(argc >= 2){
        string _imgname(argv[1]);
        imgname = _imgname;
    }
    if(argc >=3){
        use_fake_yuv = true;
    }
    cout << "reading: " << imgname << endl;
    bool use_yuv = false;
    
    if(imgname.find(".yuv")>=0 && imgname.find(".yuv")!=std::string::npos){
        std::cout << "yuv image input" << std::endl; 
        use_yuv = true;
    }
    bool use_rgb = false;
    if(imgname.find(".rgb")>=0 && imgname.find(".rgb")!=std::string::npos){
        std::cout << "rgb image input" << std::endl; 
        use_rgb = true;
    }

    //  检测相关参数
    float threshold = 0.8;
    float nms_threshold = 0.2;
    // 设置为0时, 判断条件不起作用
    ucloud::TvaiResolution maxTraget={0,0};
    ucloud::TvaiResolution minTarget={0,0};
    std::vector<ucloud::TvaiRect> pRoi;

    // 调用API接口
    cout << "create detection handle" << endl;
    AlgoAPISPtr ptrDetectorHandle = ucloud::AICoreFactory::getAlgoAPI(config[0].apiName);
    AlgoAPISPtr ptrExtractorHandle = ucloud::AICoreFactory::getAlgoAPI(config[1].apiName);
    cout << "load model" << endl;
    ret = ptrDetectorHandle->init(config[0].modelName);
    cout << "detector init return = " << ret << endl;
    ret = ptrExtractorHandle->init(config[1].modelName);
    cout << "extractor init return = " << ret << endl;
    ret = ptrDetectorHandle->set_param(threshold, nms_threshold, maxTraget, minTarget, pRoi);

    //  开始检测
    // 借用OpenCV读入数据, 并转为BGR数据输入
    TvaiImageFormat fmt;
    int sz;
    int width, height, stride;
    unsigned char* ptrImg = nullptr;
    unsigned char* ptrImg2 = nullptr;
    int width2, height2, stride2;
    if(!use_yuv){
        if (use_rgb){
            width = 256;
            height = 256;
            stride = width;
            ptrImg = ucloud::rgb_reader(imgname, width, height);
            ptrImg2 = (unsigned char*)malloc(width*height*3*sizeof(unsigned char));
            memcpy(ptrImg2, ptrImg, (width*height*3*sizeof(unsigned char)));
            fmt = TvaiImageFormat::TVAI_IMAGE_FORMAT_RGB;
            sz = width*height*3*sizeof(unsigned char);
            width2 = width; height2 = height; stride2 = stride;
        }
        else{
            if(use_fake_yuv){
                ptrImg = ucloud::readImg_to_NV21(imgname, width, height, stride);
                sz = stride*height/2*3*sizeof(unsigned char);
                ptrImg2 = ucloud::readImg(imgname,width2,height2);
                fmt = TvaiImageFormat::TVAI_IMAGE_FORMAT_NV21;
                stride2 = width2;
            } else{
                ptrImg = ucloud::readImg(imgname,width,height);
                stride = width;
                ptrImg2 = (unsigned char*)malloc(width*height*3*sizeof(unsigned char));
                memcpy(ptrImg2, ptrImg, (width*height*3*sizeof(unsigned char)));
                fmt = TvaiImageFormat::TVAI_IMAGE_FORMAT_BGR;
                sz = width*height*3*sizeof(unsigned char);
                width2 = width; height2 = height; stride2 = stride;
            }
        }
    } else {
        width = 1920;
        height = 1080;
        ptrImg = ucloud::yuv_reader(imgname, width, height);
        stride = width;
        fmt = TvaiImageFormat::TVAI_IMAGE_FORMAT_NV21;
        sz = height/2*3*width*sizeof(unsigned char);
    }
    if (ptrImg==nullptr){
        cout << "No image is found!!!" << endl;
        return -1;
    }

    VecObjBBox detectResult;
    cout << "run detection";
    ucloud::TvaiImage tvimage = {fmt, width, height, stride, ptrImg, sz};
    std::cout << "tvimage(WHS): " << tvimage.width << ", " << tvimage.height << ", " << tvimage.stride << std::endl;
    
    auto start = chrono::system_clock::now();
    ret = ptrDetectorHandle->run(tvimage, detectResult);
    auto end = chrono::system_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end-start);
    double tm_cost = double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den;
    cout << "detector return = " << ret << endl; 
    cout << "detection cost: " << tm_cost << "s" << endl;
    cout << detectResult.size() << " faces detected" << endl;
    //将检测结果返回到图像上
    if(!use_yuv){
        ucloud::drawImg(ptrImg2, width2, height2, detectResult, true);
        ucloud::writeImg("result/test_factory_face_recognition.jpg", ptrImg2 ,width2, height2,true);
        // ucloud::freeImg(&ptrImg);
    }


    cout << "run extraction" << endl;
    // if(!use_yuv){
    //     tvimage.pData = ptrImg2; //因为上面的图像画了框, 会干扰到特征提取, 所以用了副本
    //     // tvimage.format = ucloud::TvaiImageFormat::TVAI_IMAGE_FORMAT_RGB;
    // }
        

    std::vector<ucloud::TvaiFeature> extractorResult; 
    start = chrono::system_clock::now();
    ret = ptrExtractorHandle->run(tvimage,detectResult);
    cout <<  "extractor return = " << ret << endl;
    end = chrono::system_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end-start);
    tm_cost = double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den;
    cout << "extraction cost: " << tm_cost << "s, " << tm_cost/detectResult.size() << "s per face" << endl;

    for(int i = 0; i < detectResult.size(); i++ ){
        extractorResult.push_back(detectResult[i].feat);
    }
    //图像中出现的人脸进行内部相似度比较
    cout << "inner similarity" << endl;
    std::vector<std::vector<float>> similarityScore;
    ucloud::calcSimilarity(extractorResult,extractorResult,similarityScore);
    std::ofstream outfile;
    outfile.open("similarity.dat", std::ios::out | std::ios::trunc );
    for(int a = 0; a < similarityScore.size(); a++){
        for(int b = 0; b < similarityScore[a].size(); b++ ){
        outfile << similarityScore[a][b] << ", ";
        }
        outfile << "\r\n";
    }
    outfile.close();
    // releaseTvaiFeature(extractorResult);
    releaseVecObjBBox(detectResult);
    if (!use_yuv)
        ucloud::freeImg(&ptrImg2);
    else{
        free(ptrImg);
    }
}
