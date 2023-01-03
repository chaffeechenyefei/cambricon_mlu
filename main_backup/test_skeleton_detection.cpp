#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <assert.h>
#include <thread>
#include <stdio.h>
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
 * ./test_skeleton xxx.jpg 1
 * exe {imagename} {fake_yuv if given}
 * */
int main(int argc, char* argv[]) {
    string imgname = "test.jpg";
    if(argc>=2){
        string _tmp(argv[1]);
        imgname = _tmp;
    }
    bool fake_yuv = false;
    if(argc>=3){
        cout << "## use fake yuv" << endl;
        fake_yuv = true;
    }

    int objIdx = 2;
    int skIdx = 3;
    AlgoAPISPtr apiObjDetector = ucloud::AICoreFactory::getAlgoAPI(config[objIdx].apiName);
    AlgoAPISPtr apiSkDetector = ucloud::AICoreFactory::getAlgoAPI(config[skIdx].apiName);
    RET_CODE ret;
    ret = apiObjDetector->init(config[objIdx].modelName);
    cout << "apiObjDetector init return " << ret << endl;
    ret = apiSkDetector->init(config[skIdx].modelName);
    cout << "apiSkDetector init return " << ret << endl;

    bool use_yuv = false;
    unsigned char* ptrImg = nullptr;
    int width, height, stride;
    TvaiImageFormat inpFmt;
    int inpSz;
    if(imgname.find(".yuv")>=0 && imgname.find(".yuv")!=std::string::npos){
        std::cout << "yuv image input" << std::endl;
        inpFmt = TVAI_IMAGE_FORMAT_NV21;
        width = 1920;
        height = 1080;
        ptrImg = yuv_reader(imgname, width, height);
        inpSz = 3*width*height/2*sizeof(unsigned char);
        stride = width;
        use_yuv = true;
    }
    bool use_rgb = false;
    if(imgname.find(".rgb")>=0 && imgname.find(".rgb")!=std::string::npos){
        inpFmt = TVAI_IMAGE_FORMAT_RGB;
        width = 256;
        height = 256;
        ptrImg = rgb_reader(imgname, width, height);
        inpSz = 3*width*height*sizeof(unsigned char);
        std::cout << "rgb image input" << std::endl; 
        stride = width;
        use_rgb = true;
    }
    if(!use_yuv&&!use_rgb){
        if(fake_yuv){
            inpFmt = TVAI_IMAGE_FORMAT_NV21;
            ptrImg = readImg_to_NV21(imgname, width, height, stride );
            std::cout << "fake yuv input: " << width << ", " << height << ", "<< stride << ", "<< std::endl; 
            inpSz = 3*stride*height/2*sizeof(unsigned char);
        } else {
            inpFmt = TVAI_IMAGE_FORMAT_BGR;
            ptrImg = readImg(imgname, width, height);
            inpSz = 3*width*height*sizeof(unsigned char);
            stride = width;
        }

    }

    TvaiImage inpImg(inpFmt,width,height,stride,ptrImg,inpSz);
    //names: [ 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor' ]
    // CLS_TYPE yolov5s_conv_9[] = {CLS_TYPE::PEDESTRIAN, CLS_TYPE::NONCAR, CLS_TYPE::CAR, CLS_TYPE::CAR, CLS_TYPE::CAR, CLS_TYPE::NONCAR, CLS_TYPE::NONCAR, CLS_TYPE::CAR, CLS_TYPE::NONCAR};
    vector<CLS_TYPE> class_to_be_detected;

    ucloud::TvaiResolution maxTraget={0,0};
    ucloud::TvaiResolution minTarget={0,0};
    std::vector<ucloud::TvaiRect> pRoi;

    VecObjBBox objbboxes;
    VecObjBBox pedbboxes;
    ret = apiObjDetector->get_class_type(class_to_be_detected);
    cout << "Class will be detected: ";
    for (int i = 0; i < class_to_be_detected.size(); i++ ){
        cout << class_to_be_detected[i] << ", ";
    }
    cout << endl;
    ret = apiObjDetector->run(inpImg, objbboxes);
    cout << "apiObjDetector->run return " << ret << endl;
    cout << "# " << objbboxes.size() << "obj detected" << endl;
    for (int i = 0 ; i < objbboxes.size(); i++ ){
        if(objbboxes[i].objtype == CLS_TYPE::PEDESTRIAN)
            pedbboxes.push_back(objbboxes[i]);
    }

    ret = apiSkDetector->run(inpImg, pedbboxes);
    cout << "apiSkDetector->run return " << ret << endl;
    if(!pedbboxes.empty()){
        for(int i = 0; i < pedbboxes.size(); i++){
            cout << "x: " << pedbboxes[i].rect.x 
            << ", y: " << pedbboxes[i].rect.y 
            << ", w: " << pedbboxes[i].rect.width 
            << ", h: " << pedbboxes[i].rect.height 
            << ", x0: " << pedbboxes[i].x0 
            << ", y0: " << pedbboxes[i].y0
            << ", x1: " << pedbboxes[i].x1
            << ", y1: " << pedbboxes[i].y1
            << endl;
        }

    }
    
    if(!use_yuv){
        if(fake_yuv){
            freeImg(&ptrImg);
            ptrImg = readImg(imgname, width, height);
        }
        drawImg(ptrImg, width, height, pedbboxes, true, true);
        writeImg( "result/test_skeleton_detection.jpg" , ptrImg, width, height,true);
    }

    freeImg(&ptrImg);
        
}