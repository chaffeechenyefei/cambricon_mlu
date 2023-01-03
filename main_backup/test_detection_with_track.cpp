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
#include <memory.h>

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
  APIConfig(AlgoAPIName::GENERAL_TRACKOR, "feature_extract_4c4b_argb_220_v1.5.0.cambricon"),
  APIConfig(AlgoAPIName::FACE_EXTRACTOR, "resnet101_112x112_mlu220_fp16.cambricon"),
  APIConfig(AlgoAPIName::GENERAL_DETECTOR, "yolov5s-conv-9_736x416_mlu220_bs1c1_fp16.cambricon"),
  APIConfig(AlgoAPIName::SKELETON_DETECTOR, "pose_resnet_50_256x192_mlu220_bs1c1_fp16.cambricon")
};
#else//mlu270
vector<APIConfig> config = {
  APIConfig(AlgoAPIName::FACE_DETECTOR, "/project/workspace/samples/mlu_videofacerec/weights/face_det/retinaface_736x416_mlu270.cambricon"),
  APIConfig(AlgoAPIName::GENERAL_TRACKOR, "/project/workspace/samples/cambricon_offline_repo/feature_extract_4c4b_argb_270_v1.5.0.cambricon"),
  APIConfig(AlgoAPIName::FACE_EXTRACTOR, "/project/workspace/samples/mlu_videofacerec/weights/face_rec/resnet101_mlu270.cambricon"),
  APIConfig(AlgoAPIName::GENERAL_DETECTOR, "/project/workspace/samples/yolov5/mlu270/yolov5s-conv-9_736x416_mlu270_bs1c1_fp16.cambricon"),
  APIConfig(AlgoAPIName::SKELETON_DETECTOR, "/project/workspace/samples/deep-high-resolution-net/mlu270/pose_resnet_50_256x192_mlu270_bs1c1_fp16.cambricon")
};
#endif

/**
 * ./test_factory {apiIdx}
 */
int main(int argc, char* argv[]) {
    int apiIdx = 3;
    if(argc >= 2){
        apiIdx = atoi(argv[1]);
        assert(apiIdx < config.size());
    }
    AlgoAPISPtr apiHandlePtr = AICoreFactory::getAlgoAPI(config[apiIdx].apiName);
    RET_CODE ret;
    ret = apiHandlePtr->init(config[apiIdx].modelName, config[1].modelName);
    // ret = apiHandlePtr->init(config[apiIdx].modelName);
    std::cout << "init return " << ret << std::endl;

    ucloud::TvaiResolution maxTraget={0,0};
    ucloud::TvaiResolution minTarget={0,0};
    std::vector<ucloud::TvaiRect> pRoi;

    // ret = apiHandlePtr->set_param(0.8,0.2, maxTraget, minTarget , pRoi);
    // ret = apiHandlePtr->set_param(0.4,0.6, maxTraget, minTarget , pRoi);
    std::cout << "set param return " << ret << std::endl;

    auto start = chrono::system_clock::now();
    //TODO
    auto end = chrono::system_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end-start);
    double tm_cost = double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den;

    std::string datapath = "monitor/";
    std::ifstream infile;
    std::string filename = datapath + "list.txt";

    infile.open(filename, std::ios::in);
    std::string imgname;
    std::vector<string> vec_imgnames;
    bool use_yuv = false;
    while(infile >> imgname){
        if(imgname.find(".yuv")>=0 && imgname.find(".yuv")!=std::string::npos){
            use_yuv = true;
        }
        vec_imgnames.push_back(imgname);
    }

    double total_tm_cost = 0;
    for(auto iter=vec_imgnames.begin(); iter!=vec_imgnames.end(); iter++){
        unsigned char* ptrImg = nullptr;
        unsigned char* ptrShow = nullptr;
        int width, height, stride;
        int _width, _height, _stride;
        int inpSz;
        TvaiImageFormat inpFmt = TVAI_IMAGE_FORMAT_NV21;
        std::string imgname_full = datapath + *iter;
        if(!use_yuv){
            ptrImg = readImg_to_NV21(imgname_full, width, height, stride );
            ptrShow = readImg(imgname_full, _width, _height);
            _stride = _width;
        }
        else{
            width = 1920; height = 1080; stride = width;
            ptrImg = yuv_reader(imgname_full, width, height);
        }
        std::shared_ptr<unsigned char> u_ptrShow;
        std::shared_ptr<unsigned char> u_ptrImg;
        u_ptrImg.reset(ptrImg, free);
        u_ptrShow.reset(ptrShow, free);
        inpSz = 3*stride*height/2*sizeof(unsigned char);
        TvaiImage inpImg(inpFmt,width,height,stride,ptrImg,inpSz);
        VecObjBBox objbboxes;
        
        start = chrono::system_clock::now();
        ret = apiHandlePtr->run(inpImg, objbboxes);
        end = chrono::system_clock::now();
        duration = chrono::duration_cast<chrono::microseconds>(end-start);
        tm_cost = double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den;
        if(ret!=RET_CODE::SUCCESS)
            std::cout << "apiHandlePtr->run return " << ret << std::endl;
        int track_num = 0;
        std::cout << "track id = [";
        for (auto iter_box = objbboxes.begin(); iter_box != objbboxes.end(); iter_box++ ){
            if(iter_box->track_id >= 0) std::cout << iter_box->track_id << ", ";
            track_num += (iter_box->track_id >= 0) ? 1:0;
        }
        std::cout << "]" << std::endl;
        std::cout << "# " << objbboxes.size() << " faces detected; " <<  track_num << " faces tracked." << std::endl;
        total_tm_cost += tm_cost;

        if(!use_yuv){
            drawImg(u_ptrShow.get(), _width, _height, objbboxes, true, true);
            std::string savename = "monitor_result/" + *iter;
            // std::cout << "...." << savename << std::endl;
            writeImg( savename ,u_ptrShow.get(), _width, _height, true);
        }



    }
}