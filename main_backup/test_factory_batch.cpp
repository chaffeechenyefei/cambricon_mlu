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
  APIConfig(AlgoAPIName::BATCH_GENERAL_DETECTOR, "yolov5s-conv-9_auto_736x416_mlu220_bs8c4_fp16.cambricon")
};
#else//mlu270
vector<APIConfig> config = {
  APIConfig(AlgoAPIName::BATCH_GENERAL_DETECTOR, "/project/workspace/samples/yolov5/mlu270/yolov5s-conv-9_736x416_mlu270_bs8c4_fp16.cambricon")
};
#endif

/**
 * ./test_factory_batch
 */
int main(int argc, char* argv[]) {
    int batchSZ = 8;
    int apiIdx = 0;
    vector<string> imgnames = {"web001.jpg", "web003.jpg","web004.jpg","web005.jpg","web006.jpg","test.jpg", "fail.jpg","test.jpg"};

    cout << "using #" << config[apiIdx].apiName << " api" << endl;
    AlgoAPISPtr apiHandlePtr = AICoreFactory::getAlgoAPI(config[apiIdx].apiName);
    RET_CODE ret;
    ret = apiHandlePtr->init(config[apiIdx].modelName);
    cout << "init return " << ret << endl;

    unsigned char* ptrImg = nullptr;
    int width, height, stride;

    BatchImageIN batchTvimage;
    BatchBBoxOUT batchBBox;
    //用于画结果
    vector<unsigned char*> batchPtr;
    vector<int> batchW;
    vector<int> batchH;
    for(int i = 0; i < batchSZ; i++ ){
        ptrImg = readImg_to_NV21(imgnames[i], width, height, stride );
        TvaiImage tvimage(TVAI_IMAGE_FORMAT_NV21, width, height, stride, ptrImg, 3*stride*height/2*sizeof(unsigned char));
        batchTvimage.push_back(tvimage);
        ptrImg = readImg(imgnames[i], width, height);
        batchPtr.push_back(ptrImg);
        batchW.push_back(width);
        batchH.push_back(height);
    }
    vector<CLS_TYPE> class_type_to_be_detected;

    auto start = chrono::system_clock::now();
    //TODO
    auto end = chrono::system_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end-start);
    double tm_cost = double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den;

    ret = apiHandlePtr->get_class_type(class_type_to_be_detected);
    cout << "get_class_type return " << ret << endl;
    cout << "Class will be detected: ";
    for (int i = 0; i < class_type_to_be_detected.size(); i++ ){
        cout << class_type_to_be_detected[i] << ", ";
    }
    cout << endl;
    start = chrono::system_clock::now();
    ret = apiHandlePtr->run_batch(batchTvimage, batchBBox);
    end = chrono::system_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end-start);
    tm_cost = double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den;
    cout << "run_batch return " << ret << endl;
    cout << tm_cost << "s cost, " << tm_cost/batchTvimage.size() << "s per image"<< endl;

    cout << "batch size: " << batchBBox.size() << " returned" << endl;
    for (int i = 0 ; i < batchBBox.size(); i++ ){
        cout << batchBBox[i].size() <<" detected" << endl;
        for(int j = 0; j < batchBBox[i].size(); j++ ){
            if(j==0){
                cout << "#" << batchBBox[i][j].objtype << " : " << batchBBox[i][j].confidence << endl;
            }
        }
        drawImg(batchPtr[i], batchW[i], batchH[i], batchBBox[i], true, true);
        writeImg("result/batch_test.jpg", batchPtr[i] , batchW[i], batchH[i], true);
    }


    for (int i = 0; i < batchTvimage.size(); i++ ){
        free(batchTvimage[i].pData);
        free(batchPtr[i]);
    }

}