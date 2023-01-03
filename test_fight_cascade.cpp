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
#include "config.hpp"
using namespace std;
using namespace ucloud;

/**
 * ./test_fight_cascade
 */
int main(int argc, char* argv[]) {
    float threshold, nms_threshold;
    AlgoAPIName mainApiName, subApiName;
    ucloud::TvaiResolution maxTarget={0,0};
    ucloud::TvaiResolution minTarget={0,0};
    std::vector<ucloud::TvaiRect> pRoi;
    int use_batch = 0;
    task_parser(TASKNAME::FIGHT_DET, threshold,nms_threshold, mainApiName, subApiName, use_batch);
    AlgoAPISPtr detectHandle = AICoreFactory::getAlgoAPI(subApiName);
    RET_CODE ret = detectHandle->init(modelInfo[subApiName]);
    std::cout << "@@ detect init return = " << ret << std::endl;
    ret = detectHandle->set_param(0.3, 0.1, maxTarget, minTarget, pRoi);

    AlgoAPISPtr mainHandle = AICoreFactory::getAlgoAPI(mainApiName);
    ret = mainHandle->init(modelInfo[mainApiName]);
    std::cout << "@@ action fight detect model init return = " << ret << std::endl;
    ret = mainHandle->set_param(threshold, nms_threshold, maxTarget, minTarget, pRoi);

    std::string datapath = "fight/";
    std::ifstream infile;
    std::string filename = datapath + "list.txt";

    //DATA
    infile.open(filename, std::ios::in);
    std::string imgname;
    std::vector<string> vec_imgnames;
    bool use_yuv = false;
    while(infile >> imgname){
        if(imgname.find(".yuv")>=0 && imgname.find(".yuv")!=std::string::npos){
            use_yuv = true;
        }
        if(imgname.find("fi003")>=0 && imgname.find("fi003")!=std::string::npos){
            vec_imgnames.push_back(imgname);
        }
    }

    double total_tm_cost = 0;
    BatchImageIN batch_tvimages;
    BatchBBoxIN batch_bboxes;
    const int batch_size_in = use_batch;
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
        inpSz = 3*stride*height/2*sizeof(unsigned char);
        TvaiImage inpImg(inpFmt,width,height,stride,ptrImg,inpSz);
        batch_tvimages.push_back(inpImg);

        VecObjBBox box;
        ret = detectHandle->run(inpImg,box);
        batch_bboxes.push_back(box);

        if(batch_tvimages.size()==batch_size_in){
            VecObjBBox result;
            ret = mainHandle->run(batch_tvimages,batch_bboxes ,result);
            cout << "@@ run return = " << ret << endl;
            if(!result.empty()){
                cout << "fight prob = ";
                for(auto _iter=result.begin(); _iter!=result.end(); _iter++){
                    cout << _iter->confidence << ", ";
                }
                cout << endl;
            }
            for(auto iter=batch_tvimages.begin(); iter!=batch_tvimages.end(); iter++ ){
                free(iter->pData);
            }
            batch_tvimages.clear();
            batch_bboxes.clear();
        }
    }

    for(auto iter=batch_tvimages.begin(); iter!=batch_tvimages.end(); iter++ ){
        free(iter->pData);
    }

    batch_bboxes.clear();
    batch_tvimages.clear();

}