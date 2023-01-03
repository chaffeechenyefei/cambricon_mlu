#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <assert.h>
#include <thread>
#include <stdio.h>
#include <queue>
#include "libai_core.hpp"
// #include "config.hpp"
using namespace std;
using namespace ucloud;

/**
 *
 */
int main(int argc, char* argv[]) {
    float threshold, nms_threshold;
    ucloud::TvaiResolution maxTarget={0,0};
    ucloud::TvaiResolution minTarget={0,0};
    std::vector<ucloud::TvaiRect> pRoi;
    threshold = 0.2;
    nms_threshold = 0.2;
    AlgoAPISPtr algoHandle = AICoreFactory::getAlgoAPI(AlgoAPIName::MOD_DETECTOR);
    int batch_size_in = algoHandle->get_batchsize();
    //VIRTUAL
    //MLU220: diffunet_2022xxxx_736x416_mlu220_t2bs1c1_fp16_int8.cambricon
    //diffunet_2022xxxx_736x416_mlu270_t2bs1c1_fp16_int8.cambricon
    //====================================================================================
    //REAL
    //diffunet_20220106_736x416_mlu220_t2bs1c1_fp16_int8.cambricon
    //diffunet_20220106_736x416_mlu270_t2bs1c1_fp16_int8.cambricon
    //需要修改为mlu220的模型地址
    RET_CODE ret = algoHandle->init("/project/workspace/samples/3d_unet_virtual/mlu270/diffunet_2022xxxx_736x416_mlu270_t2bs1c1_fp16_int8.cambricon");
    cout << "@@ init return = " << ret << endl;
    ret = algoHandle->set_param(threshold, nms_threshold, maxTarget, minTarget, pRoi);

    //需要修改实际的数据地址
    std::string subcase = "tv04/";
    std::string datapath = "/project/data/gkpw/"+subcase;
    std::ifstream infile;
    std::string filename = datapath + "list.txt";

    //DATA
    infile.open(filename, std::ios::in);
    std::string imgname;
    std::vector<string> vec_imgnames;

    while(infile >> imgname){
        vec_imgnames.push_back(imgname);
    }

    double total_tm_cost = 0;
    BatchImageIN batch_tvimages;
    
    int step = 3;
    int step_cnt = 0;
    for(auto iter=vec_imgnames.begin(); iter!=vec_imgnames.end(); iter++){
        unsigned char* ptrImg = nullptr;
        unsigned char* ptrShow = nullptr;
        int width, height, stride;
        int _width, _height, _stride;
        int inpSz;
        TvaiImageFormat inpFmt = TVAI_IMAGE_FORMAT_NV21;
        std::string imgname_full = datapath + *iter;

        ptrImg = readImg_to_NV21(imgname_full, width, height, stride );
        ptrShow = readImg(imgname_full, _width, _height);
        _stride = _width;

        inpSz = 3*stride*height/2*sizeof(unsigned char);
        TvaiImage inpImg(inpFmt,width,height,stride,ptrImg,inpSz);

        batch_tvimages.push_back(inpImg);
        if(batch_tvimages.size()==batch_size_in){
            //ALGO
            BatchBBoxIN batch_bboxes;
            VecObjBBox result;
            /**
             * 结果中包含 FALLING_OBJ, FALLING_OBJ_UNCERTAIN, 只需要展示FALLING_OBJ的结果. FALLING_OBJ_UNCERTAIN表示有移动物体, 但是轨迹不确定, 不能认定抛物.
             */
            ret = algoHandle->run(batch_tvimages, result);
            cout << *iter << "@@ run return = " << ret << endl;
            if(!result.empty()){
                cout << "mod rect num = " << result.size() << endl;
                drawImg(ptrShow, _width, _height, result, false, false, false, 1 );
                //需要修改存储结果的路径地址
                writeImg("../gkpw/"+subcase+*iter, ptrShow, _width, _height, true);
            }
            free(batch_tvimages[0].pData);
            batch_tvimages.erase(batch_tvimages.begin(), batch_tvimages.begin()+1);
        }
        free(ptrShow);
    }

    for(auto iter=batch_tvimages.begin(); iter!=batch_tvimages.end(); iter++ ){
        free(iter->pData);
    }

}