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
  APIConfig(AlgoAPIName::GENERAL_DETECTOR, "yolov5s-conv-9-20210927_736x416_mlu220_bs1c1_fp16.cambricon"),
  APIConfig(AlgoAPIName::SKELETON_DETECTOR, "pose_resnet_50_256x192_mlu220_bs1c1_fp16.cambricon"),
  APIConfig(AlgoAPIName::PED_DETECTOR, "yolov5s-conv-people-aug-fall_736x416_mlu220_bs1c1_fp16.cambricon"),
  APIConfig(AlgoAPIName::FIRE_CLASSIFIER, "resnet34fire_62_224x224_mlu220_bs1c1_fp16.cambricon"),//6
  APIConfig(AlgoAPIName::FIRE_DETECTOR, "yolov5s-conv-fire_736x416_mlu220_bs1c1_fp16.cambricon"),//7
  APIConfig(AlgoAPIName::WATER_DETECTOR, "unetwater_293_224x224_mlu220_bs1c1_fp16.cambricon"),//8
  APIConfig(AlgoAPIName::FIRE_DETECTOR_X, ""),//9
  APIConfig(AlgoAPIName::PED_FALL_DETECTOR, "yolov5s-conv-fall-ped_736x416_mlu220_bs1c1_fp16.cambricon"),//10
  APIConfig(AlgoAPIName::SAFETY_HAT_DETECTOR, "yolov5s-conv-safety-hat_736x416_mlu220_bs1c1_fp16.cambricon"),//11
  APIConfig(AlgoAPIName::TRASH_BAG_DETECTOR, "yolov5s-conv-trashbag_736x416_mlu220_bs1c1_fp16.cambricon"),//12
};
#else//mlu270
vector<APIConfig> config = {
  APIConfig(AlgoAPIName::FACE_DETECTOR, "/project/workspace/samples/mlu_videofacerec/weights/face_det/retinaface_736x416_mlu270.cambricon"),//0
  APIConfig(AlgoAPIName::FACE_EXTRACTOR, "/project/workspace/samples/mlu_videofacerec/weights/face_rec/resnet101_mlu270.cambricon"),//1
  APIConfig(AlgoAPIName::GENERAL_DETECTOR, "/project/workspace/samples/yolov5/mlu270/yolov5s-conv-9_736x416_mlu270_bs1c1_fp16.cambricon"),//2
  APIConfig(AlgoAPIName::GENERAL_DETECTOR, "/project/workspace/samples/yolov5/mlu270/yolov5s-conv-9-20210927_736x416_mlu270_bs1c1_fp16.cambricon"),//3
  APIConfig(AlgoAPIName::SKELETON_DETECTOR, "/project/workspace/samples/deep-high-resolution-net/mlu270/pose_resnet_50_256x192_mlu270_bs1c1_fp16.cambricon"),//4
  APIConfig(AlgoAPIName::PED_DETECTOR, "/project/workspace/samples/yolov5/mlu270/yolov5s-conv-people-aug-fall_736x416_mlu270_bs1c1_fp16.cambricon"),//5
  APIConfig(AlgoAPIName::FIRE_CLASSIFIER, "/project/workspace/samples/3d_unet_virtual/mlu270/resnet34fire_62_224x224_mlu270_bs1c1_fp16.cambricon"),//6
  APIConfig(AlgoAPIName::FIRE_DETECTOR, "/project/workspace/samples/yolov5/mlu270/yolov5s-conv-fire_736x416_mlu270_bs1c1_fp16.cambricon"),//7
  APIConfig(AlgoAPIName::WATER_DETECTOR, "/project/workspace/samples/3d_unet_virtual/mlu270/unetwater_293_224x224_mlu270_bs1c1_fp16.cambricon"),//8
  APIConfig(AlgoAPIName::FIRE_DETECTOR_X, ""),//9
  APIConfig(AlgoAPIName::PED_FALL_DETECTOR, "/project/workspace/samples/yolov5/mlu270/yolov5s-conv-fall-ped_736x416_mlu270_bs1c1_fp16.cambricon"),//10
  APIConfig(AlgoAPIName::SAFETY_HAT_DETECTOR, "/project/workspace/samples/yolov5/mlu270/yolov5s-conv-safety-hat_736x416_mlu270_bs1c1_fp16.cambricon"),//11
  APIConfig(AlgoAPIName::TRASH_BAG_DETECTOR, "/project/workspace/samples/yolov5/mlu270/yolov5s-conv-trashbag_736x416_mlu270_bs1c1_fp16.cambricon"),//12
};
#endif

using std::cout;

/**apiIdx
 * 0 : FACE_DETECTOR, ????????????
 * 1 : FACE_EXTRACTOR, ??????????????????
 * 2 : GENERAL_DETECTOR, ???????????????
 * 3 : GENERAL_DETECTOR, ???????????????(??????????????????)
 * 4 : SKELETON_DETECTOR, ????????????
 * 5 : PED_DETECTOR, ????????????
 * 6 : FIRE_CLASSIFIER, ????????????, ??????bbox????????????????????????
 * 7 : FIRE_DETECTOR, ????????????
 * 8 : WATER_DETECTOR, ????????????
 * 9 : FIRE_DETECTOR_X, ?????????????????????
 * 10: PED_FALL_DETECTOR, ????????????
 * 11: SAFETY_HAT_DETECTOR, ???????????????
 * 12: TRASH_BAG_DETECTOR, ???????????????
 */

/**usage
 * ./test_factory {imgname} {apiIdx} {use fake yuv if given} 
 */
int main(int argc, char* argv[]) {
    int apiIdx = 0;
    string imgname = "test.jpg";
    if(argc>=2){
        string _tmp(argv[1]);
        imgname = _tmp;
    }
    std::cout << "reading image " << imgname << endl;
    if(argc >= 3){
        apiIdx = atoi(argv[2]);
        assert(apiIdx < config.size());
    }
    bool fake_yuv = false;
    if(argc>=4){
        //?????????????????????yuv-nv21??????
        std::cout << "## use fake yuv" << endl;
        fake_yuv = true;
    }
    std::cout << "using #" << apiIdx << " api" << endl;
    AlgoAPISPtr apiHandlePtr = AICoreFactory::getAlgoAPI(config[apiIdx].apiName);
    RET_CODE ret;
    if(config[apiIdx].apiName != AlgoAPIName::FIRE_DETECTOR_X )
        ret = apiHandlePtr->init(config[apiIdx].modelName);
    else{
        cout << "special init for AlgoAPIName::FIRE_DETECTOR_X" << endl;
        //??????????????????????????????
        ret = apiHandlePtr->init(config[6].modelName);
        //??????????????????????????????
        ret = apiHandlePtr->init(config[7].modelName, "");
    }
    std::cout << "init return " << ret << endl;

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
    VecObjBBox objbboxes;
    BBox objbbox;

    vector<CLS_TYPE> class_type_to_be_detected;

    ucloud::TvaiResolution maxTraget={0,0};
    ucloud::TvaiResolution minTarget={0,0};
    std::vector<ucloud::TvaiRect> pRoi;

    auto start = chrono::system_clock::now();
    //TODO
    auto end = chrono::system_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end-start);
    double tm_cost = double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den;

    VecRect _tmp_;

    switch (config[apiIdx].apiName)
    {
    case AlgoAPIName::FACE_DETECTOR ://????????????
        ret = apiHandlePtr->set_param(0.8,0.2, maxTraget, minTarget , pRoi);
        cout << "set param return " << ret << endl;
        start = chrono::system_clock::now();
        ret = apiHandlePtr->run(inpImg, objbboxes);
        end = chrono::system_clock::now();
        duration = chrono::duration_cast<chrono::microseconds>(end-start);
        tm_cost = double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den;
        cout << tm_cost << "s cost" << endl;
        cout << "# " << objbboxes.size() << "faces detected" << endl;
        for (int i = 0 ; i < objbboxes.size(); i++ ){
            cout <<objbboxes[i].confidence << ", ";
        }
        cout << endl;
        if(!use_yuv){
            if(fake_yuv){
                freeImg(&ptrImg);
                ptrImg = readImg(imgname, width, height);
            }
            drawImg(ptrImg, width, height, objbboxes, true, true);
        }
        break;
    case AlgoAPIName::FACE_EXTRACTOR ://??????????????????
        objbbox.rect.x = 0; objbbox.rect.y = 0;
        objbbox.rect.width = width; objbbox.rect.height = height;
        objbbox.objtype = CLS_TYPE::FACE;
        objbboxes.push_back(objbbox);
        start = chrono::system_clock::now();
        ret = apiHandlePtr->run(inpImg, objbboxes);
        end = chrono::system_clock::now();
        duration = chrono::duration_cast<chrono::microseconds>(end-start);
        tm_cost = double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den;
        cout << tm_cost << "s cost" << endl;
        cout << "# " << objbboxes.size() << "face features extracted" << endl;
        break;

    case AlgoAPIName::FIRE_CLASSIFIER://????????????, 0-1??????, ??????????????????
        objbbox.rect.x = 0; objbbox.rect.y = 0;
        objbbox.rect.width = width; objbbox.rect.height = height;
        objbbox.objtype = CLS_TYPE::FIRE;
        objbboxes.push_back(objbbox);
        start = chrono::system_clock::now();
        ret = apiHandlePtr->run(inpImg, objbboxes);
        end = chrono::system_clock::now();
        duration = chrono::duration_cast<chrono::microseconds>(end-start);
        tm_cost = double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den;
        cout << tm_cost << "s cost" << endl;
        cout << "Fire probs = " << objbboxes[0].confidence << endl;     
        break;

    case AlgoAPIName::GENERAL_DETECTOR ://???????????????
    case AlgoAPIName::FIRE_DETECTOR://????????????
    case AlgoAPIName::WATER_DETECTOR://????????????
    case AlgoAPIName::PED_DETECTOR://?????????????????????
    case AlgoAPIName::PED_FALL_DETECTOR://????????????
    case AlgoAPIName::FIRE_DETECTOR_X://????????????, ?????????
    case AlgoAPIName::SAFETY_HAT_DETECTOR://???????????????
    case AlgoAPIName::TRASH_BAG_DETECTOR://???????????????
        apiHandlePtr->set_param(0.2,0.6,TvaiResolution{0,0},TvaiResolution{0,0}, _tmp_);
        ret = apiHandlePtr->get_class_type(class_type_to_be_detected);
        cout << "get_class_type return " << ret << endl;
        cout << "Class will be detected: ";
        for (int i = 0; i < class_type_to_be_detected.size(); i++ ){
            cout << class_type_to_be_detected[i] << ", ";
        }
        cout << endl;
        start = chrono::system_clock::now();
        ret = apiHandlePtr->run(inpImg, objbboxes);
        end = chrono::system_clock::now();
        duration = chrono::duration_cast<chrono::microseconds>(end-start);
        tm_cost = double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den;
        cout << tm_cost << "s cost" << endl;
        cout << "# " << objbboxes.size() << "obj detected" << endl;
        cout << "List of confidence:" << endl;
        for (int i = 0 ; i < objbboxes.size(); i++ ){
            cout << "#" << objbboxes[i].objtype << " = " <<objbboxes[i].confidence << ", ";
        }
        cout << endl;
        if(!use_yuv){
            if(fake_yuv){
                freeImg(&ptrImg);
                ptrImg = readImg(imgname, width, height);
            }
            drawImg(ptrImg, width, height, objbboxes, true, true);
        }
        break;
    default:
        ret = RET_CODE::FAILED;
        break;
    }

    if(!use_yuv && ret == RET_CODE::SUCCESS)
        writeImg( "result/test_factory.jpg" , ptrImg, width, height,true);
    freeImg(&ptrImg);
    cout << "run return " << ret << endl;



}