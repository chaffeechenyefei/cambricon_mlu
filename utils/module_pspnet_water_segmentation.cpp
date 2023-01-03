#include "module_pspnet_water_segmentation.hpp"
#include "glog/logging.h"
#include <opencv2/opencv.hpp>
#include <math.h>
#include "basic.hpp"
#include "../inner_utils/inner_basic.hpp"
#include <fstream>

#ifdef DEBUG
#include <chrono>
#include <sys/time.h>
#include "../inner_utils/module.hpp"
#endif


// #include <future>
using namespace ucloud;
using namespace cv;

using std::vector;

/*******************************************************************************
 * inner function
*******************************************************************************/
static void saveImg(std::string filename, std::string ext ,Mat &img){
    static int n = 0;
    std::string filename_full = filename + std::to_string(n++) + ext;
    if(!img.empty())
        imwrite(filename_full, img);
}

/*******************************************************************************
PSPNetWaterSegmentV4 基于分割算法的积水检测
chaffee.chen@2022-10-
*******************************************************************************/
RET_CODE PSPNetWaterSegmentV4::init(const std::string &modelpath){
    LOGI << "-> PSPNetWaterSegmentV4::init";
    bool pad_both_side = false;//单边预留
    bool keep_aspect_ratio = true;//保持长宽比
    BASE_CONFIG config(MODEL_INPUT_FORMAT::RGBA, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
    RET_CODE ret = m_net->base_init(modelpath, config);
    //Self param
    return ret;
}

RET_CODE PSPNetWaterSegmentV4::init(std::map<InitParam, std::string> &modelpath){
    if(use_auto_model){
        RET_CODE ret = auto_model_file_search(m_roots, m_models_startswith, modelpath);
        if(ret!=RET_CODE::SUCCESS) {
            printf("auto_model_file_search failed, return %d\n",ret);
            return ret;
        }
    }
    if(modelpath.find(InitParam::BASE_MODEL) == modelpath.end())
        return RET_CODE::ERR_INIT_PARAM_FAILED;
    RET_CODE ret = PSPNetWaterSegmentV4::init(modelpath[InitParam::BASE_MODEL]);
    return ret;
}

//clear self param
PSPNetWaterSegmentV4::~PSPNetWaterSegmentV4(){LOGI << "-> PSPNetWaterSegmentV4::~PSPNetWaterSegmentV4";}


RET_CODE PSPNetWaterSegmentV4::run_bgr_on_cpu(TvaiImage& tvimage, VecObjBBox &bboxes){
    LOGI << "-> PSPNetWaterSegmentV4::run_bgr_on_cpu";
    RET_CODE ret = RET_CODE::FAILED;
    float aspect_ratio_x, aspect_ratio_y, _aspect_ratio;
    float** model_output = nullptr;
    {
        ret = m_net->general_preprocess_bgr_on_cpu(tvimage, _aspect_ratio, aspect_ratio_x, aspect_ratio_y);
        if(ret!=RET_CODE::SUCCESS) return ret;
        model_output = m_net->general_mlu_infer();
    }
    // int outputDim = m_net->m_outputShape[0].DataCount();
    //TODO
    postprocess(tvimage, model_output[0], bboxes, _aspect_ratio);
    m_net->cpu_free(model_output);
    return ret;
}

RET_CODE PSPNetWaterSegmentV4::run_yuv_on_mlu(TvaiImage &tvimage,VecObjBBox &bboxes){
    LOGI << "-> PSPNetWaterSegmentV4::run_yuv_on_mlu";
    RET_CODE ret = RET_CODE::FAILED;
    float preprocess_time{0}, npu_inference_time{0}, postprocess_time{0};
    float aspect_ratio_x, aspect_ratio_y, _aspect_ratio;
    float** model_output = nullptr;
    {
        m_Tk.start();
        ret = m_net->general_preprocess_yuv_on_mlu_union(tvimage,TvaiRect{0,0,tvimage.width, tvimage.height} , _aspect_ratio, aspect_ratio_x, aspect_ratio_y);
        preprocess_time = m_Tk.end("preprocess", false);
        if(ret!=RET_CODE::SUCCESS) return ret;
        m_Tk.start();
        model_output = m_net->general_mlu_infer();
        npu_inference_time = m_Tk.end("npu inference", false);
    }
    //TODO
#ifdef VERBOSE
    // visual(tvimage, model_output[0], _aspect_ratio);
#endif
    m_Tk.start();
    postprocess(tvimage, model_output[0], bboxes, _aspect_ratio);
    postprocess_time = m_Tk.end("postprocess", false);
    m_net->cpu_free(model_output);
    if(!bboxes.empty()){
        bboxes[0].tmInfo = {preprocess_time, npu_inference_time, postprocess_time};
    }
    return ret;
}

RET_CODE PSPNetWaterSegmentV4::run(TvaiImage &tvimage,VecObjBBox &bboxes, float threshold, float nms_threshold){
    RET_CODE ret = RET_CODE::FAILED;
    if(tvimage.format == TVAI_IMAGE_FORMAT_NV21 || tvimage.format == TVAI_IMAGE_FORMAT_NV12 ){
        ret = run_yuv_on_mlu(tvimage ,bboxes);
    }
    else if(tvimage.format==TVAI_IMAGE_FORMAT_BGR || tvimage.format == TVAI_IMAGE_FORMAT_RGB ){
        ret = run_bgr_on_cpu(tvimage, bboxes);
    }
    else
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    return ret;
}

void PSPNetWaterSegmentV4::visual(TvaiImage& tvimage, float* model_output, float aspect_ratio){
    int _oW = m_net->m_outputShape[0].W();
    int _oH = m_net->m_outputShape[0].H();
    int _W = m_net->m_inputShape[0].W();
    int _H = m_net->m_inputShape[0].H();
    Mat cv_predict(Size(_oW, _oH), CV_32FC2, model_output);
    vector<Mat> cv_predict_planes;
    split(cv_predict, cv_predict_planes);

    // std::cout << "visual:: " << sum(cv_predict)[0] << "," << sum(cv_predict)[1] << std::endl;
    Mat cv_mask = 255 - cv_predict_planes[1]*255;//0-255, 255: land, 0: water
    cv_mask.convertTo(cv_mask, CV_8UC1);
    resize(cv_mask, cv_mask, Size(_W, _H));
    cv_mask = (cv_mask > 128)/255; //0-
    cv_mask.convertTo(cv_mask, CV_32FC1);

    int vW = MIN( int(tvimage.width*aspect_ratio), _W);
    int vH = MIN( int(tvimage.height*aspect_ratio),_H);
    cv_mask = cv_mask(Rect(0,0,vW,vH));

    Mat cv_image(Size(tvimage.stride, 3*tvimage.height/2), CV_8UC1, tvimage.pData );
    Mat cv_bgr;
    cvtColor(cv_image, cv_bgr, COLOR_YUV2BGR_NV21);
    resize(cv_bgr, cv_bgr, Size(vW,vH));
    cv_bgr.convertTo(cv_bgr, CV_32FC3);
    vector<Mat> cv_bgr_planes;
    split(cv_bgr, cv_bgr_planes);
    cv_bgr_planes[1] = cv_bgr_planes[1].mul(cv_mask);
    cv_bgr_planes[2] = cv_bgr_planes[2].mul(cv_mask);
    merge(cv_bgr_planes, cv_bgr);
    cv_bgr.convertTo(cv_bgr, CV_8UC3);

    saveImg("tmp/", ".png", cv_bgr);
}

RET_CODE PSPNetWaterSegmentV4::postprocess(TvaiImage& tvimage, float* model_output, VecObjBBox &bboxes, float aspect_ratio){
    int featDim = m_net->m_outputShape[0].DataCount();//1,h,w,2
    int _oW = m_net->m_outputShape[0].W();
    int _oH = m_net->m_outputShape[0].H();
    int _W = m_net->m_inputShape[0].W();
    int _H = m_net->m_inputShape[0].H();
    Mat cv_predict(Size(_oW, _oH), CV_32FC2, model_output);
    vector<Mat> cv_predict_planes;
    split(cv_predict, cv_predict_planes);

    float model_ratio_x = (float(_oW)) / _W;
    float model_ratio_y = (float(_oH)) / _H;

    Mat cv_mask = cv_predict_planes[1] > m_predict_threshold;

    cv_mask.convertTo(cv_mask, CV_32FC1);
    cv_mask /= 255;//0-1, 1: flood, 0: land
    if( 1 /*m_pAoiRect.empty()*/){
        int vW = MIN(int(tvimage.width*aspect_ratio*model_ratio_x), _oW);
        int vH = MIN(int(tvimage.height*aspect_ratio*model_ratio_y), _oH);
        Mat cv_tmp = cv_mask(Rect(0,0,vW,vH));
        Scalar v = mean(cv_tmp);
        // LOGI << v << "," << vW << "," << vH << "," << aspect_ratio << "," << model_ratio_x << "," << tvimage.width;
        BBox bbox;
        bbox.objtype = _cls_;
        bbox.rect = TvaiRect{0,0,tvimage.width, tvimage.height};
        bbox.confidence = v[0];
        bbox.objectness = v[0];
        bboxes.push_back(bbox);
    }
    // for(auto iter=m_pAoiRect.begin(); iter!=m_pAoiRect.end(); iter++){
    //     int x = iter->x * aspect_ratio*model_ratio_x;
    //     int y = iter->y * aspect_ratio*model_ratio_y;
    //     int w = iter->width * aspect_ratio*model_ratio_x;
    //     int h = iter->height * aspect_ratio*model_ratio_y;
    //     w = MIN(_oW-x-1, w);
    //     h = MIN(_oH-y-1, h);

    //     Rect rect = Rect( iter->x, iter->y, iter->width, iter->height );
    //     Mat cv_tmp = cv_mask(rect);
    //     Scalar v = mean(cv_tmp);
    //     BBox bbox;
    //     bbox.objtype = _cls_;
    //     bbox.rect = *iter;
    //     bbox.confidence = v[0];
    //     bbox.objectness = v[0];
    //     bboxes.push_back(bbox);
    // }
    return RET_CODE::SUCCESS;
}



RET_CODE PSPNetWaterSegmentV4::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    valid_clss.push_back(_cls_);
    return RET_CODE::SUCCESS;
};


/*******************************************************************************
PSPNetWaterSegment 基于分割算法的积水检测
chaffee.chen@2022-10-
*******************************************************************************/
// RET_CODE PSPNetWaterSegment::init(const std::string &modelpath){
//     LOGI << "-> PSPNetWaterSegment::init";
//     bool pad_both_side = false;//单边预留
//     bool keep_aspect_ratio = true;//保持长宽比
//     BASE_CONFIG config(MODEL_INPUT_FORMAT::RGBA, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
//     RET_CODE ret = BaseModel::base_init(modelpath, config);

//     std::vector<TvaiRect> pAoi;
//     set_param(m_predict_threshold, 0, TvaiResolution{0,0}, TvaiResolution{0,0}, pAoi);
//     //Self param
//     return ret;
// }

// RET_CODE PSPNetWaterSegment::auto_model_file_search(std::map<InitParam, std::string> &modelpath){
//     std::string basemodelfile, trackmodelfile;
//     for(auto &&m_root: m_roots){
//         std::vector<std::string> modelfiles;
//         ls_files(m_root, modelfiles, ".cambricon");
//         for(auto &&modelfile: modelfiles){
//             if(hasBegining(modelfile, m_basemodel_startswith)) basemodelfile = m_root + modelfile;
//         }
//     }

//     std::cout << "auto loading model from: " << basemodelfile << std::endl;
//     if(basemodelfile.empty() || basemodelfile==""){
//         return RET_CODE::ERR_MODEL_FILE_NOT_EXIST;
//     }

//     modelpath = { {InitParam::BASE_MODEL, basemodelfile} };
//     return RET_CODE::SUCCESS;
// }

// RET_CODE PSPNetWaterSegment::init(std::map<InitParam, std::string> &modelpath){
//     if(use_auto_model){
//         RET_CODE ret = auto_model_file_search(modelpath);
//         if(ret!=RET_CODE::SUCCESS) return ret;
//     }
    

//     if(modelpath.find(InitParam::BASE_MODEL) == modelpath.end())
//         return RET_CODE::ERR_INIT_PARAM_FAILED;
//     RET_CODE ret = PSPNetWaterSegment::init(modelpath[InitParam::BASE_MODEL]);
//     return ret;
// }

// //clear self param
// PSPNetWaterSegment::~PSPNetWaterSegment(){LOGI << "-> PSPNetWaterSegment::~PSPNetWaterSegment";}


// RET_CODE PSPNetWaterSegment::run_bgr_on_cpu(TvaiImage& tvimage, VecObjBBox &bboxes){
//     LOGI << "-> PSPNetWaterSegment::run_bgr_on_cpu";
//     RET_CODE ret = RET_CODE::FAILED;
//     float aspect_ratio_x, aspect_ratio_y, _aspect_ratio;
//     float* model_output = nullptr;
//     {
//         std::lock_guard<std::mutex> lk(_mlu_mutex);
//         ret = BaseModel::general_preprocess_bgr_on_cpu(tvimage, _aspect_ratio, aspect_ratio_x, aspect_ratio_y);
//         if(ret!=RET_CODE::SUCCESS) return ret;
//         model_output = BaseModel::general_mlu_infer();
//     }
//     int outputDim = _oW*_oH*_oC;
//     //TODO
//     postprocess(tvimage, model_output, bboxes, _aspect_ratio);
//     free(model_output);
//     return ret;
// }

// RET_CODE PSPNetWaterSegment::run_yuv_on_mlu(TvaiImage &tvimage,VecObjBBox &bboxes){
//     LOGI << "-> PSPNetWaterSegment::run_yuv_on_mlu";
//     RET_CODE ret = RET_CODE::FAILED;
//     float aspect_ratio_x, aspect_ratio_y, _aspect_ratio;
//     float* model_output = nullptr;
//     {
//         std::lock_guard<std::mutex> lk(_mlu_mutex);
//         if(tvimage.usePhyAddr)
//             ret = BaseModel::general_preprocess_yuv_on_mlu_phyAddr(tvimage, _aspect_ratio, aspect_ratio_x, aspect_ratio_y);
//         else
//             ret = BaseModel::general_preprocess_yuv_on_mlu(tvimage, _aspect_ratio, aspect_ratio_x, aspect_ratio_y);
//         if(ret!=RET_CODE::SUCCESS) return ret;
//         model_output = BaseModel::general_mlu_infer();
//     }
//     int outputDim = _oW*_oH*_oC;
//     //TODO
// #ifdef VERBOSE
//     visual(tvimage, model_output, _aspect_ratio);
// #endif
//     postprocess(tvimage, model_output, bboxes, _aspect_ratio);
//     free(model_output);
//     return ret;
// }

// RET_CODE PSPNetWaterSegment::run(BatchImageIN &batch_tvimages, VecObjBBox &bboxes){
//     if(batch_tvimages.empty()) return RET_CODE::SUCCESS;
//     RET_CODE ret = run(batch_tvimages[0], bboxes);
//     return ret;
// }

// RET_CODE PSPNetWaterSegment::run(TvaiImage &tvimage,VecObjBBox &bboxes){
//     RET_CODE ret = RET_CODE::FAILED;
//     if(tvimage.format == TVAI_IMAGE_FORMAT_NV21 || tvimage.format == TVAI_IMAGE_FORMAT_NV12 ){
//         ret = run_yuv_on_mlu(tvimage ,bboxes);
//     }
//     else if(tvimage.format==TVAI_IMAGE_FORMAT_BGR || tvimage.format == TVAI_IMAGE_FORMAT_RGB ){
//         ret = run_bgr_on_cpu(tvimage, bboxes);
//     }
//     else
//         ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
//     return ret;
// }

// void PSPNetWaterSegment::visual(TvaiImage& tvimage, float* model_output, float aspect_ratio){
//     Mat cv_predict(Size(_oW, _oH), CV_32FC2, model_output);
//     vector<Mat> cv_predict_planes;
//     split(cv_predict, cv_predict_planes);

//     // std::cout << "visual:: " << sum(cv_predict)[0] << "," << sum(cv_predict)[1] << std::endl;
//     Mat cv_mask = 255 - cv_predict_planes[1]*255;//0-255, 255: land, 0: water
//     cv_mask.convertTo(cv_mask, CV_8UC1);
//     resize(cv_mask, cv_mask, Size(_W, _H));
//     cv_mask = (cv_mask > 128)/255; //0-
//     cv_mask.convertTo(cv_mask, CV_32FC1);

//     int vW = MIN( int(tvimage.width*aspect_ratio), _W);
//     int vH = MIN( int(tvimage.height*aspect_ratio),_H);
//     cv_mask = cv_mask(Rect(0,0,vW,vH));

//     Mat cv_image(Size(tvimage.stride, 3*tvimage.height/2), CV_8UC1, tvimage.pData );
//     Mat cv_bgr;
//     cvtColor(cv_image, cv_bgr, COLOR_YUV2BGR_NV21);
//     resize(cv_bgr, cv_bgr, Size(vW,vH));
//     cv_bgr.convertTo(cv_bgr, CV_32FC3);
//     vector<Mat> cv_bgr_planes;
//     split(cv_bgr, cv_bgr_planes);
//     cv_bgr_planes[1] = cv_bgr_planes[1].mul(cv_mask);
//     cv_bgr_planes[2] = cv_bgr_planes[2].mul(cv_mask);
//     merge(cv_bgr_planes, cv_bgr);
//     cv_bgr.convertTo(cv_bgr, CV_8UC3);

//     saveImg("tmp/", ".png", cv_bgr);
// }

// RET_CODE PSPNetWaterSegment::postprocess(TvaiImage& tvimage, float* model_output, VecObjBBox &bboxes, float aspect_ratio){
//     int featDim = _oH*_oW*_oC;//1,h,w,2
//     Mat cv_predict(Size(_oW, _oH), CV_32FC2, model_output);
//     vector<Mat> cv_predict_planes;
//     split(cv_predict, cv_predict_planes);

//     float model_ratio_x = (float(_oW)) / _W;
//     float model_ratio_y = (float(_oH)) / _H;

//     Mat cv_mask = cv_predict_planes[1] > m_predict_threshold;

//     cv_mask.convertTo(cv_mask, CV_32FC1);
//     cv_mask /= 255;//0-1, 1: flood, 0: land
//     if(m_pAoiRect.empty()){
//         int vW = MIN(int(tvimage.width*aspect_ratio*model_ratio_x), _oW);
//         int vH = MIN(int(tvimage.height*aspect_ratio*model_ratio_y), _oH);
//         Mat cv_tmp = cv_mask(Rect(0,0,vW,vH));
//         Scalar v = mean(cv_tmp);
//         // LOGI << v << "," << vW << "," << vH << "," << aspect_ratio << "," << model_ratio_x << "," << tvimage.width;
//         BBox bbox;
//         bbox.objtype = _cls_;
//         bbox.rect = TvaiRect{0,0,tvimage.width, tvimage.height};
//         bbox.confidence = v[0];
//         bbox.objectness = v[0];
//         bboxes.push_back(bbox);
//     }
//     for(auto iter=m_pAoiRect.begin(); iter!=m_pAoiRect.end(); iter++){
//         int x = iter->x * aspect_ratio*model_ratio_x;
//         int y = iter->y * aspect_ratio*model_ratio_y;
//         int w = iter->width * aspect_ratio*model_ratio_x;
//         int h = iter->height * aspect_ratio*model_ratio_y;
//         w = MIN(_oW-x-1, w);
//         h = MIN(_oH-y-1, h);

//         Rect rect = Rect( iter->x, iter->y, iter->width, iter->height );
//         Mat cv_tmp = cv_mask(rect);
//         Scalar v = mean(cv_tmp);
//         BBox bbox;
//         bbox.objtype = _cls_;
//         bbox.rect = *iter;
//         bbox.confidence = v[0];
//         bbox.objectness = v[0];
//         bboxes.push_back(bbox);
//     }
//     return RET_CODE::SUCCESS;
// }



// RET_CODE PSPNetWaterSegment::get_class_type(std::vector<CLS_TYPE> &valid_clss){
//     valid_clss.push_back(_cls_);
//     return RET_CODE::SUCCESS;
// };

// RET_CODE PSPNetWaterSegment::set_param(float threshold, float nms_threshold, TvaiResolution maxTargetSize, TvaiResolution minTargetSize, 
// std::vector<TvaiRect> &pAoiRect){
//     if(float_in_range(threshold,1,0))
//         m_predict_threshold = threshold;
//     else
//         return RET_CODE::ERR_INIT_PARAM_FAILED;
//     if(!pAoiRect.empty()){
//         m_pAoiRect = pAoiRect;
//     } else {
//         m_pAoiRect.clear();
//     }
//     m_maxTargetSize = base_get_valid_maxSize(maxTargetSize);
//     m_minTargetSize = minTargetSize;
//     return RET_CODE::SUCCESS;    
// }
