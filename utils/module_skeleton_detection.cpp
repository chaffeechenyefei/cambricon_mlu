#include "module_skeleton_detection.hpp"
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

/*******************************************************************************
inner function
*******************************************************************************/
static void get_max_coords_of_heatmap(float* heatmap, int C, int H, int W, std::vector<Point2f>& coords);
// static void rescale_coords_to_sklandmark(std::vector<Point2f>& coords, TvaiRect offset, float aX, float aY, float zX, float zY ,SkLandmark &landmk);
static void rescale_coords_to_sklandmark(std::vector<Point2f>& coords, TvaiRect offset, float aX, float aY, float zX, float zY ,LandMark &landmk);
static bool is_cls_valid(CLS_TYPE dst_cls, std::vector<CLS_TYPE> &cls_set);
/**
 * @ PARAM:
 * heatmap: in 1CHW order
 * return Point2f
 * */
void get_max_coords_of_heatmap(float* heatmap, int C, int H, int W, std::vector<Point2f>& coords){
    float dr = 0.25;
    for(int c = 0; c < C; c++){
        float* _hmap = heatmap + c*H*W;
        int max_idx; float max_val;
        argmax(_hmap, H*W, max_idx, max_val);
        int x = max_idx%W;
        int y = max_idx/W;
        float fx = 1.0*x;
        float fy = 1.0*y;
        // if(fx>2000 || fy>2000)
        //     LOGI << "max_idx = " << max_idx;
        if( 1 < x && x < W - 1 && 1 < y && y < H - 1){
            float dx = _hmap[x+1 + y*W] - _hmap[x-1 + y*W];
            float dy = _hmap[x+(y+1)*W] - _hmap[x+(y-1)*W];
            fx += dx*dr;
            fy += dy*dr;
        }
        coords.push_back(Point2f(fx,fy));
    }
}

// void rescale_coords_to_sklandmark(std::vector<Point2f>& coords,TvaiRect offset , float aX, float aY, float zX, float zY, SkLandmark &landmk){
//     for(int i = 0; i < coords.size(); i++ ){
//         landmk.x[i] = zX*coords[i].x /aX + offset.x;
//         landmk.y[i] = zY*coords[i].y /aY + offset.y;
//         // LOGI << "x = " << landmk.x[i] << ", y = " << landmk.y[i];
//         // if(landmk.x[i]>2000 || landmk.y[i]>2000)
//         //     LOGI << aspect_ratio << ", " << zX << ", " << zY << ", " << coords[i].x << ", " << coords[i].y 
//         //     << ", "<< offset.x << ", " << offset.y;
//      }
// }

void rescale_coords_to_sklandmark(std::vector<Point2f>& coords,TvaiRect offset , float aX, float aY, float zX, float zY, LandMark &landmk){
    if(!landmk.pts.empty()){
        landmk.pts.clear();
    }
    for(int i = 0; i < coords.size(); i++ ){
        landmk.pts.push_back(uPoint(zX*coords[i].x /aX + offset.x, zY*coords[i].y /aY + offset.y));
     }
     landmk.refcoord = RefCoord::IMAGE_ORIGIN;
     landmk.type = LandMarkType::SKELETON;
}

bool is_cls_valid(CLS_TYPE dst_cls, std::vector<CLS_TYPE> &cls_set){
    for(auto &&cls: cls_set){
        if(cls==dst_cls) return true;
    }
    return false;
}

/*******************************************************************************
SkeletonDetectorV4 remove BaseModel
chaffee.chen@2022-10-08
*******************************************************************************/

RET_CODE SkeletonDetectorV4::init(const std::string &modelpath){
    LOGI << "-> SkeletonDetectorV4::init";
    bool pad_both_side = false;
    bool keep_aspect_ratio = false;//不保持长宽比
    BASE_CONFIG config(MODEL_INPUT_FORMAT::RGBA, MODEL_OUTPUT_ORDER::NCHW, pad_both_side, keep_aspect_ratio);
    RET_CODE ret = m_net->base_init(modelpath, config);
    //Self param
    return ret;
}

RET_CODE SkeletonDetectorV4::init(std::map<InitParam, std::string> &modelpath){
    if(modelpath.find(InitParam::BASE_MODEL) == modelpath.end()) return RET_CODE::ERR_INIT_PARAM_FAILED;
    return init(modelpath[InitParam::BASE_MODEL]);
}

SkeletonDetectorV4::~SkeletonDetectorV4(){LOGI << "-> SkeletonDetectorV4::~SkeletonDetectorV4";}

RET_CODE SkeletonDetectorV4::run_yuv_on_mlu_phyAddr(TvaiImage &tvimage, VecObjBBox &bboxes){
    LOGI << "-> SkeletonDetectorV4::run_yuv_on_mlu_phyAddr";
    RET_CODE ret = RET_CODE::SUCCESS;
    for(int i = 0; i < bboxes.size(); i++){
        if( !is_cls_valid(bboxes[i].objtype, _cls_) ) continue;
        TvaiRect roiRect = bboxes[i].rect;
        float aspect_ratio = 1.0;
        float aX, aY;
        float** model_output = nullptr;
        {
            ret = m_net->general_preprocess_yuv_on_mlu_union(tvimage, roiRect, aspect_ratio, aX , aY);
            if(ret!=RET_CODE::SUCCESS) return ret;
            model_output = m_net->general_mlu_infer();
        }
        
        //TODO post process 
        ret = postprocess(model_output[0], roiRect ,bboxes[i].Pts, aX, aY);
        if(ret!=RET_CODE::SUCCESS){ m_net->cpu_free(model_output); return ret; }
        m_net->cpu_free(model_output);
    }
    return RET_CODE::SUCCESS;
}

RET_CODE SkeletonDetectorV4::run_bgr_on_cpu(TvaiImage& tvimage, VecObjBBox &bboxes){
    LOGI << "-> SkeletonDetectorV4::run_bgr_on_cpu";
    RET_CODE ret = RET_CODE::FAILED;
    std::vector<CLS_TYPE> valid_cls = {CLS_TYPE::PEDESTRIAN};
    std::vector<float*> model_outputs;
    std::vector<float> aspect_ratios;
    {
        ret = m_net->general_preprocess_infer_bgr_on_cpu(tvimage, bboxes ,model_outputs, aspect_ratios, valid_cls);
    }
    if(ret!=RET_CODE::SUCCESS) return ret;
    //TODO post processing
    for(int i=0; i< model_outputs.size(); i++){
        if(model_outputs[i]==nullptr) continue;
        ret = postprocess(model_outputs[i], bboxes[i].rect, bboxes[i].Pts, aspect_ratios[2*i], aspect_ratios[2*i+1]);
        if(ret!=RET_CODE::SUCCESS)
            break;        
    }
    for(int i=0; i< model_outputs.size(); i++){
        if(model_outputs[i]!=nullptr)
            free(model_outputs[i]);
    }
    model_outputs.clear();
    return ret;
}

RET_CODE SkeletonDetectorV4::run(TvaiImage &tvimage, VecObjBBox &bboxes,float threshold, float nms_threshold){
    RET_CODE ret = RET_CODE::FAILED;
    if(tvimage.format == TVAI_IMAGE_FORMAT_RGB || tvimage.format == TVAI_IMAGE_FORMAT_BGR){
        ret = run_bgr_on_cpu(tvimage, bboxes);
    }
    else if(tvimage.format == TVAI_IMAGE_FORMAT_NV21 || tvimage.format == TVAI_IMAGE_FORMAT_NV12 ){
        ret = run_yuv_on_mlu_phyAddr(tvimage, bboxes);
    }
    else
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    return ret;
}

RET_CODE SkeletonDetectorV4::postprocess(float* model_output, TvaiRect pedRect, LandMark &kypts, float aX, float aY){
    // LOGI << "-> SkeletonDetector::postprocess";
    int _W = m_net->m_inputShape[0].W();
    int _H = m_net->m_inputShape[0].H();
    int _oW = m_net->m_outputShape[0].W();
    int _oH = m_net->m_outputShape[0].H();
    int _oC = m_net->m_outputShape[0].C();
    std::vector<Point2f> coords;
    float zX = (1.0*_W)/_oW;
    float zY = (1.0*_H)/_oH;
    get_max_coords_of_heatmap(model_output, _oC, _oH, _oW , coords);
    rescale_coords_to_sklandmark(coords, pedRect , aX, aY, zX, zY, kypts);
    return RET_CODE::SUCCESS;
}

RET_CODE SkeletonDetectorV4::set_output_cls_order(std::vector<CLS_TYPE> &output_clss){
    if(!output_clss.empty())
        _cls_ = output_clss;
    else _cls_ = {CLS_TYPE::PEDESTRIAN};
}

RET_CODE SkeletonDetectorV4::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    valid_clss = _cls_;
    return RET_CODE::SUCCESS;
}





/*******************************************************************************
SkeletonDetector based on BaseModel
*******************************************************************************/

// RET_CODE SkeletonDetector::init(const std::string &modelpath){
//     LOGI << "-> SkeletonDetector::init";
//     bool pad_both_side = false;
//     bool keep_aspect_ratio = false;//不保持长宽比
//     BASE_CONFIG config(MODEL_INPUT_FORMAT::RGBA, MODEL_OUTPUT_ORDER::NCHW, pad_both_side, keep_aspect_ratio);
//     RET_CODE ret = BaseModel::base_init(modelpath, config);
//     // if (ret!=RET_CODE::SUCCESS) return ret;
//     //Self param
//     return ret;
// }

// RET_CODE SkeletonDetector::init(std::map<InitParam, std::string> &modelpath){
//     if(modelpath.find(InitParam::BASE_MODEL) == modelpath.end()) return RET_CODE::ERR_INIT_PARAM_FAILED;
//     return init(modelpath[InitParam::BASE_MODEL]);
// }

// //clear self param
// SkeletonDetector::~SkeletonDetector(){LOGI << "-> SkeletonDetector::~SkeletonDetector";}

// RET_CODE SkeletonDetector::run_yuv_on_mlu_phyAddr(TvaiImage &tvimage, VecObjBBox &bboxes){
//     LOGI << "-> SkeletonDetector::run_yuv_on_mlu_phyAddr";
//     RET_CODE ret = RET_CODE::FAILED;
//     for(int i = 0; i < bboxes.size(); i++){
//         if( !is_cls_valid(bboxes[i].objtype, _cls_) ) continue;
//         TvaiRect roiRect = bboxes[i].rect;
//         float aspect_ratio = 1.0;
//         float aX, aY;
//         float* model_output = nullptr;
//         {
//             std::lock_guard<std::mutex> lk(_mlu_mutex);
//             ret = BaseModel::general_preprocess_yuv_on_mlu_phyAddr(tvimage, roiRect, aspect_ratio, aX , aY);
//             if(ret!=RET_CODE::SUCCESS) return ret;
//             model_output = BaseModel::general_mlu_infer();
//         }
        
//         //TODO post process 
//         ret = postprocess(model_output, roiRect ,bboxes[i].Pts, aX, aY);
//         if(ret!=RET_CODE::SUCCESS){ free(model_output); return ret; }
//         free(model_output);
//     }
//     return RET_CODE::SUCCESS;
// }

// RET_CODE SkeletonDetector::run_yuv_on_mlu(TvaiImage& tvimage, VecObjBBox &bboxes){
//     LOGI << "-> SkeletonDetector::run_yuv_on_mlu";
//     RET_CODE ret = RET_CODE::FAILED;
//     for(int i = 0; i < bboxes.size(); i++){
//         if( !is_cls_valid(bboxes[i].objtype, _cls_) ) continue;
//         TvaiRect roiRect = bboxes[i].rect;
//         float aspect_ratio = 1.0;
//         float aX,aY;
//         float* model_output = nullptr;
//         {
//             std::lock_guard<std::mutex> lk(_mlu_mutex);
//             ret = BaseModel::general_preprocess_yuv_on_mlu(tvimage, roiRect, aspect_ratio, aX , aY);
//             if(ret!=RET_CODE::SUCCESS) return ret;
//             model_output = BaseModel::general_mlu_infer();
//         }
//         //TODO post process 
//         ret = postprocess(model_output, roiRect , bboxes[i].Pts, aX, aY);
//         if(ret!=RET_CODE::SUCCESS){ free(model_output); return ret; }
//         free(model_output);
//     }
//     return RET_CODE::SUCCESS;
// }

// RET_CODE SkeletonDetector::run_bgr_on_cpu(TvaiImage& tvimage, VecObjBBox &bboxes){
//     LOGI << "-> SkeletonDetector::run_bgr_on_cpu";
//     RET_CODE ret = RET_CODE::FAILED;
//     // std::vector<TvaiRect> roiRects;
//     // for(int i=0; i < bboxes.size(); i++){
//     //     roiRects.push_back(bboxes[i].rect);
//     // }
//     std::vector<CLS_TYPE> valid_cls = {CLS_TYPE::PEDESTRIAN};
//     std::vector<float*> model_outputs;
//     std::vector<float> aspect_ratios;
//     {
//         std::lock_guard<std::mutex> lk(_mlu_mutex);
//         ret = BaseModel::general_preprocess_infer_bgr_on_cpu(tvimage, bboxes ,model_outputs, aspect_ratios, valid_cls);
//         // ret = BaseModel::general_preprocess_infer_bgr_on_cpu(tvimage, roiRects,model_outputs, aspect_ratios);
//     }
//     if(ret!=RET_CODE::SUCCESS) return ret;
//     //TODO post processing
//     for(int i=0; i< model_outputs.size(); i++){
//         if(model_outputs[i]==nullptr) continue;
//         ret = postprocess(model_outputs[i], bboxes[i].rect, bboxes[i].Pts, aspect_ratios[2*i], aspect_ratios[2*i+1]);
//         if(ret!=RET_CODE::SUCCESS)
//             break;        
//     }
//     for(int i=0; i< model_outputs.size(); i++){
//         if(model_outputs[i]!=nullptr)
//             free(model_outputs[i]);
//     }
//     model_outputs.clear();
//     return ret;
// }

// RET_CODE SkeletonDetector::run(BatchImageIN &batch_tvimages, VecObjBBox &bboxes){
//     if(batch_tvimages.empty()) return RET_CODE::SUCCESS;
//     return run(batch_tvimages[0], bboxes);
// }

// RET_CODE SkeletonDetector::run(TvaiImage &tvimage, VecObjBBox &bboxes){
//     RET_CODE ret = RET_CODE::FAILED;
//     if(tvimage.format == TVAI_IMAGE_FORMAT_RGB || tvimage.format == TVAI_IMAGE_FORMAT_BGR){
//         ret = run_bgr_on_cpu(tvimage, bboxes);
//     }
//     else if(tvimage.format == TVAI_IMAGE_FORMAT_NV21 || tvimage.format == TVAI_IMAGE_FORMAT_NV12 ){
//         if(tvimage.usePhyAddr)
//             ret = run_yuv_on_mlu_phyAddr(tvimage, bboxes);
//         else
//             ret = run_yuv_on_mlu(tvimage,bboxes);
//     }
//     else
//         ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
//     return ret;
// }



// RET_CODE SkeletonDetector::postprocess(float* model_output, TvaiRect pedRect, SkLandmark &kypts, float aX, float aY){
//     // LOGI << "-> SkeletonDetector::postprocess";
//     std::vector<Point2f> coords;
//     if( sizeof(kypts.x)/sizeof(float) != _oC ){
//         LOGI << "ERR_MODEL_NOT_MATCH: " << sizeof(kypts.x)/sizeof(float) << "!=" << _oC;
//         return RET_CODE::ERR_MODEL_NOT_MATCH;
//     }
//     float zX = (1.0*_W)/_oW;
//     float zY = (1.0*_H)/_oH;
//     get_max_coords_of_heatmap(model_output, _oC, _oH, _oW , coords);
//     rescale_coords_to_sklandmark(coords, pedRect , aX, aY, zX, zY, kypts);
//     return RET_CODE::SUCCESS;
// }

// RET_CODE SkeletonDetector::postprocess(float* model_output, TvaiRect pedRect, LandMark &kypts, float aX, float aY){
//     // LOGI << "-> SkeletonDetector::postprocess";
//     std::vector<Point2f> coords;
//     float zX = (1.0*_W)/_oW;
//     float zY = (1.0*_H)/_oH;
//     get_max_coords_of_heatmap(model_output, _oC, _oH, _oW , coords);
//     rescale_coords_to_sklandmark(coords, pedRect , aX, aY, zX, zY, kypts);
//     return RET_CODE::SUCCESS;
// }

// RET_CODE SkeletonDetector::get_class_type(std::vector<CLS_TYPE> &valid_clss){
//     valid_clss = _cls_;
//     return RET_CODE::SUCCESS;
// }

// RET_CODE SkeletonDetector::set_output_cls_order(std::vector<CLS_TYPE> &output_clss){
//     if(!output_clss.empty())
//         _cls_ = output_clss;
//     else _cls_ = {CLS_TYPE::PEDESTRIAN};
// }






