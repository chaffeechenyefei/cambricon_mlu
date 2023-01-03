#include "module_base.hpp"
#include "glog/logging.h"
#include <opencv2/opencv.hpp>
#include <cnrt.h>
#include <device/mlu_context.h>
#include <easybang/resize_and_colorcvt.h>
#include <easyinfer/easy_infer.h>
#include <easyinfer/mlu_memory_op.h>
#include <easyinfer/model_loader.h>
#include <math.h>
#include "basic.hpp"
#include "../inner_utils/inner_basic.hpp"
#include <fstream>
#include <cn_api.h>


#ifdef DEBUG
#include <chrono>
#include <sys/time.h>
#endif
#include "../inner_utils/module.hpp"

// #include <future>
using namespace ucloud;
using namespace cv;

/**
 * inner function
 * */
static bool check_valid_class( CLS_TYPE objCls ,std::vector<CLS_TYPE> &valid_class);

bool check_valid_class( CLS_TYPE objCls ,std::vector<CLS_TYPE> &valid_class){
    if(valid_class.empty())
        return true;
    else{
        for(int i = 0; i < valid_class.size(); i++ ){
            if(objCls==valid_class[i])
                return true;
        }
    }
    return false;
};

/////////////////////////////////////////////////////////////////////
// Class PrivateContextV2 
/////////////////////////////////////////////////////////////////////
PrivateContextV2::PrivateContextV2(){
    env_ = new edk::MluContext();
    // LOG(INFO) << "set mlu device";
    if(env_->GetDeviceNum()<=0){
        LOGI << "No mlu device founded";
        status_ = false;
        delete env_;
        env_ = nullptr;
    } else{
        printf("** Total [%d] devices found, default using device 0\n",env_->GetDeviceNum());
        env_->SetDeviceId(0);
        env_->BindDevice();
        // LOG(INFO) << env->GetDeviceNum();
        status_ = true;
    }
}

PrivateContextV2::~PrivateContextV2(){
    delete env_;
    env_ = nullptr;
    status_ = false;
};
/////////////////////////////////////////////////////////////////////
// End of Class PrivateContext 
/////////////////////////////////////////////////////////////////////

mluColorMode getMLUTransColorMode(MODEL_INPUT_FORMAT model_input_fmt, TvaiImageFormat image_input_fmt){
    if(image_input_fmt==TVAI_IMAGE_FORMAT_NV21){
        if(model_input_fmt==MODEL_INPUT_FORMAT::BGRA)
            return mluColorMode::YUV2BGRA_NV21;
        else if(model_input_fmt==MODEL_INPUT_FORMAT::ABGR)
            return mluColorMode::YUV2ABGR_NV21;
        else if(model_input_fmt==MODEL_INPUT_FORMAT::ARGB)
            return mluColorMode::YUV2ARGB_NV21;
        else//RGBA
            return mluColorMode::YUV2RGBA_NV21;
    } else {
        if(model_input_fmt==MODEL_INPUT_FORMAT::BGRA)
            return mluColorMode::YUV2BGRA_NV12;
        else if(model_input_fmt==MODEL_INPUT_FORMAT::ABGR)
            return mluColorMode::YUV2ABGR_NV12;
        else if(model_input_fmt==MODEL_INPUT_FORMAT::ARGB)
            return mluColorMode::YUV2ARGB_NV12;
        else//RGBA
            return mluColorMode::YUV2RGBA_NV12;
    }
}

static void getCPUTransColorMode(MODEL_INPUT_FORMAT model_input_fmt, TvaiImageFormat image_input_fmt, bool &output_rgba, bool &input_rgb){
    if(image_input_fmt==TVAI_IMAGE_FORMAT_RGB)
        input_rgb = true;
    else
        input_rgb = false;
    if(model_input_fmt==MODEL_INPUT_FORMAT::RGBA)
        output_rgba = true;
    else
        output_rgba = false;
}


template<typename T>
bool base_sortBox(const T& a, const T& b) {
  return  a.confidence > b.confidence;
}
template<typename T>
void base_nmsBBox(std::vector<T>& input, float threshold, int type, std::vector<T>& output) {
  std::sort(input.begin(), input.end(), base_sortBox<T>);
  std::vector<int> bboxStat(input.size(), 0);
  for (size_t i=0; i<input.size(); ++i) {
    if (bboxStat[i] == 1) continue;
    output.push_back(input[i]);
    float area0 = (input[i].y1 - input[i].y0 + 1e-3)*(input[i].x1 - input[i].x0 + 1e-3);
    for (size_t j=i+1; j<input.size(); ++j) {
      if (bboxStat[j] == 1) continue;
      float roiWidth = std::min(input[i].x1, input[j].x1) - std::max(input[i].x0, input[j].x0);
      float roiHeight = std::min(input[i].y1, input[j].y1) - std::max(input[i].y0, input[j].y0);
      if (roiWidth < 0 || roiHeight < 0) continue;
      float area1 = (input[j].y1 - input[j].y0 + 1e-3)*(input[j].x1 - input[j].x0 + 1e-3);
      float ratio = 0.0;
      if (type == NMS_UNION) {
        ratio = roiWidth*roiHeight/(area0 + area1 - roiWidth*roiHeight);
      } else {
        ratio = roiWidth*roiHeight / std::min(area0, area1);
      }

      if (ratio > threshold) bboxStat[j] = 1; 
    }
  }
  return;
}
/**
 * Multi Class
 **/ 
void base_nmsBBox(std::vector<VecObjBBox> &input, float threshold, int type, VecObjBBox &output){
    if (input.empty()){
        VecObjBBox().swap(output);
        return;
    }
    for (int i = 0; i < input.size(); i++ ){
        base_nmsBBox(input[i], threshold, type, output);
    }
    return;
}
void base_output2ObjBox_multiCls(float* output ,std::vector<VecObjBBox> &vecbox, CLS_TYPE* cls_map, std::map<CLS_TYPE, int> &unique_cls_map ,int nbboxes ,int stride ,float threshold, int dimOffset){
    //xywh+objectness+nc (xywh=centerXY,WH)
    int nc = stride - dimOffset;
    for (int i=0; i<unique_cls_map.size(); i++){
        vecbox.push_back(VecObjBBox());
    }
    for( int i=0; i < nbboxes; i++ ){
        float* _output = &output[i*stride];
        float objectness = _output[4];
        if( objectness < threshold )
            continue;
        else {
            BBox fbox;
            float cx = *_output++;
            float cy = *_output++;
            float w = *_output++;
            float h = *_output++;
            fbox.x0 = cx - w/2;
            fbox.y0 = cy - h/2;
            fbox.x1 = cx + w/2;
            fbox.y1 = cy + h/2;
            fbox.x = fbox.x0; fbox.y = fbox.y0; fbox.w = w; fbox.h = h;
            _output++;//skip objectness
            fbox.objectness = objectness;
            int maxid = -1;
            float max_confidence = 0;
            float* confidence = _output;
            argmax(confidence, nc , maxid, max_confidence);
            fbox.confidence = objectness*max_confidence;
            fbox.quality = max_confidence;//++quality using max_confidence instead for object detection
            if (maxid < 0 || cls_map == nullptr)
                fbox.objtype = CLS_TYPE::UNKNOWN;
            else
                fbox.objtype = cls_map[maxid];
            if(unique_cls_map.find(fbox.objtype)!=unique_cls_map.end())
                vecbox[unique_cls_map[fbox.objtype]].push_back(fbox);
        }
    }
    return;
}
void base_output2ObjBox_multiCls(float* output ,std::vector<VecObjBBox> &vecbox, std::vector<CLS_TYPE> &cls_map, std::map<CLS_TYPE, int> &unique_cls_map ,int nbboxes ,int stride ,float threshold, int dimOffset){
    //xywh+objectness+nc (xywh=centerXY,WH)
    int nc = stride - dimOffset;
    for (int i=0; i<unique_cls_map.size(); i++){
        vecbox.push_back(VecObjBBox());
    }
    for( int i=0; i < nbboxes; i++ ){
        float* _output = &output[i*stride];
        float objectness = _output[4];
        if( objectness < threshold )
            continue;
        else {
            BBox fbox;
            float cx = *_output++;
            float cy = *_output++;
            float w = *_output++;
            float h = *_output++;
            fbox.x0 = cx - w/2;
            fbox.y0 = cy - h/2;
            fbox.x1 = cx + w/2;
            fbox.y1 = cy + h/2;
            fbox.x = fbox.x0; fbox.y = fbox.y0; fbox.w = w; fbox.h = h;
            _output++;//skip objectness
            fbox.objectness = objectness;
            int maxid = -1;
            float max_confidence = 0;
            float* confidence = _output;
            argmax(confidence, nc , maxid, max_confidence);
            fbox.confidence = objectness*max_confidence;
            fbox.quality = max_confidence;//++quality using max_confidence instead for object detection
            if (maxid < 0 || cls_map.empty())
                fbox.objtype = CLS_TYPE::UNKNOWN;
            else
                fbox.objtype = cls_map[maxid];
            if(unique_cls_map.find(fbox.objtype)!=unique_cls_map.end())
                vecbox[unique_cls_map[fbox.objtype]].push_back(fbox);
        }
    }
    return;
}
void base_output2ObjBox_multiCls(float* output ,std::vector<VecObjBBox> &vecbox, std::vector<std::string> &cls_map, std::map<std::string, int> &unique_cls_map ,int nbboxes ,int stride ,float threshold, int dimOffset){
    //xywh+objectness+nc (xywh=centerXY,WH)
    int nc = stride - dimOffset;
    for (int i=0; i<unique_cls_map.size(); i++){
        vecbox.push_back(VecObjBBox());
    }
    for( int i=0; i < nbboxes; i++ ){
        float* _output = &output[i*stride];
        float objectness = _output[4];
        if( objectness < threshold )
            continue;
        else {
            BBox fbox;
            float cx = *_output++;
            float cy = *_output++;
            float w = *_output++;
            float h = *_output++;
            fbox.x0 = cx - w/2;
            fbox.y0 = cy - h/2;
            fbox.x1 = cx + w/2;
            fbox.y1 = cy + h/2;
            fbox.x = fbox.x0; fbox.y = fbox.y0; fbox.w = w; fbox.h = h;
            _output++;//skip objectness
            fbox.objectness = objectness;
            int maxid = -1;
            float max_confidence = 0;
            float* confidence = _output;
            argmax(confidence, nc , maxid, max_confidence);
            fbox.confidence = objectness*max_confidence;
            fbox.quality = max_confidence;//++quality using max_confidence instead for object detection
            if (maxid < 0 || cls_map.empty())
                fbox.objtype = CLS_TYPE::UNKNOWN;
            else{
                fbox.objtype = CLS_TYPE::TARGET; //cls_map[maxid];
                fbox.objname = cls_map[maxid];
            }
            if(unique_cls_map.find(fbox.objname)!=unique_cls_map.end())
                vecbox[unique_cls_map[fbox.objname]].push_back(fbox);
        }
    }
    return;
}



void base_output2ObjBox_multiCls_yoloface(float* output ,std::vector<VecObjBBox> &vecbox, std::vector<CLS_TYPE> &cls_map, std::map<CLS_TYPE, int> &unique_cls_map ,int nbboxes ,int stride ,float threshold, int dimOffset){
    //xywh+objectness+nc (xywh=centerXY,WH)
    int nc = stride - dimOffset;
    // LOGI << "nc: " << nc << ", stride: " << stride << ", unique_cls_map.size(): " << unique_cls_map.size() << ", nbboxes: " << nbboxes;
    for (int i=0; i<unique_cls_map.size(); i++){
        vecbox.push_back(VecObjBBox());
    }
    for( int i=0; i < nbboxes; i++ ){
        float* _output = &output[i*stride];
        float objectness = _output[4];
        if( objectness < threshold )
            continue;
        else {
            BBox fbox;
            float cx = _output[0];
            float cy = _output[1];
            float w = _output[2];
            float h = _output[3];
            fbox.x0 = cx - w/2;
            fbox.y0 = cy - h/2;
            fbox.x1 = cx + w/2;
            fbox.y1 = cy + h/2;
            fbox.x = fbox.x0; fbox.y = fbox.y0; fbox.w = w; fbox.h = h;
            // _output++;//skip objectness

            for(int j = 0; j < 5 ; j++){
                float px = _output[5 + j*2];
                float py = _output[6 + j*2];
                fbox.Pts.pts.push_back(uPoint(px,py));
            }
            fbox.Pts.refcoord = RefCoord::IMAGE_ORIGIN;//cur is unscaled image origin
            fbox.Pts.type = LandMarkType::FACE_5PTS;

            fbox.objectness = objectness;
            int maxid = -1;
            float max_confidence = 0;
            float* confidence = _output;
            argmax(confidence, nc , maxid, max_confidence);
            fbox.confidence = objectness*max_confidence;
            fbox.quality = max_confidence;//++quality using max_confidence instead for object detection
            if (maxid < 0 || cls_map.empty())
                fbox.objtype = CLS_TYPE::UNKNOWN;
            else
                fbox.objtype = cls_map[maxid];
            if(unique_cls_map.find(fbox.objtype)!=unique_cls_map.end())
                vecbox[unique_cls_map[fbox.objtype]].push_back(fbox);
        }
    }
    return;
}

void base_output2ObjBox_multiCls_yoloface(float* output ,std::vector<VecObjBBox> &vecbox, CLS_TYPE* cls_map, std::map<CLS_TYPE, int> &unique_cls_map ,int nbboxes ,int stride ,float threshold, int dimOffset){
    //xywh+objectness+nc (xywh=centerXY,WH)
    int nc = stride - dimOffset;
    // LOGI << "nc: " << nc << ", stride: " << stride << ", unique_cls_map.size(): " << unique_cls_map.size() << ", nbboxes: " << nbboxes;
    for (int i=0; i<unique_cls_map.size(); i++){
        vecbox.push_back(VecObjBBox());
    }
    for( int i=0; i < nbboxes; i++ ){
        float* _output = &output[i*stride];
        float objectness = _output[4];
        if( objectness < threshold )
            continue;
        else {
            BBox fbox;
            float cx = _output[0];
            float cy = _output[1];
            float w = _output[2];
            float h = _output[3];
            fbox.x0 = cx - w/2;
            fbox.y0 = cy - h/2;
            fbox.x1 = cx + w/2;
            fbox.y1 = cy + h/2;
            fbox.x = fbox.x0; fbox.y = fbox.y0; fbox.w = w; fbox.h = h;
            // _output++;//skip objectness

            for(int j = 0; j < 5 ; j++){
                float px = _output[5 + j*2];
                float py = _output[6 + j*2];
                fbox.Pts.pts.push_back(uPoint(px,py));
            }
            fbox.Pts.refcoord = RefCoord::IMAGE_ORIGIN;//cur is unscaled image origin
            fbox.Pts.type = LandMarkType::FACE_5PTS;

            fbox.objectness = objectness;
            int maxid = -1;
            float max_confidence = 0;
            float* confidence = _output;
            argmax(confidence, nc , maxid, max_confidence);
            fbox.confidence = objectness*max_confidence;
            fbox.quality = max_confidence;//++quality using max_confidence instead for object detection
            if (maxid < 0 || cls_map == nullptr)
                fbox.objtype = CLS_TYPE::UNKNOWN;
            else
                fbox.objtype = cls_map[maxid];
            if(unique_cls_map.find(fbox.objtype)!=unique_cls_map.end())
                vecbox[unique_cls_map[fbox.objtype]].push_back(fbox);
        }
    }
    return;
}


/**
 * 20211109
 * YuvCropResizeModel: 提供一些通用基础组件, 减少代码重复度, 便于后期维护
 * 没有模型, 仅进行mlu上的图像crop
 * 继承
 * - PrivateContext: MLU通用环境变量
 * - AlgoAPI: 抽象类, 主要用于暴露接口, 无任何私有变量
 * */
////////////////////////////////////////////////////////////////////////////////////////////////////
// YuvCropResizeModel
////////////////////////////////////////////////////////////////////////////////////////////////////
RET_CODE YuvCropResizeModel::base_init( int dstH, int dstW,BASE_CONFIG config){
    if(!status_){
        return RET_CODE::ERR_NPU_INIT_FAILED;
    }
    release();
    //config
    _model_input_fmt = config.model_input_fmt;
    _pad_both_side = config.pad_both_side;
    _keep_aspect_ratio = config.keep_aspect_ratio;
    // set mlu environment
    PtrHandle* ptrHandle = new PtrHandle();
    _ptrHandle = ptrHandle;

    _H = dstH;
    _W = dstW;
    _C = 4;
    _N = 1;
    if(_C != 4){
        LOGI << "ERR_MODEL_NOT_MATCH: model input should be 4 channels.";
        return RET_CODE::ERR_MODEL_NOT_MATCH;
    }
    // _oH = ptrHandle->outputShape_.H();
    // _oW = ptrHandle->outputShape_.W();
    // _oC = ptrHandle->outputShape_.C();
    //mlu resize
    {
        mluColorMode colorMode = mluColorMode::YUV2RGBA_NV21;
        switch (_model_input_fmt)
        {
        case MODEL_INPUT_FORMAT::RGBA :
            colorMode = mluColorMode::YUV2RGBA_NV21;
            break;
        case MODEL_INPUT_FORMAT::BGRA :
            colorMode = mluColorMode::YUV2BGRA_NV21;
            break;        
        default:
            break;
        }
        LOGI << "INIT MLU RESIZE OP";
        edk::MluContext* env = reinterpret_cast<edk::MluContext*>(env_);
        create_mlu_resize_func_light(ptrHandle, env, colorMode, _H, _W ,_pad_both_side, _keep_aspect_ratio);
    }
    return RET_CODE::SUCCESS;
}

void YuvCropResizeModel::release(){
    if(_ptrHandle!=nullptr){
        PtrHandle* ptrHandle = reinterpret_cast<PtrHandle*>(_ptrHandle);
        ptrHandle->rc_op_mlu_.Destroy();
        delete reinterpret_cast<PtrHandle*>(_ptrHandle);
        _ptrHandle = nullptr;
    }
}

YuvCropResizeModel::~YuvCropResizeModel(){
    LOGI << "-> ~YuvCropResizeModel()";
    release();
}


RET_CODE YuvCropResizeModel::general_preprocess_yuv_on_mlu_phyAddr(TvaiImage &tvimage, TvaiRect roiRect, cv::Mat& cropped_img, float &aspect_ratio, float &aX, float &aY){
    if(tvimage.format != TVAI_IMAGE_FORMAT_NV12 && tvimage.format !=TVAI_IMAGE_FORMAT_NV21 )
        return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    if (_ptrHandle == nullptr )
        return RET_CODE::ERR_MODEL_NOT_INIT;    
    PtrHandle* ptrHandle = reinterpret_cast<PtrHandle*>(_ptrHandle);
    assert(_C==4);
    mluColorMode colormode = getMLUTransColorMode(_model_input_fmt, tvimage.format);
    float expand_ratio = 1.0;
    {
        if (colormode != ptrHandle->rc_attr_.color_mode){
            LOGI << "re-initial mlu resize op";
            create_mlu_resize_func_light(ptrHandle, reinterpret_cast<edk::MluContext*>(env_), colormode, _H, _W, _pad_both_side, _keep_aspect_ratio);
        }
        cropped_img = Mat::zeros(Size(_W,_H), CV_8UC4);
        void* rc_output_mlu;
        cnrtMalloc(&rc_output_mlu, _C*_H*_W);;
        edk::MluResizeConvertOp::InputData input;
        input.planes[0] = (void*)tvimage.u64PhyAddr[0];
        input.planes[1] = (void*)tvimage.u64PhyAddr[1];
        input.src_w = tvimage.width;
        input.src_h = tvimage.height;
        input.src_stride = tvimage.stride;
        roiRect = get_valid_rect(roiRect, tvimage.width, tvimage.height);
        input.crop_x = roiRect.x; input.crop_y = roiRect.y;
        input.crop_w = roiRect.width; input.crop_h = roiRect.height;
        ptrHandle->rc_op_mlu_.BatchingUp(input);
        if (!ptrHandle->rc_op_mlu_.SyncOneOutput(rc_output_mlu)) {
        THROW_EXCEPTION(edk::Exception::INTERNAL, ptrHandle->rc_op_mlu_.GetLastError());}
        aX = (1.0*_W)/roiRect.width;
        aY = (1.0*_H)/roiRect.height;
        aspect_ratio = MIN( aX , aY );

        cnrtMemcpy(cropped_img.data, rc_output_mlu,  _C*_H*_W, CNRT_MEM_TRANS_DIR_DEV2HOST);
        cnrtFree(rc_output_mlu);
    }
    return RET_CODE::SUCCESS;
}

RET_CODE YuvCropResizeModel::general_preprocess_yuv_on_mlu(TvaiImage &tvimage, TvaiRect roiRect, cv::Mat& cropped_img ,float &aspect_ratio,float &aX, float &aY){
    if(tvimage.format != TVAI_IMAGE_FORMAT_NV12 && tvimage.format !=TVAI_IMAGE_FORMAT_NV21 )
        return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    if (_ptrHandle == nullptr )
        return RET_CODE::ERR_MODEL_NOT_INIT;    
    PtrHandle* ptrHandle = reinterpret_cast<PtrHandle*>(_ptrHandle);

    // std::lock_guard<std::mutex> lk(_mlu_mutex);
    mluColorMode colormode = getMLUTransColorMode(_model_input_fmt, tvimage.format);
    float expand_ratio = 1.0;
    {
        if (colormode != ptrHandle->rc_attr_.color_mode){
            LOGI << "re-initial mlu resize op";
            create_mlu_resize_func_light(ptrHandle, reinterpret_cast<edk::MluContext*>(env_), colormode, _H, _W , _pad_both_side, _keep_aspect_ratio);
        }
        void* rc_input_mlu, *rc_output_mlu;
        assert(sizeof(unsigned char)==1);
        cnrtMalloc(&rc_output_mlu, _C*_H*_W);
        cnrtMalloc(&rc_input_mlu, 3*tvimage.stride* tvimage.height/2);
        cnrtMemcpy(rc_input_mlu, tvimage.pData, 3*tvimage.stride*tvimage.height/2, CNRT_MEM_TRANS_DIR_HOST2DEV);
        cropped_img = Mat::zeros(Size(_W,_H), CV_8UC4);

        edk::MluResizeConvertOp::InputData input;
        input.planes[0] = rc_input_mlu;
        input.planes[1] = rc_input_mlu+tvimage.height*(tvimage.stride);
        input.src_w = tvimage.width;
        input.src_h = tvimage.height;
        input.src_stride = tvimage.stride;
        roiRect = get_valid_rect(roiRect, tvimage.width, tvimage.height);
        input.crop_x = roiRect.x; input.crop_y = roiRect.y;
        input.crop_w = roiRect.width; input.crop_h = roiRect.height;
        ptrHandle->rc_op_mlu_.BatchingUp(input);
        if (!ptrHandle->rc_op_mlu_.SyncOneOutput(rc_output_mlu)) {
        THROW_EXCEPTION(edk::Exception::INTERNAL, ptrHandle->rc_op_mlu_.GetLastError());}
        
        aX = (1.0*_W)/roiRect.width;
        aY = (1.0*_H)/roiRect.height;
        aspect_ratio = MIN( aX , aY );

        cnrtMemcpy(cropped_img.data, rc_output_mlu,  _C*_H*_W, CNRT_MEM_TRANS_DIR_DEV2HOST);
        
        cnrtFree(rc_input_mlu);
        cnrtFree(rc_output_mlu);
    }
    return RET_CODE::SUCCESS;
}

RET_CODE YuvCropResizeModel::general_preprocess_infer_bgr_on_cpu(TvaiImage &tvimage, VecObjBBox& bboxes, std::vector<cv::Mat> &cropped_imgs, 
    std::vector<float> &aspect_ratios, std::vector<CLS_TYPE> &valid_class){
    if(tvimage.format != TVAI_IMAGE_FORMAT_BGR && tvimage.format !=TVAI_IMAGE_FORMAT_RGB )
        return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    if (_ptrHandle == nullptr )
        return RET_CODE::ERR_MODEL_NOT_INIT;    
    PtrHandle* ptrHandle = reinterpret_cast<PtrHandle*>(_ptrHandle);
    assert(_C==4);
    bool input_rgb, output_rgb;
    float _aspect_ratio = 1.0;
    getCPUTransColorMode(_model_input_fmt, tvimage.format, output_rgb, input_rgb);

    Mat cvimage(tvimage.height, tvimage.width, CV_8UC3, tvimage.pData);
    for(int i = 0; i < bboxes.size(); i++){
        if( !check_valid_class(bboxes[i].objtype ,valid_class) ) {
            cropped_imgs.push_back(Mat());
            continue;
        }
        TvaiRect roiRect = bboxes[i].rect;
        Mat sub_cvimage, target_cvimage;
        getRectSubPix(cvimage, Size(roiRect.width, roiRect.height), 
            Point2f(float(roiRect.x + (1.0*roiRect.width)/2), float(roiRect.y + (1.0*roiRect.height)/2)) , sub_cvimage);
        
        if(_keep_aspect_ratio){
            target_cvimage = resize(sub_cvimage, Size(_W,_H),input_rgb, output_rgb, _pad_both_side, _aspect_ratio);
            // imwrite("1.bmp", target_cvimage);
            aspect_ratios.push_back(_aspect_ratio);
        } else {
            float aX=1.0; float aY=1.0;
            target_cvimage = resize_no_aspect(sub_cvimage, Size(_W,_H),input_rgb, output_rgb, aX, aY);
            aspect_ratios.push_back(aX);
            aspect_ratios.push_back(aY);
        }
        cropped_imgs.push_back(target_cvimage);
    }
    return RET_CODE::SUCCESS;
}

RET_CODE YuvCropResizeModel::general_preprocess_infer_bgr_on_cpu(TvaiImage &tvimage, std::vector<TvaiRect>& roiRects, std::vector<cv::Mat> &cropped_imgs, std::vector<float> &aspect_ratios){
    if(tvimage.format != TVAI_IMAGE_FORMAT_BGR && tvimage.format !=TVAI_IMAGE_FORMAT_RGB )
        return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    if (_ptrHandle == nullptr )
        return RET_CODE::ERR_MODEL_NOT_INIT;    
    PtrHandle* ptrHandle = reinterpret_cast<PtrHandle*>(_ptrHandle);
    assert(_C==4);
    bool input_rgb, output_rgb;
    float _aspect_ratio = 1.0;
    getCPUTransColorMode(_model_input_fmt, tvimage.format, output_rgb, input_rgb);

    Mat cvimage(tvimage.height, tvimage.width, CV_8UC3, tvimage.pData);
    for(int i = 0; i < roiRects.size(); i++){
        TvaiRect roiRect = roiRects[i];
        Mat sub_cvimage, target_cvimage;
        getRectSubPix(cvimage, Size(roiRect.width, roiRect.height), 
            Point2f(float(roiRect.x + (1.0*roiRect.width)/2), float(roiRect.y + (1.0*roiRect.height)/2)) , sub_cvimage);
        
        if(_keep_aspect_ratio){
            target_cvimage = resize(sub_cvimage, Size(_W,_H),input_rgb, output_rgb, _pad_both_side, _aspect_ratio);
            // imwrite("1.bmp", target_cvimage);
            aspect_ratios.push_back(_aspect_ratio);
        } else {
            float aX=1.0; float aY=1.0;
            target_cvimage = resize_no_aspect(sub_cvimage, Size(_W,_H),input_rgb, output_rgb, aX, aY);
            aspect_ratios.push_back(aX);
            aspect_ratios.push_back(aY);
        }
        cropped_imgs.push_back(target_cvimage);
    }
    return RET_CODE::SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// YuvCropResizeModel END
////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * BaseModelV2: 提供一些通用基础组件, 减少代码重复度, 便于后期维护
 * 设计目的: 需要支持多输入多输出的模型
 * 继承
 * - PrivateContext: MLU通用环境变量
 * */
////////////////////////////////////////////////////////////////////////////////////////////////////
// BaseModelV2 BEGIN
////////////////////////////////////////////////////////////////////////////////////////////////////
void BaseModelV2::release(){
    if(_ptrHandle!=nullptr){
        if (nullptr != mlu_output_) _ptrHandle->mem_op_.FreeMluOutput(mlu_output_);
        if (nullptr != cpu_output_) _ptrHandle->mem_op_.FreeCpuOutput(cpu_output_);
        if (nullptr != mlu_input_) _ptrHandle->mem_op_.FreeMluInput(mlu_input_);
        if (nullptr != cpu_input_) _ptrHandle->mem_op_.FreeCpuOutput(cpu_input_);
        // if(m_mem_pool!=nullptr) m_mem_pool->~MLUMemPool();
        _ptrHandle->rc_op_mlu_.Destroy();
        delete _ptrHandle;
        _ptrHandle = nullptr;
        mlu_output_ = nullptr;
        mlu_input_ = nullptr;
        cpu_input_ = nullptr;
        cpu_output_ = nullptr;
    }
}

BaseModelV2::~BaseModelV2(){
    LOGI << "-> ~BaseModelV2()";
    release();
}

RET_CODE BaseModelV2::base_init(WeightData wdata, BASE_CONFIG config){
    if(!status_){
        return RET_CODE::ERR_NPU_INIT_FAILED;
    }
    if (wdata.pData==nullptr){
        printf("**[%s][%d] ERR: model ptr is null\n",__FILE__, __LINE__);
        return RET_CODE::ERR_MODEL_FILE_NOT_EXIST;
    }

    release();
    //config
    _model_input_fmt = config.model_input_fmt;
    _model_output_order = config.model_output_order;
    _pad_both_side = config.pad_both_side;
    _keep_aspect_ratio = config.keep_aspect_ratio;
    // set mlu environment
    PtrHandleV2* ptrHandle = new PtrHandleV2();
    _ptrHandle = ptrHandle;

    ptrHandle->model_ = std::make_shared<edk::ModelLoader>(wdata.pData, "subnet0" );
    _MI = ptrHandle->model_->InputNum();
    _MO = ptrHandle->model_->OutputNum();
    //Reset cpu layout and get input output shape; default order is NHWC
    if(config.model_output_order==MODEL_OUTPUT_ORDER::NCHW){
        edk::DataLayout cpuoutputLayOut{ edk::DataType::FLOAT32, edk::DimOrder::NCHW };
        for(int i = 0; i < _MO; i++ )
            ptrHandle->model_->SetCpuOutputLayout(cpuoutputLayOut, i);
    }
    for(int i = 0; i < _MI; i++ ){
        ptrHandle->inputShape_.push_back(ptrHandle->model_->InputShape(i));
        if(ptrHandle->inputShape_[i].C() != 4 ){
            printf("** ERR_MODEL_NOT_MATCH: model input should be 4 channels.\n");
            return RET_CODE::ERR_MODEL_NOT_MATCH;
        }
    }
    for(int i = 0; i < _MO; i++ ){
        ptrHandle->outputShape_.push_back(ptrHandle->model_->OutputShape(i));
    }
    m_inputShape = ptrHandle->inputShape_;
    m_outputShape = ptrHandle->outputShape_;

    ptrHandle->mem_op_.SetModel(ptrHandle->model_);
    ptrHandle->infer_.Init(ptrHandle->model_,0);
    // std::cout << ptrHandle->model_->InputShape(0).C() << std::endl;
    // m_mem_pool = new MLUMemPool();
    // m_mem_pool->bind_mem_handle(ptrHandle);
    cpu_input_ = ptrHandle->mem_op_.AllocCpuInput();
    mlu_input_ = ptrHandle->mem_op_.AllocMluInput();
    mlu_output_ = ptrHandle->mem_op_.AllocMluOutput();
    cpu_output_ = ptrHandle->mem_op_.AllocCpuOutput();  

    //mlu resize
    {
        mluColorMode colorMode = mluColorMode::YUV2RGBA_NV21;
        switch (_model_input_fmt)
        {
        case MODEL_INPUT_FORMAT::RGBA :
            colorMode = mluColorMode::YUV2RGBA_NV21;
            break;
        case MODEL_INPUT_FORMAT::BGRA :
            colorMode = mluColorMode::YUV2BGRA_NV21;
            break;        
        default:
            break;
        }
        LOGI << "INIT MLU RESIZE OP";
        create_mlu_resize_func(ptrHandle, env_, colorMode, _pad_both_side, _keep_aspect_ratio);
    }
    return RET_CODE::SUCCESS;
}


RET_CODE BaseModelV2::base_init(const std::string &modelpath, BASE_CONFIG config){
    if(!status_){
        return RET_CODE::ERR_NPU_INIT_FAILED;
    }
    if (!exists_file(modelpath)){
        printf("** ERR: model file: %s not exist\n", modelpath.c_str());
        return RET_CODE::ERR_MODEL_FILE_NOT_EXIST;
    }

    release();
    //config
    _model_input_fmt = config.model_input_fmt;
    _model_output_order = config.model_output_order;
    _pad_both_side = config.pad_both_side;
    _keep_aspect_ratio = config.keep_aspect_ratio;
    // set mlu environment
    PtrHandleV2* ptrHandle = new PtrHandleV2();
    _ptrHandle = ptrHandle;

    ptrHandle->model_ = std::make_shared<edk::ModelLoader>(modelpath, "subnet0" );
    _MI = ptrHandle->model_->InputNum();
    _MO = ptrHandle->model_->OutputNum();
    //Reset cpu layout and get input output shape; default order is NHWC
    if(config.model_output_order==MODEL_OUTPUT_ORDER::NCHW){
        edk::DataLayout cpuoutputLayOut{ edk::DataType::FLOAT32, edk::DimOrder::NCHW };
        for(int i = 0; i < _MO; i++ )
            ptrHandle->model_->SetCpuOutputLayout(cpuoutputLayOut, i);
    }
    for(int i = 0; i < _MI; i++ ){
        ptrHandle->inputShape_.push_back(ptrHandle->model_->InputShape(i));
        if(ptrHandle->inputShape_[i].C() != 4 ){
            printf("** ERR_MODEL_NOT_MATCH: model input should be 4 channels.\n");
            return RET_CODE::ERR_MODEL_NOT_MATCH;
        }
    }
    for(int i = 0; i < _MO; i++ ){
        ptrHandle->outputShape_.push_back(ptrHandle->model_->OutputShape(i));
    }
    m_inputShape = ptrHandle->inputShape_;
    m_outputShape = ptrHandle->outputShape_;

    ptrHandle->mem_op_.SetModel(ptrHandle->model_);
    ptrHandle->infer_.Init(ptrHandle->model_,0);
    // std::cout << ptrHandle->model_->InputShape(0).C() << std::endl;
    // m_mem_pool = new MLUMemPool();
    // m_mem_pool->bind_mem_handle(ptrHandle);
    cpu_input_ = ptrHandle->mem_op_.AllocCpuInput();
    mlu_input_ = ptrHandle->mem_op_.AllocMluInput();
    mlu_output_ = ptrHandle->mem_op_.AllocMluOutput();
    cpu_output_ = ptrHandle->mem_op_.AllocCpuOutput();  

    //mlu resize
    {
        mluColorMode colorMode = mluColorMode::YUV2RGBA_NV21;
        switch (_model_input_fmt)
        {
        case MODEL_INPUT_FORMAT::RGBA :
            colorMode = mluColorMode::YUV2RGBA_NV21;
            break;
        case MODEL_INPUT_FORMAT::BGRA :
            colorMode = mluColorMode::YUV2BGRA_NV21;
            break;        
        default:
            break;
        }
        LOGI << "INIT MLU RESIZE OP";
        create_mlu_resize_func(ptrHandle, env_, colorMode, _pad_both_side, _keep_aspect_ratio);
    }
    return RET_CODE::SUCCESS;
}


float** BaseModelV2::general_mlu_infer(){
    LOGI << "-> BaseModelV2::general_mlu_infer";
    
    float** cpu_data = nullptr;
    cpu_data = (float**)malloc(sizeof(float*)*_MO);
    
    _ptrHandle->infer_.Run(mlu_input_,mlu_output_);
    _ptrHandle->mem_op_.MemcpyOutputD2H(cpu_output_, mlu_output_); //NHWC
    //HWC->CHW Already set by init()
    for(int i = 0; i< _MO ; i++ ){
        int outputSize = sizeof(float)*_ptrHandle->outputShape_[i].BatchDataCount();//20210903 适应batch推理
        float *cpu_chw = (float*)malloc(outputSize);
        memcpy(cpu_chw, reinterpret_cast<float*>(cpu_output_[i]),outputSize);
        cpu_data[i] = cpu_chw;
    }
    return cpu_data;     
}

// float** BaseModelV2::general_mlu_infer(MLUMemNode* dataPtr){
//     LOGI << "-> BaseModelV2::general_mlu_infer";
    
//     float** cpu_data = nullptr;
//     cpu_data = (float**)malloc(sizeof(float*)*_MO);
//     _ptrHandle->infer_.Run(dataPtr->mlu_input_ptr,dataPtr->mlu_output_ptr);
//     for(int i = 0; i< _MO ; i++ ){
//         int outputSize = sizeof(float)*_ptrHandle->outputShape_[i].BatchDataCount();//20210903 适应batch推理
//         float *cpu_chw = (float*)malloc(outputSize);
//         // memcpy(cpu_chw, reinterpret_cast<float*>(cpu_output_[i]),outputSize);
//         cpu_data[i] = cpu_chw;
//     }
//     _ptrHandle->mem_op_.MemcpyOutputD2H( reinterpret_cast<void**>(cpu_data), dataPtr->mlu_output_ptr); //NHWC
//     m_mem_pool->free(dataPtr);//用完即归还
//     return cpu_data;     
// }

void BaseModelV2::cpu_free(float **ptrX){
    if(ptrX!=nullptr){
        for(int i=0; i < _MO ; i++){
            free( ptrX[i] );
        }
        free(ptrX);
    }
};

/** 
 * BASEMODEL: V2
 * TRANS:
 * ([Img]x1, [ROI]x1) -> [1,C,H,W]
 * 通用前处理: 将输入的图像数据处理后存入类共享的mlu空间.
 * 仅支持yuv格式, 因为yuv格式下可以端到端实现前处理及推理, 所以前处理及推理可以放在一个循环中进行.
 * IN:
 * tvimage: 输入图像数据
 * roiRect: 限定目标区域
 * OUT:
 * aspect_ratio: 保持长宽比例的缩放比例
 * aX: 不保持长宽比例下, X方向的缩放比例
 * aY: 不保持长宽比例下, Y方向的缩放比例
 * */
RET_CODE BaseModelV2::general_preprocess_yuv_on_mlu_union(TvaiImage &tvimage, TvaiRect roiRect, float &aspect_ratio, float &aX, float &aY){
    LOGI << "-> general_preprocess_yuv_on_mlu_union";
    if(tvimage.format != TVAI_IMAGE_FORMAT_NV12 && tvimage.format !=TVAI_IMAGE_FORMAT_NV21 )
        return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
        
    if (_ptrHandle == nullptr ) return RET_CODE::ERR_MODEL_NOT_INIT;    
    
    mluColorMode colormode = getMLUTransColorMode(_model_input_fmt, tvimage.format);
    float expand_ratio = 1.0;
    {
        if (colormode != _ptrHandle->rc_attr_.color_mode){
            LOGI << "re-initial mlu resize op";
            create_mlu_resize_func(_ptrHandle, env_, colormode, _pad_both_side, _keep_aspect_ratio);
        }

        for(int i = 0; i < _MI; i++ ){//loop MI:: Multiple Input
            int _W = _ptrHandle->inputShape_[i].W();
            int _H = _ptrHandle->inputShape_[i].H();
            edk::MluResizeConvertOp::InputData input;
            void* rc_input_mlu = nullptr;
            if(!tvimage.usePhyAddr){
                cnrtMalloc(&rc_input_mlu, 3*tvimage.stride* tvimage.height/2);//++
                cnrtMemcpy(rc_input_mlu, tvimage.pData, 3*tvimage.stride*tvimage.height/2, CNRT_MEM_TRANS_DIR_HOST2DEV);
                input.planes[0] = rc_input_mlu;
                input.planes[1] = rc_input_mlu+tvimage.height*tvimage.stride;
            } else {
                input.planes[0] = (void*)tvimage.u64PhyAddr[0];
                input.planes[1] = (void*)tvimage.u64PhyAddr[1];
            }
            
            input.src_w = tvimage.width;
            input.src_h = tvimage.height;
            input.src_stride = tvimage.stride;
            roiRect = get_valid_rect(roiRect, tvimage.width, tvimage.height);
            input.crop_x = roiRect.x; input.crop_y = roiRect.y;
            input.crop_w = roiRect.width; input.crop_h = roiRect.height;
            _ptrHandle->rc_op_mlu_.BatchingUp(input);
            void* rc_output_mlu = mlu_input_[i];
            if (!_ptrHandle->rc_op_mlu_.SyncOneOutput(rc_output_mlu)) {
            THROW_EXCEPTION(edk::Exception::INTERNAL, _ptrHandle->rc_op_mlu_.GetLastError());}
            aX = (1.0*_W)/roiRect.width;
            aY = (1.0*_H)/roiRect.height;
            aspect_ratio = MIN( aX , aY );
            if(rc_input_mlu) cnrtFree(rc_input_mlu);
        }//loop MI
    }
    LOGI << "<- general_preprocess_yuv_on_mlu_union";
    return RET_CODE::SUCCESS;
}

RET_CODE BaseModelV2::general_preprocess_bgr_on_cpu(TvaiImage &tvimage, float &aspect_ratio, float &aX, float &aY){
    if(tvimage.format != TVAI_IMAGE_FORMAT_BGR && tvimage.format !=TVAI_IMAGE_FORMAT_RGB )
        return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    if (_ptrHandle == nullptr )
        return RET_CODE::ERR_MODEL_NOT_INIT;    

    int data_index = 0;
    int _W = _ptrHandle->inputShape_[data_index].W();
    int _H = _ptrHandle->inputShape_[data_index].H();

    int inputSize = _ptrHandle->inputShape_[data_index].DataCount()*sizeof(float);
    bool input_rgb, output_rgb;
    getCPUTransColorMode(_model_input_fmt, tvimage.format, output_rgb, input_rgb);
    LOGI << "output_rgb = " << output_rgb << ", input_rgb = " << input_rgb;
    Mat src_cvimage(Size(tvimage.width,tvimage.height), CV_8UC3, tvimage.pData);
    Mat target_cvimage;
    if(_keep_aspect_ratio){
        target_cvimage = resize(src_cvimage, Size(_W,_H),input_rgb, output_rgb, _pad_both_side, aspect_ratio);
    } else {
        target_cvimage = resize_no_aspect(src_cvimage, Size(_W,_H),input_rgb, output_rgb, aX, aY);
    }
    target_cvimage.convertTo(target_cvimage, CV_32F);//20210901::fix(CV_32FC1)
    if(!target_cvimage.isContinuous())
        target_cvimage = target_cvimage.clone();
    memcpy(cpu_input_[data_index],target_cvimage.data, inputSize);
    _ptrHandle->mem_op_.MemcpyInputH2D(mlu_input_, cpu_input_);//20210811++
    LOGI << "<- general_preprocess_bgr_on_cpu";
    return RET_CODE::SUCCESS;
}

/**
 * ([Img]x1, [ROI]xT) -> [1,C,H,W]xT -> [1,oC,oH,oW]xT
 */
RET_CODE BaseModelV2::general_preprocess_infer_bgr_on_cpu(TvaiImage &tvimage, VecObjBBox& bboxes, std::vector<float*> &model_output, 
    std::vector<float> &aspect_ratios, std::vector<CLS_TYPE> &valid_class){
    if(tvimage.format != TVAI_IMAGE_FORMAT_BGR && tvimage.format !=TVAI_IMAGE_FORMAT_RGB ) return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    if (_ptrHandle == nullptr ) return RET_CODE::ERR_MODEL_NOT_INIT;

    int data_index = 0;
    int _W = _ptrHandle->inputShape_[data_index].W();
    int _H = _ptrHandle->inputShape_[data_index].H();
    int outputSize = _ptrHandle->outputShape_[data_index].DataCount()*sizeof(float); 
    int inputSize = _ptrHandle->inputShape_[data_index].DataCount()*sizeof(float);

    bool input_rgb, output_rgb;
    float _aspect_ratio = 1.0;
    getCPUTransColorMode(_model_input_fmt, tvimage.format, output_rgb, input_rgb);

    Mat cvimage(tvimage.height, tvimage.width, CV_8UC3, tvimage.pData);
    for(int i = 0; i < bboxes.size(); i++){
        if( !check_valid_class(bboxes[i].objtype ,valid_class) ) {
            model_output.push_back(nullptr);
            continue;
        }
        TvaiRect roiRect = bboxes[i].rect;
        Mat sub_cvimage, target_cvimage;
        getRectSubPix(cvimage, Size(roiRect.width, roiRect.height), 
            Point2f(float(roiRect.x + (1.0*roiRect.width)/2), float(roiRect.y + (1.0*roiRect.height)/2)) , sub_cvimage);
        
        if(_keep_aspect_ratio){
            target_cvimage = resize(sub_cvimage, Size(_W,_H),input_rgb, output_rgb, _pad_both_side, _aspect_ratio);
            aspect_ratios.push_back(_aspect_ratio);
        } else {
            float aX=1.0; float aY=1.0;
            target_cvimage = resize_no_aspect(sub_cvimage, Size(_W,_H),input_rgb, output_rgb, aX, aY);
            aspect_ratios.push_back(aX);
            aspect_ratios.push_back(aY);
        }

        target_cvimage.convertTo(target_cvimage, CV_32F);
        if(!target_cvimage.isContinuous())
            target_cvimage = target_cvimage.clone();

        float *_model_output = (float*)malloc(outputSize);
        {// mutex
            memcpy(cpu_input_[data_index],target_cvimage.data, inputSize);
            _ptrHandle->mem_op_.MemcpyInputH2D(mlu_input_, cpu_input_);
            _ptrHandle->infer_.Run(mlu_input_,mlu_output_);
            _ptrHandle->mem_op_.MemcpyOutputD2H(cpu_output_, mlu_output_); //NHWC 1,1,1,512
            //HWC->CHW Already set by init()
            memcpy(_model_output, cpu_output_[data_index] ,outputSize);
            model_output.push_back(_model_output);
        }
    }
    return RET_CODE::SUCCESS;
}

/**
 * ([Img]x1, [ROI]xT) -> [1,C,H,W]xT -> [1,oC,oH,oW]xT
 */
RET_CODE BaseModelV2::general_preprocess_infer_bgr_on_cpu(TvaiImage &tvimage, std::vector<TvaiRect>& roiRects, std::vector<float*> &model_output, std::vector<float> &aspect_ratios){
    if(tvimage.format != TVAI_IMAGE_FORMAT_BGR && tvimage.format !=TVAI_IMAGE_FORMAT_RGB )
        return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    if (_ptrHandle == nullptr )
        return RET_CODE::ERR_MODEL_NOT_INIT;    

    int data_index = 0;
    int _W = _ptrHandle->inputShape_[data_index].W();
    int _H = _ptrHandle->inputShape_[data_index].H();
    int outputSize = _ptrHandle->outputShape_[data_index].DataCount()*sizeof(float); 
    int inputSize = _ptrHandle->inputShape_[data_index].DataCount()*sizeof(float);    

    bool input_rgb, output_rgb;
    float _aspect_ratio = 1.0;
    getCPUTransColorMode(_model_input_fmt, tvimage.format, output_rgb, input_rgb);

    Mat cvimage(tvimage.height, tvimage.width, CV_8UC3, tvimage.pData);
    for(int i = 0; i < roiRects.size(); i++){
        TvaiRect roiRect = roiRects[i];
        Mat sub_cvimage, target_cvimage;
        getRectSubPix(cvimage, Size(roiRect.width, roiRect.height), 
            Point2f(float(roiRect.x + (1.0*roiRect.width)/2), float(roiRect.y + (1.0*roiRect.height)/2)) , sub_cvimage);
        
        if(_keep_aspect_ratio){
            target_cvimage = resize(sub_cvimage, Size(_W,_H),input_rgb, output_rgb, _pad_both_side, _aspect_ratio);
            aspect_ratios.push_back(_aspect_ratio);
        } else {
            float aX=1.0; float aY=1.0;
            target_cvimage = resize_no_aspect(sub_cvimage, Size(_W,_H),input_rgb, output_rgb, aX, aY);
            aspect_ratios.push_back(aX);
            aspect_ratios.push_back(aY);
        }
        target_cvimage.convertTo(target_cvimage, CV_32F);
        if(!target_cvimage.isContinuous())
            target_cvimage = target_cvimage.clone();

        float *_model_output = (float*)malloc(outputSize);
        {// mutex
            memcpy(cpu_input_[0],target_cvimage.data, inputSize);
            _ptrHandle->mem_op_.MemcpyInputH2D(mlu_input_, cpu_input_);
            _ptrHandle->infer_.Run(mlu_input_,mlu_output_);
            _ptrHandle->mem_op_.MemcpyOutputD2H(cpu_output_, mlu_output_); //NHWC 1,1,1,512
            //HWC->CHW Already set by init()
            memcpy(_model_output, cpu_output_[0] ,outputSize);
            model_output.push_back(_model_output);
        }
    }
    return RET_CODE::SUCCESS;
}


/**
 * ([Img]x1,[ROI]xB) -> [B,C,H,W], where B > 1 and B = _N
 * BATCH操作, yuv的物理地址和虚拟地址合并
 * 两种类型的BATCH
 * 第二种: 输入单个图像, 对ROI区域进行BATCH操作
 */
RET_CODE BaseModelV2::general_batch_preprocess_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox& bboxes,std::vector<float> &batch_aspect_ratio, int offset){
    LOGI << "-> BaseModelV2::general_batch_preprocess_yuv_on_mlu" ;
    if(tvimage.format != TVAI_IMAGE_FORMAT_NV12 && tvimage.format !=TVAI_IMAGE_FORMAT_NV21 )
        return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    if (_ptrHandle == nullptr )
        return RET_CODE::ERR_MODEL_NOT_INIT;    
    mluColorMode colormode = getMLUTransColorMode(_model_input_fmt, tvimage.format);
    if (colormode != _ptrHandle->rc_attr_.color_mode){
        LOGI << "re-initial mlu resize op";
        create_mlu_resize_func(_ptrHandle, env_, colormode, _pad_both_side, _keep_aspect_ratio);
    }
    //assert the first bbox is not null
    if(bboxes.empty()) return RET_CODE::ERR_EMPTY_BOX;
    float expand_ratio = 1.0;
    std::vector<void*> rc_input_mlu_list;

    int _N = m_inputShape[0].BatchSize();
    int _W = m_inputShape[0].W();
    int _H = m_inputShape[0].H();

    int imW = tvimage.width;
    int imH = tvimage.height;
    int imS = tvimage.stride;
    int L = MIN( _N +offset, bboxes.size());
    for(int b = offset; b < L; b++ )
    {
        edk::MluResizeConvertOp::InputData input;
        if(!tvimage.usePhyAddr){
            void* rc_input_mlu;
            cnrtMalloc(&rc_input_mlu, 3*imS* imH/2);//++
            rc_input_mlu_list.push_back(rc_input_mlu);
            // tempPool.add(rc_input_mlu, cnrtFree);
            cnrtMemcpy(rc_input_mlu, tvimage.pData, 3*imS*imH/2, CNRT_MEM_TRANS_DIR_HOST2DEV);
            input.planes[0] = rc_input_mlu;
            input.planes[1] = rc_input_mlu+imH*imS;
        } else {
            input.planes[0] = reinterpret_cast<void*>(tvimage.u64PhyAddr[0]);
            input.planes[1] = reinterpret_cast<void*>(tvimage.u64PhyAddr[1]);
        }
        input.src_w = imW;
        input.src_h = imH;
        input.src_stride = imS;
        int cur_box_index = b;
        TvaiRect validROI = get_valid_rect(bboxes[cur_box_index].rect, imW, imH);
        input.crop_x = validROI.x; input.crop_y = validROI.y;
        input.crop_w = validROI.width; input.crop_h = validROI.height;
        _ptrHandle->rc_op_mlu_.BatchingUp(input);
        batch_aspect_ratio.push_back(MIN( (1.0*_W)/imW , (1.0*_H)/imH ));
    }
    void* rc_output = mlu_input_[0];
    if (!_ptrHandle->rc_op_mlu_.SyncOneOutput(rc_output)) {
        THROW_EXCEPTION(edk::Exception::INTERNAL, _ptrHandle->rc_op_mlu_.GetLastError());}
    // std::cout << std::endl;
    for(int i=0; i < rc_input_mlu_list.size(); i++)
        cnrtFree(rc_input_mlu_list[i]);//--
    return RET_CODE::SUCCESS;
}

/**
 * ([Img]xB,[ROI]xB) -> [B,C,H,W], where B > 1 and B = _N
 */
RET_CODE BaseModelV2::general_batch_preprocess_yuv_on_mlu(BatchImageIN &batch_tvimage, 
std::vector<TvaiRect> &batch_roiRect, std::vector<float> &batch_aspect_ratio){
    if( batch_tvimage.empty() ) return RET_CODE::SUCCESS;
    int N = batch_tvimage.size();
    int _N = m_inputShape[0].BatchSize();
    int _W = m_inputShape[0].W();
    int _H = m_inputShape[0].H();
    if( N != _N ) { 
        printf("not equal N[%d] and _N[%d]\n", N, _N);
        return RET_CODE::ERR_MODEL_NOT_MATCH; }
    if(batch_tvimage[0].format != TVAI_IMAGE_FORMAT_NV12 && batch_tvimage[0].format !=TVAI_IMAGE_FORMAT_NV21 )
        return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    if (_ptrHandle == nullptr )
        return RET_CODE::ERR_MODEL_NOT_INIT;    
    mluColorMode colormode = getMLUTransColorMode(_model_input_fmt, batch_tvimage[0].format);
    if (colormode != _ptrHandle->rc_attr_.color_mode){
        LOGI << "re-initial mlu resize op";
        create_mlu_resize_func(_ptrHandle, env_, colormode, _pad_both_side, _keep_aspect_ratio);
    }
    bool use_whole_image = false;
    if(batch_roiRect.empty()) use_whole_image = true;

    float expand_ratio = 1.0;
    std::vector<void*> rc_input_mlu_list;
    for(int b = 0; b < N; b++ )
    {
        int imW = batch_tvimage[b].width;
        int imH = batch_tvimage[b].height;
        int imS = batch_tvimage[b].stride;
        edk::MluResizeConvertOp::InputData input;

        if(!batch_tvimage[b].usePhyAddr){
            void* rc_input_mlu;
            cnrtMalloc(&rc_input_mlu, 3*imS* imH/2);//++
            rc_input_mlu_list.push_back(rc_input_mlu);
            cnrtMemcpy(rc_input_mlu, batch_tvimage[b].pData, 3*imS*imH/2, CNRT_MEM_TRANS_DIR_HOST2DEV);
            input.planes[0] = rc_input_mlu;
            input.planes[1] = rc_input_mlu+imH*imS;
        } else {
            input.planes[0] = reinterpret_cast<void*>(batch_tvimage[b].u64PhyAddr[0]);
            input.planes[1] = reinterpret_cast<void*>(batch_tvimage[b].u64PhyAddr[1]);
        }
        input.src_w = imW;
        input.src_h = imH;
        input.src_stride = imS;
        if(!use_whole_image){
            TvaiRect roiRect = get_valid_rect(batch_roiRect[b], imW, imH);
            input.crop_x = roiRect.x; input.crop_y = roiRect.y;
            input.crop_w = roiRect.width; input.crop_h = roiRect.height;
        }
        _ptrHandle->rc_op_mlu_.BatchingUp(input);
        batch_aspect_ratio.push_back(MIN( (1.0*_W)/imW , (1.0*_H)/imH ));
    }
    void* rc_output = mlu_input_[0];
    if (!_ptrHandle->rc_op_mlu_.SyncOneOutput(rc_output)) {
        THROW_EXCEPTION(edk::Exception::INTERNAL, _ptrHandle->rc_op_mlu_.GetLastError());}
    for(int i=0; i < rc_input_mlu_list.size(); i++)
        cnrtFree(rc_input_mlu_list[i]);//--
    return RET_CODE::SUCCESS;
}


////////////////////////////////////////////////////////////////////////////////////////////////////
// BaseModelV2 END
////////////////////////////////////////////////////////////////////////////////////////////////////

static std::map<std::string, CLS_TYPE> map_str_to_type = {
    {"person",  CLS_TYPE::PEDESTRIAN},
    {"vehicle", CLS_TYPE::CAR},
    {"non_vehicle",CLS_TYPE::NONCAR},
    {"face",    CLS_TYPE::FACE},
    {"ebycycle", CLS_TYPE::EBYCYCLE},
    {"bycycle", CLS_TYPE::BYCYCLE},
    {"banner", CLS_TYPE::BANNER},
    {"fire", CLS_TYPE::FIRE},
    {"water_puddle", CLS_TYPE::WATER_PUDDLE},
    {"person_fall", CLS_TYPE::PEDESTRIAN_FALL},
    {"person_safety_hat", CLS_TYPE::PED_SAFETY_HAT},
    {"person_head", CLS_TYPE::PED_HEAD},
    {"trashbag", CLS_TYPE::TRASH_BAG},
    {"smoking", CLS_TYPE::SMOKING},
    {"fighting", CLS_TYPE::FIGHT},
    {"falling_object", CLS_TYPE::FALLING_OBJ},
    {"falling_object_uncertain", CLS_TYPE::FALLING_OBJ_UNCERTAIN},
};

void transform_string_to_cls_type(std::vector<std::string> &vec_str, std::vector<CLS_TYPE> &vec_t){
    for(auto &&str: vec_str){
        if(map_str_to_type.find(str)!=map_str_to_type.end()){
            vec_t.push_back(map_str_to_type[str]);
        } else{
            vec_t.push_back(CLS_TYPE::OTHERS);
        }
    }
}


/**
 * IN: "/xxx/xxx/xxx.cambricon"
 * OUT: "/xxx/xxx/"
 */
static std::string get_root_path(std::string &filepath){
    auto t = filepath.find_last_of('/');
    if( t >= filepath.length() ) return "";
    return filepath.substr(0,t+1);
}
static void get_unique_path(std::vector<std::string> &pathIN, std::vector<std::string> &uniquePathOUT){
    std::map<std::string,bool> pathSet;
    for(auto &&path: pathIN){
        pathSet[path] = true;
    }
    for(auto path: pathSet){
        uniquePathOUT.push_back(path.first);
    }
}
RET_CODE auto_model_file_search(std::vector<std::string> &roots, std::map<InitParam, std::string> &fileBeginNameIN, std::map<InitParam, std::string> &modelpathOUT){
    std::vector<std::string> _roots, __roots;
    for(auto &&modelpath: modelpathOUT){
        _roots.push_back(get_root_path(modelpath.second));
    }
    _roots.insert(_roots.end(),roots.begin(), roots.end() );
    get_unique_path(_roots, __roots);
    // for(auto &&root: __roots){
    //     std::cout << root << std::endl;
    // }
    modelpathOUT.clear();
    for(auto &&m_root: __roots){//loop 所有根目录
        std::vector<std::string> modelfiles;
        ls_files(m_root, modelfiles, ".cambricon");//获取根目录下所有模型文件
        for(auto &&fileBeginName: fileBeginNameIN ){//loop 指定的文件名开始字符串
            for(auto &&modelfile: modelfiles){//loop 目录下所有模型文件所有模型文件,判断是否匹配
                if(hasBegining(modelfile, fileBeginName.second )) modelpathOUT[fileBeginName.first] = m_root + modelfile;
            }
        }
    }

    //展示找到的对应的模型文件
    for(auto &&modelpath: modelpathOUT){
        std::cout << "InitParam[" << int(modelpath.first) << "] auto loading model from: " << modelpath.second << std::endl;
    }

    //判断是否有缺失
    RET_CODE ret = RET_CODE::SUCCESS;
    for(auto &&fileBeginName: fileBeginNameIN){
        if( modelpathOUT.find(fileBeginName.first) == modelpathOUT.end() ){
            std::cout << "InitParm[" << int(fileBeginName.first) << "]: " << fileBeginName.second << " is not matched" << std::endl;
            ret = RET_CODE::ERR_MODEL_FILE_NOT_EXIST;
        }
    }

    return ret;
}


TvaiRect globalscaleTvaiRect(TvaiRect &rect, float scale, int W, int H){
    /**
     * H,W is the border of image 
     */
    TvaiRect output;
    float cx = rect.x + rect.width/2;
    float cy = rect.y + rect.height/2;
    output.width = rect.width*scale;
    output.height = rect.height*scale;
    output.x = std::max(cx - output.width/2,  0.f);
    output.y = std::max(cy - output.height/2, 0.f);
    output.width = std::min(W - output.x, output.width);
    output.height = std::min(H - output.y, output.height);
    return output;
}


/*******************************************************************************
MLUMemPool
适配寒武纪
*******************************************************************************/
MLUMemPool::~MLUMemPool(){
    MLUMemNode* nodePtr=nullptr;
    while(freeNodeHeader){
        nodePtr = freeNodeHeader->next;
        _free_(freeNodeHeader);
        freeNodeHeader = nodePtr;
    }
    if(!occupiedNodes.empty()){
        printf("Error: nodes in mem pool is still occupied!\n");
        for(auto &node: occupiedNodes){
            _free_(node);
        }
        printf("manually released\n");
    }
}

/*---内部函数, 释放节点内部开辟的空间---*/
void MLUMemPool::_free_(MLUMemNode* ptr){
    if(ptr!=nullptr){
        // if(ptr->cpu_input_ptr) memHandle->mem_op_.FreeCpuInput(ptr->cpu_input_ptr);
        // if(ptr->cpu_output_ptr) memHandle->mem_op_.FreeCpuOutput(ptr->cpu_output_ptr);
        if(ptr->mlu_input_ptr) memHandle->mem_op_.FreeMluInput(ptr->mlu_input_ptr);
        if(ptr->mlu_output_ptr) memHandle->mem_op_.FreeMluOutput(ptr->mlu_output_ptr);
        // ptr->cpu_input_ptr = nullptr;
        // ptr->cpu_output_ptr = nullptr;
        ptr->mlu_input_ptr = nullptr;
        ptr->mlu_output_ptr = nullptr;
    }
}
/*---内部函数, 在节点内开辟空间, 输入必须new开辟过空间---*/
bool MLUMemPool::_malloc_(MLUMemNode* ptr){
    if(ptr!=nullptr){
        // ptr->cpu_input_ptr = memHandle->mem_op_.AllocCpuInput();
        // ptr->cpu_output_ptr = memHandle->mem_op_.AllocCpuOutput();
        ptr->mlu_input_ptr = memHandle->mem_op_.AllocMluInput();
        ptr->mlu_output_ptr = memHandle->mem_op_.AllocMluOutput();
        return true;
    }else{
        return false;
    }
}

// void MLUMemPool::init(int num_of_nodes){
//     LOGI << "-> creating nodes in MLUMemPool::init()";
//     freeNodeHeader = create(num_of_nodes);
//     LOGI << "<- creating nodes in MLUMemPool::int()";
// }

MLUMemNode* MLUMemPool::create(int num_of_nodes){
    MLUMemNode* curFreeNodeHeader = nullptr;
    for(int i=0; i< num_of_nodes; i++){
        MLUMemNode* newNode = new MLUMemNode;
        _malloc_(newNode);
        if(curFreeNodeHeader){
            MLUMemNode* tmpNode;
            tmpNode = curFreeNodeHeader;
            curFreeNodeHeader = newNode;
            newNode->next = tmpNode;
        }else{
            curFreeNodeHeader = newNode;
        }
    }
    numOfNodes += num_of_nodes;
    return curFreeNodeHeader;
}

MLUMemNode* MLUMemPool::malloc(){
    std::lock_guard<std::mutex> lock(_mlock_);
    if(freeNodeHeader==nullptr){
        //TODO需要加开空间
        LOGI << "creating more nodes";
        freeNodeHeader = create(1);
    }
    MLUMemNode* freeNode = freeNodeHeader;
    freeNodeHeader = freeNodeHeader->next;
    occupiedNodes.insert(freeNode);
    LOGI << "used nodes = " << occupiedNodes.size();
    return freeNode;
}

void MLUMemPool::free(MLUMemNode* nodePtr){
    std::lock_guard<std::mutex> lock(_mlock_);
    if(nodePtr!=nullptr){
        nodePtr->next = freeNodeHeader;
        freeNodeHeader = nodePtr;
        occupiedNodes.erase(nodePtr);
        LOGI << "remaining used nodes = " << occupiedNodes.size();
    }
}


/**
 * BaseModel: 提供一些通用基础组件, 减少代码重复度, 便于后期维护
 * 继承
 * - PrivateContext: MLU通用环境变量
 * - AlgoAPI: 抽象类, 主要用于暴露接口, 无任何私有变量
 * */
/////////////////////////////////////////////////////////////////////
// Begin of Class BaseModel 
/////////////////////////////////////////////////////////////////////
// RET_CODE BaseModel::base_init(const std::string &modelpath, BASE_CONFIG config){
//     if(!status_){
//         return RET_CODE::ERR_NPU_INIT_FAILED;
//     }
//     if (!exists_file(modelpath)){
//         std::cout << "model file: " << modelpath << " not exist" << std::endl;
//         return RET_CODE::ERR_MODEL_FILE_NOT_EXIST;
//     }

//     release();
//     //config
//     _model_input_fmt = config.model_input_fmt;
//     _model_output_order = config.model_output_order;
//     _pad_both_side = config.pad_both_side;
//     _keep_aspect_ratio = config.keep_aspect_ratio;
//     // set mlu environment
//     PtrHandle* ptrHandle = new PtrHandle();
//     _ptrHandle = ptrHandle;

//     ptrHandle->model_ = std::make_shared<edk::ModelLoader>(modelpath, "subnet0" );
//     _MI = ptrHandle->model_->InputNum();
//     //Reset cpu layout and get input output shape
//     if(config.model_output_order==MODEL_OUTPUT_ORDER::NCHW){
//         edk::DataLayout cpuoutputLayOut{ edk::DataType::FLOAT32, edk::DimOrder::NCHW };
//         ptrHandle->model_->SetCpuOutputLayout(cpuoutputLayOut, 0);
//         assert(edk::DimOrder::NHWC == ptrHandle->model_->GetCpuInputLayout(0).order);
//         assert(edk::DimOrder::NCHW == ptrHandle->model_->GetCpuOutputLayout(0).order);
//     }
//     ptrHandle->inputShape_ = ptrHandle->model_->InputShape(0);//默认所有输入大小相同
//     ptrHandle->outputShape_ = ptrHandle->model_->OutputShape(0);

//     _H = ptrHandle->inputShape_.H();
//     _W = ptrHandle->inputShape_.W();
//     _C = ptrHandle->inputShape_.C();
//     _N = ptrHandle->inputShape_.N();
//     // LOGI << _N << "inside batch size";
//     _oH = ptrHandle->outputShape_.H();
//     _oW = ptrHandle->outputShape_.W();
//     _oC = ptrHandle->outputShape_.C();
//     if(_C != 4){
//         LOGI << "ERR_MODEL_NOT_MATCH: model input should be 4 channels.";
//         return RET_CODE::ERR_MODEL_NOT_MATCH;
//     }
    
//     ptrHandle->mem_op_.SetModel(ptrHandle->model_);

//     ptrHandle->infer_.Init(ptrHandle->model_,0);

//     // std::cout << ptrHandle->model_->InputShape(0).C() << std::endl;
//     cpu_input_ = ptrHandle->mem_op_.AllocCpuInput();
//     mlu_input_ = ptrHandle->mem_op_.AllocMluInput();
//     mlu_output_ = ptrHandle->mem_op_.AllocMluOutput();
//     cpu_output_ = ptrHandle->mem_op_.AllocCpuOutput();  

//     //mlu resize
//     {
//         mluColorMode colorMode = mluColorMode::YUV2RGBA_NV21;
//         switch (_model_input_fmt)
//         {
//         case MODEL_INPUT_FORMAT::RGBA :
//             colorMode = mluColorMode::YUV2RGBA_NV21;
//             break;
//         case MODEL_INPUT_FORMAT::BGRA :
//             colorMode = mluColorMode::YUV2BGRA_NV21;
//             break;        
//         default:
//             break;
//         }
//         LOGI << "INIT MLU RESIZE OP";
//         edk::MluContext* env = reinterpret_cast<edk::MluContext*>(env_);
//         create_mlu_resize_func(ptrHandle, env, colorMode, _pad_both_side, _keep_aspect_ratio);
//     }
//     return RET_CODE::SUCCESS;
// }

// void BaseModel::release(){
//     if(_ptrHandle!=nullptr){
//         PtrHandle* ptrHandle = reinterpret_cast<PtrHandle*>(_ptrHandle);
//         if (nullptr != mlu_output_) ptrHandle->mem_op_.FreeMluOutput(mlu_output_);
//         if (nullptr != cpu_output_) ptrHandle->mem_op_.FreeCpuOutput(cpu_output_);
//         if (nullptr != mlu_input_) ptrHandle->mem_op_.FreeMluInput(mlu_input_);
//         if (nullptr != cpu_input_) ptrHandle->mem_op_.FreeCpuOutput(cpu_input_);
//         ptrHandle->rc_op_mlu_.Destroy();
//         delete reinterpret_cast<PtrHandle*>(_ptrHandle);
//         _ptrHandle = nullptr;
//     }
// }

// BaseModel::~BaseModel(){
//     LOGI << "-> ~BaseModel()";
//     release();
// }

// RET_CODE BaseModel::general_preprocess_yuv_on_mlu_union(TvaiImage &tvimage, TvaiRect roiRect, float &aspect_ratio, float &aX, float &aY){
//     LOGI << "-> general_preprocess_yuv_on_mlu_union";
//     if(tvimage.usePhyAddr)
//         return general_preprocess_yuv_on_mlu_phyAddr(tvimage, roiRect, aspect_ratio, aX, aY);
//     else
//         return general_preprocess_yuv_on_mlu(tvimage, roiRect, aspect_ratio, aX, aY);     
// }

// RET_CODE BaseModel::general_preprocess_yuv_on_mlu_phyAddr(TvaiImage &tvimage, TvaiRect roiRect, float &aspect_ratio, float &aX, float &aY){
//     if(tvimage.format != TVAI_IMAGE_FORMAT_NV12 && tvimage.format !=TVAI_IMAGE_FORMAT_NV21 )
//         return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
//     // if(tvimage.u64PhyAddr[0]==0 || tvimage.u64PhyAddr[1]==0)
//     //     return RET_CODE::ERR_PHYADDR_EMPTY;
//     if (_ptrHandle == nullptr )
//         return RET_CODE::ERR_MODEL_NOT_INIT;    
//     PtrHandle* ptrHandle = reinterpret_cast<PtrHandle*>(_ptrHandle);
//     assert(_C==4);
    
//     mluColorMode colormode = getMLUTransColorMode(_model_input_fmt, tvimage.format);
//     float expand_ratio = 1.0;
//     {
//         if (colormode != ptrHandle->rc_attr_.color_mode){
//             LOGI << "re-initial mlu resize op";
//             create_mlu_resize_func(ptrHandle, reinterpret_cast<edk::MluContext*>(env_), colormode, _pad_both_side, _keep_aspect_ratio);
//         }

//         void* rc_output_mlu = mlu_input_[0];
//         edk::MluResizeConvertOp::InputData input;
//         input.planes[0] = (void*)tvimage.u64PhyAddr[0];
//         input.planes[1] = (void*)tvimage.u64PhyAddr[1];
//         input.src_w = tvimage.width;
//         input.src_h = tvimage.height;
//         input.src_stride = tvimage.stride;
//         roiRect = get_valid_rect(roiRect, tvimage.width, tvimage.height);
//         input.crop_x = roiRect.x; input.crop_y = roiRect.y;
//         input.crop_w = roiRect.width; input.crop_h = roiRect.height;
//         ptrHandle->rc_op_mlu_.BatchingUp(input);
//         if (!ptrHandle->rc_op_mlu_.SyncOneOutput(rc_output_mlu)) {
//         THROW_EXCEPTION(edk::Exception::INTERNAL, ptrHandle->rc_op_mlu_.GetLastError());}
//         aX = (1.0*_W)/roiRect.width;
//         aY = (1.0*_H)/roiRect.height;
//         aspect_ratio = MIN( aX , aY );
//     }
//     LOGI << "<- general_preprocess_yuv_on_mlu_phyAddr";
//     return RET_CODE::SUCCESS;
// }

// RET_CODE BaseModel::general_preprocess_yuv_on_mlu_phyAddr(TvaiImage &tvimage, float &aspect_ratio, float &aX, float &aY){
//     TvaiRect roiRect{0,0,tvimage.width,tvimage.height};
//     return general_preprocess_yuv_on_mlu_phyAddr(tvimage, roiRect, aspect_ratio, aX, aY);
// }


// RET_CODE BaseModel::general_preprocess_yuv_on_mlu(TvaiImage &tvimage, TvaiRect roiRect, float &aspect_ratio,float &aX, float &aY){
//     if(tvimage.format != TVAI_IMAGE_FORMAT_NV12 && tvimage.format !=TVAI_IMAGE_FORMAT_NV21 )
//         return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
//     if (_ptrHandle == nullptr )
//         return RET_CODE::ERR_MODEL_NOT_INIT;    
//     PtrHandle* ptrHandle = reinterpret_cast<PtrHandle*>(_ptrHandle);

//     // std::lock_guard<std::mutex> lk(_mlu_mutex);
//     mluColorMode colormode = getMLUTransColorMode(_model_input_fmt, tvimage.format);
//     float expand_ratio = 1.0;
//     {
//         if (colormode != ptrHandle->rc_attr_.color_mode){
//             LOGI << "re-initial mlu resize op";
//             create_mlu_resize_func(ptrHandle, reinterpret_cast<edk::MluContext*>(env_), colormode, _pad_both_side, _keep_aspect_ratio);
//         }
//         void* rc_output = mlu_input_[0];
//         void* rc_input_mlu;
//         cnrtMalloc(&rc_input_mlu, 3*tvimage.stride* tvimage.height/2);
//         cnrtMemcpy(rc_input_mlu, tvimage.pData, 3*tvimage.stride*tvimage.height/2, CNRT_MEM_TRANS_DIR_HOST2DEV);

//         edk::MluResizeConvertOp::InputData input;
//         input.planes[0] = rc_input_mlu;
//         input.planes[1] = rc_input_mlu+tvimage.height*(tvimage.stride);
//         input.src_w = tvimage.width;
//         input.src_h = tvimage.height;
//         input.src_stride = tvimage.stride;
//         roiRect = get_valid_rect(roiRect, tvimage.width, tvimage.height);
//         // LOGI << "roiRect: " << roiRect.x << ", "<< roiRect.y  << ", "<< roiRect.width  << ", "<< roiRect.height;
//         input.crop_x = roiRect.x; input.crop_y = roiRect.y;
//         input.crop_w = roiRect.width; input.crop_h = roiRect.height;
//         ptrHandle->rc_op_mlu_.BatchingUp(input);
//         if (!ptrHandle->rc_op_mlu_.SyncOneOutput(rc_output)) {
//         THROW_EXCEPTION(edk::Exception::INTERNAL, ptrHandle->rc_op_mlu_.GetLastError());}
        
//         aX = (1.0*_W)/roiRect.width;
//         aY = (1.0*_H)/roiRect.height;
//         aspect_ratio = MIN( aX , aY );
        
//         cnrtFree(rc_input_mlu);
//     }
//     LOGI << "<- general_preprocess_yuv_on_mlu" ;
//     return RET_CODE::SUCCESS;
// }


// RET_CODE BaseModel::general_preprocess_yuv_on_mlu(TvaiImage &tvimage, float &aspect_ratio, float &aX, float &aY){
//     LOGI << "-> BaseModel::general_preprocess_yuv_on_mlu";
//     TvaiRect roiRect{0,0,tvimage.width, tvimage.height};
//     return general_preprocess_yuv_on_mlu(tvimage, roiRect, aspect_ratio, aX, aY);
// }

// /**
//  * ([Img]x1, [ROI]xT) -> [1,C,H,W]xT -> [1,oC,oH,oW]xT
//  */
// RET_CODE BaseModel::general_preprocess_infer_bgr_on_cpu(TvaiImage &tvimage, VecObjBBox& bboxes, std::vector<float*> &model_output, 
//     std::vector<float> &aspect_ratios, std::vector<CLS_TYPE> &valid_class){
//     if(tvimage.format != TVAI_IMAGE_FORMAT_BGR && tvimage.format !=TVAI_IMAGE_FORMAT_RGB )
//         return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
//     if (_ptrHandle == nullptr )
//         return RET_CODE::ERR_MODEL_NOT_INIT;    
//     PtrHandle* ptrHandle = reinterpret_cast<PtrHandle*>(_ptrHandle);
//     assert(_C==4);
//     int outputSize = ptrHandle->outputShape_.DataCount()*sizeof(float); 
//     int inputSize = ptrHandle->inputShape_.DataCount()*sizeof(float);
//     bool input_rgb, output_rgb;
//     float _aspect_ratio = 1.0;
//     getCPUTransColorMode(_model_input_fmt, tvimage.format, output_rgb, input_rgb);

//     Mat cvimage(tvimage.height, tvimage.width, CV_8UC3, tvimage.pData);
//     for(int i = 0; i < bboxes.size(); i++){
//         if( !check_valid_class(bboxes[i].objtype ,valid_class) ) {
//             model_output.push_back(nullptr);
//             continue;
//         }
//         TvaiRect roiRect = bboxes[i].rect;
//         Mat sub_cvimage, target_cvimage;
//         getRectSubPix(cvimage, Size(roiRect.width, roiRect.height), 
//             Point2f(float(roiRect.x + (1.0*roiRect.width)/2), float(roiRect.y + (1.0*roiRect.height)/2)) , sub_cvimage);
        
//         if(_keep_aspect_ratio){
//             target_cvimage = resize(sub_cvimage, Size(_W,_H),input_rgb, output_rgb, _pad_both_side, _aspect_ratio);
//             // imwrite("1.bmp", target_cvimage);
//             aspect_ratios.push_back(_aspect_ratio);
//         } else {
//             float aX=1.0; float aY=1.0;
//             target_cvimage = resize_no_aspect(sub_cvimage, Size(_W,_H),input_rgb, output_rgb, aX, aY);
//             aspect_ratios.push_back(aX);
//             aspect_ratios.push_back(aY);
//         }
//         // imwrite("p.bmp", target_cvimage);
//         // LOGI << "target_cvimage: " << target_cvimage.cols << ", " << target_cvimage.rows;
//         //Infer
//         target_cvimage.convertTo(target_cvimage, CV_32F);
//         if(!target_cvimage.isContinuous())
//             target_cvimage = target_cvimage.clone();

//         float *_model_output = (float*)malloc(outputSize);
//         {// mutex
//             memcpy(cpu_input_[0],target_cvimage.data, inputSize);
//             ptrHandle->mem_op_.MemcpyInputH2D(mlu_input_, cpu_input_);
//             ptrHandle->infer_.Run(mlu_input_,mlu_output_);
//             ptrHandle->mem_op_.MemcpyOutputD2H(cpu_output_, mlu_output_); //NHWC 1,1,1,512
//             //HWC->CHW Already set by init()
//             memcpy(_model_output, cpu_output_[0] ,outputSize);
//             model_output.push_back(_model_output);
//         }
//     }
//     return RET_CODE::SUCCESS;
// }

// /**
//  * ([Img]x1, [ROI]xT) -> [1,C,H,W]xT -> [1,oC,oH,oW]xT
//  */
// RET_CODE BaseModel::general_preprocess_infer_bgr_on_cpu(TvaiImage &tvimage, std::vector<TvaiRect>& roiRects, std::vector<float*> &model_output, std::vector<float> &aspect_ratios){
//     if(tvimage.format != TVAI_IMAGE_FORMAT_BGR && tvimage.format !=TVAI_IMAGE_FORMAT_RGB )
//         return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
//     if (_ptrHandle == nullptr )
//         return RET_CODE::ERR_MODEL_NOT_INIT;    
//     PtrHandle* ptrHandle = reinterpret_cast<PtrHandle*>(_ptrHandle);
//     assert(_C==4);
//     int outputSize = ptrHandle->outputShape_.DataCount()*sizeof(float); 
//     int inputSize = ptrHandle->inputShape_.DataCount()*sizeof(float);
//     bool input_rgb, output_rgb;
//     float _aspect_ratio = 1.0;
//     getCPUTransColorMode(_model_input_fmt, tvimage.format, output_rgb, input_rgb);

//     Mat cvimage(tvimage.height, tvimage.width, CV_8UC3, tvimage.pData);
//     for(int i = 0; i < roiRects.size(); i++){
//         TvaiRect roiRect = roiRects[i];
//         Mat sub_cvimage, target_cvimage;
//         getRectSubPix(cvimage, Size(roiRect.width, roiRect.height), 
//             Point2f(float(roiRect.x + (1.0*roiRect.width)/2), float(roiRect.y + (1.0*roiRect.height)/2)) , sub_cvimage);
        
//         if(_keep_aspect_ratio){
//             target_cvimage = resize(sub_cvimage, Size(_W,_H),input_rgb, output_rgb, _pad_both_side, _aspect_ratio);
//             // imwrite("1.bmp", target_cvimage);
//             aspect_ratios.push_back(_aspect_ratio);
//         } else {
//             float aX=1.0; float aY=1.0;
//             target_cvimage = resize_no_aspect(sub_cvimage, Size(_W,_H),input_rgb, output_rgb, aX, aY);
//             aspect_ratios.push_back(aX);
//             aspect_ratios.push_back(aY);
//         }
//         // imwrite("p.bmp", target_cvimage);
//         // LOGI << "target_cvimage: " << target_cvimage.cols << ", " << target_cvimage.rows;
//         //Infer
//         target_cvimage.convertTo(target_cvimage, CV_32F);
//         if(!target_cvimage.isContinuous())
//             target_cvimage = target_cvimage.clone();

//         float *_model_output = (float*)malloc(outputSize);
//         {// mutex
//             memcpy(cpu_input_[0],target_cvimage.data, inputSize);
//             ptrHandle->mem_op_.MemcpyInputH2D(mlu_input_, cpu_input_);
//             ptrHandle->infer_.Run(mlu_input_,mlu_output_);
//             ptrHandle->mem_op_.MemcpyOutputD2H(cpu_output_, mlu_output_); //NHWC 1,1,1,512
//             //HWC->CHW Already set by init()
//             memcpy(_model_output, cpu_output_[0] ,outputSize);
//             model_output.push_back(_model_output);
//         }
//     }
//     return RET_CODE::SUCCESS;
// }

// //默认使用keep aspect ratio, 需要后期修改, 已修改,
// RET_CODE BaseModel::general_preprocess_bgr_on_cpu(TvaiImage &tvimage, float &aspect_ratio, float &aX, float &aY){
//     if(tvimage.format != TVAI_IMAGE_FORMAT_BGR && tvimage.format !=TVAI_IMAGE_FORMAT_RGB )
//         return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
//     if (_ptrHandle == nullptr )
//         return RET_CODE::ERR_MODEL_NOT_INIT;    
//     PtrHandle* ptrHandle = reinterpret_cast<PtrHandle*>(_ptrHandle);
//     assert(_C==4);
//     int inputSize = ptrHandle->inputShape_.DataCount()*sizeof(float);
//     bool input_rgb, output_rgb;
//     getCPUTransColorMode(_model_input_fmt, tvimage.format, output_rgb, input_rgb);
//     LOGI << "output_rgb = " << output_rgb << ", input_rgb = " << input_rgb;
//     Mat src_cvimage(Size(tvimage.width,tvimage.height), CV_8UC3, tvimage.pData);
//     Mat target_cvimage;
//     if(_keep_aspect_ratio){
//         // LOGI << "keep aspect ratio";
//         target_cvimage = resize(src_cvimage, Size(_W,_H),input_rgb, output_rgb, _pad_both_side, aspect_ratio);
//         // imwrite("1.bmp", target_cvimage);
//     } else {
//         // LOGI << "dont keep aspect ratio";
//         target_cvimage = resize_no_aspect(src_cvimage, Size(_W,_H),input_rgb, output_rgb, aX, aY);
//     }
//     target_cvimage.convertTo(target_cvimage, CV_32F);//20210901::fix(CV_32FC1)
//     if(!target_cvimage.isContinuous())
//         target_cvimage = target_cvimage.clone();
//     memcpy(cpu_input_[0],target_cvimage.data, inputSize);
//     ptrHandle->mem_op_.MemcpyInputH2D(mlu_input_, cpu_input_);//20210811++
//     return RET_CODE::SUCCESS;
// }

// float* BaseModel::general_mlu_infer(){
//     LOGI << "-> BaseModel::general_mlu_infer";
//     PtrHandle* ptrHandle = reinterpret_cast<PtrHandle*>(_ptrHandle);
//     int outputSize = sizeof(float)*ptrHandle->outputShape_.BatchDataCount();//20210903 适应batch推理
//     float *cpu_chw = (float*)malloc(outputSize);
//     ptrHandle->infer_.Run(mlu_input_,mlu_output_);
//     ptrHandle->mem_op_.MemcpyOutputD2H(cpu_output_, mlu_output_); //NHWC
//     //HWC->CHW Already set by init()
//     memcpy(cpu_chw, reinterpret_cast<float*>(cpu_output_[0]),outputSize);
//     return cpu_chw;     
// }

// std::shared_ptr<float> BaseModel::general_mlu_infer_share_ptr(){
//     float* _ptrFeat = general_mlu_infer();
//     std::shared_ptr<float> ptrFeature;
//     ptrFeature.reset(_ptrFeat, free);
//     return ptrFeature;
// }

// /**
//  * ([Img]x1,[ROI]xB) -> [B,C,H,W], where B > 1 and B = _N
//  * BATCH操作, yuv的物理地址和虚拟地址合并
//  * 两种类型的BATCH
//  * 第二种: 输入单个图像, 对ROI区域进行BATCH操作
//  */
// RET_CODE BaseModel::general_batch_preprocess_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox& bboxes,std::vector<float> &batch_aspect_ratio, int offset){
//     LOGI << "-> BaseModel::general_batch_preprocess_yuv_on_mlu" ;
//     if(tvimage.format != TVAI_IMAGE_FORMAT_NV12 && tvimage.format !=TVAI_IMAGE_FORMAT_NV21 )
//         return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
//     if (_ptrHandle == nullptr )
//         return RET_CODE::ERR_MODEL_NOT_INIT;    
//     PtrHandle* ptrHandle = reinterpret_cast<PtrHandle*>(_ptrHandle);
//     mluColorMode colormode = getMLUTransColorMode(_model_input_fmt, tvimage.format);
//     if (colormode != ptrHandle->rc_attr_.color_mode){
//         LOGI << "re-initial mlu resize op";
//         create_mlu_resize_func(ptrHandle, reinterpret_cast<edk::MluContext*>(env_), colormode, _pad_both_side, _keep_aspect_ratio);
//     }
//     //assert the first bbox is not null
//     if(bboxes.empty() || bboxes[offset].objtype == CLS_TYPE::UNKNOWN) return RET_CODE::ERR_EMPTY_BOX;
//     float expand_ratio = 1.0;
//     std::vector<void*> rc_input_mlu_list;
//     // TemporalPtrPool<cnrtRet_t> tempPool;

//     int imW = tvimage.width;
//     int imH = tvimage.height;
//     int imS = tvimage.stride;
//     int L = MIN(_N+offset, bboxes.size());
//     std::vector<int> index_empty_data;
//     for(int b = offset; b < L; b++ )
//     {
//         edk::MluResizeConvertOp::InputData input;
//         if(!tvimage.usePhyAddr){
//             void* rc_input_mlu;
//             cnrtMalloc(&rc_input_mlu, 3*imS* imH/2);//++
//             rc_input_mlu_list.push_back(rc_input_mlu);
//             // tempPool.add(rc_input_mlu, cnrtFree);
//             cnrtMemcpy(rc_input_mlu, tvimage.pData, 3*imS*imH/2, CNRT_MEM_TRANS_DIR_HOST2DEV);
//             input.planes[0] = rc_input_mlu;
//             input.planes[1] = rc_input_mlu+imH*imS;
//         } else {
//             input.planes[0] = reinterpret_cast<void*>(tvimage.u64PhyAddr[0]);
//             input.planes[1] = reinterpret_cast<void*>(tvimage.u64PhyAddr[1]);
//         }
//         input.src_w = imW;
//         input.src_h = imH;
//         input.src_stride = imS;
//         int cur_box_index = b;
//         if(bboxes[cur_box_index].objtype == CLS_TYPE::UNKNOWN) {
//             index_empty_data.push_back(cur_box_index-offset);
//             cur_box_index = offset;
//         }//空数据情况下，使用第一个box，默认第一个box非空
//         TvaiRect validROI = get_valid_rect(bboxes[cur_box_index].rect, imW, imH);
//         input.crop_x = validROI.x; input.crop_y = validROI.y;
//         input.crop_w = validROI.width; input.crop_h = validROI.height;
//         ptrHandle->rc_op_mlu_.BatchingUp(input);
//         batch_aspect_ratio.push_back(MIN( (1.0*_W)/imW , (1.0*_H)/imH ));
//     }
//     void* rc_output = mlu_input_[0];
//     if (!ptrHandle->rc_op_mlu_.SyncOneOutput(rc_output)) {
//         THROW_EXCEPTION(edk::Exception::INTERNAL, ptrHandle->rc_op_mlu_.GetLastError());}
//     for(auto ind: index_empty_data){//ATT::offset 是否存在问题?
//         cnrtMemset(mlu_input_[0] + ind*_H*_W*_C*size_t(1), 0, _H*_W*_C*size_t(1));
//     }
//     // std::cout << std::endl;
//     for(int i=0; i < rc_input_mlu_list.size(); i++)
//         cnrtFree(rc_input_mlu_list[i]);//--
//     return RET_CODE::SUCCESS;
// }

// /**
//  * ([Img]xB,[ROI]xB) -> [B,C,H,W], where B > 1 and B = _N
//  */
// RET_CODE BaseModel::general_batch_preprocess_yuv_on_mlu(BatchImageIN &batch_tvimage, 
// std::vector<TvaiRect> &batch_roiRect, std::vector<float> &batch_aspect_ratio){
//     if( batch_tvimage.empty() ) return RET_CODE::SUCCESS;
//     int N = batch_tvimage.size();
//     if( N != _N ) { 
//         LOGI << "no equal N: " << N << " _N: " << _N;
//         return RET_CODE::ERR_MODEL_NOT_MATCH; }
//     if(batch_tvimage[0].format != TVAI_IMAGE_FORMAT_NV12 && batch_tvimage[0].format !=TVAI_IMAGE_FORMAT_NV21 )
//         return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
//     if (_ptrHandle == nullptr )
//         return RET_CODE::ERR_MODEL_NOT_INIT;    
//     PtrHandle* ptrHandle = reinterpret_cast<PtrHandle*>(_ptrHandle);
//     mluColorMode colormode = getMLUTransColorMode(_model_input_fmt, batch_tvimage[0].format);
//     if (colormode != ptrHandle->rc_attr_.color_mode){
//         LOGI << "re-initial mlu resize op";
//         create_mlu_resize_func(ptrHandle, reinterpret_cast<edk::MluContext*>(env_), colormode, _pad_both_side, _keep_aspect_ratio);
//     }
//     bool use_whole_image = false;
//     if(batch_roiRect.empty()) use_whole_image = true;

//     float expand_ratio = 1.0;
//     std::vector<void*> rc_input_mlu_list;
//     for(int b = 0; b < N; b++ )
//     {
//         int imW = batch_tvimage[b].width;
//         int imH = batch_tvimage[b].height;
//         int imS = batch_tvimage[b].stride;
//         edk::MluResizeConvertOp::InputData input;

//         if(!batch_tvimage[b].usePhyAddr){
//             void* rc_input_mlu;
//             cnrtMalloc(&rc_input_mlu, 3*imS* imH/2);//++
//             rc_input_mlu_list.push_back(rc_input_mlu);
//             cnrtMemcpy(rc_input_mlu, batch_tvimage[b].pData, 3*imS*imH/2, CNRT_MEM_TRANS_DIR_HOST2DEV);
//             input.planes[0] = rc_input_mlu;
//             input.planes[1] = rc_input_mlu+imH*imS;
//         } else {
//             input.planes[0] = reinterpret_cast<void*>(batch_tvimage[b].u64PhyAddr[0]);
//             input.planes[1] = reinterpret_cast<void*>(batch_tvimage[b].u64PhyAddr[1]);
//         }
//         input.src_w = imW;
//         input.src_h = imH;
//         input.src_stride = imS;
//         if(!use_whole_image){
//             TvaiRect roiRect = get_valid_rect(batch_roiRect[b], imW, imH);
//             input.crop_x = roiRect.x; input.crop_y = roiRect.y;
//             input.crop_w = roiRect.width; input.crop_h = roiRect.height;
//         }
//         ptrHandle->rc_op_mlu_.BatchingUp(input);
//         batch_aspect_ratio.push_back(MIN( (1.0*_W)/imW , (1.0*_H)/imH ));
//     }
//     void* rc_output = mlu_input_[0];
//     if (!ptrHandle->rc_op_mlu_.SyncOneOutput(rc_output)) {
//         THROW_EXCEPTION(edk::Exception::INTERNAL, ptrHandle->rc_op_mlu_.GetLastError());}
//     for(int i=0; i < rc_input_mlu_list.size(); i++)
//         cnrtFree(rc_input_mlu_list[i]);//--
//     return RET_CODE::SUCCESS;
// }

// /**
//  * TRANS:
//  * ([Img]x1, [ROI]xN) -> [1,C,H,W]xN, where N > 1 and N = _MI
//  * MIMO操作, multiple input tensor, single output tensor
//  * 第一种: 单个图像输入, 单个图像中多个ROI区域组成一组输入
//  */
// RET_CODE BaseModel::general_preprocess_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox& bboxes,std::vector<float> &batch_aspect_ratio, int offset){
//     LOGI << "-> BaseModel::general_preprocess_yuv_on_mlu[mimo]" ;
//     if(tvimage.format != TVAI_IMAGE_FORMAT_NV12 && tvimage.format !=TVAI_IMAGE_FORMAT_NV21 )
//         return RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
//     if (_ptrHandle == nullptr )
//         return RET_CODE::ERR_MODEL_NOT_INIT;    
//     PtrHandle* ptrHandle = reinterpret_cast<PtrHandle*>(_ptrHandle);
//     mluColorMode colormode = getMLUTransColorMode(_model_input_fmt, tvimage.format);
//     if (colormode != ptrHandle->rc_attr_.color_mode){
//         LOGI << "re-initial mlu resize op";
//         create_mlu_resize_func(ptrHandle, reinterpret_cast<edk::MluContext*>(env_), colormode, _pad_both_side, _keep_aspect_ratio);
//     }
//     //assert the first bbox is not null
//     if(bboxes.empty() || bboxes[offset].objtype == CLS_TYPE::UNKNOWN) return RET_CODE::ERR_EMPTY_BOX;
//     float expand_ratio = 1.0;

//     int imW = tvimage.width;
//     int imH = tvimage.height;
//     int imS = tvimage.stride;
//     int L = MIN(_MI+offset, bboxes.size());

//     void* rc_input_mlu = nullptr;
//     edk::MluResizeConvertOp::InputData input;
//     if(!tvimage.usePhyAddr){
//         cnrtMalloc(&rc_input_mlu, 3*imS* imH/2);//++
//         cnrtMemcpy(rc_input_mlu, tvimage.pData, 3*imS*imH/2, CNRT_MEM_TRANS_DIR_HOST2DEV);
//         input.planes[0] = rc_input_mlu;
//         input.planes[1] = rc_input_mlu+imH*imS;
//     } else {
//         input.planes[0] = reinterpret_cast<void*>(tvimage.u64PhyAddr[0]);
//         input.planes[1] = reinterpret_cast<void*>(tvimage.u64PhyAddr[1]);
//     }

//     for(int b = offset; b < L; b++ )
//     {
//         int output_index = (b - offset)%_MI;
//         void* rc_output = mlu_input_[output_index];
//         if(bboxes[b].objtype == CLS_TYPE::UNKNOWN) {
//             //空数据情况下
//             cnrtMemset(rc_output,0, _H*_W*_C*_N*size_t(1));//assert _N == 1
//             batch_aspect_ratio.push_back(0);
//         } else{//not empty
//             input.src_w = imW;
//             input.src_h = imH;
//             input.src_stride = imS;
//             TvaiRect validROI = get_valid_rect(bboxes[b].rect, imW, imH);
//             input.crop_x = validROI.x; input.crop_y = validROI.y;
//             input.crop_w = validROI.width; input.crop_h = validROI.height;
//             ptrHandle->rc_op_mlu_.BatchingUp(input);
//             batch_aspect_ratio.push_back(MIN( (1.0*_W)/imW , (1.0*_H)/imH ));

//             if (!ptrHandle->rc_op_mlu_.SyncOneOutput(rc_output)) {
//                 THROW_EXCEPTION(edk::Exception::INTERNAL, ptrHandle->rc_op_mlu_.GetLastError());}
//         }    
//     }//loop of input number
//     if(rc_input_mlu!=nullptr)
//         cnrtFree(rc_input_mlu);//--
//     return RET_CODE::SUCCESS;
// }
/////////////////////////////////////////////////////////////////////
// End of Class BaseModel 
/////////////////////////////////////////////////////////////////////