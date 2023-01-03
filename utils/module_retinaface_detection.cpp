#include "module_retinaface_detection.hpp"
#include "glog/logging.h"
#include <opencv2/opencv.hpp>
#include <math.h>
#include "basic.hpp"
#include "../inner_utils/inner_basic.hpp"
#include <fstream>
// #include <cn_api.h>

#include "../inner_utils/ip_iqa_blur.hpp"
#include "../inner_utils/ip_iqa_pose.hpp"

#include "json_encoder/json_encoder.hpp"


#ifdef DEBUG
#include <chrono>
#include <sys/time.h>
#include "../inner_utils/module.hpp"
#endif

// #include <future>
using namespace ucloud;
using namespace cv;

/*******************************************************************************
comman function for postprocess
*******************************************************************************/
static void output2FaceBox(float* output ,VecObjBBox &vecbox, int nbboxes ,int stride=15, float threshold=0.8){
    //score(1)+xyxy(4)+landmarks(10)
    for( int i=0; i < nbboxes; i++ ){
        float* _output = &output[i*stride];
        float score = *_output++;
        if( score < threshold )
            continue;
        else {
            BBox fbox;
            fbox.x0 = *_output++;
            fbox.y0 = *_output++;
            fbox.x1 = *_output++;
            fbox.y1 = *_output++;
            fbox.confidence = score;
            fbox.objectness = score;
            for(int j = 0; j < 5 ; j++){
                float px = *_output++;
                float py = *_output++;
                fbox.Pts.pts.push_back(uPoint(px,py));
            }
            fbox.Pts.refcoord = RefCoord::IMAGE_ORIGIN;//cur is unscaled image origin
            fbox.Pts.type = LandMarkType::FACE_5PTS;
            fbox.objtype = CLS_TYPE::FACE;
            vecbox.push_back(fbox);
        }
    }
    return;
}
static void _transform_xyxy_xyhw(VecObjBBox &vecbox, float expand_ratio ,float aspect_ratio){
    for (int i=0 ; i < vecbox.size(); i++ ){
        float cx = (vecbox[i].x0 + vecbox[i].x1)/(2*aspect_ratio);
        float cy = (vecbox[i].y0 + vecbox[i].y1)/(2*aspect_ratio);
        float w = (vecbox[i].x1 - vecbox[i].x0)*expand_ratio/aspect_ratio;
        float h = (vecbox[i].y1 - vecbox[i].y0)*expand_ratio/aspect_ratio;
        float _x0 = cx - w/2;
        float _y0 = cy - h/2;

        vecbox[i].rect.x = int(_x0);
        vecbox[i].rect.y = int(_y0);
        vecbox[i].rect.width = int(w);
        vecbox[i].rect.height = int(h);
        for(int j=0;j<vecbox[i].Pts.pts.size(); j++){
            vecbox[i].Pts.pts[j].x /= aspect_ratio;
            vecbox[i].Pts.pts[j].y /= aspect_ratio;
        }
    }
};

/*******************************************************************************
FaceDetectionV4 使用BaseModelV2 不含跟踪
*******************************************************************************/
FaceDetectionV4::FaceDetectionV4(){
    m_net = std::make_shared<BaseModelV2>();
}

RET_CODE FaceDetectionV4::init(const std::string &modelpath){
    std::map<InitParam, std::string> modelconfig = {{InitParam::BASE_MODEL,modelpath}};
    return this->init(modelconfig);
}

RET_CODE FaceDetectionV4::init(std::map<InitParam, std::string> &modelpath){
    LOGI << "-> FaceDetectionV4::init";
    if(use_auto_model){
        RET_CODE ret = auto_model_file_search(m_roots, m_models_startswith, modelpath);
        if(ret!=RET_CODE::SUCCESS) return ret;
    }
        
    if(modelpath.find(InitParam::BASE_MODEL)==modelpath.end())
        return RET_CODE::ERR_INIT_PARAM_FAILED;
    bool pad_both_side = false;
    bool keep_aspect_ratio = true;
    BASE_CONFIG config(MODEL_INPUT_FORMAT::RGBA, MODEL_OUTPUT_ORDER::NCHW, pad_both_side, keep_aspect_ratio);
    RET_CODE ret = m_net->base_init(modelpath[InitParam::BASE_MODEL], config);
    if (ret!=RET_CODE::SUCCESS) return ret;
    iqa_evaluator.init(112,112);

    if(modelpath.find(InitParam::SUB_MODEL)!=modelpath.end()){
        FaceAttributionV4* _ptr_ = new FaceAttributionV4();
        ret= _ptr_->init(modelpath[InitParam::SUB_MODEL]);
        m_faceAttrExtractor.reset(_ptr_);
    } else m_faceAttrExtractor = nullptr;
    LOGI << "<- FaceDetectionV4::init";
    return ret;
}

RET_CODE FaceDetectionV4::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    valid_clss.push_back(_cls_);
    return RET_CODE::SUCCESS;
}

float FaceDetectionV4::get_box_expand_ratio(){
    return _expand_ratio;
}

RET_CODE FaceDetectionV4::run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    RET_CODE ret = RET_CODE::FAILED;
    threshold = clip_threshold(threshold);
    nms_threshold = clip_threshold(nms_threshold);

    float preprocess_time{0}, npu_inference_time{0}, postprocess_time{0};

    float aspect_ratio = 1.0;
    float expand_ratio = _expand_ratio;
    float aX,aY;
    float** model_output = nullptr;

    TvaiRect roi{0,0,tvimage.width, tvimage.height};
    m_Tk.start();
    if(tvimage.format == TVAI_IMAGE_FORMAT_RGB || tvimage.format == TVAI_IMAGE_FORMAT_BGR){
        ret = m_net->general_preprocess_bgr_on_cpu(tvimage, aspect_ratio, aX , aY);
    }
    else if(tvimage.format == TVAI_IMAGE_FORMAT_NV21 || tvimage.format == TVAI_IMAGE_FORMAT_NV12 ){
        ret = m_net->general_preprocess_yuv_on_mlu_union(tvimage, roi ,aspect_ratio, aX, aY);
        if(ret!=RET_CODE::SUCCESS) return ret;
    }
    else
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    preprocess_time = m_Tk.end("preprocess", false);
    m_Tk.start();
    model_output = m_net->general_mlu_infer();
    npu_inference_time = m_Tk.end("npu inference", false);
    m_Tk.start();
    ret = postprocess(model_output[0], bboxes, threshold, nms_threshold, expand_ratio, aspect_ratio, tvimage.width, tvimage.height);    
    m_net->cpu_free(model_output);
    if(ret!=RET_CODE::SUCCESS) return ret;
    ret = iqa_quality(tvimage, bboxes);

    if(m_faceAttrExtractor){
        ret = m_faceAttrExtractor->run(tvimage, bboxes);
        if(ret!=RET_CODE::SUCCESS) {
            printf("ERR: m_faceAttrExtractor return [%d]\n", ret);
            return ret;
        }
    }
    postprocess_time = m_Tk.end("postprocess", false);

    // LOGI << "====JSON====";
    for(auto &&box: bboxes){
        UcloudJsonEncoder jsonWriter;
        jsonWriter.initial_context_with_string(box.desc);
        jsonWriter.add_context(tagJSON_ROOT::OTHERS, tagJSON_ATTR::NOTE , "retinaface" );
        std::string json_file = jsonWriter.output_to_string();
        box.desc = json_file;
        LOGI << json_file;
    }    
    if(!bboxes.empty()){
        bboxes[0].tmInfo = {preprocess_time, npu_inference_time, postprocess_time};
    }
    return ret;
}

RET_CODE FaceDetectionV4::postprocess(float* model_output, VecObjBBox &bboxes, float threshold, float nms_threshold, float expand_ratio, float aspect_ratio, int imgW, int imgH){
    LOGI << "-> FaceDetectionV4::postprocess";
    int nBBox = m_net->m_outputShape[0].C();
    int featLen = m_net->m_outputShape[0].H();
    VecObjBBox vecBox;
    VecObjBBox vecBox_after_nms;
    output2FaceBox(model_output, vecBox, nBBox, featLen, threshold);
    base_nmsBBox(vecBox,nms_threshold, NMS_MIN ,vecBox_after_nms );
    LOGI << "after nms " << vecBox_after_nms.size() << std::endl;
    _transform_xyxy_xyhw(vecBox_after_nms, expand_ratio, aspect_ratio);
    LOGI << "after filter " << bboxes.size() << std::endl;
    bboxes.insert(bboxes.end(), vecBox_after_nms.begin(), vecBox_after_nms.end());
    VecObjBBox().swap(vecBox);
    VecObjBBox().swap(vecBox_after_nms);
    vecBox.clear();
    // free(cpu_chw);
    return RET_CODE::SUCCESS;     
}

RET_CODE FaceDetectionV4::iqa_quality(TvaiImage &tvimage, VecObjBBox &bboxes){
    RET_CODE ret = iqa_evaluator.run(tvimage, bboxes);
    return ret;
}

float FaceDetectionV4::clip_threshold(float x){
    if(x < 0) return m_default_threshold;
    if(x > 1) return m_default_threshold;
    return x;
}
float FaceDetectionV4::clip_nms_threshold(float x){
    if(x < 0) return m_default_nms_threshold;
    if(x > 1) return m_default_nms_threshold;
    return x;
}

/*******************************************************************************
FaceDetectionV2
*******************************************************************************/

// float FaceDetectionV2::get_box_expand_ratio(){
//     return _expand_ratio;
// }

// RET_CODE FaceDetectionV2::init(const std::string &modelpath){
//     bool pad_both_side = false;
//     bool keep_aspect_ratio = true;
//     BASE_CONFIG config(MODEL_INPUT_FORMAT::RGBA, MODEL_OUTPUT_ORDER::NCHW, pad_both_side, keep_aspect_ratio);
//     RET_CODE ret = BaseModel::base_init(modelpath, config);
//     if (ret!=RET_CODE::SUCCESS) return ret;
//     std::vector<TvaiRect> pAoi;
//     set_param(_threshold, _nms_threshold, TvaiResolution{0,0}, TvaiResolution{0,0}, pAoi);
//     // PtrHandle* ptrHandle = reinterpret_cast<PtrHandle*>(_ptrHandle);
//     iqa_evaluator.init(112,112);
//     return ret;
// }

// RET_CODE FaceDetectionV2::create_trackor(int uuid_cam){
//     if(m_Trackors.find(uuid_cam)==m_Trackors.end()){
//         edk::FeatureMatchTrack *track = new edk::FeatureMatchTrack;
//         track->SetParams(m_max_cosine_distance, m_nn_budget, m_max_iou_distance, m_fps*2, m_n_init);
//         std::shared_ptr<edk::EasyTrack> m_trackor_t;
//         m_trackor_t.reset(track);
//         m_Trackors.insert(std::pair<int,std::shared_ptr<edk::EasyTrack>>(uuid_cam,m_trackor_t));
//     }
//     return RET_CODE::SUCCESS;
// }

// RET_CODE FaceDetectionV2::init_trackor(const std::string &trackmodelpath){
//     create_trackor(-1);
//     m_trackFeatExtractor.reset(new ObjFeatureExtraction());
//     if (!exists_file(trackmodelpath)) return RET_CODE::ERR_MODEL_FILE_NOT_EXIST;
//     RET_CODE ret = m_trackFeatExtractor->init(trackmodelpath); //@cambricon official
//     // RET_CODE ret = m_trackFeatExtractor->init(trackmodelpath, MODEL_INPUT_FORMAT::RGBA, false, false); //@lihui
//     return ret;
// }

// RET_CODE FaceDetectionV2::init(const std::string &modelpath, const std::string &trackmodelpath){
//     RET_CODE ret = FaceDetectionV2::init(modelpath);
//     if(ret!=RET_CODE::SUCCESS) return ret;
//     ret = init_trackor(trackmodelpath);
//     return ret;
// }

// RET_CODE FaceDetectionV2::init(std::map<InitParam, std::string> &modelpath){
//     if(use_auto_model){
//         RET_CODE ret = auto_model_file_search(m_roots, m_models_startswith, modelpath);
//         if(ret!=RET_CODE::SUCCESS) return ret;
//     }
        
//     if(modelpath.find(InitParam::BASE_MODEL)==modelpath.end())
//         return RET_CODE::ERR_INIT_PARAM_FAILED;
//     bool pad_both_side = false;
//     bool keep_aspect_ratio = true;
//     BASE_CONFIG config(MODEL_INPUT_FORMAT::RGBA, MODEL_OUTPUT_ORDER::NCHW, pad_both_side, keep_aspect_ratio);
//     RET_CODE ret = BaseModel::base_init(modelpath[InitParam::BASE_MODEL], config);
//     if (ret!=RET_CODE::SUCCESS) return ret;
//     std::vector<TvaiRect> pAoi;
//     set_param(_threshold, _nms_threshold, TvaiResolution{0,0}, TvaiResolution{0,0}, pAoi);
//     iqa_evaluator.init(112,112);

//     if(modelpath.find(InitParam::TRACK_MODEL)!=modelpath.end())
//         ret = init_trackor(modelpath[InitParam::TRACK_MODEL]);    

//     if(modelpath.find(InitParam::SUB_MODEL)!=modelpath.end()){
//         FaceAttribution* _ptr_ = new FaceAttribution();
//         ret= _ptr_->init(modelpath[InitParam::SUB_MODEL]);
//         m_faceAttrExtractor.reset(_ptr_);
//     }

//     return ret;
// }


// FaceDetectionV2::~FaceDetectionV2(){}


// RET_CODE FaceDetectionV2::run_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox &bboxes){
//     LOGI << "-> FaceDetectionV2::run_yuv_on_mlu_phyAddr";
//     RET_CODE ret = RET_CODE::FAILED;
//     float aspect_ratio = 1.0;
//     float expand_ratio = _expand_ratio;
//     float aX,aY;
//     float* model_output = nullptr;
//     TvaiRect roi{0,0,tvimage.width, tvimage.height};
//     {
//         std::lock_guard<std::mutex> lk(_mlu_mutex);
//         ret = BaseModel::general_preprocess_yuv_on_mlu_union(tvimage, roi ,aspect_ratio, aX, aY);
//         // ret = BaseModel::general_preprocess_yuv_on_mlu_phyAddr(tvimage, aspect_ratio, aX, aY);
//         if(ret!=RET_CODE::SUCCESS) return ret;
//         model_output = BaseModel::general_mlu_infer();
//     }
//     ret = postprocess(model_output, bboxes, expand_ratio, aspect_ratio, tvimage.width, tvimage.height);
//     free(model_output);
//     if(ret!=RET_CODE::SUCCESS) return ret;
//     //图像质量分析
//     ret = iqa_quality(tvimage, bboxes);
//     if(m_trackFeatExtractor){
//         std::lock_guard<std::mutex> lk(_mlu_mutex);
//         trackprocess(tvimage, bboxes);
//     }
//     if(m_faceAttrExtractor){
//         ret = m_faceAttrExtractor->run(tvimage, bboxes);
//         if(ret!=RET_CODE::SUCCESS) {
//             LOGI << "m_faceAttrExtractor err...";
//             return ret;
//         }
//     }
//     return ret;
// }

// RET_CODE FaceDetectionV2::run_bgr_on_cpu(TvaiImage &tvimage, VecObjBBox &bboxes){
//     LOGI << "-> FaceDetectionV2::run_bgr_on_cpu";
//     RET_CODE ret = RET_CODE::FAILED;
//     float aspect_ratio = 1.0;
//     float expand_ratio = _expand_ratio;
//     float aX,aY;
//     float* model_output = nullptr;
//     {
//         std::lock_guard<std::mutex> lk(_mlu_mutex);
//         ret = BaseModel::general_preprocess_bgr_on_cpu(tvimage, aspect_ratio, aX , aY);
//         if(ret!=RET_CODE::SUCCESS) return ret;
//         model_output = BaseModel::general_mlu_infer();
//     }
//     ret = postprocess(model_output, bboxes, expand_ratio, aspect_ratio, tvimage.width, tvimage.height);
//     free(model_output);
//     //图像质量分析
//     ret = iqa_quality(tvimage, bboxes);
//     //cpu上不做跟踪算法
//     return ret;
// }

// // RET_CODE FaceDetectionV2::run(BatchImageIN &batch_tvimages, VecObjBBox &bboxes){
// //     if(batch_tvimages.empty()) return RET_CODE::SUCCESS;
// //     return run(batch_tvimages[0], bboxes);
// // }

// RET_CODE FaceDetectionV2::run(TvaiImage &tvimage, VecObjBBox &bboxes){
//     RET_CODE ret = RET_CODE::FAILED;
//     if(tvimage.format == TVAI_IMAGE_FORMAT_RGB || tvimage.format == TVAI_IMAGE_FORMAT_BGR){
//         ret = run_bgr_on_cpu(tvimage, bboxes);
//     }
//     else if(tvimage.format == TVAI_IMAGE_FORMAT_NV21 || tvimage.format == TVAI_IMAGE_FORMAT_NV12 ){
//         ret = run_yuv_on_mlu(tvimage, bboxes);
//     }
//     else
//         ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;

//     LOGI << "====JSON====";
//     for(auto &&box: bboxes){
//             UcloudJsonEncoder jsonWriter;
//             jsonWriter.initial_context_with_string(box.desc);
//             jsonWriter.add_context(tagJSON_ROOT::OTHERS, tagJSON_ATTR::NOTE , "retinaface" );
//             std::string json_file = jsonWriter.output_to_string();
//             box.desc = json_file;
//             LOGI << json_file;
//     }    
//     return ret;
// }

// RET_CODE FaceDetectionV2::set_param(float threshold, float nms_threshold, TvaiResolution maxTargetSize, TvaiResolution minTargetSize, 
// std::vector<TvaiRect> &pAoiRect){
//     if(float_in_range(threshold,1,0))
//         _threshold = threshold;
//     else
//         return RET_CODE::ERR_INIT_PARAM_FAILED;
//     if(float_in_range(nms_threshold,1,0))
//         _nms_threshold = nms_threshold;
//     else
//         return RET_CODE::ERR_INIT_PARAM_FAILED;
//     _maxTargeSize = base_get_valid_maxSize(maxTargetSize);
//     _minTargeSize = minTargetSize;
//     std::vector<TvaiRect>().swap(_pAoiRect);
//     _pAoiRect.clear();
//     for(int i=0; i<pAoiRect.size(); i++ ){
//         _pAoiRect.push_back(pAoiRect[i]);
//     }
//     return RET_CODE::SUCCESS;    
// }

// // object filter
// template<typename T>
// bool check_rect_resolution(T _box, TvaiResolution minSz, TvaiResolution maxSz){
//     int width = _box.rect.width;
//     int height = _box.rect.height;
//     if (width < minSz.width || width > maxSz.width)
//         return false;
//     if (height < minSz.height || height > maxSz.height)
//         return false;
//     return true;
// }
// template<typename T>
// bool check_in_valid_region(T _box, const std::vector<TvaiRect> &pAoiRegion){
//     int x_center = _box.rect.x + _box.rect.width/2;
//     int y_center = _box.rect.y + _box.rect.height/2;
//     if(pAoiRegion.size()==0){
//         return true;
//     }
//     for (int i = 0; i < pAoiRegion.size(); i++ ){
//         //  std::cout << pAoiRegion[i].x << "," << pAoiRegion[i].y << "," << std::endl;
//         if (x_center>=pAoiRegion[i].x && x_center<=pAoiRegion[i].x+pAoiRegion[i].width &&\
//             y_center>=pAoiRegion[i].y && y_center<=pAoiRegion[i].y+pAoiRegion[i].height
//         )
//             return true;
//     }
//     return false;
// }
// void FaceDetectionV2::object_filter(VecObjBBox &input_bboxes, VecObjBBox &output_bboxes, int imgW, int imgH){
//     VecObjBBox().swap(output_bboxes);
//     output_bboxes.clear();
//     std::vector<TvaiRect> imgRois;
//     TvaiRect imgRoi{0,0,imgW,imgH};
//     imgRois.push_back(imgRoi);
//     for (int i = 0 ; i < input_bboxes.size() ; i++ ){
//         BBox _box = input_bboxes[i];
//         if (!check_rect_resolution(_box, _minTargeSize, _maxTargeSize)){
//             // std::cout << "resolution failed" << _box.rect.width <<  "," << _box.rect.height << ", " <<_minTargeSize.height << "," << _minTargeSize.width << ","
//             // << _maxTargeSize.height << "," << _maxTargeSize.width << std::endl;
//             continue;
//         }
//         if (!check_in_valid_region(_box, _pAoiRect)){
//             // std::cout << "region failed" << std::endl;
//             continue;
//         }
//         if (!check_in_valid_region(_box, imgRois)){
//             continue;
//         }
//         output_bboxes.push_back(_box);
//     }    
// }


// RET_CODE FaceDetectionV2::postprocess(float* model_output, VecObjBBox &bboxes, float expand_ratio, float aspect_ratio, int imgW, int imgH){
//     LOGI << "-> FaceDetectionV2::postprocess";
//     int nBBox = _oC;
//     int featLen = _oH;
//     VecObjBBox vecBox;
//     VecObjBBox vecBox_after_nms;
//     output2FaceBox(model_output, vecBox, nBBox, featLen, _threshold);
//     base_nmsBBox(vecBox,_nms_threshold, NMS_MIN ,vecBox_after_nms );
//     LOGI << "after nms " << vecBox_after_nms.size() << std::endl;
//     _transform_xyxy_xyhw(vecBox_after_nms, expand_ratio, aspect_ratio);
//     object_filter(vecBox_after_nms, bboxes, imgW, imgH);
//     LOGI << "after filter " << bboxes.size() << std::endl;
//     VecObjBBox().swap(vecBox);
//     VecObjBBox().swap(vecBox_after_nms);
//     vecBox.clear();
//     // free(cpu_chw);
//     return RET_CODE::SUCCESS;     
// }

// RET_CODE FaceDetectionV2::get_class_type(std::vector<CLS_TYPE> &valid_clss){
//     valid_clss.push_back(_cls_);
//     return RET_CODE::SUCCESS;
// }


// RET_CODE FaceDetectionV2::trackprocess(TvaiImage &tvimage, VecObjBBox &bboxes_in){
//     LOGI << "-> FaceDetectionV2::trackprocess";
//     m_trackFeatExtractor->run(tvimage, bboxes_in);
//     float imgW = tvimage.width;
//     float imgH = tvimage.height;
//     std::vector<edk::DetectObject> in, out;
//     for(auto iter = bboxes_in.begin(); iter != bboxes_in.end(); iter++){
//         edk::DetectObject obj;
//         float x = CLIP(iter->rect.x/imgW);
//         float y = CLIP(iter->rect.y/imgH);
//         float w = CLIP(iter->rect.width/imgW);
//         float h = CLIP(iter->rect.height/imgH);
//         w = (x + w > 1.0) ? (1.0 - x) : w;
//         h = (y + h > 1.0) ? (1.0 - y) : h;
//         obj.label = iter->objtype;
//         obj.score = iter->confidence;
//         obj.bbox.x = x;
//         obj.bbox.y = y;
//         obj.bbox.width = w;
//         obj.bbox.height = h;
//         obj.feature = iter->trackfeat;
//         in.push_back(obj);
//     }
//     edk::TrackFrame tframe;
//     create_trackor(tvimage.uuid_cam);
//     m_Trackors[tvimage.uuid_cam]->UpdateFrame(tframe, in , &out);
//     LOGI << "after track " << out.size();
//     for(int i = 0; i < out.size() ; i++){
//         bboxes_in[out[i].detect_id].track_id = out[i].track_id;
//     }
//     return RET_CODE::SUCCESS;
// }

// RET_CODE FaceDetectionV2::iqa_quality(TvaiImage &tvimage, VecObjBBox &bboxes){
//     RET_CODE ret = iqa_evaluator.run(tvimage, bboxes);
//     return ret;
// }


/*******************************************************************************
FaceDetectionV4DeepSort
*******************************************************************************/