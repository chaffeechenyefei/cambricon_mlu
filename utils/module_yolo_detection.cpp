#include "module_yolo_detection.hpp"
// #include "glog/logging.h"
// #include <opencv2/opencv.hpp>
// #include <math.h>
// #include "basic.hpp"
// #include "../inner_utils/inner_basic.hpp"
// #include <fstream>
// #include <iostream>

// /**
//  * jsoncpp https://github.com/open-source-parsers/jsoncpp/tree/jsoncpp_version
//  * tag: 1.9.5
// */
// #include "json/json.h"
// #include "json_encoder/json_encoder.hpp"

// #ifdef DEBUG
// #include <chrono>
// #include <sys/time.h>
// #include "../inner_utils/module.hpp"
// #endif

// #ifdef VERBOSE
// #define LOGI LOG(INFO)
// #else
// #define LOGI 0 && LOG(INFO)
// #endif

// #define NMS_UNION 0
// #define NMS_MIN 1

// #define CLIP(x) ((x) < 0 ? 0 : ((x) > 1 ? 1 : (x)))

// // #include <future>
// using namespace ucloud;
// using namespace cv;

// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// // YoloV2
// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// RET_CODE YoloDetectionV2::create_trackor(int uuid_cam){
//     if(m_Trackors.find(uuid_cam)==m_Trackors.end()){
//         LOGI << "-> YoloDetectionV2::create_trackor+";
//         edk::FeatureMatchTrack *track = new edk::FeatureMatchTrack;
//         track->SetParams(m_max_cosine_distance, m_nn_budget, m_max_iou_distance, m_fps*2, m_n_init);
//         std::shared_ptr<edk::EasyTrack> m_trackor_t;
//         m_trackor_t.reset(track);
//         m_Trackors.insert(std::pair<int,std::shared_ptr<edk::EasyTrack>>(uuid_cam,m_trackor_t));
//     } else LOGI << "-> YoloDetectionV2::create_trackor";
//     return RET_CODE::SUCCESS;
// }

// RET_CODE YoloDetectionV2::init_trackor(const std::string &trackmodelpath){
//     create_trackor(-1);
//     m_trackFeatExtractor.reset(new ObjFeatureExtraction());
//     if (!exists_file(trackmodelpath)) return RET_CODE::ERR_MODEL_FILE_NOT_EXIST;
//     m_trackFeatExtractor->init(trackmodelpath); //@cambricon official
//     // m_trackFeatExtractor->init(trackmodelpath, MODEL_INPUT_FORMAT::RGBA, false, false); //@lihui
//     return RET_CODE::SUCCESS;
// }

// RET_CODE YoloDetectionV2::init(const std::string &modelpath){
//     std::map<InitParam, std::string> modelConfig = {{InitParam::BASE_MODEL, modelpath}};
//     return this->init(modelConfig);
// }

// RET_CODE YoloDetectionV2::init(std::map<InitParam, std::string> &modelpath){
//     LOGI << "-> YoloDetectionV2::init";
//     if(use_auto_model){
//         RET_CODE ret = auto_model_file_search(m_roots, m_models_startswith, modelpath);
//         if(ret!=RET_CODE::SUCCESS) return ret;
//     }

//     if(modelpath.find(InitParam::BASE_MODEL) == modelpath.end())
//         return RET_CODE::ERR_INIT_PARAM_FAILED;
//     bool pad_both_side = false;
//     bool keep_aspect_ratio = true;
//     BASE_CONFIG config(MODEL_INPUT_FORMAT::RGBA, MODEL_OUTPUT_ORDER::NCHW, pad_both_side, keep_aspect_ratio);
//     RET_CODE ret = BaseModel::base_init( modelpath[InitParam::BASE_MODEL], config);
//     if (ret!=RET_CODE::SUCCESS) {
//         LOGI << "RET_CODE ret = BaseModel::base_init( modelpath[InitParam::BASE_MODEL], config); return" << ret;
//         return ret;
//     }

//     _output_cls_num = _oH - 5;
//     if(modelpath.find(InitParam::TRACK_MODEL) != modelpath.end())
//         ret = init_trackor(modelpath[InitParam::TRACK_MODEL]);
//     LOGI << "<- YoloDetectionV2::init";
//     return ret;
// }

// // RET_CODE YoloDetectionV2::init_json(const std::string &configpath){
// //     RET_CODE ret = RET_CODE::FAILED;
// //     Json::Value root;
// //     std::ifstream ifs;
// //     Json::CharReaderBuilder builder;
// //     // builder["collectComments"] = true;
// //     JSONCPP_STRING errs;
// //     if(hasEnding(configpath, ".json")){
// //         //.json文件的处理
// //         std::ifstream ifs;
// //         ifs.open(configpath);
// //         if (!parseFromStream(builder, ifs, &root, &errs)) {
// //             std::cout << errs << std::endl;
// //             return RET_CODE::ERR_INIT_PARAM_FAILED;
// //         }
// //     } else {
// //         //string json格式输入的处理
// //         std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
// //         if (!reader->parse(configpath.data(), configpath.data() + configpath.size(), &root, &errs)) {
// //             std::cout << errs << std::endl;
// //             return RET_CODE::ERR_INIT_PARAM_FAILED;
// //         }
// //     }
// //     //1. init model path
// //     // std::cout << "1. " << std::endl;
// //     std::string modelpath = "";
// //     std::string trackpath = "";
// //     if(!root.isMember("base_model")){
// //         std::cout << "json key::base_model not found" << std::endl;
// //         return RET_CODE::ERR_INIT_PARAM_FAILED;
// //     } else modelpath = root["base_model"].asString();
// //     if(root.isMember("track_model")){
// //         trackpath = root["track_model"].asString();
// //         ret = init(modelpath, trackpath);
// //     } else{
// //         ret = init(modelpath);
// //     }
// //     if(ret!=RET_CODE::SUCCESS) return ret;
// //     //2. init output class
// //     // std::cout << "2. " << std::endl;
// //     if(!root.isMember("class_order")){
// //         std::cout << "json key::class_order not found" << std::endl;
// //         return RET_CODE::ERR_INIT_PARAM_FAILED;
// //     } else{
// //         // std::cout << "json key::class_order found" << std::endl;
// //         std::vector<std::string> cls_ord;
// //         std::vector<CLS_TYPE> cls_type;
// //         for(int i = 0; i < root["class_order"].size(); i++){
// //             cls_ord.push_back(root["class_order"][i].asString());
// //         }
// //         transform_string_to_cls_type(cls_ord, cls_type);
// //         set_output_cls_order(cls_type);
// //     }
// //     //3. init threshold
// //     // std::cout << "3. " << std::endl;
// //     if(root.isMember("threshold")){
// //         _threshold = root["threshold"].asFloat();
// //     }
// //     if(root.isMember("threshold_nms")){
// //         _nms_threshold = root["threshold_nms"].asFloat();
// //     }
// //     return RET_CODE::SUCCESS;
// // }


// YoloDetectionV2::~YoloDetectionV2(){
//     LOGI << "~YoloDetectionV2()";
// }

// // RET_CODE YoloDetectionV2::run_batch_yuv_on_mlu(BatchImageIN &batch_tvimages, BatchBBoxOUT &batch_bboxes){
// //     LOGI << "-> YoloDetectionV2::run_batch_yuv_on_mlu";
// //     RET_CODE ret = RET_CODE::FAILED;
// //     float expand_ratio = 1.0;
// //     std::vector<float> batch_aspect_ratio;
// //     float* batch_model_output = nullptr;
// //     {
// //         std::lock_guard<std::mutex> lk(_mlu_mutex);
// //         std::vector<TvaiRect> batch_rects;
// //         ret = BaseModel::general_batch_preprocess_yuv_on_mlu(batch_tvimages, batch_rects, batch_aspect_ratio);
// //         if(ret!=RET_CODE::SUCCESS) return ret;
// //         batch_model_output = BaseModel::general_mlu_infer();
// //     }
// //     int outputDim = _oW*_oH*_oC;
// //     for(int i = 0; i < _N; i++ ){
// //         float* model_output = batch_model_output+outputDim*i;
// //         VecObjBBox bboxes;
// //         ret = postprocess(model_output, bboxes, expand_ratio, batch_aspect_ratio[i], batch_tvimages[i].width, batch_tvimages[i].height);
// //         batch_bboxes.push_back(bboxes);
// //     }
// //     free(batch_model_output);
// //     return ret;
// // }


// RET_CODE YoloDetectionV2::run_yuv_on_mlu_phyAddr(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
//     LOGI << "-> YoloDetectionV2::run_yuv_on_mlu_phyAddr";
//     RET_CODE ret = RET_CODE::FAILED;
//     float aspect_ratio = 1.0;
//     float aX,aY;
//     float* model_output = nullptr;
//     {
//         std::lock_guard<std::mutex> lk(_mlu_mutex);
//         ret = BaseModel::general_preprocess_yuv_on_mlu_phyAddr(tvimage, aspect_ratio, aX, aY);
//         if(ret!=RET_CODE::SUCCESS) return ret;
//         model_output = BaseModel::general_mlu_infer();
//     }
//     ret = postprocess(model_output, bboxes, threshold, nms_threshold, 1.0, aspect_ratio, tvimage.width, tvimage.height);
//     free(model_output);
//     if(ret!=RET_CODE::SUCCESS) return ret;
//     if(m_trackFeatExtractor){
//         std::lock_guard<std::mutex> lk(_mlu_mutex);
//         trackprocess(tvimage, bboxes);
//     }
//     return ret;
// }

// RET_CODE YoloDetectionV2::run_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
//     LOGI << "-> YoloDetectionV2::run_yuv_on_mlu";
//     RET_CODE ret = RET_CODE::FAILED;
//     float aspect_ratio = 1.0;
//     float aX,aY;
//     float* model_output = nullptr;
//     {
//         std::lock_guard<std::mutex> lk(_mlu_mutex);
//         ret = BaseModel::general_preprocess_yuv_on_mlu(tvimage, aspect_ratio, aX, aY);
//         if(ret!=RET_CODE::SUCCESS) return ret;
//         model_output = BaseModel::general_mlu_infer();
//     }

//     ret = postprocess(model_output, bboxes, threshold, nms_threshold, 1.0, aspect_ratio, tvimage.width, tvimage.height);
//     free(model_output);
//     if(ret!=RET_CODE::SUCCESS) return ret;
//     if(m_trackFeatExtractor){
//         std::lock_guard<std::mutex> lk(_mlu_mutex);
//         trackprocess(tvimage, bboxes);
//     }
//     return ret;
// }

// RET_CODE YoloDetectionV2::run_yuv_on_mlu(TvaiImage &tvimage, TvaiRect tvrect ,VecObjBBox &bboxes, float threshold, float nms_threshold){
//     LOGI << "-> YoloDetectionV2::run_yuv_on_mlu";
//     RET_CODE ret = RET_CODE::FAILED;
//     float aspect_ratio = 1.0;
//     float aX,aY;
//     float* model_output = nullptr;
//     {
//         std::lock_guard<std::mutex> lk(_mlu_mutex);
//         ret = BaseModel::general_preprocess_yuv_on_mlu_union(tvimage, tvrect ,aspect_ratio, aX, aY);
//         if(ret!=RET_CODE::SUCCESS) return ret;
//         model_output = BaseModel::general_mlu_infer();
//     }

//     ret = postprocess(model_output, bboxes, tvrect ,threshold, nms_threshold, 1.0, aspect_ratio, tvimage.width, tvimage.height);
//     free(model_output);
//     if(ret!=RET_CODE::SUCCESS) return ret;
//     if(m_trackFeatExtractor){
//         std::lock_guard<std::mutex> lk(_mlu_mutex);
//         trackprocess(tvimage, bboxes);
//     }
//     return ret;
// }

// RET_CODE YoloDetectionV2::run_bgr_on_cpu(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
//     LOGI << "-> YoloDetectionV2::run_bgr_on_cpu";
//     RET_CODE ret = RET_CODE::FAILED;
//     float aspect_ratio = 1.0;
//     float aX,aY;
//     float* model_output = nullptr;
//     {
//         std::lock_guard<std::mutex> lk(_mlu_mutex);
//         ret = BaseModel::general_preprocess_bgr_on_cpu(tvimage, aspect_ratio, aX , aY);
//         if(ret!=RET_CODE::SUCCESS) return ret;
//         model_output = BaseModel::general_mlu_infer();
//     }
//     ret = postprocess(model_output, bboxes, threshold, nms_threshold, 1.0, aspect_ratio, tvimage.width, tvimage.height);
//     free(model_output);
//     return ret;
// }

// // RET_CODE YoloDetectionV2::run_batch(BatchImageIN &batch_tvimages, BatchBBoxOUT &batch_bboxes){
// //     RET_CODE ret = RET_CODE::FAILED;
// //     for(int i = 0; i < batch_tvimages.size(); i++ ){
// //         if(batch_tvimages[i].format != TVAI_IMAGE_FORMAT_NV21 && batch_tvimages[i].format != TVAI_IMAGE_FORMAT_NV12 ){
// //             ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
// //             return ret;
// //         }
// //     }
// //     ret = run_batch_yuv_on_mlu(batch_tvimages, batch_bboxes);
// //     return ret;
// // }


// RET_CODE YoloDetectionV2::run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
//     RET_CODE ret = RET_CODE::FAILED;
//     threshold = clip_threshold(threshold);
//     nms_threshold = clip_threshold(nms_threshold);
//     LOGI << "-> YoloDetectionV2::threshold =" << threshold;

//     if(tvimage.format == TVAI_IMAGE_FORMAT_RGB || tvimage.format == TVAI_IMAGE_FORMAT_BGR){
//         ret = run_bgr_on_cpu(tvimage, bboxes, threshold, nms_threshold);
//     }
//     else if(tvimage.format == TVAI_IMAGE_FORMAT_NV21 || tvimage.format == TVAI_IMAGE_FORMAT_NV12 ){
//         if(tvimage.usePhyAddr)
//             ret = run_yuv_on_mlu_phyAddr(tvimage, bboxes, threshold, nms_threshold);
//         else
//             ret = run_yuv_on_mlu(tvimage,bboxes, threshold, nms_threshold);
//     }
//     else
//         ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;    

//     return ret;
// }

// RET_CODE YoloDetectionV2::run(TvaiImage &tvimage, TvaiRect tvrect, VecObjBBox &bboxes, float threshold, float nms_threshold){
//     RET_CODE ret = RET_CODE::FAILED;
//     threshold = clip_threshold(threshold);
//     nms_threshold = clip_threshold(nms_threshold);
//     if(tvimage.format == TVAI_IMAGE_FORMAT_NV21 || tvimage.format == TVAI_IMAGE_FORMAT_NV12 ){
//         ret = run_yuv_on_mlu(tvimage, tvrect, bboxes, threshold, nms_threshold);
//     }
//     else
//         ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
//     return ret;
// }

// RET_CODE YoloDetectionV2::get_class_type(std::vector<CLS_TYPE> &valid_clss){
//     LOGI << "-> get_class_type: inner_class_num = " << _output_cls_num;
//     CLS_TYPE* _ptr = _output_cls_order.get();
//     if(_output_cls_num<=0) return RET_CODE::ERR_MODEL_NOT_INIT;
//     if(_output_cls_order==nullptr) return RET_CODE::ERR_MODEL_NOT_INIT;
//     for(int i = 0 ; i < _output_cls_num; i++ ){
//         bool FLAG_exsit_class = false;
//         for( int j = 0; j < valid_clss.size(); j++){
//             if(_ptr[i]==valid_clss[j]){
//                 FLAG_exsit_class = true;
//                 break;
//             }
//         }
//         if(!FLAG_exsit_class) valid_clss.push_back(_ptr[i]);
//     }
//     return RET_CODE::SUCCESS;
// }

// static inline int get_unique_cls_num(CLS_TYPE* output_clss, int len_output_clss, std::map<CLS_TYPE,int> &unique_cls_order){
//     unique_cls_order.clear();
//     std::vector<CLS_TYPE> unique_cls;
//     for(int i=0; i < len_output_clss ; i++){
//         bool conflict = false;
//         for(auto iter=unique_cls.begin(); iter!=unique_cls.end(); iter++){
//             if(output_clss[i] == *iter ){
//                 conflict = true;
//                 break;
//             }
//         }
//         if(!conflict) unique_cls.push_back(output_clss[i]);
//     }
//     for(int i=0; i < unique_cls.size(); i++ ){
//         unique_cls_order.insert(std::pair<CLS_TYPE,int>(unique_cls[i],i));
//     }
//     return unique_cls.size();
// }
// RET_CODE YoloDetectionV2::set_output_cls_order(CLS_TYPE* output_clss, int len_output_clss){
//     // if(_output_cls_num <= 0)
//     //     return RET_CODE::ERR_MODEL_NOT_INIT;
//     _output_cls_num = len_output_clss;
//     if(_output_cls_num > len_output_clss){
//         std::cout << "Model have " << _output_cls_num << " classes, but only " << len_output_clss << " given" << std::endl;
//         return RET_CODE::ERR_OUTPUT_CLS_INIT_FAILED;
//     }
//     // if(_output_cls_order!=nullptr){
//     //     free(_output_cls_order);
//     // }
//     CLS_TYPE* output_cls_order = (CLS_TYPE*)malloc(_output_cls_num*sizeof(CLS_TYPE));
//     memcpy(output_cls_order, output_clss, sizeof(CLS_TYPE)*_output_cls_num);
//     _output_cls_order.reset(output_cls_order, free );

//     _unique_cls_num = get_unique_cls_num(output_clss, len_output_clss, _unique_cls_order);
//     return RET_CODE::SUCCESS;
// }

// static inline int get_unique_cls_num(std::vector<CLS_TYPE>& output_clss, std::map<CLS_TYPE,int> &unique_cls_order ){
//     unique_cls_order.clear();
//     std::vector<CLS_TYPE> unique_cls;
//     for(auto i=output_clss.begin(); i !=output_clss.end(); i++){
//         bool conflict = false;
//         for(auto iter=unique_cls.begin(); iter!=unique_cls.end(); iter++){
//             if( *i == *iter ){
//                 conflict = true;
//                 break;
//             }
//         }
//         if(!conflict) unique_cls.push_back(*i);
//     }
//     for(int i=0; i < unique_cls.size(); i++ ){
//         unique_cls_order.insert(std::pair<CLS_TYPE,int>(unique_cls[i],i));
//     }
//     return unique_cls.size();
// }
// RET_CODE YoloDetectionV2::set_output_cls_order(std::vector<CLS_TYPE>& output_clss){
//     _output_cls_num = output_clss.size();
//     CLS_TYPE* output_cls_order = (CLS_TYPE*)malloc(_output_cls_num*sizeof(CLS_TYPE));
//     memcpy(output_cls_order, &output_clss[0], sizeof(CLS_TYPE)*_output_cls_num);
//     _output_cls_order.reset(output_cls_order, free );

//     _unique_cls_num = get_unique_cls_num(output_clss, _unique_cls_order);
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
// // void YoloDetectionV2::object_filter(VecObjBBox &input_bboxes, VecObjBBox &output_bboxes, int imgW, int imgH){
// //     // VecObjBBox().swap(output_bboxes);
// //     // output_bboxes.clear();
// //     std::vector<TvaiRect> imgRois;
// //     TvaiRect imgRoi{0,0,imgW,imgH};
// //     imgRois.push_back(imgRoi);
// //     for (int i = 0 ; i < input_bboxes.size() ; i++ ){
// //         BBox _box = input_bboxes[i];
// //         if (!check_rect_resolution(_box, _minTargeSize, _maxTargeSize)){
// //             // std::cout << "resolution failed" << _box.rect.width <<  "," << _box.rect.height << ", " <<_minTargeSize.height << "," << _minTargeSize.width << ","
// //             // << _maxTargeSize.height << "," << _maxTargeSize.width << std::endl;
// //             continue;
// //         }
// //         if (!check_in_valid_region(_box, _pAoiRect)){
// //             // std::cout << "region failed" << std::endl;
// //             continue;
// //         }
// //         if (!check_in_valid_region(_box, imgRois)){
// //             continue;
// //         }
// //         output_bboxes.push_back(_box);
// //     }    
// // }

// RET_CODE YoloDetectionV2::postprocess(float* model_output, VecObjBBox &bboxes, float threshold, float nms_threshold ,
//     float expand_ratio, float aspect_ratio, int imgW, int imgH)
// {
//     LOGI << "-> YoloDetectionV2::postprocess";
//     int nBBox = _oC;
//     int featLen = _oH;
//     std::vector<VecObjBBox> vecBox;
//     VecObjBBox vecBox_after_nms;
//     base_output2ObjBox_multiCls(model_output, vecBox, _output_cls_order.get(), _unique_cls_order , nBBox, featLen, threshold);
//     base_nmsBBox(vecBox,nms_threshold, NMS_MIN ,vecBox_after_nms );
//     LOGI << "after nms " << vecBox_after_nms.size() << std::endl;
//     base_transform_xyxy_xyhw(vecBox_after_nms, 1.0, aspect_ratio);
//     // object_filter(vecBox_after_nms, bboxes, imgW, imgH);
//     // LOGI << "after filter " << bboxes.size() << std::endl;
//     std::vector<VecObjBBox>().swap(vecBox);
//     bboxes.insert(bboxes.end(), vecBox_after_nms.begin(), vecBox_after_nms.end());
//     VecObjBBox().swap(vecBox_after_nms);
//     vecBox.clear();
//     return RET_CODE::SUCCESS;     
// }

// static void shift_box_from_roi_to_org(VecObjBBox &bboxes, TvaiRect &roirect){
//     for(auto &&bbox: bboxes){
//         bbox.rect.x += roirect.x;
//         bbox.rect.y += roirect.y;
//     }
// }
// RET_CODE YoloDetectionV2::postprocess(float* model_output, VecObjBBox &bboxes, TvaiRect tvrect, float threshold, float nms_threshold, 
//     float expand_ratio, float aspect_ratio, int imgW, int imgH)
// {
//     LOGI << "-> YoloDetectionV2::postprocess";
//     int nBBox = _oC;
//     int featLen = _oH;
//     std::vector<VecObjBBox> vecBox;
//     VecObjBBox vecBox_after_nms;
//     base_output2ObjBox_multiCls(model_output, vecBox, _output_cls_order.get(), _unique_cls_order , nBBox, featLen, threshold);
//     base_nmsBBox(vecBox,nms_threshold, NMS_MIN ,vecBox_after_nms );
//     LOGI << "after nms " << vecBox_after_nms.size() << std::endl;
//     base_transform_xyxy_xyhw(vecBox_after_nms, 1.0, aspect_ratio);
//     shift_box_from_roi_to_org(vecBox_after_nms, tvrect);
//     // object_filter(vecBox_after_nms, bboxes, imgW, imgH);
//     // LOGI << "after filter " << bboxes.size() << std::endl;
//     bboxes.insert(bboxes.end(), vecBox_after_nms.begin(), vecBox_after_nms.end());
//     std::vector<VecObjBBox>().swap(vecBox);
//     VecObjBBox().swap(vecBox_after_nms);
//     vecBox.clear();
//     return RET_CODE::SUCCESS;     
// }


// RET_CODE YoloDetectionV2::trackprocess(TvaiImage &tvimage, VecObjBBox &bboxes_in){
//     LOGI << "-> YoloDetectionV2::trackprocess";
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

// float YoloDetectionV2::clip_threshold(float x){
//     if(x < 0) return m_default_threshold;
//     if(x > 1) return m_default_threshold;
//     return x;
// }
// float YoloDetectionV2::clip_nms_threshold(float x){
//     if(x < 0) return m_default_nms_threshold;
//     if(x > 1) return m_default_nms_threshold;
//     return x;
// }

// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// // YoloV3
// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// YoloDetectionV3::YoloDetectionV3(){
//     m_detector = std::make_shared<YoloDetectionV2>();
//     m_trackor = std::make_shared<ByteTrackOriginPool>(m_fps, m_nn_buf);
// }

// RET_CODE YoloDetectionV3::init(std::map<InitParam, std::string> &modelpath){
//     // std::cout << "1..." << std::endl;
//     LOGI << "-> YoloDetectionV3::init";
//     if(use_auto_model){
//         RET_CODE ret = auto_model_file_search(m_roots, m_models_startswith, modelpath);
//         if(ret!=RET_CODE::SUCCESS) return ret;
//     }
//     RET_CODE ret = m_detector->init(modelpath);
//     if(ret!=RET_CODE::SUCCESS) {
//         LOGI << "m_detector->init(modelpath) failed return" << ret;
//         return ret;
//     }
//     LOGI << "<- YoloDetectionV3::init";
//     return RET_CODE::SUCCESS;
// }



// RET_CODE YoloDetectionV3::get_class_type(std::vector<CLS_TYPE> &valid_clss){
//     RET_CODE ret = m_detector->get_class_type(valid_clss);
//     return ret;
// }

// RET_CODE YoloDetectionV3::set_output_cls_order(std::vector<CLS_TYPE> &output_clss){
//     RET_CODE ret = m_detector->set_output_cls_order(output_clss);
//     return ret;
// }

// RET_CODE YoloDetectionV3::run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
//     threshold = clip_threshold(threshold);
//     nms_threshold = clip_threshold(nms_threshold);
//     BYTETRACKPARM track_param = { threshold, threshold+0.1f};
//     RET_CODE ret = m_detector->run(tvimage, bboxes, threshold, nms_threshold);
//     if(ret!=RET_CODE::SUCCESS) return ret;
//     if(m_trackor){
//         m_trackor->update(tvimage, bboxes, track_param);
//     }
// }

// float YoloDetectionV3::clip_threshold(float x){
//     if(x < 0) return m_default_threshold;
//     if(x > 1) return m_default_threshold;
//     return x;
// }
// float YoloDetectionV3::clip_nms_threshold(float x){
//     if(x < 0) return m_default_nms_threshold;
//     if(x > 1) return m_default_nms_threshold;
//     return x;
// }