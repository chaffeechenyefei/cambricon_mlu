#ifndef _MODULE_YOLO_DETECTION_HPP_
#define _MODULE_YOLO_DETECTION_HPP_
#include "module_base.hpp"

// #include "module_obj_feature_extraction.hpp"
// #include <easytrack/easy_track.h>
// #include "module_track.hpp"
// #include <mutex>
////////////////////////////////////////////////////////////////////////////////////////////////////////
//通用检测
////////////////////////////////////////////////////////////////////////////////////////////////////////

// using namespace ucloud;

// class YoloDetectionV2: public BaseModel{
// public:
//     YoloDetectionV2(){};
//     /**
//      * 20211117
//      * 新接口形式
//      */
//     RET_CODE init(std::map<InitParam, std::string> &modelpath);
//     RET_CODE init(const std::string &modelpath);
//     ~YoloDetectionV2();
//     /**
//      * @IN:
//      *  tvimage: BGR/YUV_NV21 format data
//      * @OUT:
//      *  bboxes: bounding box
//      * @DESC:
//      *  Support model: firstconv(input channel=4, uint8) only.
//      *  When NV21 is input, resize and crop ops are done on mlu.
//      *  Postprocess id done on cpu. Will be moved to mlu.
//      **/
//     RET_CODE run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold=0.55, float nms_threshold=0.6);
//     RET_CODE run(TvaiImage &tvimage, TvaiRect tvrect, VecObjBBox &bboxes, float threshold=0.55, float nms_threshold=0.6);

//     RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);
//     // RET_CODE run_batch(BatchImageIN &batch_tvimages, BatchBBoxOUT &batch_bboxes);
//     //inner use
//     RET_CODE set_output_cls_order(CLS_TYPE* output_clss, int len_output_clss);
//     //@overload
//     RET_CODE set_output_cls_order(std::vector<CLS_TYPE> &output_clss);

// protected:
//     float clip_threshold(float x);
//     float clip_nms_threshold(float x);

// private:
//     /**
//      * @param:
//      * input_bboxes: run函数返回的检测框
//      * output_bboxes: 过滤后的检测框(框的大小过滤)
//      **/
//     // void object_filter(VecObjBBox &input_bboxes, VecObjBBox &output_bboxes, int imgW, int imgH);
//     RET_CODE run_yuv_on_mlu_phyAddr(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold);
//     RET_CODE run_bgr_on_cpu(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold);
//     RET_CODE run_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold);
//     RET_CODE postprocess(float* model_output, VecObjBBox &bboxes, float threshold, float nms_threshold,
//         float expand_ratio, float aspect_ratio, int imgW, int imgH);

//     RET_CODE run_yuv_on_mlu(TvaiImage &tvimage, TvaiRect tvrect ,VecObjBBox &bboxes, float threshold, float nms_threshold);
//     RET_CODE postprocess(float* model_output, VecObjBBox &bboxes, TvaiRect tvrect, float threshold, float nms_threshold, 
//         float expand_ratio, float aspect_ratio, int imgW, int imgH);

//     // RET_CODE run_batch_yuv_on_mlu(BatchImageIN &batch_tvimages, BatchBBoxOUT &batch_bboxes);

//     //Tracking相关
//     /**
//      * trackprocess
//      * default 版本的跟踪, 基于featureMatch. cpu模式下的匹配.
//      */
//     RET_CODE trackprocess(TvaiImage &tvimage, VecObjBBox &bboxes_in);
//     /**
//      * init_trackor
//      * 初始化所有跟踪相关的模块和变量
//      */
//     RET_CODE init_trackor(const std::string &trackmodelpath);
//     RET_CODE create_trackor(int uuid_cam=-1);
    
//     std::shared_ptr<ObjFeatureExtraction> m_trackFeatExtractor = nullptr;
//     // std::unique_ptr<edk::EasyTrack> m_Trackor = nullptr;
//     std::map<int,std::shared_ptr<edk::EasyTrack>> m_Trackors;

//     //当传入的参数超过边界时,采用默认数值
//     float m_default_threshold = 0.55;
//     float m_default_nms_threshold = 0.6;
//     // CLS_TYPE* _output_cls_order{nullptr};
//     std::shared_ptr<CLS_TYPE> _output_cls_order;
//     std::map<CLS_TYPE,int> _unique_cls_order;
//     int _unique_cls_num = 0;
//     int _output_cls_num = 0;

// private://tracking param
// #ifdef MLU220
//     int m_fps = 4;
//     float m_max_cosine_distance = 0.5;//0.2 for fps=25
// #else
// #ifdef SIM_MLU220
//     int m_fps = 4;
//     float m_max_cosine_distance = 0.5;//0.2 for fps=25    
// #else
//     int m_fps = 25;
//     float m_max_cosine_distance = 0.2;
// #endif
// #endif
//     int m_nn_budget = 25;
//     float m_max_iou_distance = 0.5;
//     int m_n_init = 2;

// public://yolo检测系列, 开通自动模型推导, 根据根目录, 以及文件开头进行推导.
//     std::vector<std::string> m_roots = {"/cambricon/model/"};
//     std::map<ucloud::InitParam, std::string> m_models_startswith = {
//         {InitParam::BASE_MODEL, "yolov5s-conv-9"},
//         {InitParam::TRACK_MODEL, "feature_extract_4c4b"},
//     };        
//     bool use_auto_model = false;
// };

// /**
//  * YoloDetectionV3
//  * based on YoloDetectionV2, but tracking is replaced by ByteTrackor
//  */
// typedef std::shared_ptr<YoloDetectionV2> Detector_Ptr;
// class YoloDetectionV3: public AlgoAPI{
// public:
//     YoloDetectionV3();
//      ~YoloDetectionV3(){}
//     RET_CODE init(std::map<InitParam, std::string> &modelpath);
//     RET_CODE run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold=0.55, float nms_threshold=0.6);
//     RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss);
//     RET_CODE set_output_cls_order(std::vector<CLS_TYPE> &output_clss);

// protected:
//     float clip_threshold(float x);
//     float clip_nms_threshold(float x);


// ///Auto file search
// public://yolo检测系列, 开通自动模型推导, 根据根目录, 以及文件开头进行推导.
//     std::vector<std::string> m_roots = {"/cambricon/model/"};
//     // std::string m_basemodel_startswith = "yolov5s-conv-9";
//     std::map<InitParam, std::string> m_models_startswith = {
//         {InitParam::BASE_MODEL, "yolov5s-conv-9"},
//     };
//     bool use_auto_model = false;

// protected:
//     Detector_Ptr m_detector = nullptr;
//     std::shared_ptr<TrackPoolAPI<BYTETRACKPARM>> m_trackor = nullptr;

// private://tracking param
// #ifdef MLU220
//     int m_fps = 4;
//     int m_nn_buf = 10;
// #else
// #ifdef SIM_MLU220
//     int m_fps = 4;
//     int m_nn_buf = 10;
// #else
//     int m_fps = 25;//25
//     int m_nn_buf = 30;
// #endif
// #endif
//     //当传入的参数超过边界时,采用默认数值
//     float m_default_threshold = 0.55;
//     float m_default_nms_threshold = 0.6;
// };


#endif