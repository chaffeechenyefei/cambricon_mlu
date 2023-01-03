#include "libai_core.hpp"
#include "glog/logging.h"
#include <opencv2/opencv.hpp>
#include "utils/basic.hpp"
#include "inner_utils/inner_basic.hpp"
#include <fstream>
//precision
#include <iomanip>
#include <sstream>

#include <fstream>
#include <iostream>
/**
 * jsoncpp https://github.com/open-source-parsers/jsoncpp/tree/jsoncpp_version
 * tag: 1.9.5
*/
#include "json/json.h"

#ifdef DEBUG
#include <chrono>
#include <sys/time.h>
#include "inner_utils/module.hpp"
#endif

#include "utils/module_base.hpp"
#include "utils/module_yolo_detection_v2.hpp"
#include "utils/module_skeleton_detection.hpp"
#include "utils/module_retinaface_detection.hpp"
#include "utils/module_face_feature_extraction.hpp"
#include "utils/module_action_tsn_classify.hpp"
#include "utils/module_mod_unet2dshift.hpp"
#include "utils/module_binary_classification.hpp"
#include "utils/module_cascade_detection.hpp"
#include "utils/module_pspnet_water_segmentation.hpp"
#include "utils/module_mod_traditional.hpp"
#include "utils/module_smoking_detection.hpp"
#include "utils/module_phoning_detection.hpp"
#include "utils/module_falling_detection.hpp"
#include "utils/module_sos_action_detection.hpp"
#include "utils/module_yoloface_detection.hpp"
#include "utils/module_general_inference.hpp"


// #include <future>
using namespace ucloud;
using namespace cv;


#define MODEL_REPO "/cambricon/model/"
#define YOLO_REPO "/project/workspace/samples/yolov5/mlu270/"
#define FACE_DET_REPO "/project/workspace/samples/mlu_videofacerec/weights/face_det/"
#define FACE_REC_REPO "/project/workspace/samples/mlu_videofacerec/weights/face_rec/"
#define MLU_ORG_REPO "/project/workspace/samples/cambricon_offline_repo/"
#define OTHER_REPO "/project/workspace/samples/3d_unet_virtual/mlu270/"

static PrivateContextV2 global_init_env;

/////////////////////////////////////////////////////////////////////
// Class Factory
/////////////////////////////////////////////////////////////////////
AICoreFactory::AICoreFactory(){LOGI << "AICoreFactory Constructor";}
AICoreFactory::~AICoreFactory(){}

AlgoAPISPtr AICoreFactory::getAlgoAPI(AlgoAPIName apiName){
#ifdef BUILD_VERSION
    printf("\033[31m------------------------LIBAI_CORE-------------------------------\033[0m\n");
    printf("\033[31m-----------------------------------------------------------------\033[0m\n");
    printf("\033[31mcurrent version of libai_core.so is \"%s\"\033[0m\n", BUILD_VERSION);
    printf("\033[31m-----------------------------------------------------------------\033[0m\n");
    printf("\033[31m-----------------------------------------------------------------\033[0m\n");
#endif    
    AlgoAPISPtr apiHandle;
    switch (apiName)
    {
    case AlgoAPIName::UDF_JSON:
        apiHandle = std::make_shared<YoloDetectionV4>();
        break;
////////////////////
///人脸检测
////////////////////
    case AlgoAPIName::FACE_DETECTOR :
        {
            // FaceDetectionV2* _ptr = new FaceDetectionV2();
            FaceDetectionV4DeepSort* _ptr = new FaceDetectionV4DeepSort();
        #ifdef USE_STATIC_MODEL
        #ifdef MLU220
            _ptr->m_roots =  {MODEL_REPO};
        #else
            _ptr->m_roots =  {FACE_DET_REPO, MLU_ORG_REPO};
        #endif
            _ptr->use_auto_model = true;
            _ptr->m_models_startswith = {
                {InitParam::BASE_MODEL, "retinaface"},
                {InitParam::TRACK_MODEL, "feature_extract_4c4b"},
            };            
        #endif
            apiHandle.reset(_ptr);
        }
        break;

    case AlgoAPIName::FACE_DETECTORV2 :
        {
            // FaceDetectionV2* _ptr = new FaceDetectionV2();
            FaceDetectionV4ByteTrack* _ptr = new FaceDetectionV4ByteTrack();
        #ifdef USE_STATIC_MODEL
        #ifdef MLU220
            _ptr->m_roots =  {MODEL_REPO};
        #else
            _ptr->m_roots =  {FACE_DET_REPO, MLU_ORG_REPO};
        #endif
            _ptr->use_auto_model = true;
            _ptr->m_models_startswith = {
                {InitParam::BASE_MODEL, "retinaface"},
            };            
        #endif
            apiHandle.reset(_ptr);
        }
        break;        
////////////////////
///人脸检测 带属性
////////////////////        
    case AlgoAPIName::FACE_DETECTOR_ATTR :
        {
            FaceDetectionV4DeepSort* _ptr = new FaceDetectionV4DeepSort();
        #ifdef USE_STATIC_MODEL
        #ifdef MLU220
            _ptr->m_roots =  {MODEL_REPO};
        #else
            _ptr->m_roots =  {FACE_DET_REPO, MLU_ORG_REPO};
        #endif
            _ptr->use_auto_model = true;
            _ptr->m_models_startswith = {
                {InitParam::BASE_MODEL, "retinaface"},
                {InitParam::SUB_MODEL, "faceattr"},
                {InitParam::TRACK_MODEL, "feature_extract_4c4b"},
            };            
        #endif
            apiHandle.reset(_ptr);
        }
        break;
    case AlgoAPIName::FACE_DETECTORV2_ATTR :
        {
            FaceDetectionV4ByteTrack* _ptr = new FaceDetectionV4ByteTrack();
        #ifdef USE_STATIC_MODEL
        #ifdef MLU220
            _ptr->m_roots =  {MODEL_REPO};
        #else
            _ptr->m_roots =  {FACE_DET_REPO,OTHER_REPO};
        #endif
            _ptr->use_auto_model = true;
            _ptr->m_models_startswith = {
                {InitParam::BASE_MODEL, "retinaface"},
                {InitParam::SUB_MODEL, "faceattr"},
            };            
        #endif
            apiHandle.reset(_ptr);
        }
        break;        
////////////////////
///人脸特征提取
////////////////////        
    case AlgoAPIName::FACE_EXTRACTOR :
        {
            FaceExtractionV4* _ptr = new FaceExtractionV4();
        #ifdef USE_STATIC_MODEL
        #ifdef MLU220
            _ptr->m_roots =  {MODEL_REPO};
        #else
            _ptr->m_roots =  {FACE_REC_REPO};
        #endif        
            _ptr->use_auto_model = true;
            _ptr->m_models_startswith = {
                {InitParam::BASE_MODEL, "resnet101"},
            };   
        #endif  
            apiHandle.reset(_ptr);
        }
        break;
////////////////////
///人车非检测
////////////////////          
    case AlgoAPIName::GENERAL_DETECTOR ://通用物体检测, 基于yolov5s-conv + DeepSort
        {
            //names: [ 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor' ]
            YoloDetectionV4DeepSort *_ptr = new YoloDetectionV4DeepSort();
            std::vector<CLS_TYPE> yolov5s_conv_9 = {CLS_TYPE::PEDESTRIAN, CLS_TYPE::NONCAR, CLS_TYPE::CAR, CLS_TYPE::CAR, CLS_TYPE::CAR, CLS_TYPE::NONCAR, CLS_TYPE::NONCAR, CLS_TYPE::CAR, CLS_TYPE::NONCAR};
            _ptr->set_output_cls_order(yolov5s_conv_9);
        #ifdef USE_STATIC_MODEL
        #ifdef MLU220
            _ptr->m_roots =  {MODEL_REPO};
        #else
            _ptr->m_roots =  {YOLO_REPO};
        #endif
            _ptr->use_auto_model = true;
            _ptr->m_models_startswith = {
                {InitParam::BASE_MODEL, "yolov5s-conv-9"},
                {InitParam::TRACK_MODEL, "feature_extract_4c4b"},
            };
        #endif
            apiHandle.reset(_ptr);
        }
        break;
////////////////////
///人车非检测V2
////////////////////          
    case AlgoAPIName::GENERAL_DETECTORV2 ://通用物体检测, 基于yolov5s-conv + ByteTrack
        {
            //names: [ 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor' ]
            YoloDetectionV4ByteTrack *_ptr = new YoloDetectionV4ByteTrack();
            std::vector<CLS_TYPE> yolov5s_conv_9 =  {CLS_TYPE::PEDESTRIAN, CLS_TYPE::NONCAR, CLS_TYPE::CAR, CLS_TYPE::CAR, CLS_TYPE::CAR, CLS_TYPE::NONCAR, CLS_TYPE::NONCAR, CLS_TYPE::CAR, CLS_TYPE::NONCAR};
            _ptr->set_output_cls_order(yolov5s_conv_9);
        #ifdef USE_STATIC_MODEL
        #ifdef MLU220
            _ptr->m_roots =  {MODEL_REPO};
        #else
            _ptr->m_roots =  {YOLO_REPO};
        #endif
            _ptr->use_auto_model = true;
            _ptr->m_models_startswith = {
                {InitParam::BASE_MODEL, "yolov5s-conv-9"},
            };
        #endif
            apiHandle.reset(_ptr);
        }
        break;   
    case AlgoAPIName::GENERAL_DETECTORV3 ://通用物体检测, 基于yolov5s-conv + ByteTrack no reid
        {
            //names: [ 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor' ]
            YoloDetectionV4ByteTrack *_ptr = new YoloDetectionV4ByteTrack();
            std::vector<CLS_TYPE> yolov5s_conv_9 =  {CLS_TYPE::PEDESTRIAN, CLS_TYPE::NONCAR, CLS_TYPE::CAR, CLS_TYPE::CAR, CLS_TYPE::CAR, CLS_TYPE::NONCAR, CLS_TYPE::NONCAR, CLS_TYPE::CAR, CLS_TYPE::NONCAR};
            _ptr->set_output_cls_order(yolov5s_conv_9);
            _ptr->set_trackor(TRACKMETHOD::BYTETRACK_NO_REID);
        #ifdef USE_STATIC_MODEL
        #ifdef MLU220
            _ptr->m_roots =  {MODEL_REPO};
        #else
            _ptr->m_roots =  {YOLO_REPO};
        #endif
            _ptr->use_auto_model = true;
            _ptr->m_models_startswith = {
                {InitParam::BASE_MODEL, "yolov5s-conv-9"},
            };
        #endif
            apiHandle.reset(_ptr);
        }
        break;                 
    case AlgoAPIName::GENERAL_DETECTORV4 ://通用物体检测, 基于yolov5s-conv + ByteTrack + 轨迹输出
        {
            //names: [ 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor' ]
            YoloDetectionV4ByteTrack_POST_RULE_HOVER *_ptr = new YoloDetectionV4ByteTrack_POST_RULE_HOVER();
            std::vector<CLS_TYPE> yolov5s_conv_9 =  {CLS_TYPE::PEDESTRIAN, CLS_TYPE::NONCAR, CLS_TYPE::CAR, CLS_TYPE::CAR, CLS_TYPE::CAR, CLS_TYPE::NONCAR, CLS_TYPE::NONCAR, CLS_TYPE::CAR, CLS_TYPE::NONCAR};
            _ptr->set_output_cls_order(yolov5s_conv_9);
        #ifdef USE_STATIC_MODEL
        #ifdef MLU220
            _ptr->m_roots =  {MODEL_REPO};
        #else
            _ptr->m_roots =  {YOLO_REPO};
        #endif
            _ptr->use_auto_model = true;
            _ptr->m_models_startswith = {
                {InitParam::BASE_MODEL, "yolov5s-conv-9"},
            };
        #endif
            apiHandle.reset(_ptr);
        }
        break;       
    case AlgoAPIName::GENERAL_DETECTORV5 ://通用物体检测, 基于yolov5s-conv + deepsort + 轨迹输出
        {
            //names: [ 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor' ]
            YoloDetectionV4DeepSort_POST_RULE_HOVER *_ptr = new YoloDetectionV4DeepSort_POST_RULE_HOVER();
            std::vector<CLS_TYPE> yolov5s_conv_9 =  {CLS_TYPE::PEDESTRIAN, CLS_TYPE::NONCAR, CLS_TYPE::CAR, CLS_TYPE::CAR, CLS_TYPE::CAR, CLS_TYPE::NONCAR, CLS_TYPE::NONCAR, CLS_TYPE::CAR, CLS_TYPE::NONCAR};
            _ptr->set_output_cls_order(yolov5s_conv_9);
            apiHandle.reset(_ptr);
        }
        break;                
////////////////////
///行人检测加强版
////////////////////           
    case AlgoAPIName::PED_DETECTOR://行人检测加强版
        {
            printf("\033[32m AlgoAPIName::PED_DETECTOR\n\033[0m");
            YoloDetectionV4ByteTrack *_ptr = new YoloDetectionV4ByteTrack();
            std::vector<CLS_TYPE> yolov5s_conv_cls = {CLS_TYPE::PEDESTRIAN};
            _ptr->set_output_cls_order(yolov5s_conv_cls);
        #ifdef USE_STATIC_MODEL
        #ifdef MLU220
            _ptr->m_roots =  { MODEL_REPO };
        #else
            _ptr->m_roots =  { YOLO_REPO , };
        #endif        
            _ptr->use_auto_model = true;
            _ptr->m_models_startswith = {
                {InitParam::BASE_MODEL, "yolov5s-conv-people"},
            };            
        #endif
            apiHandle.reset(_ptr);
        }
        break;
////////////////////
///摔倒检测
////////////////////          
    case AlgoAPIName::PED_FALL_DETECTOR://行人摔倒检测, 只检测摔倒状态的行人
        {
            PedFallingDetection* _ptr = new PedFallingDetection();//20220406->bytetrack
        #ifdef USE_STATIC_MODEL
        #ifdef MLU220
            _ptr->m_roots =  {MODEL_REPO};
        #else
            _ptr->m_roots =  {YOLO_REPO, OTHER_REPO};
        #endif        
            _ptr->use_auto_model = true;
            _ptr->m_models_startswith = {
                {InitParam::BASE_MODEL, "yolov5s-conv-fall-ped-2022"},
                {InitParam::SUB_MODEL, "posenet"},
            };
        #endif
            apiHandle.reset(_ptr);            
        }
        // apiHandle = std::make_shared<PedFallingDetection>();
        break;
////////////////////
///行人骨架弯腰检测
////////////////////          
    case AlgoAPIName::PED_SK_DETECTOR://行人骨架弯腰检测
        {
            PedSkeletonDetection* _ptr = new PedSkeletonDetection();
        #ifdef USE_STATIC_MODEL
        #ifdef MLU220
            _ptr->m_roots =  {MODEL_REPO};
        #else
            _ptr->m_roots =  {YOLO_REPO, OTHER_REPO};
        #endif        
            _ptr->use_auto_model = true;
            _ptr->m_models_startswith = {
                {InitParam::BASE_MODEL, "yolov5s-conv-people-aug-fall"},
                {InitParam::SUB_MODEL, "posenet"},
            };
        #endif
            apiHandle.reset(_ptr);            
        }
        break;        
////////////////////
///测试
////////////////////          
    case AlgoAPIName::FIRE_CLASSIFIER:
        {
            BinaryClassificationV4 *_ptr = new BinaryClassificationV4();
            std::vector<CLS_TYPE> filter_cls{CLS_TYPE::FIRE};
            _ptr->set_filter_cls(filter_cls);
            _ptr->set_primary_output_cls(1, CLS_TYPE::FIRE);
            apiHandle.reset(_ptr);
        }
        break;
////////////////////
///火焰检测
////////////////////          
    case AlgoAPIName::FIRE_DETECTOR://火焰检测
        {
            YoloDetectionV4ByteTrack *_ptr = new YoloDetectionV4ByteTrack();//20220406->bytetrack
            std::vector<CLS_TYPE> yolov5s_conv_cls = {CLS_TYPE::FIRE};
            _ptr->set_output_cls_order(yolov5s_conv_cls);
        #ifdef USE_STATIC_MODEL
        #ifdef MLU220
            _ptr->m_roots =  {MODEL_REPO};
        #else
            _ptr->m_roots =  {YOLO_REPO};
        #endif        
            _ptr->use_auto_model = true;
            _ptr->m_models_startswith = {
                {InitParam::BASE_MODEL, "yolov5s-conv-fire"},};
        #endif        
            apiHandle.reset(_ptr);
        }
        break;
////////////////////
///火焰检测加强版
////////////////////          
    case AlgoAPIName::FIRE_DETECTOR_X://火焰检测加强版
        {
            CascadeDetection* _ptr = new CascadeDetection();//20220406->bytetrack
        #ifdef USE_STATIC_MODEL
        #ifdef MLU220
            _ptr->m_roots =  {MODEL_REPO};
        #else
            _ptr->m_roots = { YOLO_REPO, OTHER_REPO };
        #endif
            _ptr->use_auto_model = true;
            _ptr->m_models_startswith = {
                {InitParam::BASE_MODEL, "yolov5s-conv-fire"},
                {InitParam::SUB_MODEL, "resnet34fire"},
            };
        #endif            
            apiHandle.reset(_ptr);            
        }
        break;
////////////////////
///安全帽检测
////////////////////          
    case AlgoAPIName::SAFETY_HAT_DETECTOR ://安全帽检测
        {
            YoloDetectionV4ByteTrack *_ptr = new YoloDetectionV4ByteTrack();
            std::vector<CLS_TYPE> yolov5s_conv_cls = {CLS_TYPE::PED_SAFETY_HAT, CLS_TYPE::PED_HEAD};
            _ptr->set_output_cls_order(yolov5s_conv_cls);
        #ifdef USE_STATIC_MODEL
        #ifdef MLU220
            _ptr->m_roots =  {MODEL_REPO};
        #else
            _ptr->m_roots =  {YOLO_REPO};
        #endif                
            _ptr->use_auto_model = true;
            _ptr->m_models_startswith = {
                {InitParam::BASE_MODEL, "yolov5s-conv-safety-hat"},
            };            
        #endif            
            apiHandle.reset(_ptr);
        }
        break;
////////////////////
///同济大学安全帽检测416x416
////////////////////          
    case AlgoAPIName::TJ_HELMET_DETECTOR ://安全帽检测
        {
            YoloDetectionV4ByteTrack *_ptr = new YoloDetectionV4ByteTrack();
            std::vector<CLS_TYPE> yolov5s_conv_cls = {CLS_TYPE::PED_SAFETY_HAT};
            _ptr->set_output_cls_order(yolov5s_conv_cls);
        #ifdef USE_STATIC_MODEL
        #ifdef MLU220
            _ptr->m_roots =  {MODEL_REPO};
        #else
            _ptr->m_roots =  {YOLO_REPO};
        #endif                
            _ptr->use_auto_model = true;
            _ptr->m_models_startswith = {
                {InitParam::BASE_MODEL, "yolov5s-conv-safety-hat-tongji"},
            };            
        #endif            
            apiHandle.reset(_ptr);
        }
        break;        
////////////////////
///人头检测
////////////////////          
    case AlgoAPIName::HEAD_DETECTOR ://人头检测
        {
            YoloDetectionV4ByteTrack *_ptr = new YoloDetectionV4ByteTrack();
            std::vector<CLS_TYPE> yolov5s_conv_cls = {CLS_TYPE::PED_HEAD};
            _ptr->set_output_cls_order(yolov5s_conv_cls);
        #ifdef USE_STATIC_MODEL
        #ifdef MLU220
            _ptr->m_roots =  {MODEL_REPO};
        #else
            _ptr->m_roots =  {YOLO_REPO};
        #endif                
            _ptr->use_auto_model = true;
            _ptr->m_models_startswith = {
                {InitParam::BASE_MODEL, "yolov5s-conv-head"},
                {InitParam::TRACK_MODEL, "feature_extract_4c4b"},
            };            
        #endif            
            apiHandle.reset(_ptr);
        }
        break;
////////////////////
///电瓶车检测
////////////////////          
    case AlgoAPIName::NONCAR_DETECTOR ://非机动车检测加强版, 针对非机动车进电梯开发
        {
            YoloDetectionV4ByteTrack *_ptr = new YoloDetectionV4ByteTrack();//20220406->bytetrack
            std::vector<CLS_TYPE> yolov5s_conv_cls = {CLS_TYPE::EBYCYCLE, CLS_TYPE::BYCYCLE};
            _ptr->set_output_cls_order(yolov5s_conv_cls);
        #ifdef USE_STATIC_MODEL
        #ifdef MLU220
            _ptr->m_roots =  {MODEL_REPO};
        #else
            _ptr->m_roots =  {YOLO_REPO};
        #endif                
            _ptr->use_auto_model = true;
            _ptr->m_models_startswith = {
                {InitParam::BASE_MODEL, "yolov5s-conv-motor"},};            
            // _ptr->m_basemodel_startswith = "yolov5s-conv-motor";
            // _ptr->m_trackmodel_startswith = "none";
        #endif            
            apiHandle.reset(_ptr);
        }
        break;
////////////////////
///抽烟检测
////////////////////          
    case AlgoAPIName::SMOKING_DETECTOR:
        {
            SmokingDetectionV2* _ptr = new SmokingDetectionV2();
        #ifdef USE_STATIC_MODEL
        #ifdef MLU220
            _ptr->m_roots =  {MODEL_REPO};
        #else
            _ptr->m_roots =  {YOLO_REPO};
        #endif                
            _ptr->use_auto_model = true;
            _ptr->m_models_startswith = {
                {InitParam::BASE_MODEL, "retinaface"},
                {InitParam::SUB_MODEL, "yolov5s-conv-cig"},};
        #endif
            apiHandle.reset(_ptr);
        }
        // apiHandle = std::make_shared<SmokingDetectionV2>();
        break;
////////////////////
///打电话检测
////////////////////          
    case AlgoAPIName::PHONING_DETECTOR:
        {
            PhoningDetection* _ptr = new PhoningDetection();
        #ifdef USE_STATIC_MODEL
        #ifdef MLU220
            _ptr->m_roots =  {MODEL_REPO};
        #else
            _ptr->m_roots =  {YOLO_REPO};
        #endif                
            _ptr->use_auto_model = true;
            _ptr->m_models_startswith = {
                {InitParam::BASE_MODEL, "yolov5s-conv-9"},
                {InitParam::SUB_MODEL, "phoning"},};            
        #endif
            apiHandle.reset(_ptr);
        }
        // apiHandle = std::make_shared<PhoningDetection>();
        break;
////////////////////
///物品遗留
////////////////////           
    case AlgoAPIName::TARGERT_REMAIN_DETECTOR:{
            printf("\033[32m AlgoAPIName::TARGERT_REMAIN_DETECTOR\n\033[0m");
            IMP_OBJECT_REMAIN* _ptr_ = new IMP_OBJECT_REMAIN();
            apiHandle.reset(_ptr_);
        }
        break;
        
////////////////////
///高空抛物
////////////////////         
    case AlgoAPIName::MOD_DETECTOR:
        {
            MovementSegment* _ptr = new MovementSegment();
        #ifdef USE_STATIC_MODEL
        #ifdef MLU220
            _ptr->m_roots =  {MODEL_REPO};
        #else
            _ptr->m_roots =  {YOLO_REPO};
        #endif                
            _ptr->use_auto_model = true;
            _ptr->m_models_startswith = {
                {InitParam::BASE_MODEL, "diffunet"},};      
        #endif    
            apiHandle.reset(_ptr);           
        }
        // apiHandle = std::make_shared<UNet2DShiftSegment>();
        break;
////////////////////
///传统高空抛物
////////////////////  
    case AlgoAPIName::MOD_MOG2_DETECTOR:
        apiHandle = std::make_shared<BackgroundSegment>();
        break;    
    case AlgoAPIName::MOD_MOG2_DETECTORV2:
        apiHandle = std::make_shared<BackgroundSegmentV4ByteTrack>();
        break;        

////////////////////
///垃圾检测
////////////////////         
    case AlgoAPIName::TRASH_BAG_DETECTOR ://垃圾袋检测
        {
            YoloDetectionV4ByteTrack *_ptr = new YoloDetectionV4ByteTrack();//20220406->bytetrack
            //垃圾袋，纸盒，瓶子，海面漂浮垃圾堆
            std::vector<CLS_TYPE> yolov5s_conv_cls = {CLS_TYPE::TRASH_BAG, CLS_TYPE::OTHERS, CLS_TYPE::OTHERS, CLS_TYPE::TRASH_BAG};
            _ptr->set_output_cls_order(yolov5s_conv_cls);
        #ifdef USE_STATIC_MODEL
        #ifdef MLU220
            _ptr->m_roots =  {MODEL_REPO};
        #else
            _ptr->m_roots =  {YOLO_REPO};
        #endif          
            _ptr->use_auto_model = true;
            _ptr->m_models_startswith = {
                {InitParam::BASE_MODEL, "yolov5s-conv-trashbag"},};               
            // _ptr->m_basemodel_startswith = "yolov5s-conv-trashbag";
            // _ptr->m_trackmodel_startswith = "none";
        #endif
            apiHandle.reset(_ptr);
        }
        break;
////////////////////
///横幅检测
////////////////////         
    case AlgoAPIName::BANNER_DETECTOR ://横幅标语检测
        {
            YoloDetectionV4ByteTrack *_ptr = new YoloDetectionV4ByteTrack();//20220406->bytetrack
            std::vector<CLS_TYPE> yolov5s_conv_cls = {CLS_TYPE::BANNER};
            _ptr->set_output_cls_order(yolov5s_conv_cls);
        #ifdef USE_STATIC_MODEL
        #ifdef MLU220
            _ptr->m_roots =  {MODEL_REPO};
        #else
            _ptr->m_roots =  {YOLO_REPO};
        #endif          
            _ptr->use_auto_model = true;
            _ptr->m_models_startswith = {
                {InitParam::BASE_MODEL, "yolov5s-conv-banner"},}; 
            // _ptr->m_basemodel_startswith = "yolov5s-conv-banner";
            // _ptr->m_trackmodel_startswith = "none";
        #endif            
            apiHandle.reset(_ptr);
        }
        break;    
////////////////////
///道路积水检测
////////////////////         
    // case AlgoAPIName::WATER_DETECTOR_OLD://积水检测
    //     {
    //         UNetWaterSegment* _ptr = new UNetWaterSegment();
    //     #ifdef USE_STATIC_MODEL
    //     #ifdef MLU220
    //         _ptr->m_roots =  {MODEL_REPO};
    //     #else
    //         _ptr->m_roots =  {OTHER_REPO};
    //     #endif 
    //         _ptr->use_auto_model = true;
    //         _ptr->m_basemodel_startswith = "unetwater";
    //     #endif            
    //         apiHandle.reset(_ptr);            
    //     }
    //     // apiHandle = std::make_shared<UNetWaterSegment>();
    //     break;
    case AlgoAPIName::WATER_DETECTOR://积水检测
        {   
            PSPNetWaterSegmentV4* _ptr = new PSPNetWaterSegmentV4();
        #ifdef USE_STATIC_MODEL
        #ifdef MLU220
            _ptr->m_roots =  {MODEL_REPO};
        #else
            _ptr->m_roots =  {OTHER_REPO};
        #endif         
            _ptr->use_auto_model = true;
            _ptr->m_models_startswith = {
                {InitParam::BASE_MODEL, "pspwater"},}; 
        #endif            
            apiHandle.reset(_ptr);              
        }
        break;
////////////////////
///骨架定位
////////////////////         
    case AlgoAPIName::SKELETON_DETECTOR://骨架检测, 基于resnet50(hrnet离线量化有问题)
        apiHandle = std::make_shared<SkeletonDetectorV4>();
        break;
////////////////////
///打斗
////////////////////         
    case AlgoAPIName::ACTION_CLASSIFIER://行为识别: 打斗
        apiHandle = std::make_shared<TSNActionClassifyV4>();
        break;

// #ifndef MLU220
////////////////////
///SOS举手求救
//////////////////// 
    case AlgoAPIName::SOS_DETECTOR:
        {
            SOSDetectionV2* _ptr = new SOSDetectionV2();//20220407->bytetrack
        #ifdef USE_STATIC_MODEL
        #ifdef MLU220
            _ptr->m_roots =  {MODEL_REPO};
        #else
            _ptr->m_roots =  {YOLO_REPO};
        #endif              
            _ptr->use_auto_model = true;
            _ptr->m_models_startswith = {
                {InitParam::BASE_MODEL, "yolov5s-conv-9"},
                {InitParam::SUB_MODEL, "yolov5s-conv-hand"},
            };
        #endif    
            apiHandle.reset(_ptr);              
        }
        // apiHandle = std::make_shared<SOSDetectionV2>();
        break;
////////////////////
///车牌检测
////////////////////          
    case AlgoAPIName::LICPLATE_DETECTOR ://车牌检测
        {
            YoloFaceDetection *_ptr = new YoloFaceDetection();
            std::vector<CLS_TYPE> yolov5s_conv_cls = {CLS_TYPE::LICPLATE_BLUE, CLS_TYPE::LICPLATE_SGREEN, CLS_TYPE::LICPLATE_BGREEN, CLS_TYPE::LICPLATE_YELLOW};
            _ptr->set_output_cls_order(yolov5s_conv_cls);
        #ifdef USE_STATIC_MODEL
        #ifdef MLU220
            _ptr->m_roots =  {MODEL_REPO};
        #else
            _ptr->m_roots =  {YOLO_REPO};
        #endif                
            _ptr->use_auto_model = true;
            _ptr->m_models_startswith = {
                {InitParam::BASE_MODEL, "yolov5s-face-licplate"},
                // {InitParam::SUB_MODEL, "licplate-recog"},
            };            
        #endif            
            apiHandle.reset(_ptr);
        }
        break;
////////////////////
///车牌识别
////////////////////        
    case AlgoAPIName::LICPLATE_RECOGNIZER :
        apiHandle = std::make_shared<LicplateDetRec>();
        break;

////////////////////
///手的检测
////////////////////         
    case AlgoAPIName::HAND_DETECTOR ://人手检测, 一般用于内部, 不单独使用
        {
            YoloDetectionV4 *_ptr = new YoloDetectionV4();
            std::vector<CLS_TYPE> yolov5s_conv_cls = {CLS_TYPE::HAND};
            _ptr->set_output_cls_order(yolov5s_conv_cls);
        #ifdef USE_STATIC_MODEL
        #ifdef MLU220
            _ptr->m_roots =  {MODEL_REPO};
        #else
            _ptr->m_roots =  {YOLO_REPO};
        #endif                
            _ptr->use_auto_model = true;
            _ptr->m_models_startswith = {
                {InitParam::BASE_MODEL, "yolov5s-conv-hand"},
            };            
        #endif              
            apiHandle.reset(_ptr);
        }
        break;        
/*******************************************************************************
 * GENERAL_YOLOV5_DETECTOR = 2001,//通用yolov5检测器
*******************************************************************************/ 
    case AlgoAPIName::GENERAL_YOLOV5_DETECTOR:
        apiHandle = std::make_shared<YoloDetectionV4>();
        break;    
/*******************************************************************************
 * GENERAL_CLASSIFY        = 2002,//通用分类器
*******************************************************************************/         
    case AlgoAPIName::GENERAL_CLASSIFY:
        apiHandle = std::make_shared<ClassificationV4>();
        break;
/*******************************************************************************
 * GENERAL_INFER           = 2003,//通用推理接口, 返回内容自行解析
*******************************************************************************/      
    case AlgoAPIName::GENERAL_INFER:
        apiHandle = std::make_shared<GeneralInferenceSIMO>();
        break;
// #endif
    default:
        printf("ERROR: Current API code [%d] is not ready yet!\n", apiName);
        break;
    }
    return apiHandle;
}
/////////////////////////////////////////////////////////////////////
// End of Class Factory 
/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
// 其他功能函数
/////////////////////////////////////////////////////////////////////
unsigned char* ucloud::yuv_reader(std::string filename, int w, int h){
    std::ifstream fin(filename, std::ios::binary);
    int l = fin.tellg();
    fin.seekg(0, std::ios::end);
    int m = fin.tellg();
    fin.seekg(0,std::ios::beg);
    // cout << "file size " << (m-l) << " bytes" << endl;
    assert(m-l == w*h*1.5);
    int stride = w;
    int wh = w*h;
    unsigned char* yuvdata = (unsigned char*)malloc(int(wh/2*3)*sizeof(unsigned char));
    fin.read( reinterpret_cast<char*>(yuvdata) , int(wh/2*3)*sizeof(unsigned char));
    fin.close();
    return yuvdata;
}

unsigned char* ucloud::rgb_reader(std::string filename, int w, int h){
    std::ifstream fin(filename, std::ios::binary);
    int l = fin.tellg();
    fin.seekg(0, std::ios::end);
    int m = fin.tellg();
    fin.seekg(0,std::ios::beg);
    assert(m-l == w*h*3);
    int stride = w;
    int wh = w*h;
    unsigned char* rgbdata = (unsigned char*)malloc(int(wh*3)*sizeof(unsigned char));
    fin.read( reinterpret_cast<char*>(rgbdata) , int(wh*3)*sizeof(unsigned char));
    fin.close();
    return rgbdata;
}

/*--------------Read/Write API------------------*/

unsigned char* ucloud::readImg(std::string filepath, int &width, int &height){
    Mat im = imread(filepath);
    unsigned char* dst_ptr = nullptr;
    if (im.empty())
        return dst_ptr;
    dst_ptr = (unsigned char*)malloc(im.total()*3);
    memcpy(dst_ptr, im.data, im.total()*3);
    width = im.cols;
    height = im.rows;
    return dst_ptr;
}

unsigned char* ucloud::readImg_to_NV21(std::string filepath, int &width, int &height, int &stride){
    Mat im = imread(filepath);
    unsigned char* dst_ptr = nullptr;
    if (im.empty())
        return dst_ptr;
    dst_ptr = BGR2YUV_nv21_with_stride(im, width, height, stride, 2);
    return dst_ptr;
}

unsigned char* ucloud::readImg_to_NV21(std::string filepath, int w, int h,int &width, int &height, int &stride){
    Mat im = imread(filepath);
    if(im.empty()){
        printf("%s not found\n", filepath.c_str());
        return nullptr;
    }
    float ar = 1.0;
    im = resize_aspect(im,Size(w,h), false, ar );
    unsigned char* dst_ptr = nullptr;
    if (im.empty())
        return dst_ptr;
    dst_ptr = BGR2YUV_nv21_with_stride(im, width, height, stride, 2);
    return dst_ptr;    
}

unsigned char* ucloud::readImg_to_RGB(std::string filepath, int &width, int &height){
    Mat im = imread(filepath);
    unsigned char* dst_ptr = nullptr;
    if (im.empty())
        return dst_ptr;
    cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
    dst_ptr = (unsigned char*)malloc(im.total()*3);
    memcpy(dst_ptr, im.data, im.total()*3);
    width = im.cols;
    height = im.rows;
    return dst_ptr;
}

unsigned char* ucloud::readImg_to_BGR(std::string filepath, int &width, int &height){
    Mat im = imread(filepath);
    unsigned char* dst_ptr = nullptr;
    if (im.empty())
        return dst_ptr;
    dst_ptr = (unsigned char*)malloc(im.total()*3);
    memcpy(dst_ptr, im.data, im.total()*3);
    width = im.cols;
    height = im.rows;
    return dst_ptr;
}

unsigned char* ucloud::readImg_to_RGB(std::string filepath, int w, int h,int &width, int &height){
    Mat im = imread(filepath);
    unsigned char* dst_ptr = nullptr;
    if (im.empty())
        return dst_ptr;
    cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
    float ar = 1.0;
    im = resize_aspect(im,Size(w,h), false, ar );         
    dst_ptr = (unsigned char*)malloc(im.total()*3);
    memcpy(dst_ptr, im.data, im.total()*3);
    width = im.cols;
    height = im.rows;
    return dst_ptr;
}

unsigned char* ucloud::readImg_to_BGR(std::string filepath, int w, int h, int &width, int &height){
    Mat im = imread(filepath);
    unsigned char* dst_ptr = nullptr;
    if (im.empty())
        return dst_ptr;
    float ar = 1.0;
    im = resize_aspect(im,Size(w,h), false, ar );        
    dst_ptr = (unsigned char*)malloc(im.total()*3);
    memcpy(dst_ptr, im.data, im.total()*3);
    width = im.cols;
    height = im.rows;
    return dst_ptr;
}










static int global_rand_color[256] = {0};
static bool global_rand_color_init = false;
static int global_landmark_color[] = {//bgr::5
    255,0,0,
    0,255,0,
    0,0,255,
    255,255,0,
    0,255,255,
};
void ucloud::drawImg(unsigned char* img, int width, int height, VecObjBBox &bboxs, 
    bool disp_landmark, bool disp_label, bool use_rand_color, int color_for_trackid_or_cls){
    int thickness = (width/640) + 2;
    if(!global_rand_color_init){
        for(int i = 0; i < sizeof(global_rand_color)/sizeof(int); i++ ){
            global_rand_color[i] = rand()%255;
        }
        // global_rand_color[0] = 0; global_rand_color[1] = 0; global_rand_color[2] = 0; 
        global_rand_color_init = true;
    }
    if(!bboxs.empty()){
        int* rand_color = (int*)malloc(bboxs.size()*3*sizeof(int));
        if(use_rand_color){
            for(int i = 0; i < bboxs.size()*3; i++ )
                rand_color[i] = rand()%255;
        }
        Mat im(height,width,CV_8UC3, img);
        for(int i = 0; i < bboxs.size(); i++){
            Scalar color = use_rand_color ? Scalar(rand_color[i*3],rand_color[i*3+1],rand_color[i*3+2]): Scalar(0,255,0) ;
            if(bboxs[i].track_id >=0 && color_for_trackid_or_cls == 0){
                int gi = bboxs[i].track_id%(sizeof(global_rand_color)/sizeof(int)/3);
                color = Scalar(global_rand_color[gi*3],global_rand_color[gi*3+1],global_rand_color[gi*3+2]);
                // std::cout << gi << "," << global_rand_color[gi*3] << std::endl;
            } else if(bboxs[i].track_id >=0 && color_for_trackid_or_cls == 1){
                int gi = int(bboxs[i].objtype)%(sizeof(global_rand_color)/sizeof(int)/3);
                color = Scalar(global_rand_color[gi*3],global_rand_color[gi*3+1],global_rand_color[gi*3+2]);
                // std::cout << gi << "," << global_rand_color[gi*3] << std::endl;
            } else if(color_for_trackid_or_cls == 1 ){
                color = Scalar(0,255,0);
            }

            TvaiRect _rect = bboxs[i].rect;
            rectangle(im, Rect(_rect.x,_rect.y,_rect.width,_rect.height), color,thickness);
            if (disp_label){
                std::string track_id = "";
                if(bboxs[i].track_id >= 0 )
                    track_id = ": " + std::to_string(bboxs[i].track_id);
                std::stringstream stream;
                stream << std::fixed << std::setprecision(2) << bboxs[i].objectness;
                std::string score = " ," + stream.str();
                putText(im, std::to_string(bboxs[i].objtype)+track_id+score, Point(_rect.x, _rect.y+25), FONT_ITALIC, 0.8, color, thickness-1);
            }
                
            if (disp_landmark){
                for(int j = 0; j < bboxs[i].Pts.pts.size(); j++){
                    int gj = j%5;
                    cv::Scalar lmk_color = Scalar(global_landmark_color[3*gj],global_landmark_color[3*gj+1],global_landmark_color[3*gj+2]);
                    circle(im, Point2f(bboxs[i].Pts.pts[j].x, bboxs[i].Pts.pts[j].y),3, lmk_color,2);
                }
            }
            // std::cout << bboxs[i].trace.size()-1 << std::endl;
            for(int j = 0; j < int(bboxs[i].trace.size())-1; j++){
                // std::cout << "j: " << j << "---" << bboxs[i].trace.size() << std::endl;
                cv::line(im, cv::Point2f(bboxs[i].trace[j].x,bboxs[i].trace[j].y), cv::Point2f(bboxs[i].trace[j+1].x, bboxs[i].trace[j+1].y) , color, 2 );
                // std::cout << "j: " << j << std::endl;
            }
        }
        free(rand_color);
    }
}

void ucloud::writeImg(std::string filepath , unsigned char* img, int width, int height, bool overwrite){
    static int image_cnt = 0;
    std::string _filepath = filepath;
    if (!overwrite){
        while(exists_file(_filepath)){
            _filepath = filepath + "_" + std::to_string(image_cnt) + ".jpg";
            image_cnt++;
        }
    }
    Mat im(height, width, CV_8UC3, img);
    imwrite(_filepath, im);
}

void ucloud::freeImg(unsigned char** imgPtr){
    free(*imgPtr);
    *imgPtr = nullptr;
}


//视频读取基于opencv
void vidReader::release(){
    if(handle_t!=nullptr){
        VideoCapture* vid = reinterpret_cast<VideoCapture*>(handle_t);
        vid->release();    
        handle_t = nullptr;
        m_len = 0;
    }
}

bool vidReader::init(std::string filename){
    release();
    VideoCapture* vid = new VideoCapture();
    bool ret = vid->open(filename);
    if(!ret) {
        std::cout << "video open failed"<<std::endl;
        vid->release();
        return ret;
    }
    m_len = vid->get(CV_CAP_PROP_FRAME_COUNT);
    handle_t = reinterpret_cast<void*>(vid);
    return ret;
}

unsigned char* vidReader::getbgrImg(int &width, int &height){
    VideoCapture* vid = reinterpret_cast<VideoCapture*>(handle_t);
    Mat frame, img;
    bool ret = vid->isOpened();
    if(!ret) { std::cout<< "open failed" << std::endl; return nullptr;}
    ret = vid->read(frame);
    if(!ret || frame.empty() ){
        return nullptr;
    }
    frame.copyTo(img);
    width = img.cols;
    height = img.rows;
    unsigned char* buf = (unsigned char*)malloc(width*height*3*sizeof(unsigned char));
    memcpy(buf, img.data, img.total()*3);
    return buf;
}

unsigned char* vidReader::getyuvImg(int &width, int &height, int &stride){
    VideoCapture* vid = reinterpret_cast<VideoCapture*>(handle_t);
    Mat frame, img;
    bool ret = vid->isOpened();
    if(!ret) { std::cout<< "open failed" << std::endl; return nullptr;}
    ret = vid->read(frame);
    if(!ret || frame.empty() ){
        return nullptr;
    }
    frame.copyTo(img);
    unsigned char* dst_ptr = nullptr;
    dst_ptr = BGR2YUV_nv21_with_stride(img, width, height, stride, 2);
    return dst_ptr;
}

VIDOUT* vidReader::getImg(){
    VideoCapture* vid = reinterpret_cast<VideoCapture*>(handle_t);
    Mat frame, img;
    bool ret = vid->isOpened();
    if(!ret) { std::cout<< "open failed" << std::endl;}
    ret = vid->read(frame);
    if(!ret || frame.empty() ){
        return nullptr;
    }
    frame.copyTo(img);

    unsigned char* bgrbuf = (unsigned char*)malloc(img.total()*3*sizeof(unsigned char));
    memcpy(bgrbuf, img.data, img.total()*3);

    int width, height, stride;
    unsigned char* yuvbuf = nullptr;
    yuvbuf = BGR2YUV_nv21_with_stride(img, width, height, stride, 2);

    VIDOUT* rett = new VIDOUT();
    rett->bgrbuf = bgrbuf;
    rett->yuvbuf = yuvbuf;
    rett->w = width;
    rett->h = height;
    rett->s = stride;
    rett->_w = img.cols;
    rett->_h = img.rows;
    return rett;
}

int vidReader::width(){
    VideoCapture* vid = reinterpret_cast<VideoCapture*>(handle_t);
    return vid->get(CV_CAP_PROP_FRAME_WIDTH);
}

int vidReader::height(){
    VideoCapture* vid = reinterpret_cast<VideoCapture*>(handle_t);
    return vid->get(CV_CAP_PROP_FRAME_HEIGHT);    
}

int vidReader::fps(){
    VideoCapture* vid = reinterpret_cast<VideoCapture*>(handle_t);
    return vid->get(CV_CAP_PROP_FPS);
}

bool vidWriter::init(std::string filename, int width, int height, int fps){
    release();
    VideoWriter* vid = new VideoWriter();
    // bool ret = vid->open(filename, CV_FOURCC('D','I','V','X'), fps, Size(width, height));
    bool ret = vid->open(filename, CV_FOURCC('H','2','6','4'), fps, Size(width, height));
    if(!ret) {
        vid->release();
        return ret;
    }
    m_fps = fps;
    m_height = height;
    m_width = width;
    handle_t = reinterpret_cast<void*>(vid);
    return ret;
}

void vidWriter::release(){
    if(handle_t!=nullptr){
        VideoWriter* vid = reinterpret_cast<VideoWriter*>(handle_t);
        vid->release();    
        handle_t = nullptr;
    }    
}

void vidWriter::writeImg(unsigned char* buf, int bufw, int bufh){
    VideoWriter* vid = reinterpret_cast<VideoWriter*>(handle_t);
    Mat img( Size(bufw, bufh),CV_8UC3, buf);
    Mat img_fit;
    resize(img, img_fit, Size(m_width, m_height));
    vid->write(img_fit);
}




template<typename T>
void nmsBBox(std::vector<T>& input, float threshold, int type, std::vector<T>& output);


unsigned char* ucloud::tvaiImageToMatData(ucloud::TvaiImage input , int &width, int &height){
    unsigned char* inputbuf = input.pData;
    width = input.width;
    height = input.height;
    int bufSz = width*height*3*sizeof(unsigned char);
    unsigned char* outputbuf = (unsigned char*)malloc(bufSz);

    if (input.format == TvaiImageFormat::TVAI_IMAGE_FORMAT_BGR || input.format == TvaiImageFormat::TVAI_IMAGE_FORMAT_RGB ){
        if(input.width==input.stride){
            memcpy(outputbuf, inputbuf, bufSz);
        } else{
            unsigned char* _inbuf, *_outbuf;
            _inbuf = inputbuf;
            _outbuf = outputbuf;
            for(int i = 0; i < height; i++){
                memcpy(_outbuf,_inbuf, width*3*sizeof(unsigned char) );
                _inbuf += input.stride*3;
                _outbuf += width*3;
            }
        }
        
    } else if( input.format == TvaiImageFormat::TVAI_IMAGE_FORMAT_NV21 ) {
        cv::Mat bgr(height, width, CV_8UC3, outputbuf);
        YUV2BGR_n21(inputbuf, width, height, input.stride, bgr);
    } else{
        std::cout << "FORMAT NOT SUPPORTED" << std::endl;
        free(outputbuf);
        return nullptr;
    }
    return outputbuf;
}

/////////////////////////////////////////////////////////////////////
// End of Basic Function
/////////////////////////////////////////////////////////////////////
template<typename T>
void output2ObjBox_singleCls(float* output ,std::vector<T> &vecbox, int nbboxes, int stride ,float threshold=0.8){
    //xywh+objectness+nc (xywh=centerXY,WH)
    int nc = 1;
    assert(stride == 4+1+nc);
    for( int i=0; i < nbboxes; i++ ){
        float* _output = &output[i*stride];
        float objectness = _output[4];
        if( objectness < threshold )
            continue;
        else {
            T fbox;
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
            fbox.confidence = objectness*(*_output++);
            fbox.objtype = CLS_TYPE::UNKNOWN;
            vecbox.push_back(fbox);
        }
    }
    return;
}

template<typename T>
void nmsBBox(std::vector<T>& input, float threshold, int type, std::vector<T>& output);
void nmsBBox(std::vector<VecObjBBox> &input, float threshold, int type, VecObjBBox &output);
template<typename T>
void transform_xyxy_xyhw(std::vector<T> &vecbox, float expand_ratio=1.3 ,float aspect_ratio=1.0);
void output2ObjBox_multiCls(float* output ,std::vector<VecObjBBox> &vecbox, CLS_TYPE* cls_map ,int nbboxes ,int stride ,float threshold=0.8);

inline TvaiResolution get_valid_maxSize(TvaiResolution rect){
    // if width or height is set zero, then this condition will be ignored.
    unsigned int maxVal = 4096;
    TvaiResolution _rect;
    _rect.height = (rect.height==0)? maxVal:rect.height;
    _rect.width = (rect.width==0)? maxVal:rect.width;
    return _rect;
}

template<typename T>
bool sortFaceBox(const T& a, const T& b) {
  return  a.confidence > b.confidence;
}
template<typename T>
void nmsBBox(std::vector<T>& input, float threshold, int type, std::vector<T>& output) {
  std::sort(input.begin(), input.end(), sortFaceBox<T>);
  std::vector<int> bboxStat(input.size(), 0);
  for (size_t i=0; i<input.size(); ++i) {
    if (bboxStat[i] == 1) continue;
    output.push_back(input[i]);
    float area0 = (input[i].y1 - input[i].y0 + 1)*(input[i].x1 - input[i].x0 + 1);
    for (size_t j=i+1; j<input.size(); ++j) {
      if (bboxStat[j] == 1) continue;
      float roiWidth = std::min(input[i].x1, input[j].x1) - std::max(input[i].x0, input[j].x0);
      float roiHeight = std::min(input[i].y1, input[j].y1) - std::max(input[i].y0, input[j].y0);
      if (roiWidth<=0 || roiHeight<=0) continue;
      float area1 = (input[j].y1 - input[j].y0 + 1)*(input[j].x1 - input[j].x0 + 1);
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
void nmsBBox(std::vector<VecObjBBox> &input, float threshold, int type, VecObjBBox &output){
    if (input.empty()){
        VecObjBBox().swap(output);
        return;
    }
    for (int i = 0; i < input.size(); i++ ){
        nmsBBox(input[i], threshold, type, output);
    }
    return;
}
template<typename T>
void transform_xyxy_xyhw(std::vector<T> &vecbox, float expand_ratio ,float aspect_ratio){
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
    }
}
void output2ObjBox_multiCls(float* output ,std::vector<VecObjBBox> &vecbox, CLS_TYPE* cls_map ,int nbboxes ,int stride ,float threshold){
    //xywh+objectness+nc (xywh=centerXY,WH)
    int nc = stride - 5;
    for (int i=0; i<nc; i++){
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
            if (maxid < 0 || cls_map == nullptr)
                fbox.objtype = CLS_TYPE::UNKNOWN;
            else
                fbox.objtype = cls_map[maxid];
            vecbox[maxid].push_back(fbox);
        }
    }
    return;
}


template<typename T>
void output2FaceBox(float* output ,std::vector<T> &vecbox, int nbboxes ,int stride=15, float threshold=0.8){
  //score(1)+xyxy(4)+landmarks(10)
  for( int i=0; i < nbboxes; i++ ){
    float* _output = &output[i*stride];
    float score = *_output++;
    if( score < threshold )
      continue;
    else {
      T fbox;
      fbox.x0 = *_output++;
      fbox.y0 = *_output++;
      fbox.x1 = *_output++;
      fbox.y1 = *_output++;
      fbox.confidence = score;
      fbox.landmark.x[0] = *_output++;
      fbox.landmark.y[0] = *_output++;
      fbox.landmark.x[1] = *_output++;
      fbox.landmark.y[1] = *_output++;
      fbox.landmark.x[2] = *_output++;
      fbox.landmark.y[2] = *_output++;
      fbox.landmark.x[3] = *_output++;
      fbox.landmark.y[3] = *_output++;
      fbox.landmark.x[4] = *_output++;
      fbox.landmark.y[4] = *_output++;
      vecbox.push_back(fbox);
    }
  }
  return;
}

/////////////////////////////////////////////////////////////////////
// Class FaceDetector 
/////////////////////////////////////////////////////////////////////
// object filter
template<typename T>
bool check_rect_resolution(T _box, TvaiResolution minSz, TvaiResolution maxSz){
    int width = _box.rect.width;
    int height = _box.rect.height;
    if (width < minSz.width || width > maxSz.width)
        return false;
    if (height < minSz.height || height > maxSz.height)
        return false;
    return true;
}
template<typename T>
bool check_in_valid_region(T _box, const std::vector<TvaiRect> &pAoiRegion){
    int x_center = _box.rect.x + _box.rect.width/2;
    int y_center = _box.rect.y + _box.rect.height/2;
    if(pAoiRegion.size()==0){
        // std::cout << "empty region" << std::endl;
        return true;
    }
    // std::cout << "region size = " << pAoiRegion.size() << std::endl;
        
    for (int i = 0; i < pAoiRegion.size(); i++ ){
        //  std::cout << pAoiRegion[i].x << "," << pAoiRegion[i].y << "," << std::endl;
        if (x_center>=pAoiRegion[i].x && x_center<=pAoiRegion[i].x+pAoiRegion[i].width &&\
            y_center>=pAoiRegion[i].y && y_center<=pAoiRegion[i].y+pAoiRegion[i].height
        )
            return true;
    }
    return false;
}
// bool check_face_angle(FaceInfo _box){
//     float ratio = ratio_p_mid_line( _box.landmark.x[2],_box.landmark.y[2], _box.landmark.x[0],_box.landmark.y[0], _box.landmark.x[1],_box.landmark.y[1]);
//     if (ratio <= 0.2)
//         return true;
//     else
//         return false;
// }
/////////////////////////////////////////////////////////////////////
// End of Class FaceDetector 
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
// Class FaceExtractor 
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
// End of Class FaceExtractor 
/////////////////////////////////////////////////////////////////////
void ucloud::releaseTvaiFeature(std::vector<ucloud::TvaiFeature>& feats){
    for(int i=0 ; i < feats.size(); i++ ){
        free(feats[i].pFeature);
        feats[i].pFeature = nullptr;
    }
    std::vector<TvaiFeature>().swap(feats);
    feats.clear();
}

void ucloud::releaseVecObjBBox(VecObjBBox &bboxs){
    for(int i = 0; i < bboxs.size(); i++){
        if(bboxs[i].feat.pFeature!=nullptr) free(bboxs[i].feat.pFeature);
        bboxs[i].feat.pFeature = nullptr;
        bboxs[i].Pts.pts.clear();
    }
    VecObjBBox().swap(bboxs);
}
/////////////////////////////////////////////////////////////////////
// Class YOLODETECTOR 
/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////
// End of Class YOLODETECTOR 
/////////////////////////////////////////////////////////////////////

/*--------------BEGIN Clocker------------------*/
Clocker::Clocker(){
    ctx = new Timer();
}

Clocker::~Clocker(){
    delete reinterpret_cast<Timer*>(ctx);
}

void Clocker::start(){
    Timer* cTx = reinterpret_cast<Timer*>(ctx);
    cTx->start();
}

double Clocker::end(std::string title, bool display){
    Timer* cTx = reinterpret_cast<Timer*>(ctx);
    return cTx->end(title, display);
}
/*--------------END Clocker------------------*/


/*******************************************************************************
输入argc argv的解析
chaffee.chen@2022-10-20
*******************************************************************************/
/*--------------BEGIN ArgParser------------------*/
bool ArgParser::add_argument(const std::string &keyword, int default_value , const std::string &helpword){
    //if(std::is_same<typename std::decay<T>::type,int>::value)
    std::string valType = "int";
    m_cmd_int.insert(std::make_pair(keyword, default_value));
    std::string _helpword = helpword+" ["+valType+"], default value = " + std::to_string(default_value);
    m_cmd_help.insert(std::make_pair(keyword, _helpword));
    printf("+ %s\n", _helpword.c_str());
    return true;
}

bool ArgParser::add_argument(const std::string &keyword, float default_value , const std::string &helpword){
    //if(std::is_same<typename std::decay<T>::type,int>::value)
    std::string valType = "float";
    m_cmd_float.insert(std::make_pair(keyword, default_value));
    std::string _helpword = helpword+" ["+valType+"], default value = " + std::to_string(default_value);
    m_cmd_help.insert(std::make_pair(keyword, _helpword));
    printf("+ %s\n", _helpword.c_str());
    return true;
}

// bool ArgParser::add_argument(const std::string &keyword, bool default_value , const std::string &helpword){
//     //if(std::is_same<typename std::decay<T>::type,int>::value)
//     std::string valType = "bool";
//     m_cmd_bool.insert(std::make_pair(keyword, default_value));
//     std::string _helpword = helpword+" ["+valType+"], default value = " + (default_value?"true":"false");
//     m_cmd_help.insert(std::make_pair(keyword, _helpword));
//     printf("+ %s\n", _helpword.c_str());
//     return true;
// }

bool ArgParser::add_argument(const std::string &keyword, const std::string &default_value , const std::string &helpword){
    std::string valType = "string";
    m_cmd_str.insert(std::make_pair(keyword, default_value));
    std::string _helpword = helpword+" ["+valType+"], default value = " + default_value;
    m_cmd_help.insert(std::make_pair(keyword, _helpword));
    printf("+ %s\n", _helpword.c_str());
    return true;
}

bool ArgParser::parser(int argc, char* argv[]){
    printf("------------------------------------------\n");
    printf("parser\n");
    printf("------------------------------------------\n");
    for(int i = 0; i < argc; i++){
        std::string keyword = std::string(argv[i]);
        // printf("%s ", keyword.c_str());
        if(keyword=="-help"||keyword=="--help"){
            print_help();
            return false;
        }
    }
    printf("start parsering...\n");
    for(int i = 0 ; i < argc -1; i++ ){
        std::string keyword = std::string(argv[i]);
        // printf("%s \n", argv[i]);
        if(m_cmd_help.find(keyword)==m_cmd_help.end()){
            // printf("%s command is not known, which will be skipped.\n", keyword.c_str());
            continue;
        }
        if(m_cmd_int.find(keyword)!=m_cmd_int.end()){
            m_cmd_int[keyword] = std::atoi(argv[i+1]);
            printf("set [%s] to %d\n", keyword.c_str() ,m_cmd_int[keyword]);
        }
        if(m_cmd_str.find(keyword)!=m_cmd_str.end()){
            m_cmd_str[keyword] = std::string(argv[i+1]);
            printf("set [%s] to %s\n", keyword.c_str() ,m_cmd_str[keyword].c_str());
        }
        // if(m_cmd_bool.find(keyword)!=m_cmd_bool.end()){
        //     m_cmd_bool[keyword] = std::atoi(argv[i+1]) > 0 ? true:false ;
        // }
        if(m_cmd_float.find(keyword)!=m_cmd_float.end()){
            m_cmd_float[keyword] = std::atof(argv[i+1]);
            printf("set [%s] to %.3f\n", keyword.c_str() ,m_cmd_float[keyword]);
        }
    }
    // print_help();
    printf("------------------------------------------\n");
    return true;
}

void ArgParser::print_help(){
    printf("------------------------------------------\n");
    printf("help\n");
    printf("------------------------------------------\n");
    for(auto&& cmd: m_cmd_help){
        printf("%s : %s\n", cmd.first.c_str(), cmd.second.c_str());
    }
    printf("------------------------------------------\n");
}

float ArgParser::get_value_float(const std::string &keyword){
    if(m_cmd_help.find(keyword)==m_cmd_help.end()){
        printf("%s keyword unknown, will be skipped and return 0\n", keyword.c_str());
        return 0;
    }
    if(m_cmd_float.find(keyword)!=m_cmd_float.end()){
        return m_cmd_float[keyword];
    }
    return 0;
}

int ArgParser::get_value_int(const std::string &keyword){
    if(m_cmd_help.find(keyword)==m_cmd_help.end()){
        printf("%s keyword unknown, will be skipped and return 0\n", keyword.c_str());
        return 0;
    }
    if(m_cmd_int.find(keyword)!=m_cmd_int.end()){
        return m_cmd_int[keyword];
    }
    return 0;
}

// bool ArgParser::get_value_bool(const std::string &keyword){
//     if(m_cmd_help.find(keyword)==m_cmd_help.end()){
//         printf("%s keyword unknown, will be skipped and return 0\n", keyword.c_str());
//         return 0;
//     }
//     if(m_cmd_bool.find(keyword)!=m_cmd_bool.end()){
//         return m_cmd_bool[keyword];
//     }
//     return 0;
// }

std::string ArgParser::get_value_string(const std::string &keyword){
    if(m_cmd_help.find(keyword)==m_cmd_help.end()){
        printf("%s keyword unknown, will be skipped and return 0\n", keyword.c_str());
        return "";
    }
    if(m_cmd_str.find(keyword)!=m_cmd_str.end()){
        return m_cmd_str[keyword];
    }
    return "";
}

bool ArgParser::is_value_exist(const std::string &keyword){
    if(m_cmd_help.find(keyword)!=m_cmd_help.end()){
        return true;
    } else{
        return false;
    }
}


// template<typename T> T get_value(const std::string &keyword){
//         if(m_cmd_help.find(keyword)!=m_cmd_help.end()){
//             printf("%s keyword unknown, will be skipped and return 0\n", keyword.c_str());
//             return 0;
//         }
//         if(m_cmd_int.find(keyword)!=m_cmd_int.end()){
//             if(std::is_same<typename std::decay<T>::type,int>::value)
//                 return m_cmd_int[keyword];
//             else{
//                 printf("return type int unmatched with template type\n");
//             }
//         }
//         if(m_cmd_str.find(keyword)!=m_cmd_str.end()){
//             if(std::is_same<typename std::decay<T>::type,std::string>::value)
//                 return m_cmd_str[keyword];
//             else{
//                 printf("return type string unmatched with template type\n");
//             }        
//         }
//         if(m_cmd_bool.find(keyword)!=m_cmd_bool.end()){
//             if(std::is_same<typename std::decay<T>::type,bool>::value)
//                 return m_cmd_bool[keyword];
//             else{
//                 printf("return type bool unmatched with template type\n");
//             }          
//         }
//         if(m_cmd_float.find(keyword)!=m_cmd_float.end()){
//             if(std::is_same<typename std::decay<T>::type,float>::value)
//                 return m_cmd_float[keyword];
//             else{
//                 printf("return type float unmatched with template type\n");
//             }          
//         }
//         return 0;
//     }




/*--------------BEGIN ArgParser------------------*/

