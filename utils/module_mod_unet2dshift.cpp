#include "module_mod_unet2dshift.hpp"
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
MovementSegment 基于分割算法的高空抛物
chaffee.chen@2022-10-
*******************************************************************************/
RET_CODE MovementSegment::init(const std::string &modelpath){
    LOGI << "-> MovementSegment::init";
    bool pad_both_side = false;//单边预留
    bool keep_aspect_ratio = false;//不保持长宽比
    BASE_CONFIG config(MODEL_INPUT_FORMAT::BGRA, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
    RET_CODE ret = m_net->base_init(modelpath, config);
    if(ret!=RET_CODE::SUCCESS) return ret;
    ret = init_trackor();
    //Self param
    return ret;
}

RET_CODE MovementSegment::init(std::map<InitParam, std::string> &modelpath){
    if(use_auto_model){
        RET_CODE ret = auto_model_file_search(m_roots, m_models_startswith, modelpath);
        if(ret!=RET_CODE::SUCCESS) {
            printf("auto_model_file_search failed, return %d\n",ret);
            return ret;
        }
    }
    if(modelpath.find(InitParam::BASE_MODEL) == modelpath.end()) return RET_CODE::ERR_INIT_PARAM_FAILED;
    return this->init(modelpath[InitParam::BASE_MODEL]);
}

//clear self param
MovementSegment::~MovementSegment(){LOGI << "-> MovementSegment::~MovementSegment";
    m_Trackors.clear();
    clear();
}

RET_CODE MovementSegment::run_yuv_on_mlu(BatchImageIN &batch_tvimages, VecObjBBox &bboxes){
    LOGI << "-> UNet2DShiftSegment::run_yuv_on_mlu";
    RET_CODE ret = RET_CODE::FAILED;
    std::vector<float> batch_aspect_ratio;

    VecRect batch_roi; 
    float** batch_model_output = nullptr;
    {
        std::vector<float> _aspect_ratio;
        ret = m_net->general_batch_preprocess_yuv_on_mlu(batch_tvimages, batch_roi, _aspect_ratio);
        if(ret!=RET_CODE::SUCCESS) return ret;
        batch_model_output = m_net->general_mlu_infer();
    }
    int _W = m_net->m_inputShape[0].W();
    int _H = m_net->m_inputShape[0].H();
    float aspect_ratio_x = (1.0*_W)/batch_tvimages[0].width;
    float aspect_ratio_y = (1.0*_H)/batch_tvimages[0].height;
    //ATT. UNet2DShiftSegment IN_BATCH_SIZE=3 OUT_BATCH_SIZE=1
    // int outputDim = m_net->m_outputShape[0].DataCount();
    //TODO
    VecObjBBox bboxes_before_filter;
    //默认所有输入图像同样尺寸, 所以aspect_ratio只用第一个
    // visual(batch_tvimages[2], batch_model_output);
    postprocess(batch_model_output[0], bboxes_before_filter, aspect_ratio_x, aspect_ratio_y);
    m_net->cpu_free(batch_model_output);
    bboxes.insert(bboxes.end(), bboxes_before_filter.begin(), bboxes_before_filter.end());
    // postfilter(bboxes_before_filter, bboxes);
    if(!m_test_mod)//开启测试时, 关闭轨迹后处理
        trackprocess(batch_tvimages[0] , bboxes);
    return ret;
}

RET_CODE MovementSegment::clear(){
    for(auto &&buf: m_tviamge_buffers){
        for(auto &&tvimage: buf.second){
            cnrtFree( (void*)(tvimage.u64PhyAddr[0]));
            cnrtFree( (void*)(tvimage.u64PhyAddr[1]));
        }
        buf.second.clear();
    }
    m_tviamge_buffers.clear();
    // for(auto &&tvimage : tvimage_buffers){
    //     cnrtFree( (void*)(tvimage.u64PhyAddr[0]));
    //     cnrtFree( (void*)(tvimage.u64PhyAddr[1]));
    // }
    // tvimage_buffers.clear();
}


RET_CODE MovementSegment::push_back(TvaiImage &tvimage){
    RET_CODE ret = RET_CODE::FAILED;
    int cam_uuid = tvimage.uuid_cam;
    if(m_tviamge_buffers.find(cam_uuid) != m_tviamge_buffers.end()){
        while( m_tviamge_buffers[cam_uuid].size()>=m_batchsize){
            cnrtFree( (void*)(m_tviamge_buffers[cam_uuid][0].u64PhyAddr[0]));
            cnrtFree( (void*)(m_tviamge_buffers[cam_uuid][0].u64PhyAddr[1]));
            m_tviamge_buffers[cam_uuid].erase(m_tviamge_buffers[cam_uuid].begin());
        }
    }
    if(tvimage.format == TVAI_IMAGE_FORMAT_NV12 || tvimage.format == TVAI_IMAGE_FORMAT_NV21 ){
        TvaiImage tmp = tvimage;
        tmp.usePhyAddr = true;
        void* u64_cp_1 = nullptr;
        void* u64_cp_2 = nullptr;
        cnrtMalloc( &u64_cp_1, tmp.stride* tmp.height);
        cnrtMalloc( &u64_cp_2, tmp.stride* tmp.height/2);
        tmp.u64PhyAddr[0] = reinterpret_cast<uint64_t>(u64_cp_1);
        tmp.u64PhyAddr[1] = reinterpret_cast<uint64_t>(u64_cp_2);
        if(tvimage.usePhyAddr){
            cnrtMemcpy(u64_cp_1, (void*)(tvimage.u64PhyAddr[0]), tmp.height*tmp.stride, CNRT_MEM_TRANS_DIR_DEV2DEV);
            cnrtMemcpy(u64_cp_2 , (void*)(tvimage.u64PhyAddr[1]), tmp.stride*tmp.height/2, CNRT_MEM_TRANS_DIR_DEV2DEV);
        } else {
            cnrtMemcpy(u64_cp_1, tvimage.pData, tmp.height*tmp.stride, CNRT_MEM_TRANS_DIR_HOST2DEV);
            cnrtMemcpy(u64_cp_2, tvimage.pData+tvimage.height*tvimage.stride, tmp.stride*tmp.height/2, CNRT_MEM_TRANS_DIR_HOST2DEV);
        }
        m_tviamge_buffers[cam_uuid].push_back(tmp);
        ret = RET_CODE::SUCCESS;
        // std::cout << "tvimage_buffers: " <<  tvimage_buffers.size() <<std::endl;
    } else {
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    }
    return ret;
}

RET_CODE MovementSegment::run(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    RET_CODE ret = RET_CODE::FAILED;
    int cam_uuid = tvimage.uuid_cam;
    if(tvimage.format == TVAI_IMAGE_FORMAT_NV12 || tvimage.format == TVAI_IMAGE_FORMAT_NV21 ){
        ret = push_back(tvimage);
        if(ret!=RET_CODE::SUCCESS) return ret;
        if(m_tviamge_buffers[cam_uuid].size()==m_batchsize)
            ret = run(m_tviamge_buffers[cam_uuid], bboxes, threshold, nms_threshold);
    } else {
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    }
    return ret;
}

RET_CODE MovementSegment::run(BatchImageIN &batch_tvimages, VecObjBBox &bboxes, float threshold, float nms_threshold){
    // std::cout <<"trigger" << std::endl;
    RET_CODE ret = RET_CODE::FAILED;
    if(batch_tvimages.empty()) return ret;//return FAILED if input empty
    if(batch_tvimages[0].format == TVAI_IMAGE_FORMAT_NV21 || batch_tvimages[0].format == TVAI_IMAGE_FORMAT_NV12 ){
        ret = run_yuv_on_mlu(batch_tvimages ,bboxes);//little trick, use nullptr to judge the condition
    }
    else
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    return ret;
}

void MovementSegment::visual(TvaiImage& tvimage, float* model_output){
    int _oW = m_net->m_outputShape[0].W();
    int _oH = m_net->m_outputShape[0].H();
    Mat cv_predict(Size(_oW, _oH), CV_32FC1, model_output);
    Mat cv_mask = 255 - cv_predict*255;
    cv_mask.convertTo(cv_mask, CV_8U);
    Mat cv_image(Size(tvimage.stride, 3*tvimage.height/2), CV_8UC1, tvimage.pData );
    Mat cv_bgr;
    cvtColor(cv_image, cv_bgr, COLOR_YUV2BGR_NV21);
    int _W = m_net->m_inputShape[0].W();
    int _H = m_net->m_inputShape[0].H();
    resize(cv_bgr, cv_bgr, Size(_W,_H));
    // Mat cv_bgras[4];
    // split(cv_bgr, cv_bgras);
    // cv_mask.copyTo(cv_bgras[3]);
    // Mat cv_tmp;
    // merge(cv_bgras, 4, cv_tmp);
    saveImg("tmp/", ".jpg", cv_mask);
}

RET_CODE MovementSegment::postprocess(float* model_output, VecObjBBox &bboxes, float aspect_ratio_x, float aspect_ratio_y){
    int featDim = m_net->m_outputShape[0].DataCount();//1,224,224,1
    int _oW = m_net->m_outputShape[0].W();
    int _oH = m_net->m_outputShape[0].H();
    Mat cv_predict(Size(_oW, _oH), CV_32FC1, model_output);
    Mat cv_mask = cv_predict > m_predict_threshold;
    // Mat cv_tmp = cv_predict*255;
    // Mat cv_tmp2;
    // cv_tmp.convertTo(cv_tmp2, CV_8U );
    // saveImg("tmp/",".jpg", cv_tmp2);
    // cv_mask *= 255;
    Mat cv_kernel = Mat::ones(Size(3,3), CV_8UC1);
    Mat cv_dilate_mask, cv_t_mask;
    vector<vector<Point>> vec_cv_contours;
    dilate(cv_mask, cv_dilate_mask , cv_kernel, Point(1,1), 3);
    // float auto_thresh = cv::threshold(cv_dilate_mask, cv_t_mask, 0, 255, THRESH_BINARY+THRESH_OTSU);
    // imwrite("tmp.jpg", cv_t_mask);
    cv_dilate_mask = cv_dilate_mask > 50;
    findContours(cv_dilate_mask,vec_cv_contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    // std::cout << vec_cv_contours.size() << std::endl;
    for(auto iter=vec_cv_contours.begin(); iter!=vec_cv_contours.end(); iter++){
        Rect rect = boundingRect(*iter);
        // float ratio = (1.0* iter->size()) / (rect.width*rect.height);
        // std::cout <<  "inner ratio = " << ratio << std::endl;
        BBox bbox;
        bbox.objtype = CLS_TYPE::FALLING_OBJ_UNCERTAIN;
        bbox.confidence = 1.0;
        bbox.objectness = bbox.confidence;
        bbox.rect.x = ((1.0*rect.x) / aspect_ratio_x); bbox.rect.width = ((1.0*rect.width) / aspect_ratio_x);
        bbox.rect.y = ((1.0*rect.y) / aspect_ratio_y); bbox.rect.height = ((1.0*rect.height) / aspect_ratio_y);
        bboxes.push_back(bbox);
    }
    // free(binary_predict);
    return RET_CODE::SUCCESS;
}

RET_CODE MovementSegment::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    valid_clss = _cls_;
    return RET_CODE::SUCCESS;
};

RET_CODE MovementSegment::create_trackor(int uuid_cam){
    if(m_Trackors.find(uuid_cam)==m_Trackors.end()){
        LOGI << "-> UNet2DShiftSegment::create_trackor+";
        std::shared_ptr<BoxTraceSet> m_trackor_t(new BoxTraceSet());
        m_Trackors.insert(std::pair<int,std::shared_ptr<BoxTraceSet>>(uuid_cam,m_trackor_t));
    } else LOGI << "-> UNet2DShiftSegment::create_trackor";
    return RET_CODE::SUCCESS;
}

RET_CODE MovementSegment::init_trackor(){
    create_trackor(-1);
    return RET_CODE::SUCCESS;
}

RET_CODE MovementSegment::trackprocess(TvaiImage &tvimage, VecObjBBox &ins){
    create_trackor(tvimage.uuid_cam);
    std::vector<BoxPoint> bpts;
    int cur_time = 1 + m_Trackors[tvimage.uuid_cam]->m_time;
    int box_cnt = 0;
    for(auto in: ins){
        BoxPoint tmp = BoxPoint(in, cur_time);
        bpts.push_back( tmp );
        box_cnt++;
        if(box_cnt >= m_max_executable_bboxes ) break;
        // std::cout << "push_back: " << in.rect.x  << "," << in.rect.y <<"," << in.rect.width <<"," << in.rect.height << std::endl;
    }
    m_Trackors[tvimage.uuid_cam]->push_back( bpts );
    std::vector<BoxPoint> marked_pts, unmarked_pts;
    m_Trackors[tvimage.uuid_cam]->output_last_point_of_trace(marked_pts, unmarked_pts);
    ins.clear();
    for(auto bxpt: marked_pts){
        BBox pt;
        pt.confidence = 1.0;
        pt.objectness = 1.0;
        pt.quality = 1.0;
        pt.objtype = CLS_TYPE::FALLING_OBJ;
        pt.rect = TvaiRect{ int(bxpt.x),int(bxpt.y),int(bxpt.w),int(bxpt.h)};
        // std::cout << "HIT:" << bxpt.x << "," << bxpt.y << "," << bxpt.w << "," << bxpt.h << std::endl;
        ins.push_back(pt);
    }
    for(auto bxpt: unmarked_pts){
        BBox pt;
        pt.confidence = 1.0;
        pt.objectness = 1.0;
        pt.quality = 1.0;
        pt.objtype = CLS_TYPE::FALLING_OBJ_UNCERTAIN;
        pt.rect = TvaiRect{ int(bxpt.x),int(bxpt.y),int(bxpt.w),int(bxpt.h)};
        // std::cout << "UNHIT:" << bxpt.x << "," << bxpt.y << "," << bxpt.w << "," << bxpt.h << std::endl;
        ins.push_back(pt);
    }
#ifndef MLU220    
    // m_Trackors[tvimage.uuid_cam]->print();
#endif
    return RET_CODE::SUCCESS;
}


/*******************************************************************************
UNet2DShiftSegment 基于分割算法的高空抛物 继承BaseModel
chaffee.chen@2021
*******************************************************************************/
// RET_CODE UNet2DShiftSegment::init(const std::string &modelpath){
//     LOGI << "-> UNet2DShiftSegment::init";
//     bool pad_both_side = false;//单边预留
//     bool keep_aspect_ratio = false;//不保持长宽比
//     BASE_CONFIG config(MODEL_INPUT_FORMAT::BGRA, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
//     RET_CODE ret = BaseModel::base_init(modelpath, config);
//     if(ret!=RET_CODE::SUCCESS) return ret;
//     ret = init_trackor();
//     std::vector<TvaiRect> pAoi;
//     set_param(m_predict_threshold, 0, TvaiResolution{0,0}, TvaiResolution{0,0}, pAoi);
//     //Self param
//     return ret;
// }

// RET_CODE UNet2DShiftSegment::auto_model_file_search(std::map<InitParam, std::string> &modelpath){
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

// RET_CODE UNet2DShiftSegment::init(std::map<InitParam, std::string> &modelpath){
//     if(use_auto_model){
//         RET_CODE ret = auto_model_file_search(modelpath);
//         if(ret!=RET_CODE::SUCCESS) return ret;
//     }

//     if(modelpath.find(InitParam::BASE_MODEL) == modelpath.end()) return RET_CODE::ERR_INIT_PARAM_FAILED;
//     return init(modelpath[InitParam::BASE_MODEL]);
// }

// //clear self param
// UNet2DShiftSegment::~UNet2DShiftSegment(){LOGI << "-> UNet2DShiftSegment::~UNet2DShiftSegment";
//     m_Trackors.clear();
//     clear();
// }

// RET_CODE UNet2DShiftSegment::run_yuv_on_mlu(BatchImageIN &batch_tvimages, VecObjBBox &bboxes){
//     LOGI << "-> UNet2DShiftSegment::run_yuv_on_mlu";
//     RET_CODE ret = RET_CODE::FAILED;
//     std::vector<float> batch_aspect_ratio;

//     VecRect batch_roi; 
//     float* batch_model_output = nullptr;
//     {
//         std::lock_guard<std::mutex> lk(_mlu_mutex);
//         std::vector<float> _aspect_ratio;
//         ret = BaseModel::general_batch_preprocess_yuv_on_mlu(batch_tvimages, batch_roi, _aspect_ratio);
//         if(ret!=RET_CODE::SUCCESS) return ret;
//         batch_model_output = BaseModel::general_mlu_infer();
//     }
//     float aspect_ratio_x = (1.0*_W)/batch_tvimages[0].width;
//     float aspect_ratio_y = (1.0*_H)/batch_tvimages[0].height;
//     //ATT. UNet2DShiftSegment IN_BATCH_SIZE=3 OUT_BATCH_SIZE=1
//     int outputDim = _oW*_oH*_oC;
//     //TODO
//     VecObjBBox bboxes_before_filter;
//     //默认所有输入图像同样尺寸, 所以aspect_ratio只用第一个
//     // visual(batch_tvimages[2], batch_model_output);
//     postprocess(batch_model_output, bboxes_before_filter, aspect_ratio_x, aspect_ratio_y);
//     free(batch_model_output);
//     postfilter(bboxes_before_filter, bboxes);
//     if(!m_test_mod)//开启测试时, 关闭轨迹后处理
//         trackprocess(batch_tvimages[0] , bboxes);
//     return ret;
// }

// RET_CODE UNet2DShiftSegment::clear(){
//     for(auto &&tvimage : tvimage_buffers){
//         cnrtFree( (void*)(tvimage.u64PhyAddr[0]));
//         cnrtFree( (void*)(tvimage.u64PhyAddr[1]));
//     }
//     tvimage_buffers.clear();
// }


// RET_CODE UNet2DShiftSegment::push_back(TvaiImage &tvimage){
//     RET_CODE ret = RET_CODE::FAILED;
//     while(tvimage_buffers.size()>=m_batchsize){
//         cnrtFree( (void*)(tvimage_buffers[0].u64PhyAddr[0]));
//         cnrtFree( (void*)(tvimage_buffers[0].u64PhyAddr[1]));
//         tvimage_buffers.erase(tvimage_buffers.begin());
//     }
//     if(tvimage.format == TVAI_IMAGE_FORMAT_NV12 || tvimage.format == TVAI_IMAGE_FORMAT_NV21 ){
//         TvaiImage tmp = tvimage;
//         tmp.usePhyAddr = true;
//         void* u64_cp_1 = nullptr;
//         void* u64_cp_2 = nullptr;
//         cnrtMalloc( &u64_cp_1, tmp.stride* tmp.height);
//         cnrtMalloc( &u64_cp_2, tmp.stride* tmp.height/2);
//         tmp.u64PhyAddr[0] = reinterpret_cast<uint64_t>(u64_cp_1);
//         tmp.u64PhyAddr[1] = reinterpret_cast<uint64_t>(u64_cp_2);
//         if(tvimage.usePhyAddr){
//             cnrtMemcpy(u64_cp_1, (void*)(tvimage.u64PhyAddr[0]), tmp.height*tmp.stride, CNRT_MEM_TRANS_DIR_DEV2DEV);
//             cnrtMemcpy(u64_cp_2 , (void*)(tvimage.u64PhyAddr[1]), tmp.stride*tmp.height/2, CNRT_MEM_TRANS_DIR_DEV2DEV);
//         } else {
//             cnrtMemcpy(u64_cp_1, tvimage.pData, tmp.height*tmp.stride, CNRT_MEM_TRANS_DIR_HOST2DEV);
//             cnrtMemcpy(u64_cp_2, tvimage.pData+tvimage.height*tvimage.stride, tmp.stride*tmp.height/2, CNRT_MEM_TRANS_DIR_HOST2DEV);
//         }
//         tvimage_buffers.push_back(tmp);
//         ret = RET_CODE::SUCCESS;
//         // std::cout << "tvimage_buffers: " <<  tvimage_buffers.size() <<std::endl;
//     } else {
//         ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
//     }
//     return ret;
// }

// RET_CODE UNet2DShiftSegment::run(TvaiImage &tvimage, VecObjBBox &bboxes){
//     RET_CODE ret = RET_CODE::FAILED;
//     if(tvimage.format == TVAI_IMAGE_FORMAT_NV12 || tvimage.format == TVAI_IMAGE_FORMAT_NV21 ){
//         ret = push_back(tvimage);
//         if(ret!=RET_CODE::SUCCESS) return ret;
//         if(tvimage_buffers.size()==m_batchsize)
//             ret = run(tvimage_buffers, bboxes);
//     } else {
//         ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
//     }
//     return ret;
// }

// RET_CODE UNet2DShiftSegment::run(BatchImageIN &batch_tvimages, VecObjBBox &bboxes){
//     // std::cout <<"trigger" << std::endl;
//     RET_CODE ret = RET_CODE::FAILED;
//     if(batch_tvimages.empty()) return ret;//return FAILED if input empty
//     if(batch_tvimages[0].format == TVAI_IMAGE_FORMAT_NV21 || batch_tvimages[0].format == TVAI_IMAGE_FORMAT_NV12 ){
//         ret = run_yuv_on_mlu(batch_tvimages ,bboxes);//little trick, use nullptr to judge the condition
//     }
//     else
//         ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
//     return ret;
// }

// void UNet2DShiftSegment::visual(TvaiImage& tvimage, float* model_output){
//     Mat cv_predict(Size(_oW, _oH), CV_32FC1, model_output);
//     Mat cv_mask = 255 - cv_predict*255;
//     cv_mask.convertTo(cv_mask, CV_8U);
//     Mat cv_image(Size(tvimage.stride, 3*tvimage.height/2), CV_8UC1, tvimage.pData );
//     Mat cv_bgr;
//     cvtColor(cv_image, cv_bgr, COLOR_YUV2BGR_NV21);
//     resize(cv_bgr, cv_bgr, Size(_W,_H));
//     // Mat cv_bgras[4];
//     // split(cv_bgr, cv_bgras);
//     // cv_mask.copyTo(cv_bgras[3]);
//     // Mat cv_tmp;
//     // merge(cv_bgras, 4, cv_tmp);
//     saveImg("tmp/", ".jpg", cv_mask);

//     // try{
//     //     for(int r=0; r<=cv_bgr.rows; r++){
//     //         Vec4b* ptrR = cv_bgr.ptr<Vec4b>(r);
//     //         uchar* ptrP = cv_mask.ptr<uchar>(r);
//     //         for(int c=0; c<=cv_bgr.cols;c++){
//     //             ptrR[c][3] = ptrP[c];
//     //         }
//     //         saveImg("tmp/", ".jpg", cv_bgr);
//     //     }
//     // }
//     // catch(const Exception &e){
//     //     std::cout << "Err" << std::endl;
//     //     return;
//     // }
// }

// RET_CODE UNet2DShiftSegment::postprocess(float* model_output, VecObjBBox &bboxes, float aspect_ratio_x, float aspect_ratio_y){
//     int featDim = _oH*_oW*_oC;//1,224,224,1
//     Mat cv_predict(Size(_oW, _oH), CV_32FC1, model_output);
//     Mat cv_mask = cv_predict > m_predict_threshold;
//     // Mat cv_tmp = cv_predict*255;
//     // Mat cv_tmp2;
//     // cv_tmp.convertTo(cv_tmp2, CV_8U );
//     // saveImg("tmp/",".jpg", cv_tmp2);
//     // cv_mask *= 255;
//     Mat cv_kernel = Mat::ones(Size(3,3), CV_8UC1);
//     Mat cv_dilate_mask, cv_t_mask;
//     vector<vector<Point>> vec_cv_contours;
//     dilate(cv_mask, cv_dilate_mask , cv_kernel, Point(1,1), 3);
//     // float auto_thresh = cv::threshold(cv_dilate_mask, cv_t_mask, 0, 255, THRESH_BINARY+THRESH_OTSU);
//     // imwrite("tmp.jpg", cv_t_mask);
//     cv_dilate_mask = cv_dilate_mask > 50;
//     findContours(cv_dilate_mask,vec_cv_contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
//     // std::cout << vec_cv_contours.size() << std::endl;
//     for(auto iter=vec_cv_contours.begin(); iter!=vec_cv_contours.end(); iter++){
//         Rect rect = boundingRect(*iter);
//         // float ratio = (1.0* iter->size()) / (rect.width*rect.height);
//         // std::cout <<  "inner ratio = " << ratio << std::endl;
//         BBox bbox;
//         bbox.objtype = CLS_TYPE::FALLING_OBJ_UNCERTAIN;
//         bbox.confidence = 1.0;
//         bbox.objectness = bbox.confidence;
//         bbox.rect.x = ((1.0*rect.x) / aspect_ratio_x); bbox.rect.width = ((1.0*rect.width) / aspect_ratio_x);
//         bbox.rect.y = ((1.0*rect.y) / aspect_ratio_y); bbox.rect.height = ((1.0*rect.height) / aspect_ratio_y);
//         bboxes.push_back(bbox);
//     }
//     // free(binary_predict);
//     return RET_CODE::SUCCESS;
// }

// static bool check_rect_resolution(BBox _box, TvaiResolution minSz, TvaiResolution maxSz){
//     int width = _box.rect.width;
//     int height = _box.rect.height;
//     if (width < minSz.width || width > maxSz.width)
//         return false;
//     if (height < minSz.height || height > maxSz.height)
//         return false;
//     return true;
// }
// static bool check_in_valid_region(BBox _box, const std::vector<TvaiRect> &pAoiRegion){
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
// RET_CODE UNet2DShiftSegment::postfilter(VecObjBBox &ins, VecObjBBox &outs){
//     //TODO: 需要增加区域限定
//     LOGI << "bboxes before filter = " << ins.size();
//     for(auto iter=ins.begin(); iter!=ins.end(); iter++){
//         if (!check_rect_resolution(*iter, m_minTargetSize, m_maxTargetSize)){
//             // std::cout<< "resolution failed" << std::endl;
//             continue;
//         }
//         if (!check_in_valid_region(*iter, m_pAoiRect)){
//             // std::cout<< "region failed" << std::endl;
//             continue;
//         }
//         if(iter->confidence > m_predict_threshold ){
//             outs.push_back(*iter);
//         }
//     }
//     LOGI << "bboxes after filter = " << outs.size();
//     return RET_CODE::SUCCESS;
// }

// RET_CODE UNet2DShiftSegment::get_class_type(std::vector<CLS_TYPE> &valid_clss){
//     valid_clss = _cls_;
//     return RET_CODE::SUCCESS;
// };

// RET_CODE UNet2DShiftSegment::set_param(float threshold, float nms_threshold, TvaiResolution maxTargetSize, TvaiResolution minTargetSize, 
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

// RET_CODE UNet2DShiftSegment::create_trackor(int uuid_cam){
//     if(m_Trackors.find(uuid_cam)==m_Trackors.end()){
//         LOGI << "-> UNet2DShiftSegment::create_trackor+";
//         std::shared_ptr<BoxTraceSet> m_trackor_t(new BoxTraceSet());
//         m_Trackors.insert(std::pair<int,std::shared_ptr<BoxTraceSet>>(uuid_cam,m_trackor_t));
//     } else LOGI << "-> UNet2DShiftSegment::create_trackor";
//     return RET_CODE::SUCCESS;
// }

// RET_CODE UNet2DShiftSegment::init_trackor(){
//     create_trackor(-1);
//     return RET_CODE::SUCCESS;
// }

// RET_CODE UNet2DShiftSegment::trackprocess(TvaiImage &tvimage, VecObjBBox &ins){
//     create_trackor(tvimage.uuid_cam);
//     std::vector<BoxPoint> bpts;
//     int cur_time = 1 + m_Trackors[tvimage.uuid_cam]->m_time;
//     int box_cnt = 0;
//     for(auto in: ins){
//         BoxPoint tmp = BoxPoint(in, cur_time);
//         bpts.push_back( tmp );
//         box_cnt++;
//         if(box_cnt >= m_max_executable_bboxes ) break;
//         // std::cout << "push_back: " << in.rect.x  << "," << in.rect.y <<"," << in.rect.width <<"," << in.rect.height << std::endl;
//     }
//     m_Trackors[tvimage.uuid_cam]->push_back( bpts );
//     std::vector<BoxPoint> marked_pts, unmarked_pts;
//     m_Trackors[tvimage.uuid_cam]->output_last_point_of_trace(marked_pts, unmarked_pts);
//     ins.clear();
//     for(auto bxpt: marked_pts){
//         BBox pt;
//         pt.confidence = 1.0;
//         pt.objectness = 1.0;
//         pt.quality = 1.0;
//         pt.objtype = CLS_TYPE::FALLING_OBJ;
//         pt.rect = TvaiRect{ int(bxpt.x),int(bxpt.y),int(bxpt.w),int(bxpt.h)};
//         // std::cout << "HIT:" << bxpt.x << "," << bxpt.y << "," << bxpt.w << "," << bxpt.h << std::endl;
//         ins.push_back(pt);
//     }
//     for(auto bxpt: unmarked_pts){
//         BBox pt;
//         pt.confidence = 1.0;
//         pt.objectness = 1.0;
//         pt.quality = 1.0;
//         pt.objtype = CLS_TYPE::FALLING_OBJ_UNCERTAIN;
//         pt.rect = TvaiRect{ int(bxpt.x),int(bxpt.y),int(bxpt.w),int(bxpt.h)};
//         // std::cout << "UNHIT:" << bxpt.x << "," << bxpt.y << "," << bxpt.w << "," << bxpt.h << std::endl;
//         ins.push_back(pt);
//     }
// #ifndef MLU220    
//     // m_Trackors[tvimage.uuid_cam]->print();
// #endif
//     return RET_CODE::SUCCESS;
// }
