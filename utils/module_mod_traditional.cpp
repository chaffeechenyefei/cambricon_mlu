#include "module_mod_traditional.hpp"
#include "glog/logging.h"
#include <opencv2/video/background_segm.hpp>
// #include <opencv2/bgsegm.hpp>
#include <math.h>
#include "basic.hpp"
#include "../inner_utils/inner_basic.hpp"
#include <fstream>

#include <exception> 

#ifdef VERBOSE
#define LOGI LOG(INFO)
#else
#define LOGI 0 && LOG(INFO)
#endif

using namespace ucloud;
using namespace cv;

using std::vector;

static float joint_bboxes_areas(BBox &a, BBox &b);



float joint_bboxes_areas(BBox &a, BBox &b){
    float ax0 = a.rect.x;
    float ay0 = a.rect.y;
    float ax1 = a.rect.x + a.rect.width;
    float ay1 = a.rect.y + a.rect.height;
    float bx0 = b.rect.x;
    float by0 = b.rect.y;
    float bx1 = b.rect.x + b.rect.width;
    float by1 = b.rect.y + b.rect.height;

    float area0 = (ay1 - ay0 + 1e-3)*(ax1 - ax0 + 1e-3);
    float roiWidth = std::min(ax1, bx1) - std::max(ax0, bx0);
    float roiHeight = std::min(ay1, by1) - std::max(ay0, by0);
    if (roiWidth < 0 || roiHeight < 0) return 0;
    float area1 = (by1 - by0 + 1e-3)*(bx1 - bx0 + 1e-3);
    return roiWidth*roiHeight/std::min(area0, area1);
}



/*******************************************************************************
BackgroundSegmentV4
chaffee.chen@2022-10-21
*******************************************************************************/
RET_CODE BackgroundSegmentV4::init(){
    LOGI << "-> BackgroundSegmentV4::init";
    bool pad_both_side = false;//单边预留
    bool keep_aspect_ratio = true;//保持长宽比
    BASE_CONFIG config(MODEL_INPUT_FORMAT::BGRA, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
    RET_CODE ret = m_net->base_init(m_dstH, m_dstW, config);
    if(ret!=RET_CODE::SUCCESS) return ret;
    // ret = init_trackor();
    // ret = init_model();
    //Self param
    LOGI << "<- BackgroundSegmentV4::init";
    return ret;
}

RET_CODE BackgroundSegmentV4::create_model(int uuid_cam){
    if(m_Models.find(uuid_cam)==m_Models.end()){
        LOGI << "-> BackgroundSegmentV4::create_model+";
#ifdef OPENCV3 //opencv3.4.6
        //一定要用cv::Ptr<BackgroundSubtractor>, 不能用BackgroundSubtractor*, 否则会自动释放出现coredump
        cv::Ptr<BackgroundSubtractor> ptM = cv::createBackgroundSubtractorMOG2(50,32,true);
        m_Models.insert(std::pair<int,cv::Ptr<BackgroundSubtractor>>(uuid_cam,ptM));
#else
        BackgroundSubtractorMOG2 *ptM = new BackgroundSubtractorMOG2(100,16,true);
        std::share_ptr<BackgroundSubtractor> model_t(ptM);
        m_Models.insert(std::pair<int,std::shared_ptr<BackgroundSubtractor>>(uuid_cam,model_t));
#endif
        LOGI << "<- BackgroundSegment4::create_model+";
    } else LOGI << "-> BackgroundSegment4::create_model trival";
    return RET_CODE::SUCCESS;
}

RET_CODE BackgroundSegmentV4::run(TvaiImage& tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    LOGI << "-> BackgroundSegmentV4::run";
    RET_CODE ret = RET_CODE::FAILED;
    switch (tvimage.format)
    {
    case TVAI_IMAGE_FORMAT_RGB:
    case TVAI_IMAGE_FORMAT_BGR:
        printf("**[%s][%d] unsupported image format\n", __FILE__, __LINE__);
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
        break;
    case TVAI_IMAGE_FORMAT_NV21:
    case TVAI_IMAGE_FORMAT_NV12:
        ret = run_yuv_on_mlu(tvimage,bboxes, threshold, nms_threshold);
        break;
    default:
        printf("**[%s][%d] unsupported image format\n", __FILE__, __LINE__);
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
        break;
    }
    if(ret!=RET_CODE::SUCCESS) return ret;
    LOGI << "<- BackgroundSegmentV4::run";
    return ret;
}

RET_CODE BackgroundSegmentV4::run_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    LOGI << "-> BackgroundSegmentV4::run_yuv_on_mlu";
    RET_CODE ret = RET_CODE::FAILED;
    TvaiRect roiRect{0,0,tvimage.width, tvimage.height};
    float aspect_ratio = 1.0;
    float aX, aY;
    Mat cropped_img;
    {
        if(tvimage.usePhyAddr)
            ret = m_net->general_preprocess_yuv_on_mlu_phyAddr(tvimage, roiRect, cropped_img, aspect_ratio, aX , aY);
        else
            ret = m_net->general_preprocess_yuv_on_mlu(tvimage, roiRect, cropped_img, aspect_ratio, aX , aY);
        if(ret!=RET_CODE::SUCCESS) return ret;
    }
    
    //TODO post process
    VecObjBBox bboxes_before_filter;
    ret = postprocess(cropped_img, bboxes_before_filter, aspect_ratio, tvimage.uuid_cam);
    postfilter(tvimage, bboxes_before_filter, bboxes, threshold);
    // trackprocess(tvimage , bboxes);
    LOGI << "<- BackgroundSegmentV4::run_yuv_on_mlu";
    return ret;
}

RET_CODE BackgroundSegmentV4::postprocess(cv::Mat &cropped_img, VecObjBBox &bboxes, float aspect_ratio, int uuid_cam){
    LOGI << "-> BackgroundSegmentV4::postprocess";
    create_model(uuid_cam);
    Mat fgmask;//0:bg 255:fg
    float lr = 0.3;
    Mat gray_img, bal_img;
    cvtColor(cropped_img, gray_img, CV_BGRA2GRAY);
    blur(gray_img, gray_img, Size(3,3));
    gray_img.convertTo(bal_img, CV_32FC1);
    cv::pow(bal_img/255,0.7,bal_img);
    bal_img = bal_img*255;
    bal_img.convertTo(gray_img,CV_8UC1);
    // imwrite("tmp.jpg", gray_img);
    // equalizeHist(gray_img, gray_img);
#ifdef OPENCV3
    // LOGI << "-> BackgroundSegment::apply";
    m_Models[uuid_cam]->apply(gray_img, fgmask, lr);
    // LOGI << "<- BackgroundSegment::apply";
#else
    (*(m_Models[uuid_cam]))(gray_img, fgmask, lr);
#endif
    erode(fgmask, fgmask, Mat::ones(3,3,CV_8UC1));
    dilate(fgmask, fgmask, Mat::ones(5,5,CV_8UC1));
    
    vector<vector<Point>> vec_cv_contours;
    findContours(fgmask,vec_cv_contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    // printf("vec_cv_contours %d\n", vec_cv_contours.size());
    for(auto iter=vec_cv_contours.begin(); iter!=vec_cv_contours.end(); iter++){
        Rect rect = boundingRect(*iter);
        BBox bbox;
        bbox.objtype = CLS_TYPE::FALLING_OBJ_UNCERTAIN;
        bbox.confidence = 1.0;
        bbox.objectness = bbox.confidence;
        bbox.rect.x = ((1.0*rect.x) / aspect_ratio); bbox.rect.width = ((1.0*rect.width) / aspect_ratio);
        bbox.rect.y = ((1.0*rect.y) / aspect_ratio); bbox.rect.height = ((1.0*rect.height) / aspect_ratio);
        bboxes.push_back(bbox);
    }
    LOGI << "<- BackgroundSegmentV4::postprocess";
    return RET_CODE::SUCCESS;
}


RET_CODE BackgroundSegmentV4::postfilter(TvaiImage &tvimage, VecObjBBox &ins, VecObjBBox &outs, float threshold){
    //TODO: 需要增加区域限定
    LOGI << "bboxes before filter = " << ins.size();
    for(auto &&box: ins){
        float box_size = (float(box.rect.width*box.rect.height))/(tvimage.width*tvimage.height);
        if(box_size > 0.5) continue;
        if(box.confidence > threshold ) outs.push_back(box);
    }
    LOGI << "bboxes after filter = " << outs.size();
    return RET_CODE::SUCCESS;
}

RET_CODE BackgroundSegmentV4::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    valid_clss = _cls_;
    return RET_CODE::SUCCESS;
};



/*******************************************************************************
BackgroundSegment
chaffee.chen@2021-xx-xx
*******************************************************************************/
RET_CODE BackgroundSegment::init(const std::string &modelpath){
    LOGI << "-> BackgroundSegment::init";
    bool pad_both_side = false;//单边预留
    bool keep_aspect_ratio = true;//保持长宽比
    BASE_CONFIG config(MODEL_INPUT_FORMAT::BGRA, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
    RET_CODE ret = YuvCropResizeModel::base_init(m_dstH, m_dstW, config);
    if(ret!=RET_CODE::SUCCESS) return ret;
    ret = init_trackor();
    ret = init_model();
    //Self param
    return ret;
}

RET_CODE BackgroundSegment::init(){
    return BackgroundSegment::init("");
}

RET_CODE BackgroundSegment::init(std::map<InitParam, std::string> &modelpath){
    return BackgroundSegment::init("");
}

RET_CODE BackgroundSegment::init_model(){
    return create_model(-1);
}

RET_CODE BackgroundSegment::create_model(int uuid_cam){
    if(m_Models.find(uuid_cam)==m_Models.end()){
        LOGI << "-> BackgroundSegment::create_model+";
#ifdef OPENCV3 //opencv3.4.6
        //一定要用cv::Ptr<BackgroundSubtractor>, 不能用BackgroundSubtractor*, 否则会自动释放出现coredump
        cv::Ptr<BackgroundSubtractor> ptM = cv::createBackgroundSubtractorMOG2(50,32,true);
        // cv::Ptr<BackgroundSubtractor> ptM = cv::bgsegm::createBackgroundSubtractorGMG(50,0.95);
        // cv::Ptr<BackgroundSubtractor> ptM = cv::bgsegm::createBackgroundSubtractorGSOC(cv::bgsegm::LSBPCameraMotionCompensation::LSBP_CAMERA_MOTION_COMPENSATION_LK);
        // cv::Ptr<BackgroundSubtractor> ptM = cv::bgsegm::createBackgroundSubtractorLSBP(0,50,16);
        m_Models.insert(std::pair<int,cv::Ptr<BackgroundSubtractor>>(uuid_cam,ptM));
#else
        BackgroundSubtractorMOG2 *ptM = new BackgroundSubtractorMOG2(100,16,true);
        std::share_ptr<BackgroundSubtractor> model_t(ptM);
        m_Models.insert(std::pair<int,std::shared_ptr<BackgroundSubtractor>>(uuid_cam,model_t));
#endif
    } else LOGI << "-> BackgroundSegment::create_model trival";
    return RET_CODE::SUCCESS;
}

RET_CODE BackgroundSegment::init_trackor(){
    create_trackor(-1);
    return RET_CODE::SUCCESS;
}

RET_CODE BackgroundSegment::create_trackor(int uuid_cam){
    if(m_Trackors.find(uuid_cam)==m_Trackors.end()){
        LOGI << "-> BackgroundSegment::create_trackor+";
        std::shared_ptr<BoxTraceSet> m_trackor_t(new BoxTraceSet());
        m_Trackors.insert(std::pair<int,std::shared_ptr<BoxTraceSet>>(uuid_cam,m_trackor_t));
    } else LOGI << "-> BackgroundSegment::create_trackor";
    return RET_CODE::SUCCESS;
}

BackgroundSegment::~BackgroundSegment(){LOGI << "-> BackgroundSegment::~BackgroundSegment";
    m_Trackors.clear();
    m_Models.clear();
}

static bool check_rect_resolution(BBox _box, TvaiResolution minSz, TvaiResolution maxSz){
    int width = _box.rect.width;
    int height = _box.rect.height;
    if (width < minSz.width || width > maxSz.width)
        return false;
    if (height < minSz.height || height > maxSz.height)
        return false;
    return true;
}
static bool check_in_valid_region(BBox _box, const std::vector<TvaiRect> &pAoiRegion){
    int x_center = _box.rect.x + _box.rect.width/2;
    int y_center = _box.rect.y + _box.rect.height/2;
    if(pAoiRegion.size()==0){
        return true;
    }
    for (int i = 0; i < pAoiRegion.size(); i++ ){
        //  std::cout << pAoiRegion[i].x << "," << pAoiRegion[i].y << "," << std::endl;
        if (x_center>=pAoiRegion[i].x && x_center<=pAoiRegion[i].x+pAoiRegion[i].width &&\
            y_center>=pAoiRegion[i].y && y_center<=pAoiRegion[i].y+pAoiRegion[i].height
        )
            return true;
    }
    return false;
}
RET_CODE BackgroundSegment::postfilter(TvaiImage &tvimage, VecObjBBox &ins, VecObjBBox &outs, float threshold){
    //TODO: 需要增加区域限定
    LOGI << "bboxes before filter = " << ins.size();
    for(auto iter=ins.begin(); iter!=ins.end(); iter++){
        float box_size = (float(iter->rect.width*iter->rect.height)) / (tvimage.width*tvimage.height);
        if(box_size > 0.5) continue;
        if(iter->confidence > threshold ){
            outs.push_back(*iter);
        }
    }
    LOGI << "bboxes after filter = " << outs.size();
    return RET_CODE::SUCCESS;
}

RET_CODE BackgroundSegment::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    valid_clss = _cls_;
    return RET_CODE::SUCCESS;
};


RET_CODE BackgroundSegment::run(BatchImageIN &batch_tvimages, VecObjBBox &bboxes){
    if(batch_tvimages.empty()) return RET_CODE::SUCCESS;
    RET_CODE ret = run(batch_tvimages[0], bboxes);
    return ret;
}

RET_CODE BackgroundSegment::run(TvaiImage& tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    LOGI << "-> BackgroundSegment::run";
    RET_CODE ret = RET_CODE::FAILED;
    if(tvimage.format == TVAI_IMAGE_FORMAT_RGB || tvimage.format == TVAI_IMAGE_FORMAT_BGR){
        // ret = run_bgr_on_cpu(tvimage, bboxes);
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    }
    else if(tvimage.format == TVAI_IMAGE_FORMAT_NV21 || tvimage.format == TVAI_IMAGE_FORMAT_NV12 ){
        ret = run_yuv_on_mlu(tvimage,bboxes, threshold, nms_threshold);
    }
    else
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
    LOGI << "<- BackgroundSegment::run";
    return ret;
}

RET_CODE BackgroundSegment::run_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    LOGI << "-> BackgroundSegment::run_yuv_on_mlu";
    RET_CODE ret = RET_CODE::FAILED;
    TvaiRect roiRect{0,0,tvimage.width, tvimage.height};
    float aspect_ratio = 1.0;
    float aX, aY;
    Mat cropped_img;
    {
        std::lock_guard<std::mutex> lk(_mlu_mutex);
        if(tvimage.usePhyAddr)
            ret = YuvCropResizeModel::general_preprocess_yuv_on_mlu_phyAddr(tvimage, roiRect, cropped_img, aspect_ratio, aX , aY);
        else
            ret = YuvCropResizeModel::general_preprocess_yuv_on_mlu(tvimage, roiRect, cropped_img, aspect_ratio, aX , aY);
        if(ret!=RET_CODE::SUCCESS) return ret;
    }
    
    //TODO post process
    VecObjBBox bboxes_before_filter;
    ret = postprocess(cropped_img, bboxes_before_filter, aspect_ratio, tvimage.uuid_cam);
    postfilter(tvimage, bboxes_before_filter, bboxes, threshold);
    trackprocess(tvimage , bboxes);
    return ret;
}

RET_CODE BackgroundSegment::postprocess(cv::Mat &cropped_img, VecObjBBox &bboxes, float aspect_ratio, int uuid_cam){
    LOGI << "-> BackgroundSegment::postprocess";
    create_model(uuid_cam);
    Mat fgmask;//0:bg 255:fg
    float lr = 0.3;
    Mat gray_img, bal_img;
    cvtColor(cropped_img, gray_img, CV_BGRA2GRAY);
    blur(gray_img, gray_img, Size(3,3));
    gray_img.convertTo(bal_img, CV_32FC1);
    cv::pow(bal_img/255,0.7,bal_img);
    bal_img = bal_img*255;
    bal_img.convertTo(gray_img,CV_8UC1);
    // imwrite("tmp.jpg", gray_img);
    // equalizeHist(gray_img, gray_img);
#ifdef OPENCV3
    // LOGI << "-> BackgroundSegment::apply";
    m_Models[uuid_cam]->apply(gray_img, fgmask, lr);
    // LOGI << "<- BackgroundSegment::apply";
#else
    (*(m_Models[uuid_cam]))(gray_img, fgmask, lr);
#endif
    erode(fgmask, fgmask, Mat::ones(3,3,CV_8UC1));
    dilate(fgmask, fgmask, Mat::ones(5,5,CV_8UC1));
    
    vector<vector<Point>> vec_cv_contours;
    findContours(fgmask,vec_cv_contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    // printf("vec_cv_contours %d\n", vec_cv_contours.size());
    for(auto iter=vec_cv_contours.begin(); iter!=vec_cv_contours.end(); iter++){
        Rect rect = boundingRect(*iter);
        BBox bbox;
        bbox.objtype = CLS_TYPE::FALLING_OBJ_UNCERTAIN;
        bbox.confidence = 1.0;
        bbox.objectness = bbox.confidence;
        bbox.rect.x = ((1.0*rect.x) / aspect_ratio); bbox.rect.width = ((1.0*rect.width) / aspect_ratio);
        bbox.rect.y = ((1.0*rect.y) / aspect_ratio); bbox.rect.height = ((1.0*rect.height) / aspect_ratio);
        bboxes.push_back(bbox);
    }
    LOGI << "<- BackgroundSegment::postprocess";
    return RET_CODE::SUCCESS;
}


RET_CODE BackgroundSegment::trackprocess(TvaiImage &tvimage, VecObjBBox &ins){
    int min_box_num = 4;//认定为轨迹至少需要几个box
    create_trackor(tvimage.uuid_cam);
    std::vector<BoxPoint> bpts;
    int cur_time = 1 + m_Trackors[tvimage.uuid_cam]->m_time;
    for(auto in: ins){
        BoxPoint tmp = BoxPoint(in, cur_time);
        bpts.push_back( tmp );
    }
    m_Trackors[tvimage.uuid_cam]->push_back( bpts );
    std::vector<BoxPoint> marked_pts, unmarked_pts;
    // m_Trackors[tvimage.uuid_cam]->output_last_point_of_trace(marked_pts, unmarked_pts, min_box_num);
    m_Trackors[tvimage.uuid_cam]->output_trace(marked_pts, unmarked_pts, min_box_num);
    ins.clear();
    for(auto bxpt: marked_pts){
        BBox pt;
        pt.confidence = 1.0;
        pt.objectness = 1.0;
        pt.objtype = CLS_TYPE::FALLING_OBJ;
        pt.rect = TvaiRect{ int(bxpt.x),int(bxpt.y),int(bxpt.w),int(bxpt.h)};
        pt.track_id = bxpt.m_trace_id;
        ins.push_back(pt);
    }
    for(auto bxpt: unmarked_pts){
        BBox pt;
        pt.confidence = 1.0;
        pt.objectness = 1.0;
        pt.objtype = CLS_TYPE::FALLING_OBJ_UNCERTAIN;
        pt.rect = TvaiRect{ int(bxpt.x),int(bxpt.y),int(bxpt.w),int(bxpt.h)};
        ins.push_back(pt);
    }
#ifndef MLU220
    // m_Trackors[tvimage.uuid_cam]->print();
#endif
    return RET_CODE::SUCCESS;
}



/*******************************************************************************
IMP_OBJECT_REMAIN
chaffee.chen@2022-11-08
*******************************************************************************/

RET_CODE IMP_OBJECT_REMAIN::init(std::map<InitParam, std::string> &modelpath){
    LOGI << "-> IMP_OBJECT_REMAIN::init";
    if(modelpath.find(InitParam::BASE_MODEL)==modelpath.end()){
        printf("**[%s][%d] base model is missing in IMP_OBJECT_REMAIN\n", __FILE__, __LINE__);
        return RET_CODE::FAILED;
    }
    RET_CODE ret = m_ped_net->init(modelpath);
    if(ret!=RET_CODE::SUCCESS){
        printf("**[%s][%d] m_ped_net init return %d in IMP_OBJECT_REMAIN\n", __FILE__, __LINE__, ret);
        return ret;
    }
    bool pad_both_side = false;//单边预留
    bool keep_aspect_ratio = true;//保持长宽比
    BASE_CONFIG config(MODEL_INPUT_FORMAT::BGRA, MODEL_OUTPUT_ORDER::NHWC, pad_both_side, keep_aspect_ratio);
    ret = m_net->base_init(m_dstH, m_dstW, config);
    if(ret!=RET_CODE::SUCCESS) return ret;
    backgroud_data = cv::Mat::zeros(m_dstH, m_dstW, CV_32FC1);
    //Self param
    LOGI << "<- IMP_OBJECT_REMAIN::init";
    return ret;
}

RET_CODE IMP_OBJECT_REMAIN::create_model(int uuid_cam){
    if(m_Models.find(uuid_cam)==m_Models.end()){
        LOGI << "-> IMP_OBJECT_REMAIN::create_model+";
#ifdef OPENCV3 //opencv3.4.6
        //一定要用cv::Ptr<BackgroundSubtractor>, 不能用BackgroundSubtractor*, 否则会自动释放出现coredump
        cv::Ptr<BackgroundSubtractor> ptM = cv::createBackgroundSubtractorMOG2(50,32,true);
        m_Models.insert(std::pair<int,cv::Ptr<BackgroundSubtractor>>(uuid_cam,ptM));
#else
        BackgroundSubtractorMOG2 *ptM = new BackgroundSubtractorMOG2(100,16,true);
        std::share_ptr<BackgroundSubtractor> model_t(ptM);
        m_Models.insert(std::pair<int,std::shared_ptr<BackgroundSubtractor>>(uuid_cam,model_t));
#endif
        LOGI << "<- IMP_OBJECT_REMAIN::create_model+";
    } else LOGI << "-> IMP_OBJECT_REMAIN::create_model trival";
    return RET_CODE::SUCCESS;
}

RET_CODE IMP_OBJECT_REMAIN::run(TvaiImage& tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    LOGI << "-> IMP_OBJECT_REMAIN::run";
    RET_CODE ret = RET_CODE::FAILED;
    switch (tvimage.format)
    {
    case TVAI_IMAGE_FORMAT_RGB:
    case TVAI_IMAGE_FORMAT_BGR:
        printf("**[%s][%d] unsupported image format\n", __FILE__, __LINE__);
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
        break;
    case TVAI_IMAGE_FORMAT_NV21:
    case TVAI_IMAGE_FORMAT_NV12:
        ret = run_yuv_on_mlu(tvimage,bboxes, threshold, nms_threshold);
        break;
    default:
        printf("**[%s][%d] unsupported image format\n", __FILE__, __LINE__);
        ret = RET_CODE::ERR_UNSUPPORTED_IMG_FORMAT;
        break;
    }
    if(ret!=RET_CODE::SUCCESS) return ret;
    LOGI << "<- IMP_OBJECT_REMAIN::run";
    return ret;
}

RET_CODE IMP_OBJECT_REMAIN::run_yuv_on_mlu(TvaiImage &tvimage, VecObjBBox &bboxes, float threshold, float nms_threshold){
    LOGI << "-> IMP_OBJECT_REMAIN::run_yuv_on_mlu";
    RET_CODE ret = RET_CODE::FAILED;
    TvaiRect roiRect{0,0,tvimage.width, tvimage.height};
    float aspect_ratio = 1.0;
    float aX, aY;
    Mat cropped_img;
    {
        if(tvimage.usePhyAddr)
            ret = m_net->general_preprocess_yuv_on_mlu_phyAddr(tvimage, roiRect, cropped_img, aspect_ratio, aX , aY);
        else
            ret = m_net->general_preprocess_yuv_on_mlu(tvimage, roiRect, cropped_img, aspect_ratio, aX , aY);
        if(ret!=RET_CODE::SUCCESS) return ret;
    }
    VecObjBBox bboxes_obj, bboxes_ped;
    ret = m_ped_net->run(tvimage, bboxes_ped, m_ped_threshold, 0.6);
    if(ret!=RET_CODE::SUCCESS){
        printf("**[%s][%d] m_ped_net->run return %d\n", __FILE__, __LINE__, ret);
        return ret;
    }
    //TODO post process
    ret = postprocess(cropped_img, bboxes_ped, bboxes_obj, aspect_ratio, tvimage.uuid_cam);
    postfilter(tvimage, bboxes_obj, bboxes, threshold);
    // trackprocess(tvimage , bboxes);
    LOGI << "<- IMP_OBJECT_REMAIN::run_yuv_on_mlu";
    return ret;
}

RET_CODE IMP_OBJECT_REMAIN::postprocess(cv::Mat &cropped_img, VecObjBBox &bboxes_ped, VecObjBBox &bboxes_obj, float aspect_ratio, int uuid_cam){
    LOGI << "-> IMP_OBJECT_REMAIN::postprocess";
    create_model(uuid_cam);
    Mat fgmask;//0:bg 255:fg
    float lr = m_background_init ? 0:0.05;//没有初始化完毕, 使用0.05
    Mat gray_img, bal_img;
    cvtColor(cropped_img, gray_img, CV_BGRA2GRAY);
    blur(gray_img, gray_img, Size(3,3));
    gray_img.convertTo(bal_img, CV_32FC1);
    cv::pow(bal_img/255,0.7,bal_img);
    bal_img = bal_img*255;
    bal_img.convertTo(gray_img,CV_8UC1);
    // imwrite("tmp.jpg", gray_img);
    // equalizeHist(gray_img, gray_img);
#ifdef OPENCV3
    // LOGI << "-> BackgroundSegment::apply";
    m_Models[uuid_cam]->apply(gray_img, fgmask, lr);
    // LOGI << "<- BackgroundSegment::apply";
#else
    (*(m_Models[uuid_cam]))(gray_img, fgmask, lr);
#endif
    erode(fgmask, fgmask, Mat::ones(3,3,CV_8UC1));
    dilate(fgmask, fgmask, Mat::ones(5,5,CV_8UC1));
    
    vector<vector<Point>> vec_cv_contours;
    findContours(fgmask,vec_cv_contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    if(!m_background_init){
        backgroud_data = m_background_rate*backgroud_data + (1-m_background_rate)*bal_img;
        m_background_history++;
        if(m_background_history > m_background_history_max){
            m_background_init = true;
        }
    } else if(vec_cv_contours.empty() && bboxes_ped.empty()){
        backgroud_data = m_background_rate*backgroud_data + (1-m_background_rate)*bal_img;
    } else {
        // backgroud_data.convertTo(gray_img,CV_8UC1);
        // lr = 0.01;
    // #ifdef OPENCV3
    //     m_Models[uuid_cam]->apply(gray_img, fgmask, lr);
    // #else
    //     (*(m_Models[uuid_cam]))(gray_img, fgmask, lr);
    // #endif        
    }

    // printf("vec_cv_contours %d\n", vec_cv_contours.size());
    for(auto iter=vec_cv_contours.begin(); iter!=vec_cv_contours.end(); iter++){
        Rect rect = boundingRect(*iter);
        BBox bbox;
        bbox.objtype = CLS_TYPE::TARGET;
        bbox.confidence = 1.0;
        bbox.objectness = bbox.confidence;
        bbox.rect.x = ((1.0*rect.x) / aspect_ratio); bbox.rect.width = ((1.0*rect.width) / aspect_ratio);
        bbox.rect.y = ((1.0*rect.y) / aspect_ratio); bbox.rect.height = ((1.0*rect.height) / aspect_ratio);
        if(bboxes_ped.empty()){
            bboxes_obj.push_back(bbox);
        } else {
            for(auto &&ped_box: bboxes_ped){
                float joint_area = joint_bboxes_areas(bbox, ped_box);
                if(joint_area >= 0.3) continue;
                else bboxes_obj.push_back(bbox);
            }
        }
    }
    LOGI << "<- IMP_OBJECT_REMAIN::postprocess";
    return RET_CODE::SUCCESS;
}


RET_CODE IMP_OBJECT_REMAIN::postfilter(TvaiImage &tvimage, VecObjBBox &ins, VecObjBBox &outs, float threshold){
    //TODO: 需要增加区域限定
    LOGI << "bboxes before filter = " << ins.size();
    for(auto &&box: ins){
        float box_size = (float(box.rect.width*box.rect.height))/(tvimage.width*tvimage.height);
        if(box_size > 0.7) continue;
        if(box.confidence > threshold ) outs.push_back(box);
    }
    LOGI << "bboxes after filter = " << outs.size();
    return RET_CODE::SUCCESS;
}

RET_CODE IMP_OBJECT_REMAIN::get_class_type(std::vector<CLS_TYPE> &valid_clss){
    valid_clss = _cls_;
    return RET_CODE::SUCCESS;
};