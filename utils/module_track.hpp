#ifndef _MODULE_TRACK_HPP_
#define _MODULE_TRACK_HPP_
#include "module_base.hpp"
#include "trackor/bytetrack_no_reid/BYTETracker_no_reid.h"
#include "trackor/bytetrack_origin/BYTETracker_origin.h"
#include <easytrack/easy_track.h>
#include <map>
#include <mutex>

/**
 * Single Class Trackor
 */
using ucloud::BBox;
using ucloud::VecObjBBox;
using ucloud::RET_CODE;
using ucloud::TvaiImage;

enum class TRACKMETHOD{
    BYTETRACK_ORIGIN           = 0,
    BYTETRACK_NO_REID,
    DEEPSORT,
};

using cam_class_uuid = std::string;
cam_class_uuid get_cam_class_uuid(int cam_uuid, ucloud::CLS_TYPE clsType);

/**
 * 跟踪器通用接口
 */
template<class PARAM>
class TrackPoolAPI{
public:
    TrackPoolAPI(){}
    virtual ~TrackPoolAPI(){}
    virtual RET_CODE init(std::string &modelpath){return RET_CODE::ERR_VIRTUAL_FUNCTION;}
    virtual void update(TvaiImage &tvimage, VecObjBBox &bboxIN, PARAM& params ){}
    virtual void set_fps(int fps){}
    virtual void clear(){}
protected:
    std::mutex m_mutex;//Lock should be used in those apis only
};

/******************************************************************************
 * DEEPSORT
 *******************************************************************************/
typedef struct _DEEPSORTPARM{
    float track_threshold;
}DEEPSORTPARM;
typedef std::shared_ptr<edk::EasyTrack> DeepSort_Ptr;
class DeepSortPool: public TrackPoolAPI<DEEPSORTPARM>{
public:
    DeepSortPool(){}
    ~DeepSortPool(){}
    DeepSortPool(int fps, int nn_budget, float max_cosine_dist):\
        m_fps(fps),\
        m_nn_budget(nn_budget),\
        m_max_cosine_dist(max_cosine_dist){}

    void set_fps(int fps){m_fps=fps;}
    void update(TvaiImage &tvimage, VecObjBBox &bboxIN, DEEPSORTPARM &params);

protected:
    void add_trackor(cam_class_uuid uuid);
private:
    std::map<cam_class_uuid, DeepSort_Ptr> m_trackors;
    int m_n_init = 2;
    int m_fps = 25;//+-
    int m_nn_budget = 25;//+-
    float m_max_cosine_dist = 0.5;//+-
    float m_max_iou_dist = 0.5;
    
    // std::mutex m_mutex;
};

/******************************************************************************
 * BYTETRACK
 *******************************************************************************/
typedef struct _BYTETRACKPARM{
    float track_threshold;
    float high_detect_threshold;
}BYTETRACKPARM;


/**
 * ByteTrack_No_ReID
 */
typedef std::shared_ptr<bytetrack_no_reid::BYTETracker> ByteTrackNoReID_Ptr;
class ByteTrackNoReIDPool: public TrackPoolAPI<BYTETRACKPARM>{
public:
    ByteTrackNoReIDPool(){}
    ~ByteTrackNoReIDPool(){}
    ByteTrackNoReIDPool(int fps, int nn_buf):\
        m_fps(fps),\
        m_nn_buf(nn_buf) {}

    void set_fps(int fps){m_fps=fps;}
    void update(TvaiImage &tvimage, VecObjBBox &bboxIN, BYTETRACKPARM &params);
    void clear();

protected:
    void add_trackor(cam_class_uuid uuid, BYTETRACKPARM params);
private:
    std::map<cam_class_uuid, ByteTrackNoReID_Ptr> m_trackors;
    std::map<cam_class_uuid, BYTETRACKPARM> m_params;
    int m_fps = 25;
    int m_nn_buf = 30;
    // std::mutex m_mutex;
};

/**
 * ByteTrack_ORIGIN
 */
typedef std::shared_ptr<bytetrack_origin::BYTETracker> ByteTrackOrigin_Ptr;
class ByteTrackOriginPool: public TrackPoolAPI<BYTETRACKPARM>{
public:
    ByteTrackOriginPool(){}
    ~ByteTrackOriginPool(){}
    ByteTrackOriginPool(int fps, int nn_buf):\
        m_fps(fps),\
        m_nn_buf(nn_buf) {}

    void set_fps(int fps){m_fps=fps;}
    void update(TvaiImage &tvimage, VecObjBBox &bboxIN, BYTETRACKPARM &params);
    void clear();

protected:
    void add_trackor(cam_class_uuid uuid, BYTETRACKPARM params);
private:
    std::map<cam_class_uuid, ByteTrackOrigin_Ptr> m_trackors;
    std::map<cam_class_uuid, BYTETRACKPARM> m_params;
    int m_fps = 25;
    int m_nn_buf = 30;
    // std::mutex m_mutex;
};



#endif