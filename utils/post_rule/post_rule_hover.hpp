#ifndef _POST_RULE_HOVER_HPP_
#define _POST_RULE_HOVER_HPP_
#include "../../libai_core.hpp"



class POST_RULE_HOVER: public ucloud::AlgoAPI{
    using trace_uuid = std::string;
    using vecPoints = std::vector<ucloud::uPoint>;
    // using quePoints = std::queue<ucloud::uPoint>;
public:
    POST_RULE_HOVER(){printf("post_rule_hover constructed\n");}
    virtual ~POST_RULE_HOVER(){}
    ucloud::RET_CODE run(ucloud::TvaiImage& tvimage, ucloud::VecObjBBox &bboxes, float threshold=0.55, float nms_threshold=0.6);


protected:
    trace_uuid get_box_trace_uuid(ucloud::TvaiImage& tvimage, ucloud::BBox &bbox);


    std::map<trace_uuid, vecPoints> m_trace_sets;
    std::map<trace_uuid, int> m_trace_lost_times;
    int max_trace_len = 20*10;//最大轨迹存储数量
    int max_lost_times = max_trace_len/2;//最大丢失次数
    int max_display_len = 10;//最大对外展示数据

};

#endif
