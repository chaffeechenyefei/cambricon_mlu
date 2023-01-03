#include "post_rule_hover.hpp"
#include "../module_base.hpp"

using namespace ucloud;
using namespace std;

#define OVER_SIZE -1000

ucloud::RET_CODE POST_RULE_HOVER::run(ucloud::TvaiImage& tvimage, ucloud::VecObjBBox &bboxes, float threshold, float nms_threshold){
    LOGI << "-> POST_RULE_HOVER::run";
    std::set<trace_uuid> updated_traceid;
    //push new data inside first
    // std::cout << "push new data inside first" << std::endl;
    for(auto &&box: bboxes){
        if(box.track_id < 0) continue;
        trace_uuid traceid = get_box_trace_uuid(tvimage, box);
        // std::cout << traceid << std::endl;
        uPoint foot;//right foot
        foot.x = box.rect.x + box.rect.width;
        foot.y = box.rect.y + box.rect.height;
        // m_trace_sets[traceid].push(foot);
        if(m_trace_sets.find(traceid)!=m_trace_sets.end()){
            m_trace_sets[traceid].push_back(foot);
        } else {
            m_trace_sets[traceid] = {foot};
        }
        m_trace_lost_times[traceid] = 0;//激活
        updated_traceid.insert(traceid);
    }
    //update and remove
    // std::cout << "update and remove" << std::endl;
    std::set<trace_uuid> trace_to_remove;
    for(auto &&trace: m_trace_sets){
        if(updated_traceid.find(trace.first) == updated_traceid.end()){
            //find a trace not updated, insert a null one
            uPoint foot(OVER_SIZE,OVER_SIZE);
            m_trace_sets[trace.first].push_back(foot);
            m_trace_lost_times[trace.first]++;//计算连续丢失次数
            if(m_trace_lost_times[trace.first] > max_lost_times){
                trace_to_remove.insert(trace.first);
                continue;
            }
        }
        if(m_trace_sets[trace.first].size() > max_trace_len ){
            m_trace_sets[trace.first].erase(m_trace_sets[trace.first].begin());
        }
    }

    for(auto &&traceid: trace_to_remove){
        m_trace_sets.erase(traceid);
    }

    // std::cout << "update bboxes" << std::endl;
    for(auto &&box: bboxes){
        if(box.track_id < 0 ) continue;
        trace_uuid traceid = get_box_trace_uuid(tvimage, box);
        // std::cout << traceid << std::endl;
        if(m_trace_sets.find(traceid)!=m_trace_sets.end()){
            int cur_trace_len = m_trace_sets[traceid].size();
            // int step = 1;
            if(max_display_len >= cur_trace_len){
                for(auto &&pt: m_trace_sets[traceid]){
                    if(pt.x > OVER_SIZE && pt.y > OVER_SIZE )
                        box.trace.push_back(pt);
                }
            }
            else{
                vecPoints tmp;
                for(auto &&pt: m_trace_sets[traceid]){
                    if(pt.x > OVER_SIZE && pt.y > OVER_SIZE )
                        tmp.push_back(pt);
                }
                if(max_display_len >= tmp.size()) tmp.swap(box.trace);
                else{
                    int step = tmp.size()/max_display_len;
                    int cnt = 0;
                    for(auto &&pt: tmp){
                        if(cnt%step==0 || cnt==tmp.size()-1)
                            box.trace.push_back(pt);
                        cnt++;
                    }
                }
            }

        }
    }
    // std::cout << "finished" << std::endl;
    LOGI << "<- POST_RULE_HOVER::run";
    return RET_CODE::SUCCESS;
}


POST_RULE_HOVER::trace_uuid POST_RULE_HOVER::get_box_trace_uuid(ucloud::TvaiImage& tvimage, ucloud::BBox &bbox){
    int cam_uuid = tvimage.uuid_cam;
    int trace_id = bbox.track_id;
    if(bbox.track_id < 0 ) return "";

    return std::to_string(cam_uuid) + "_" + std::to_string(trace_id);
}