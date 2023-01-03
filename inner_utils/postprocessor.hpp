#ifndef _POSTPROCESSOR_HPP_
#define _POSTPROCESSOR_HPP_
#include "../libai_core.hpp"

#include <queue>

// using ucloud::TvaiRect;
// using ucloud::VecObjBBox;

class DetectHeatMap{
public:
    DetectHeatMap(){};
    ~DetectHeatMap(){};
    float calc_score(ucloud::TvaiRect &rect);
    void add(ucloud::TvaiRect &rect);
    void decay();
    bool empty();

    void set_max_buffer_size(int size){max_buffer_size=size;}
    
private:
    void update_roi();

    ucloud::TvaiRect roi{0,0,0,0};
    std::vector<ucloud::TvaiRect> buffer;
    int max_buffer_size = 6;

};


class single_object_detect_trigger{
public:
    single_object_detect_trigger(){}
    ~single_object_detect_trigger(){}
    //assume all the bbox in bboxes belongs to the same CLS_TYPE
    void updateResult(ucloud::VecObjBBox &bboxes);
private:
    
};

#endif