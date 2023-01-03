#include "postprocessor.hpp"

using ucloud::TvaiRect;

float DetectHeatMap::calc_score(ucloud::TvaiRect &rect){
    if(buffer.empty()) return 0;
    float x0 = roi.x;
    float y0 = roi.y;
    float x1 = roi.width+roi.x;
    float y1 = roi.height+roi.y;

    float _x0 = rect.x;
    float _y0 = rect.y;
    float _x1 = rect.width+rect.x;
    float _y1 = rect.height+rect.y;

    float roiWidth = std::min(x1, _x1) - std::max(x0, _x0);
    float roiHeight = std::min(y1, _y1) - std::max(y0, _y0);

    if (roiWidth<=0 || roiHeight<=0) return 0;
    float area0 = (y1 - y0 + 1)*(x1 - x0 + 1);
    float area1 = (_y1 - _y0 + 1)*(_x1 - _x0 + 1);
    return roiWidth*roiHeight/(area0 + area1 - roiWidth*roiHeight+1e-3);
}

bool DetectHeatMap::empty(){
    return buffer.empty();
}

void DetectHeatMap::add(ucloud::TvaiRect &rect){
    buffer.push_back(rect);
    while(buffer.size() > max_buffer_size)
        buffer.erase(buffer.begin());
    update_roi();
}

void DetectHeatMap::decay(){
    if(!buffer.empty())   
        buffer.erase(buffer.begin());
    update_roi();
}

void DetectHeatMap::update_roi(){
    if(buffer.empty()) roi = TvaiRect{0,0,0,0};
    else {
        int x0 = buffer[0].x;
        int y0 = buffer[0].y;
        int x1 = buffer[0].width+buffer[0].x;
        int y1 = buffer[0].height+buffer[0].y;
        //TODO利用buffer更新roi
        for(auto iter=buffer.begin(); iter!=buffer.end(); iter++){
            if(iter==buffer.begin()) continue;
            x0 = std::min(x0, iter->x);
            y0 = std::min(y0, iter->y);
            x1 = std::max(x1, iter->x+iter->width);
            y1 = std::max(y1, iter->y+iter->height);
        }
        roi = TvaiRect{x0,y0,x1-x0,y1-y0};
    }
}