#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <assert.h>
#include <thread>


using namespace std;


#include "config.hpp"
using namespace ucloud;

std::mutex cmutex;
bool use_static_frame = false;
const int fps = 25;//视频实际帧率
const int interval_ms = 100;//采样间隔ms
int interval_fps = ((float)interval_ms)/1000*fps + 1;//mlu220下fps间隔

void create_thread_for_yolo_task(int thread_id, TASKNAME taskid ,string datapath ,bool use_track=false, \
bool simulate_mlu220=false, bool dont_infer=false){
    int frame_limit = 30000;
    interval_fps = simulate_mlu220 ? interval_fps:1;
    // thread thread_source([=](){
        int flag_for_trackid_or_cls = 0;
        if(/*taskid == TASKNAME::GKPW || taskid == TASKNAME::GKPW2 ||*/ use_track == false) flag_for_trackid_or_cls = 1;
        RET_CODE retcode = RET_CODE::FAILED;
        float threshold, nms_threshold;
        AlgoAPIName apiName, apiSubName;
        std::map<InitParam, std::string> init_param;
        int use_batch = 0;

        bool flag_parser = task_parser(taskid, threshold, nms_threshold, apiName, init_param, use_batch);
        if(!flag_parser) {
            std::cout << "parser failed" << std::endl;
            return;
        }

        ucloud::TvaiResolution maxTarget={0,0};
        ucloud::TvaiResolution minTarget={0,0};
        std::vector<ucloud::TvaiRect> pRoi;
        double tm_cost = 0;
        int num_result = 0;
        std::cout << "yolo::loading model for thread #" << thread_id << endl;
        //Get Algo API
        AlgoAPISPtr ptrMainHandle = ucloud::AICoreFactory::getAlgoAPI(apiName);
        std::cout << "AICoreFactory done!" << endl;
        //Initial model with loading weights
        retcode = ptrMainHandle->init(init_param);
        if( retcode != RET_CODE::SUCCESS ){ std::cout << "algo initial failed" << endl; return; }
        //Set model parameters
        // ptrMainHandle->set_param(threshold, nms_threshold, maxTarget, minTarget, pRoi );
        std::vector<ucloud::CLS_TYPE> valid_clss;
        retcode = ptrMainHandle->get_class_type(valid_clss);
        printf("valid classes: [");
        for(auto &&m: valid_clss){
            printf("%d, ", int(m));
        }
        printf("]\n");

        ifstream infile;
        string filename = datapath;
        int real_infer_num = 0;

        vidReader handle_t;
        bool ret = handle_t.init(filename);
        if(!ret) { std::cout << "vid read handle init failed" << endl; return;}
        else{ 
            std::cout << "vid read handle init success" << endl;
            std::cout << "fps = " << handle_t.fps() << ", width = " << handle_t.width() << ", height = " << handle_t.height() << endl;
        }
        vidWriter w_handle_t;
        std::string savefilename = "x";
        savefilename = filename +".mp4";//.mkv for h264
        
        int ratio = (handle_t.width() > 1920) ? 2:1; // resize video if video is too large to save
        ret = w_handle_t.init( savefilename, handle_t.width()/ratio, handle_t.height()/ratio, handle_t.fps() );
        if(!ret) { std::cout << "vid write handle init failed" << endl; return;}
        else{ 
            std::cout << "vid write handle init success" << endl;
        }
        int frame_cnt = 0;
        std::vector<VIDOUT*> frameBuf;
        unsigned char* static_frame = nullptr;
        while(1){
            if(frame_cnt%100==0){ 
                // std::cout << ((float)frame_cnt)/handle_t.len() << ": " << num_result << " detected" << endl;
                printf("[%0.1f%%] %06d/%06d : %d detected\n",((float)frame_cnt)/handle_t.len()*100, frame_cnt, handle_t.len(), num_result ); 
            }
            if(frame_cnt >= handle_t.len() || frame_cnt > frame_limit) break;
            VecObjBBox bboxes; 
            int width, height, stride;
            VIDOUT* vidtmp = nullptr;
            vidtmp = handle_t.getImg();  //bgr img for illustration; yuv img for infer
            if(vidtmp==nullptr) {
                std::cout << "vid handle get EOF" << endl;
                break;
            }
            frameBuf.push_back(vidtmp);
            frame_cnt++;

            // if(frame_cnt==10 && use_static_frame){
            //     std::cout << "using static frame" << endl;
            //     width = frameBuf[0]->w; height = frameBuf[0]->h; stride = frameBuf[0]->s;
            //     static_frame = (unsigned char*)malloc(3*width*height/2*sizeof(unsigned char));
            //     memcpy(static_frame, frameBuf[0]->yuvbuf, 3*width*height/2*sizeof(unsigned char));
            // }

            if(use_batch <= 1){//single frame
                bool flag_do_infer = true;
                //simulate mlu220情况下，只有interval_fps的需要推理，其他都不需要；mlu270都需要推理
                if( (simulate_mlu220 && (frame_cnt-1)%interval_fps != 0 ) || dont_infer) flag_do_infer = false;
                width = frameBuf[0]->w; height = frameBuf[0]->h; stride = frameBuf[0]->s;
                if( flag_do_infer ){
                    // printf("+++");
                    int inputdata_sz = 3*stride*height/2*sizeof(unsigned char);
                    TvaiImage tvimage{TVAI_IMAGE_FORMAT_NV21,width,height,stride,frameBuf[0]->yuvbuf, inputdata_sz};
                    auto start = chrono::system_clock::now();
                    RET_CODE _ret_ = ptrMainHandle->run(tvimage, bboxes);
                    // printf("[%d]\n", _ret_);
                    auto end = chrono::system_clock::now();
                    auto duration = chrono::duration_cast<chrono::microseconds>(end-start);
                    tm_cost += double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den;
                    num_result += bboxes.size();
                    real_infer_num++;
                    if(!bboxes.empty()){
                        bool flag_disp_label = true;
                        if(taskid==TASKNAME::GKPW2) flag_disp_label = false;
                        bool use_rand_color = false;
                        
                        if(taskid==TASKNAME::GKPW || taskid==TASKNAME::GKPW2){
                            VecObjBBox _bboxes;
                            for(auto &&box :bboxes){
                                if(box.objtype == CLS_TYPE::FALLING_OBJ)
                                    _bboxes.push_back(box);
                            }
                            drawImg( frameBuf[0]->bgrbuf, width, height, _bboxes, false, false, false, flag_for_trackid_or_cls);
                        } else if(taskid==TASKNAME::GKPW3){
                            VecObjBBox _bboxes;
                            for(auto &&box :bboxes){
                                if(box.track_id >= 0)
                                    _bboxes.push_back(box);
                            }
                            drawImg( frameBuf[0]->bgrbuf, width, height, _bboxes, false, false, false, flag_for_trackid_or_cls);
                        }
                        else{
                            // std::cout << "drawing" << std::endl;
                            if(!dont_infer)
                                drawImg( frameBuf[0]->bgrbuf, width, height, bboxes, true, flag_disp_label, use_rand_color, flag_for_trackid_or_cls);
                            // std::cout << "finished" << std::endl;
                        }
                            

                    }
                }
                w_handle_t.writeImg(frameBuf[0]->bgrbuf, width, height);
                frameBuf[0]->release();
                frameBuf.erase(frameBuf.begin());
            }else{// multi frame (2 or 8 frames)
                if(frameBuf.size() < use_batch*interval_fps ) continue;
                else{
                    width = frameBuf[0]->w; height = frameBuf[0]->h; stride = frameBuf[0]->s;
                    int inputdata_sz = 3*width*height/2*sizeof(unsigned char);
                    BatchImageIN tvimages;
                    int _cnt = 0;
                    for(auto iterBuf = frameBuf.begin(); iterBuf!=frameBuf.end(); iterBuf++,_cnt++){
                        if(_cnt%interval_fps == 0){
                            TvaiImage tvimage{TVAI_IMAGE_FORMAT_NV21,width,height,stride,(*iterBuf)->yuvbuf, inputdata_sz};
                            tvimages.push_back(tvimage);
                        }
                        // if(static_frame!=nullptr && iterBuf == frameBuf.begin() && use_static_frame){
                        //     if(frame_cnt==10) std::cout << "infer using static frame" << endl;
                        //     TvaiImage tvimage{TVAI_IMAGE_FORMAT_NV21,width,height,stride,static_frame, inputdata_sz};
                        //     tvimages.push_back(tvimage);
                        // } else {
                        //     TvaiImage tvimage{TVAI_IMAGE_FORMAT_NV21,width,height,stride,(*iterBuf)->yuvbuf, inputdata_sz};
                        //     tvimages.push_back(tvimage);
                        // }
                    }
                    assert(tvimages.size() == use_batch );
                    auto start = chrono::system_clock::now();
                    ptrMainHandle->run(tvimages, bboxes);
                    auto end = chrono::system_clock::now();
                    auto duration = chrono::duration_cast<chrono::microseconds>(end-start);
                    tm_cost += double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den;
                    num_result += bboxes.size();
                    real_infer_num++;
                    if(!bboxes.empty()){
                        drawImg( frameBuf[frameBuf.size()-1]->bgrbuf, width, height, bboxes, false, false, false, 1);
                    }
                    w_handle_t.writeImg(frameBuf[frameBuf.size()-1]->bgrbuf, width, height);

                    if(simulate_mlu220){//mlu220采用非连续方式150ms的两个图像输入
                        // frameBuf.erase(frameBuf.begin(),frameBuf.begin()+interval_fps);
                        // if(frameBuf.size() != interval_fps*(use_batch-1)) std::cout << "ERROR: " << frameBuf.size() << std::endl;
                        frameBuf.clear();
                    }else{//mlu270采用连续的方式进行
                        frameBuf[0]->release();
                        frameBuf.erase(frameBuf.begin());
                    }

                } // multi process
            } // single or multi frames
        } // end while

        for(auto iter=frameBuf.begin(); iter!=frameBuf.end(); iter++){
            (*iter)->release();
        }
        frameBuf.clear();
        if(static_frame!=nullptr) free(static_frame);
        std::cout << "yolo::thread #" << thread_id << " : [" << num_result << "]" << endl;
        std::cout << "avg tm cost = " << tm_cost/real_infer_num << std::endl;
    // });//end of thread
    // thread_source.detach();
}

/**
 * ./test_case_vid {mlu220/mlu270} {datapath} {taskid} {use tracking or not} {infer or not}
 * ./test_case_vid_new -data /project/data/video_ucloud -task 15 -use_track 1 -mlu220 1
 */
int main(int argc, char* argv[]) {
    bool use_track = false;
    bool simulate_mlu220 = false;
    bool dont_infer = false;
    string datapath;
    TASKNAME taskid = TASKNAME::PED_CAR_NONCAR;
    ArgParser myParser;
    myParser.add_argument("-mlu220",0,"whether fps simulate condition on mlu220");
    myParser.add_argument("-data","", "datapath of target video");
    myParser.add_argument("-task", 0, "use which algorithm(task)");
    myParser.add_argument("-track",0, "use tracking or not");
    myParser.add_argument("-infer",1, "do infer or not");
    myParser.add_argument("-list",0, "list all the tasks");

    if(!myParser.parser(argc, argv)) return -1;

    bool listAll = myParser.get_value_int("-list") > 0 ? true:false;
    if(listAll){
        print_all_task();
        return -1;
    }

    simulate_mlu220 = myParser.get_value_int("-mlu220") > 0 ? true: false;
    datapath = myParser.get_value_string("-data");
    taskid = TASKNAME(myParser.get_value_int("-task"));
    use_track = myParser.get_value_int("-track") > 0 ? true: false;
    dont_infer = myParser.get_value_int("-infer") > 0 ? false: true;


    std::cout << "==========FPS===========" << std::endl;
    if(simulate_mlu220)
        std::cout << "simulate_mlu220: interval fps = " << interval_fps << std::endl;
    else
        std::cout << "normal mlu270" << std::endl;
    std::cout << "=====================" << std::endl;        

    if(use_track) std::cout << "use tracking" << endl;
    else std::cout << "dont use tracking!!!" << endl;

    if(dont_infer) std::cout << "video will be saved only!!!" << endl;
    
    std::cout << "using datapath " << datapath << endl;
    std::cout << "taskid " << int(taskid) << endl;

    create_thread_for_yolo_task(0, taskid , datapath, use_track, simulate_mlu220, dont_infer);

    // pthread_exit(NULL);
    return 0;
}