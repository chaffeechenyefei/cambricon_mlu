#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <assert.h>
#include <thread>
#include <stdio.h>

using namespace std;


#include "config.hpp"
using namespace ucloud;

using std::cout;

/**usage
 * ./test_one {imgname} {taskID/TASKNAME} {use fake yuv if given} {threshold}
 */
int main(int argc, char* argv[]) {
    ArgParser myParser;
    myParser.add_argument("-data", "test.jpg", "input image");
    myParser.add_argument("-task", 0, "taskid");
    myParser.add_argument("-yuv", 1 , "use yuv");
    myParser.add_argument("-threshold", -1.0f, "input threshold, default use threshold from task_parser");
    myParser.add_argument("-w",1920,"input image width");
    myParser.add_argument("-h",1080,"input image height");
    myParser.add_argument("-list",0,"list all the task");
    if(!myParser.parser(argc, argv)) return -1;

    bool listAll = myParser.get_value_int("-list") > 0 ? true:false;
    if(listAll){
        print_all_task();
        return -1;
    }
        


    string imgname = myParser.get_value_string("-data");
    int taskID = myParser.get_value_int("-task");
    bool fake_yuv = myParser.get_value_int("-yuv")>0?true:false;
    float _threshold_ = myParser.get_value_float("-threshold");
    bool use_cmd_threshold = _threshold_ > 0 ? true:false;
    int W = myParser.get_value_int("-w");
    int H = myParser.get_value_int("-h");

    std::cout << "## reading image " << imgname << endl;
    if(use_cmd_threshold) std::cout << "## use cmd threshold:" << _threshold_ << endl;
    if(fake_yuv) std::cout << "## use fake yuv" << endl;

    RET_CODE ret;
    bool use_yuv = false;
    unsigned char* ptrImg = nullptr;
    int width, height, stride;
    TvaiImageFormat inpFmt;
    int inpSz;
    if(imgname.find(".yuv")>=0 && imgname.find(".yuv")!=std::string::npos){
        std::cout << "yuv image input" << std::endl;
        inpFmt = TVAI_IMAGE_FORMAT_NV21;
        width = 1920;
        height = 1080;
        ptrImg = yuv_reader(imgname, width, height);
        inpSz = 3*width*height/2*sizeof(unsigned char);
        stride = width;
        use_yuv = true;
    }
    if(!use_yuv){
        if(fake_yuv){
            inpFmt = TVAI_IMAGE_FORMAT_NV21;
            ptrImg = readImg_to_NV21(imgname, W, H, width, height, stride );
            if (ptrImg == nullptr){
                cout << "Empty image" << endl;
                return -1;
            }
            std::cout << "fake yuv input: " << width << ", " << height << ", "<< stride << ", "<< std::endl; 
            inpSz = 3*stride*height/2*sizeof(unsigned char);
        } else {
            inpFmt = TVAI_IMAGE_FORMAT_BGR;
            ptrImg = readImg_to_BGR(imgname, W,H, width, height);
            if (ptrImg == nullptr){
                cout << "Empty image" << endl;
                return -1;
            }
            inpSz = 3*width*height*sizeof(unsigned char);
            stride = width;
        }

    }

    TvaiImage inpImg(inpFmt,width,height,stride,ptrImg,inpSz);
    VecObjBBox objbboxes;
    BBox objbbox;

    vector<CLS_TYPE> class_type_to_be_detected;

    auto start = chrono::system_clock::now();
    //TODO
    auto end = chrono::system_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end-start);
    double tm_cost = double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den;

    VecRect _tmp_;
    AlgoAPIName apiName;
    std::map<InitParam, std::string> init_param;
    string taskDesc;
    float threshold_det = 0.2;
    float threshold_nms = 0.6;
    int use_batch = 1;
    //step1: 根据任务设定参数
    bool parser_flag = task_parser( TASKNAME(taskID), threshold_det, threshold_nms, apiName, init_param, use_batch );
    if( !parser_flag ) {
        std::cout << "parser failed" << std::endl;
        return -1;
    }

    //step2: 根据任务将形式上一致的任务放在一起执行
    switch ( TASKNAME(taskID) )
    {    
    //形式上一致的任务
    default:
        {
            cout << "TASK: " << taskDesc << endl;
            AlgoAPISPtr detectHandle = AICoreFactory::getAlgoAPI(apiName);
            ret = detectHandle->init(init_param);
            cout << "detect init return " << ret << endl;
            if(use_cmd_threshold) threshold_det = _threshold_;
            cout << "threshold: " << threshold_det << ", " << threshold_nms << endl;
            ret = detectHandle->get_class_type(class_type_to_be_detected);
            cout << "get_class_type return " << ret << endl;
            cout << "Class will be detected: ";
            for (int i = 0; i < class_type_to_be_detected.size(); i++ ){
                cout << class_type_to_be_detected[i] << ", ";
            }
            cout << endl;
            start = chrono::system_clock::now();
            ret = detectHandle->run(inpImg, objbboxes, threshold_det, threshold_nms);
            end = chrono::system_clock::now();
            duration = chrono::duration_cast<chrono::microseconds>(end-start);
            tm_cost = double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den;
            if(ret!=RET_CODE::SUCCESS) {
                printf("detectHandle->run(inpImg, objbboxes) return [%d]\n", ret);
                return -1;
            } else
                printf("detectHandle->run(inpImg, objbboxes) return [%d]\n", ret);
            cout << tm_cost << "s cost" << endl;
            cout << "# " << objbboxes.size() << "obj detected" << endl;
            cout << "List of confidence: \"#CLS_TYPE = objectness\" " << endl;
            for (int i = 0 ; i < objbboxes.size(); i++ ){
                cout << "#" << objbboxes[i].objtype << " = " <<objbboxes[i].objectness 
                << " [" 
                << objbboxes[i].rect.x << ", " << objbboxes[i].rect.y << ", " 
                << objbboxes[i].rect.width << ", " << objbboxes[i].rect.height
                << "], " ;
            }
            cout << endl;
            if(!use_yuv){
                if(fake_yuv){
                    freeImg(&ptrImg);
                    ptrImg = readImg_to_BGR(imgname, W, H, width, height);
                }
                drawImg(ptrImg, width, height, objbboxes, true, true);
            }
        }
        break;
    }

    //step3: 结果可视化
    if(ret == RET_CODE::SUCCESS)
        writeImg( "result/test_one.jpg" , ptrImg, width, height,true);
    freeImg(&ptrImg);
    cout << "run return " << ret << endl;
}