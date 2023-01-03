#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <assert.h>
#include <thread>
#include <stdio.h>
#include "libai_core.hpp"


using namespace std;
using namespace ucloud;

// unsigned char* yuv_reader(string filename, int w = 1920, int h=1080){
//     ifstream fin(filename, std::ios::binary);
//     int l = fin.tellg();
//     fin.seekg(0, ios::end);
//     int m = fin.tellg();
//     fin.seekg(0,ios::beg);
//     cout << "file size " << (m-l) << " bytes" << endl;
//     assert(m-l == w*h*1.5);
//     int stride = w;
//     int wh = w*h;
//     unsigned char* yuvdata = (unsigned char*)malloc(int(wh*1.5)*sizeof(unsigned char));
//     fin.read( reinterpret_cast<char*>(yuvdata) , int(wh*1.5)*sizeof(unsigned char));
//     return yuvdata;
// }

int main(int argc, char* argv[]) {
    uint32_t t = -2;
    std::cout << t << std::endl;

    if (sizeof(uint64_t) == sizeof(void*)){
        std::cout << "size of uint64_t and void* equals" << std::endl;
    } else{
        std::cout << "size of uint64_t and void* not equals" << std::endl;
    }
    std::cout << "void* = " << sizeof(void*) << ", uint64_t = " << sizeof(uint64_t) << std::endl; 

    if (sizeof(uint64_t) == sizeof(unsigned long long)){
        std::cout << "size of uint64_t and unsigned long long equals" << std::endl;
    } else{
        std::cout << "size of uint64_t and unsigned long long not equals" << std::endl;
    }
    std::cout << "unsigned long long = " << sizeof(unsigned long long) << ", uint64_t = " << sizeof(uint64_t) << std::endl;

        //  检测相关参数
    float threshold = 0.4;
    float nms_threshold = 0.6;
    // 设置为0时, 判断条件不起作用
    ucloud::TvaiResolution maxTraget={0,0};
    ucloud::TvaiResolution minTarget={0,0};
    std::vector<ucloud::TvaiRect> pRoi;


    int max_color = 50;
    int rand_color[max_color*3];

    for(int i = 0; i < max_color*3; i++ )
        rand_color[i] = rand()%255;

    string modelfile;
    string modeldir = "/project/workspace/samples/yolov5/";
    string imgname = "web.jpg";
    int N = 1;

    if(argc >= 2){
        string _imgname(argv[1]);
        imgname = _imgname;
        cout << "reading: " << imgname << endl;
    }
    if(argc >=3){
        string _strN(argv[2]);
        N = std::stoi(_strN);
        std::cout << "batch size = " << N << std::endl;
    }

    bool use_yuv = false;
    if(imgname.find(".yuv")>=0 && imgname.find(".yuv")!=std::string::npos){
        std::cout << "yuv image input" << std::endl; 
        use_yuv = true;
    }

#ifdef MLU220
    modelfile = "yolov5s-conv-people_736x416_mlu220_bs8c4_fp16.cambricon";
#else
    modelfile = modeldir + "mlu270/" + "yolov5s-conv-people_736x416_mlu270_bs1c1_fp16.cambricon";
    // modelfile = modeldir + "mlu270/" + "yolov5s-conv-people_736x416_mlu270.cambricon";
#endif
    cout << "Model File: " << modelfile << endl;

    YoloDetector* objDetector = new YoloDetector();
    RET_CODE ret = objDetector->init(modelfile);
    cout << "INITIAL = " << ret << endl;
    objDetector->set_param(threshold, nms_threshold, maxTraget, minTarget, pRoi);
    CLS_TYPE output_clss[] = {CLS_TYPE::PEDESTRIAN};
    ret = objDetector->set_output_cls_order(output_clss, 1);
    cout << "CLASS TYPE = " << ret << endl;

    VecObjBBox bboxes;
    int width, height;
    unsigned char* imgPtr;

    vector<unsigned char*> vecImgPtr;
    if(!use_yuv){
        imgPtr = ucloud::readImg(imgname,width,height);
    } else {
        width = 1920;
        height = 1080;
        for(int i=0; i<N ;i++){
            vecImgPtr.push_back(yuv_reader(imgname,width,height));
        }
        imgPtr = vecImgPtr[0];
    }
    
    if (imgPtr==nullptr)
        cout << "EMPTY IMAGE" << endl;

    // objDetector->_run_(im.data, im.cols, im.rows, bboxes);
    ucloud::TvaiImageFormat fmt = (!use_yuv) ? ucloud::TvaiImageFormat::TVAI_IMAGE_FORMAT_BGR : ucloud::TvaiImageFormat::TVAI_IMAGE_FORMAT_NV21;

    int sz = (!use_yuv)? width*height*3*sizeof(unsigned char) : height/2*3*width*sizeof(unsigned char);
    if(use_yuv){
        std::vector<TvaiImage> batch_in;
        std::vector<VecObjBBox> batch_out;
        for(int i=0; i<N ;i++){
            TvaiImage _tvimage_{fmt, width, height, width, vecImgPtr[i], sz};
            batch_in.push_back(_tvimage_);
        }
        auto start = chrono::system_clock::now();
        if(N!=1)
            ret = objDetector->run_yuv_on_mlu_batch(batch_in, batch_out);
        else
            ret = objDetector->run(batch_in[0], bboxes);
        auto end = chrono::system_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end-start);
        double tm_cost = double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den;
        cout << "detection cost: " << tm_cost << "s" << endl;
        cout << "DETECTOR = " << ret << endl;

        if(N!=1){
            for(int i = 0 ; i < batch_out.size(); i++){
                cout << "Batch #" << i << ": Object detected: " << batch_out[i].size() << endl;
                for (int j = 0; j < batch_out[i].size(); j++ ){
                    cout << "conf:" << batch_out[i][j].confidence << ", x = " << batch_out[i][j].rect.x
                    << ", y = " << batch_out[i][j].rect.y 
                    << ", w = " << batch_out[i][j].rect.width
                    << ", h = " << batch_out[i][j].rect.height << endl;
                }
            }
        } else{
            cout << "Object detected: " << bboxes.size() << endl;
            for(int k = 0; k < bboxes.size(); k++ ){
                cout << "conf:" << bboxes[k].confidence << ", x = " << bboxes[k].rect.x
                    << ", y = " << bboxes[k].rect.y 
                    << ", w = " << bboxes[k].rect.width
                    << ", h = " << bboxes[k].rect.height << endl;
            }
            cout << endl;
        }

        for(int i = 0 ; i < N ; i++ ){
            free(vecImgPtr[i]);
        }
    }
    else{
        TvaiImage tvimage{fmt, width, height, width, imgPtr, sz};
        auto start = chrono::system_clock::now();
        ret = objDetector->run(tvimage, bboxes);
        auto end = chrono::system_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end-start);
        double tm_cost = double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den;
        cout << "detection cost: " << tm_cost << "s" << endl;
        cout << "DETECTOR = " << ret << endl;
        cout << "Object detected: " << bboxes.size() << endl;
        //print confidence
        for(int k = 0; k < bboxes.size(); k++ ){
            cout << bboxes[k].objtype << ":" << bboxes[k].confidence << ", ";
        }
        cout << endl;
        drawImg(imgPtr, width, height, bboxes);
        writeImg("1.png", imgPtr, width, height);
        free(imgPtr);
    }

    
    delete objDetector;

}