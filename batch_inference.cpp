#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <assert.h>
#include <thread>
#include <stdio.h>
#include <dirent.h>
#include <sys/stat.h>
using namespace std;


#include "config.hpp"
using namespace ucloud;

using std::cout;

std::string subreplace(std::string resource_str, std::string sub_str, std::string new_str)
{
    std::string dst_str = resource_str;
    std::string::size_type pos = 0;
    while(( pos = dst_str.find(sub_str)) != std::string::npos)   //替换所有指定子串
    {
        dst_str.replace(pos, sub_str.length(), new_str);
    }
    return dst_str;
}
bool hasEnding (std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}
void ls_files( const std::string &dir_name, std::vector<std::string> &filenames, const std::string& endswith )
{
	// check the parameter !
	if( dir_name.empty() || dir_name == "" )
		return;
 
	// check if dir_name is a valid dir
	struct stat s;
	lstat( dir_name.c_str() , &s );
	if( ! S_ISDIR( s.st_mode ) )
	{
		std::cout<<"dir_name is not a valid directory !"<< std::endl;
		return;
	}
	
	struct dirent * filename;    // return value for readdir()
 	DIR * dir;                   // return value for opendir()
	dir = opendir( dir_name.c_str() );
	if( NULL == dir )
	{
		std::cout<<"Can not open dir "<<dir_name<<std::endl;
		return;
	}
	
	/* read all the files in the dir ~ */
	while( ( filename = readdir(dir) ) != NULL )
	{
		// get rid of "." and ".."
		if( strcmp( filename->d_name , "." ) == 0 || 
			strcmp( filename->d_name , "..") == 0    )
			continue;
        std::string tmp(filename->d_name);
        if(endswith.empty() || endswith == "" )
            filenames.push_back(tmp);
        else{
            if(hasEnding(tmp, endswith)) filenames.push_back(tmp);
        }
		// std::cout<<filename ->d_name <<std::endl;
	}
    closedir(dir);
} 

/**usage
 * ./batch_infer {dataPath} {taskID/TASKNAME} {threshold}[not a must]
 */
int main(int argc, char* argv[]) {
    auto start = chrono::system_clock::now();
    //TODO
    auto end = chrono::system_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end-start);
    double tm_cost = double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den;

    int taskID = 0;
    string dataRoot = "data/";
    string dataExt = ".jpg";
    if(argc>=2){
        string _tmp(argv[1]);
        dataRoot = _tmp;
    }
    printf("reading images with ending %s from %s \n",dataExt.c_str(),dataRoot.c_str());
    if(argc >= 3){
        taskID = atoi(argv[2]);
    }
    bool fake_yuv = false;
    float _threshold_ = 0;
    bool use_cmd_threshold = false;
    if(argc>=4){
        std::cout << "## use cmd threshold:" << _threshold_ << endl;
        use_cmd_threshold = true;
        _threshold_ = atof(argv[3]);
    }

    // find images
    vector<string> filenames;
    ls_files(dataRoot, filenames, dataExt);
    printf("total [%d] images found\n",filenames.size());
    // init algo
    RET_CODE ret = RET_CODE::FAILED;
    vector<CLS_TYPE> class_type_to_be_detected;
    VecRect _tmp_;
    ucloud::TvaiResolution maxTraget{0,0};
    ucloud::TvaiResolution minTarget{0,0};
    std::vector<ucloud::TvaiRect> pRoi;
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
    cout << "TASK: " << taskDesc << endl;
    AlgoAPISPtr detectHandle = AICoreFactory::getAlgoAPI(apiName);
    ret = detectHandle->init(init_param);
    cout << "detect init return " << ret << endl;
    if(ret!=RET_CODE::SUCCESS) return -1;
    if(use_cmd_threshold) threshold_det = _threshold_;
    ret = detectHandle->set_param(threshold_det,threshold_nms,maxTraget,minTarget, _tmp_);
    if(ret!=RET_CODE::SUCCESS) return -1;
    cout << "threshold: " << threshold_det << ", " << threshold_nms << endl;
    ret = detectHandle->get_class_type(class_type_to_be_detected);
    cout << "get_class_type return " << ret << endl;
    if(ret!=RET_CODE::SUCCESS) return -1;
    cout << "Class will be detected: ";
    for (int i = 0; i < class_type_to_be_detected.size(); i++ ){
        cout << class_type_to_be_detected[i] << ", ";
    }
    cout << endl;

    // loop infer
    int cnt = 0;
    int cnt_infer = 0;
    double total_cost = 0;
    unsigned char* ptrImg = nullptr;
    int width, height, stride;
    TvaiImageFormat inpFmt;
    int inpSz;
    for(auto &&filename: filenames){
        if(cnt++%1000 == 0){
            printf("[%d %] avg speed = %f s\n", int(((float)cnt)/filenames.size()*100), (cnt_infer>0)?(total_cost/cnt_infer):0);
            fflush(stdout);
        }
        string filename_full = dataRoot + '/' + filename;
        ptrImg = readImg_to_NV21(filename_full, width, height, stride );
        if(ptrImg==nullptr){
            printf("%s can not be read correctly\n",filename.c_str());
            continue;
        }
        inpFmt = TVAI_IMAGE_FORMAT_NV21;
        inpSz = 3*stride*height/2*sizeof(unsigned char);

        TvaiImage inpImg(inpFmt,width,height,stride,ptrImg,inpSz);
        VecObjBBox objbboxes;

        start = chrono::system_clock::now();
        ret = detectHandle->run(inpImg, objbboxes);
        end = chrono::system_clock::now();
        duration = chrono::duration_cast<chrono::microseconds>(end-start);
        tm_cost = double(duration.count()) * chrono::microseconds::period::num / chrono::microseconds::period::den;
        if(ret!=RET_CODE::SUCCESS) {
            printf("infer err:[%d] in %s\n", ret, filename.c_str());
            ucloud::freeImg(&ptrImg);
            continue;
        }
        total_cost += tm_cost;
        cnt_infer++;

        if(!objbboxes.empty()){
            ofstream fd;
            string savename;
            savename = subreplace(filename, dataExt, ".txt");
            // printf("savename = %s\n", savename.c_str());
            fd.open(dataRoot+"/"+savename);
            if(fd.is_open()){
                ostringstream line("");
                for(auto &&box: objbboxes){
                    float x = ((float)(box.rect.x))/width;
                    float y = ((float)box.rect.y)/height;
                    float w = ((float)box.rect.width)/width;
                    float h = ((float)box.rect.height)/height;
                    float cx = x + w/2;
                    float cy = y + h/2;
                    line << box.objtype << " " << cx << " " << cy << " " << w << " " << h << " " << box.objectness <<"\n";
                }
                fd.write(line.str().c_str(), line.str().length());
            }
            fd.close();
        }
        

        // 可视化
        // ucloud::freeImg(&ptrImg);
        // ptrImg = readImg(filename_full, width, height);
        // drawImg(ptrImg, width, height, objbboxes, true, true);
        // writeImg( "result/test_batch.jpg" , ptrImg, width, height,true);
        ucloud::freeImg(&ptrImg);
        // break;
    }

    printf("[%d] avg speed = %f s\n", filenames.size(), (cnt_infer>0)?(total_cost/cnt_infer):0);

}

    // cout << "# " << objbboxes.size() << "obj detected" << endl;
    // cout << "List of confidence: \"#CLS_TYPE = objectness\" " << endl;
    // for (int i = 0 ; i < objbboxes.size(); i++ ){
    //     cout << "#" << objbboxes[i].objtype << " = " <<objbboxes[i].objectness 
    //     << " [" 
    //     << objbboxes[i].rect.x << ", " << objbboxes[i].rect.y << ", " 
    //     << objbboxes[i].rect.width << ", " << objbboxes[i].rect.height
    //     << "], " ;
    // }
    // cout << endl;
