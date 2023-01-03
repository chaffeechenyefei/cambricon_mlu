#include "config.hpp"

#include <stdlib.h>
#include <time.h> 

#include <vector>


std::vector<MODELFILENAME> valid_model_names = {
    MODELFILENAME::FACE_DET,                //人脸检测
    MODELFILENAME::FACE_EXT,                //特征提取
    MODELFILENAME::SKELETON_DET_R18,        //骨架定位
    MODELFILENAME::FIRE_CLS,                //火焰分类
    MODELFILENAME::GENERAL_TRK_MLU,         //跟踪特征提取器
    MODELFILENAME::GENERAL_DET,             //人车非检测
    MODELFILENAME::PED_DET,                 //行人检测
    MODELFILENAME::PED_FALL_DET,            //摔倒检测
    MODELFILENAME::SAFETY_HAT_DET,          //安全帽检测
    MODELFILENAME::TRASH_BAG_DET,           //垃圾检测
    MODELFILENAME::FIRE_DET_220407,         //火焰检测
    MODELFILENAME::MOTOR_DET,               //电瓶车、自行车车检测
    MODELFILENAME::HAND_DET_736x416,        //手的检测
    MODELFILENAME::CIG_DET,                 //抽烟检测
    MODELFILENAME::MOD_DET_DIF,             //高空抛物
    MODELFILENAME::BANNER_DET,              //横幅检测
    MODELFILENAME::HEAD_DET,                //人头检测
    MODELFILENAME::LICPLATE_DET,            //车牌检测
    MODELFILENAME::LICPLATE_RECOG,          //车牌识别
};


/*
 函数说明：对字符串中所有指定的子串进行替换
 参数：
string resource_str            //源字符串
string sub_str                //被替换子串
string new_str                //替换子串
返回值: string
 */
std::string subreplace(std::string resource_str, std::string sub_str, std::string new_str, bool once=false)
{
    std::string dst_str = resource_str;
    std::string::size_type pos = 0;
    while(( pos = dst_str.find(sub_str)) != std::string::npos)   //替换所有指定子串
    {
        dst_str.replace(pos, sub_str.length(), new_str);
        if(once) break;
    }
    return dst_str;
}

#include <sys/stat.h>
inline bool exists_file(const std::string& name) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}

/**
 * 将所有用到的模型根据valid_model_names都自动拉取到一个目录下
 */
int main(int argc, char* argv[]) {
    std::string savepath220 = "model_220/";
    std::string savepath270 = "model_270/";
    std::string cmd_rm220 = "rm -r " + savepath220;
    std::string cmd_mk220 = "mkdir " + savepath220;
    std::string cmd_rm270 = "rm -r " + savepath270;
    std::string cmd_mk270 = "mkdir " + savepath270;
    system(cmd_rm220.c_str());
    system(cmd_mk220.c_str());
    system(cmd_rm270.c_str());
    system(cmd_mk270.c_str());
    for(auto &&model270: cambricon_model_file ){
        bool skip = true;
        for(auto &&_vmn: valid_model_names){
            if(_vmn == model270.first){
                skip = false;
                break;
            }
        }
        if(skip) continue;
        std::cout << "copy: " << int(model270.first) << std::endl;
        std::string modelpath = model270.second;
        modelpath = subreplace(modelpath, "mlu270", "mlu220");
        modelpath = subreplace(modelpath, "_270_", "_220_");
        // std::cout << modelpath << std::endl;
        std::string twin_modelpath = subreplace(modelpath, ".cambricon",".cambricon_twins", true);
        std::cout << "from: " << modelpath << std::endl;
        std::string command = "cp ";
        command += modelpath;
        command += " ";
        command += savepath220;
        // std::cout << command << std::endl;
        system(command.c_str());
        command = "cp " + twin_modelpath + " " + savepath220;
        // std::cout << command << std::endl;
        if(exists_file(twin_modelpath))
            system(command.c_str());

        command = "cp " + model270.second + " " + savepath270;
        // std::cout << command << std::endl;
        system(command.c_str());
        twin_modelpath = subreplace(model270.second, ".cambricon",".cambricon_twins", true);
        command = "cp " + twin_modelpath + " " + savepath270;
        // std::cout << command << std::endl;
        if(exists_file(twin_modelpath))
            system(command.c_str());
    }


    return 0;
}