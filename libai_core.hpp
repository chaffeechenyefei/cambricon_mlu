/**
 * libai_core.hpp 2022-10-21
 * libai_core.hpp 2022-04-08
 * chaffee.chen@ucloud.cn
 */
#ifndef _LIBAI_CORE_HPP_
#define _LIBAI_CORE_HPP_

#include <vector>
#include <string>
#include <mutex>
#include <memory>
#include <map>

#if __GNUC__ >= 4
    #ifdef UCLOUD_EXPORT
        #define UCLOUD_API_PUBLIC __attribute__((visibility ("default")))
        #define UCLOUD_API_LOCAL __attribute__((visibility("hidden")))
    #else
        #define UCLOUD_API_PUBLIC
        #define UCLOUD_API_LOCAL
    #endif
#else
    #error "##### requires gcc version >= 4.0 #####"
#endif

#define TEST 1

namespace ucloud{

/*******************************************************************************
目录索引
*******************************************************************************/
class UCLOUD_API_PUBLIC AICoreFactory;//算法创建工厂前置声明, 方便索引
class UCLOUD_API_PUBLIC AlgoAPI;//算法对外接口前置声明, 方便索引
using AlgoAPISPtr = std::shared_ptr<AlgoAPI> ;//算法实例的智能指针
class UCLOUD_API_PUBLIC Clocker;//计时使用
/*******************************************************************************/
// UCLOUD_API_PUBLIC void releaseTvaiFeature(VecFeat& feats);//释放返回的人脸特征
// UCLOUD_API_PUBLIC void releaseVecObjBBox(VecObjBBox &bboxs);//释放返回的bboxes中所有的指针
/*******************************************************************************/
typedef struct tagBBox BBox;//重要信息传递对象
typedef struct tagTensors Tensors;
typedef struct tagWeightData WeightData;//模型结构体
/*******************************************************************************/
//=========自测使用======================================================================
class UCLOUD_API_PUBLIC ArgParser;//argc argv的解析, 便于程序测试
class UCLOUD_API_PUBLIC VIDOUT;//视频数据, yuv, rgb
class UCLOUD_API_PUBLIC vidReader;//视频读取
class UCLOUD_API_PUBLIC vidWriter;//视频写入
/*******************************************************************************/


/*******************************************************************************
 * 支持的算法种类 
 * 注释格式: 说明 + 算法返回类型
*******************************************************************************/
enum class AlgoAPIName{
//=========人======================================================================
    FACE_DETECTOR       = 0,//人脸检测, 采用DeepSort跟踪 [CLS_TYPE::FACE]
    FACE_DETECTORV2     = 1,//人脸检测, 采用非特征跟踪, 速度更快 [CLS_TYPE::FACE]
    FACE_EXTRACTOR      = 2,//人脸特征提取 [CLS_TYPE::FACE]
    FACE_DETECTOR_ATTR  = 3,//人脸检测 [CLS_TYPE::FACE] 带人脸属性json
    FACE_DETECTORV2_ATTR  = 4,//人脸检测 [CLS_TYPE::FACE] 带人脸属性json
    PED_DETECTOR        = 5,//行人检测加强版, 针对摔倒进行数据增强, mAP高于人车非中的人 [CLS_TYPE::PEDESTRIAN]
    PED_FALL_DETECTOR   = 6,//行人摔倒检测, 只检测摔倒的行人 [CLS_TYPE::PEDESTRIAN_FALL]
    SKELETON_DETECTOR   = 7,//人体骨架/关键点检测器--后续对接可用于摔倒检测等业务 x
    HEAD_DETECTOR       = 8,//人头检测, 检测画面中人头数量, 用于密集场景人数统计 [CLS_TYPE::HEAD]
    SMOKING_DETECTOR    = 9,//抽烟行为检测 x [CLS_TYPE::SMOKING]
    PHONING_DETECTOR    = 10,//打电话/玩手机行为检测 x [CLS_TYPE::PHONING]
    SOS_DETECTOR        = 11,//SOS举手求救 [CLS_TYPE::SOS]
    PED_SK_DETECTOR     = 12,//行人弯腰检测, [CLS_TYPE::PEDESTRIAN_BEND, CLS_TYPE::PEDESTRIAN]
    ACTION_CLASSIFIER   = 13,//行为识别, 目前支持打斗 [需要数据更新模型] x 不可用

//=========车======================================================================
    GENERAL_DETECTOR    = 50,//通用人车非检测器,即yolodetector, 可用于人车非 [CLS_TYPE::PEDESTRIAN, CLS_TYPE::NONCAR, CLS_TYPE::CAR]
    GENERAL_DETECTORV2  = 51,//通用人车非检测器,跟踪器替代origin bytetrack [CLS_TYPE::PEDESTRIAN, CLS_TYPE::NONCAR, CLS_TYPE::CAR]
    GENERAL_DETECTORV3  = 52,//通用人车非检测器,跟踪器替代no reid bytetrack [CLS_TYPE::PEDESTRIAN, CLS_TYPE::NONCAR, CLS_TYPE::CAR]
    NONCAR_DETECTOR     = 53,//非机动车检测加强版, 针对非机动车进电梯开发 [CLS_TYPE::BYCYCLE, CLS_TYPE::EBYCYCLE]
    LICPLATE_DETECTOR   = 54, //车牌检测 [CLS_TYPE::LICPLATE_X, ...]
    LICPLATE_RECOGNIZER = 55, //车牌识别 [CLS_TYPE::LICPLATE_X, ...], desc json输出车牌信息
    GENERAL_DETECTORV4  = 56,//通用人车非检测器,跟踪器替代origin bytetrack + 轨迹输出  [CLS_TYPE::PEDESTRIAN, CLS_TYPE::NONCAR, CLS_TYPE::CAR]
    GENERAL_DETECTORV5  = 57,//通用人车非检测器,deepsort + 轨迹输出  [CLS_TYPE::PEDESTRIAN, CLS_TYPE::NONCAR, CLS_TYPE::CAR]

//=========物======================================================================
    FIRE_DETECTOR       = 100,//火焰检测 [CLS_TYPE::FIRE]
    FIRE_DETECTOR_X     = 101,//火焰检测加强版, 带火焰分类器 [CLS_TYPE::FIRE]
    SAFETY_HAT_DETECTOR = 102,//安全帽检测 [CLS_TYPE::SAFETY_HAT, CLS_TYPE::HEAD]
    TRASH_BAG_DETECTOR  = 103,//垃圾袋检测 x [CLS_TYPE::TRASH_BAG]
    BANNER_DETECTOR     = 104,//横幅检测 x [CLS_TYPE::BANNER]
    WATER_DETECTOR      = 105,//积水检测 x [CLS_TYPE::WATER_PUDDLE]
    
    MOD_DETECTOR        = 106,//高空抛物, Moving Object Detection(MOD)[需要改善后处理, 开放做多帧接口测试] [FALLING_OBJ, FALLING_OBJ_UNCERTAIN]
    MOD_MOG2_DETECTOR   = 107,//高空抛物, Moving Object Detection(MOD)[MoG2版本]
    MOD_MOG2_DETECTORV2   = 108,//高空抛物, Moving Object Detection(MOD)[MoG2版本] + Bytetrack 效果不佳
    TARGERT_REMAIN_DETECTOR = 109,//物品遗留检测 [CLS_TYPE::TARGET]
    
//=========保留项目======================================================================
    RESERVED1           = 1001,//保留占位符号
    RESERVED2           = 1002,
    RESERVED3           = 1003,
   

//=========高级接口======================================================================
//用户可以根据需要自行进行开发(SISO: 只针对单输入单输出)
    GENERAL_YOLOV5_DETECTOR = 2001,//通用yolov5检测器: 对输入图像进行检测, 返回目标位置, 需要设定模型输出的对应类别
    GENERAL_CLASSIFY        = 2002,//通用分类器: 对输入图像的特定区域(VecBox进行设定)进行分类, 雪瑶设定模型输出的对应类别
    GENERAL_INFER           = 2003,//通用推理接口, 返回内容自行解析, 内容返回在TvaiFeature中

//=========内部使用======================================================================
    UDF_JSON            = 5000, //用户自定义json输入 x 不可用
    GENERAL_TRACKOR     = 5001,//通用跟踪模块, 不能实例化, 但可以在内部使用
    HAND_DETECTOR       = 5002,//人手检测 736x416
    BATCH_GENERAL_DETECTOR    = 5005,//测试用
    FIRE_CLASSIFIER           = 5006,//火焰分类, 内部测试用
//=========其它======================================================================
    TJ_HELMET_DETECTOR =10007,//同济416x416安全帽检测 
    // BATCH_GENERAL_DETECTOR    = 100,//测试用
    // WATER_DETECTOR_OLD      = 1008,//积水检测(旧版unet,与新版之间存在后处理的逻辑差异)
};

/*******************************************************************************
返回值
*******************************************************************************/
typedef enum _RET_CODE{
    SUCCESS                     = 0, //成功
    FAILED                      = 1, //未知失败

    ERR_NPU_INIT_FAILED         = 2, //MLU初始化失败
    ERR_MODEL_FILE_NOT_EXIST    = 3, //模型文件不存在
    ERR_INIT_PARAM_FAILED       = 4, //参数初始化失败
    ERR_UNSUPPORTED_IMG_FORMAT  = 5, //图像输入格式不支持
    ERR_MODEL_NOT_MATCH         = 6, //载入的模型有问题, 无法推理
    ERR_MODEL_NOT_INIT          = 7,  //模型没有被加载, 无法推理, 请先调用成员函数init()
    ERR_OUTPUT_CLS_INIT_FAILED  = 8,  //检测模型输出类型(CLS_TYPE)绑定失败
    ERR_BATCHSIZE_NOT_MATCH     = 9,  //输入数据batchsize大小和模型不一致
    ERR_PHYADDR_EMPTY           = 10, //物理地址空
    ERR_VIRTUAL_FUNCTION        = 11, //虚函数, 该类不支持该接口
    ERR_EMPTY_BOX               = 12, //输入的BOX为空
}RET_CODE;

/*******************************************************************************
目标类别
*******************************************************************************/
typedef enum tagCLS_TYPE{
    PEDESTRIAN                  = 0     ,   //行人
    FACE                        = 1     ,   //人脸
    PEDESTRIAN_FALL             = 2     ,   //摔倒的行人
    HAND                        = 3     ,   //人手检测
    PEDESTRIAN_BEND             = 4     ,   //行人弯腰

    CAR                         = 10    ,   //车辆
    NONCAR                      = 100   ,   //非机动车
    BYCYCLE                             ,   //自行车
    EBYCYCLE                            ,   //电瓶车

    PET                         = 200   ,   //宠物
    PET_DOG                             ,   //宠物狗
    PET_CAT                             ,   //宠物猫

    WATER_PUDDLE                = 300   ,   //积水
    TRASH_BAG                           ,   //垃圾袋
    BANNER                              ,   //横幅标语
    FIRE                                ,   //火焰
    //行为识别
    FIGHT                       = 400   ,   //打架行为
    SMOKING                     = 410   ,   //抽烟
    PHONING                     = 411   ,   //打电话或玩手机
    SOS                         = 412   ,   //举手求救
    //安全帽
    PED_HEAD                    = 500   ,   //头
    PED_SAFETY_HAT                      ,   //安全帽
    //高空抛物
    FALLING_OBJ                 = 600   ,   //高空抛物--轨迹确定
    FALLING_OBJ_UNCERTAIN               ,   //高空抛物--轨迹未确定
    //车牌检测
    LICPLATE_BLUE               = 700   ,   //蓝牌
    LICPLATE_SGREEN                     ,   //小型新能源车（纯绿）
    LICPLATE_BGREEN                     ,   //大型新能源车（黄加绿）
    LICPLATE_YELLOW                     ,   //黄牌
    LICPLATE                            ,   //车牌


    OTHERS                      = 900   ,   //其它类别, 相当于占位符
    OTHERS_A                    = 901   ,   //自定义占位符
    OTHERS_B                    = 902   ,
    OTHERS_C                    = 903   ,
    OTHERS_D                    = 904   ,
    

    UNKNOWN                     = 1000  ,   //未定义
    TARGET                      = 1001  ,   //不管是什么, 反正是重要目标, 具体是什么, 通过外部自己自由定义
}CLS_TYPE;


//目标特征值
typedef struct TvaiFeature_S
{
    unsigned int          featureLen = 0;        /* 特征值长度, 字节数 */
    unsigned char         *pFeature = nullptr;         /* 特征值指针 */
}TvaiFeature;

//目标特征值
typedef struct tagTensors
{
    unsigned int          num_tensors = 0;        /* 模型输出Tensor的数量 */
    TvaiFeature           *tensors = nullptr;     /* tensor实际存储内容, flatten后的(view(-1)) */ 
}Tensors;


// 分辨率, 用于设定检测目标的大小范围
typedef struct TvaiResolution_S {
    unsigned int     width;                 /* 宽度 */
    unsigned int     height;                /* 高度 */
}TvaiResolution;

// 通用矩形框
typedef struct TvaiRect_S
{
    int     x;          /* 左上角X坐标 */
    int     y;          /* 左上角Y坐标 */
    int     width;      /* 区域宽度 */
    int     height;     /* 区域高度 */
}TvaiRect;

//通用关键点类型
typedef enum _LandMarkType{
    FACE_5PTS            =   0, //人脸五点
    SKELETON             =   1, //人体骨架信息
    UNKNOW_LANDMARK      =   10,//未知预留
}LandMarkType;

//关键点坐标参考系
typedef enum _RefCoord{
    IMAGE_ORIGIN        =   0,//图像坐标系, 以图像左上角为坐标原点
    ROI_ORIGIN          =   1,//以ROI区域原点为坐标原点
    HEATMAP_ORIGIN      =   2,//以模型输出的heatmap图像左上角为坐标原点
}RefCoord;

//关键点xy坐标结构体
typedef struct _uPoint{
    float x;
    float y;
    _uPoint(float _x, float _y):x(_x),y(_y){}
    _uPoint(){x=0;y=0;}
} uPoint;

//关键点信息结构
typedef struct _LandMark{
    std::vector<uPoint> pts;
    LandMarkType type;
    RefCoord refcoord;
} LandMark;

typedef struct _AInfo{
    float preprocess_time;
    float npu_inference_time;
    float postprocess_time;
}AInfo;

//通用返回信息结构
typedef struct tagBBox {
    /*****************************外部使用**************************************/
    CLS_TYPE objtype = CLS_TYPE::UNKNOWN;
    float confidence = 0; //tvai 某类别的概率(由模型输出的objectness*confidence得到)或事件概率
    float quality = 0;//图像质量分0-1, 1:高质量图像
    TvaiRect rect; //tvai -> image scale 最终输出
    LandMark Pts;//关键点位信息
    std::string desc = "";//json infomation, 目的: 预留, 用于临时情况下将信息以json字段方式输出
    int track_id = -1;//跟踪唯一标识
    TvaiFeature feat;//特征信息
    AInfo tmInfo;//内部处理时间
    std::vector<uPoint> trace;//框的历史轨迹, 如果有的话
    /*****************************高级接口 适用于AlgoName::2003**************************************/
    Tensors tensors;//模型输出的Tensor使用完毕后用户自行free所有内容
    /*****************************内部使用**************************************/
    //模型输出:
    float x0;//top left corner x -> model scale
    float y0;//top left corner y -> model scale
    float x1;//bottom right corner x -> model scale
    float y1;//bottom right corner y -> model scale
    float x,y,w,h;//top left corner + width + height -> model scale
    //最终图像输出(由模型输出经过aspect_scale, feature_map缩放得到)
    float objectness = 0; //物体概率
    std::string objname = "unknown";//objtype的文字描述, 在objtype = OTHERS(_A)的情况下, 可以进行透传. 目的: 支持临时算法改动
    std::vector<float> trackfeat;//跟踪用特征
}BBox;

// 图像格式
typedef enum _TvaiImageFormat{
    TVAI_IMAGE_FORMAT_NULL    = 0,              /* 格式为空 */
    TVAI_IMAGE_FORMAT_GRAY,                     /* 单通道灰度图像 */
    TVAI_IMAGE_FORMAT_NV12,                     /* YUV420SP_NV12：YYYYYYYY UV UV */
    TVAI_IMAGE_FORMAT_NV21,                     /* YVU420SP_NV21：YYYYYYYY VU VU */
    TVAI_IMAGE_FORMAT_RGB,                      /* 3通道，RGBRGBRGBRGB */
    TVAI_IMAGE_FORMAT_BGR,                      /* 3通道，BGRBGRBGRBGR */
    TVAI_IMAGE_FORMAT_I420,                     /* YUV420p_I420 ：YYYYYYYY UU VV */
}TvaiImageFormat;


// 输入图像结构
typedef struct TvaiImage_S
{
    TvaiImageFormat      format;      /* 图像像素格式 */
    int                  width;       /* 图像宽度 */
    int                  height;      /* 图像高度 */
    int                  stride;      /* 图像水平跨度 */
    unsigned char        *pData = nullptr; /* 图像数据。*/
    int                  dataSize;    /* 图像数据的长度 */
    uint64_t             u64PhyAddr[3]={0}; /* 数据的物理地址 */
    bool                 usePhyAddr=false;  /* 是否使用数据的物理地址 */
    int                  uuid_cam=-1;/*图像设备来源唯一标号, 用于上下文相关的任务*/

    TvaiImage_S(TvaiImageFormat _fmt, int _width, int _height, int _stride, unsigned char* _pData, int _dataSize, int _uuid_cam=-1):\
        format(_fmt), width(_width), height(_height), stride(_stride), pData(_pData), dataSize(_dataSize), uuid_cam(_uuid_cam){}
    TvaiImage_S(){
        format = TvaiImageFormat::TVAI_IMAGE_FORMAT_NULL;
        width = 0;
        height = 0;
        stride = 0;
        dataSize = 0;
        uuid_cam = -1;
    }
}TvaiImage;

//自定义简称
typedef std::vector<TvaiRect> VecRect;
// typedef std::vector<FaceInfo> VecFaceBBox;
typedef std::vector<TvaiFeature> VecFeat;
typedef std::vector<BBox> VecObjBBox;
// typedef std::vector<SkLandmark> VecSkLandmark;
typedef std::vector<TvaiImage> BatchImageIN;
typedef std::vector<VecObjBBox> BatchBBoxOUT;
typedef std::vector<VecObjBBox> BatchBBoxIN;

////////////////////////////////////////////////////////////////////////////////////////////////////////
// 方法,工厂类,枚举类型
////////////////////////////////////////////////////////////////////////////////////////////////////////
typedef enum _InitParam{
    BASE_MODEL          = 0, //基础模型(检测模型/特征提取模型/分类模型)
    TRACK_MODEL         = 1, //跟踪模型
    SUB_MODEL           = 2, //模型级联时, 主模型用于初步检测, 次模型用于二次过滤, 提高精度
    SUB_NODEL_A         = 3, 
    SUB_NODEL_B         = 4, 
    SUB_NODEL_C         = 5, 
    SUB_NODEL_D         = 6, 

    // PED_MODEL           = 10,
    // HAND_MODEL          = 11,
    // FACE_MODEL          = 12,
}InitParam;
using InitParamMap = std::map<InitParam, std::string>;

typedef enum _APIParam{
    OBJ_THRESHOLD      = 0, //目标检测阈值/分类阈值
    NMS_THRESHOLD      = 1, //NMS检测阈值
    MAX_OBJ_SIZE       = 2, //目标最大尺寸限制
    MIN_OBJ_SIZE       = 3, //目标最小尺寸限制
    VALID_REGION       = 4, //有效检测区域设定
}APIParam;

typedef struct tagWeightData{
    unsigned char* pData;
    int size;/*size_t*/
}WeightData;


//接口基类, 接口函数
class UCLOUD_API_PUBLIC AlgoAPI{
public:
    /*****************************外部使用**************************************/
    AlgoAPI(){};
    virtual ~AlgoAPI(){};
    virtual RET_CODE init(InitParamMap &modelpath){return RET_CODE::ERR_VIRTUAL_FUNCTION;}
    virtual RET_CODE init(std::map<InitParam, WeightData> &weightConfig){return RET_CODE::ERR_VIRTUAL_FUNCTION;}

    virtual RET_CODE run(TvaiImage& tvimage, VecObjBBox &bboxes, float threshold=0.55, float nms_threshold=0.6){return RET_CODE::ERR_VIRTUAL_FUNCTION;}
    
    /**
     * 返回检测的类别, 或返回适用的类别
     */
    virtual RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss){return RET_CODE::ERR_VIRTUAL_FUNCTION;};
    /*****************************高级接口 适用于AlgoName::2001-200X**************************************/
    virtual RET_CODE set_output_cls_order(std::vector<std::string> &output_clss){return RET_CODE::ERR_VIRTUAL_FUNCTION;};


    /*****************************内部使用**************************************/
    virtual RET_CODE set_output_cls_order(std::vector<CLS_TYPE> &clss){return RET_CODE::ERR_VIRTUAL_FUNCTION;};
    virtual int get_batchsize(){return 1;}
    /**
     * 普通初始化方式, 方便后面灵活运用
     */
    virtual RET_CODE init(){return RET_CODE::ERR_VIRTUAL_FUNCTION;}                            
    virtual RET_CODE init(const std::string &modelpath){//兼容模式
        std::map<InitParam, std::string> configs = {{InitParam::BASE_MODEL,modelpath}};
        return init(configs);}
    virtual RET_CODE init(WeightData weightConfig){
        std::map<InitParam, WeightData> config = {{InitParam::BASE_MODEL, weightConfig}};
        return init(config);
        }        
    /**
     * 供依赖序列图像输入的算法使用, 如: 打架
     * 该接口仅支持yuv图像输入
     * FUNC:
     *  RET_CODE run(BatchImageIN &batch_tvimages, BatchBBoxIN &batch_bboxes, VecObjBBox &bboxes)
     *  输入时序图像, 以及经过目标检测的结果, 函数根据目标位置, 对有目标的区域进行行为判断.
     * FUNC:
     *  RET_CODE run(BatchImageIN &batch_tvimages, VecObjBBox &bboxes)
     *  输入时序图像, 并对整张图进行判断
     */
    virtual RET_CODE run(BatchImageIN &batch_tvimages, BatchBBoxIN &batch_bboxes, VecObjBBox &bboxes,float threshold, float nms_threshold){return RET_CODE::ERR_VIRTUAL_FUNCTION;}
    /**
     * 高空抛物已改单帧推理模式, 多帧推理接口仍保留可使用.
     * chaffee@2022-05-17
    */
    virtual RET_CODE run(BatchImageIN &batch_tvimages, VecObjBBox &bboxes, float threshold=0.55, float nms_threshold=0.6){
        //接口兼容:兼容单帧输入的情况@2022-02-17
        if(batch_tvimages.empty()) return RET_CODE::SUCCESS;
        else return run(batch_tvimages[0], bboxes, threshold, nms_threshold);
    }

    /*****************************弃用**************************************/
    virtual RET_CODE set_param(float threshold, float nms_threshold, TvaiResolution maxTargetSize, TvaiResolution minTargetSize, std::vector<TvaiRect> &pAoiRect){return RET_CODE::ERR_VIRTUAL_FUNCTION;}
};


//工厂
class UCLOUD_API_PUBLIC AICoreFactory{
public:
    static AlgoAPISPtr getAlgoAPI(AlgoAPIName apiName);
protected:
    AICoreFactory();
    ~AICoreFactory();
};


// extern "C"{
/** tvaiImageToMatData 
 *  将输入的图像转为算法支持的BGR格式数据, 支持RGB/YUVNV21, 不支持物理地址
 *  IN: 
 * input: TvaiImage格式数据
 *  OUT: 
 * width 返回BGR图像的宽
 * height 返回BGR图像的高
 * RETURN:
 * 在内部通过malloc开辟空间, OPENCV支持的BGR格式数据
 **/
UCLOUD_API_PUBLIC unsigned char* tvaiImageToMatData(TvaiImage input, int &width, int &height);

////////////////////////////////////////////////////////////////////////////////////////////////////////
// 返回指针资源的释放, 例如人脸特征提取返回的人脸特征
////////////////////////////////////////////////////////////////////////////////////////////////////////
//释放返回的人脸特征
UCLOUD_API_PUBLIC void releaseTvaiFeature(VecFeat& feats);
//释放返回的bboxes中所有的指针
UCLOUD_API_PUBLIC void releaseVecObjBBox(VecObjBBox &bboxs);

////////////////////////////////////////////////////////////////////////////////////////////////////////
// 封装opencv的读写图片功能, 便于测试和sdk的封装
////////////////////////////////////////////////////////////////////////////////////////////////////////
//jpg png bmp等常见图像的读取
UCLOUD_API_PUBLIC unsigned char* readImg(std::string filepath, int &width, int &height);

UCLOUD_API_PUBLIC unsigned char* readImg_to_RGB(std::string filepath, int &width, int &height);
UCLOUD_API_PUBLIC unsigned char* readImg_to_BGR(std::string filepath, int &width, int &height);
UCLOUD_API_PUBLIC unsigned char* readImg_to_NV21(std::string filepath, int &width, int &height, int &stride);
// UCLOUD_API_PUBLIC unsigned char* readImg_to_NV12(std::string filepath, int &width, int &height, int &stride);
/**
 * 读入图像, 并转为yuv数据, 同时进行缩放
 * PARA:
 * w :将输入图像resize到w/h尺寸
 * h :将输入图像resize到w/h尺寸
 * width :yuv能适配的尺寸
 * height :yuv能适配的尺寸
 */
UCLOUD_API_PUBLIC unsigned char* readImg_to_RGB(std::string filepath, int w, int h, int &width, int &height);
UCLOUD_API_PUBLIC unsigned char* readImg_to_BGR(std::string filepath, int w, int h, int &width, int &height);
UCLOUD_API_PUBLIC unsigned char* readImg_to_NV21(std::string filepath, int w, int h,int &width, int &height, int &stride);
// UCLOUD_API_PUBLIC unsigned char* readImg_to_NV12(std::string filepath, int w, int h,int &width, int &height, int &stride);

//写图像, 是否采用覆盖式写入
UCLOUD_API_PUBLIC void writeImg(std::string filepath , unsigned char* imgPtr, int width, int height, bool overwrite=true);
UCLOUD_API_PUBLIC void freeImg(unsigned char** imgPtr);
/**
 * drawImg 画图内部使用opencv, 输出RGB/RGB数据
 * PARAM:
 *  img: 输入RGB/BGR数据
 *  width: 图像的宽
 *  height: 图像的长
 *  bboxs: 需要画的目标框
 *  disp_landmark: 是否将框内landmark坐标画在图上
 *  disp_label: 是否显示框的CLS_TYPE枚举值
 */
UCLOUD_API_PUBLIC void drawImg(unsigned char* img, int width, int height, VecObjBBox &bboxs, \
        bool disp_landmark=false ,bool disp_label=false, bool use_rand_color=true, int color_for_trackid_or_cls = 0);
//读取yuv和rgb的二进制文件流, 便于测试
UCLOUD_API_PUBLIC unsigned char* yuv_reader(std::string filename, int w=1920, int h=1080);
UCLOUD_API_PUBLIC unsigned char* rgb_reader(std::string filename, int w=1920, int h=1080);
//视频读取基于opencv
class UCLOUD_API_PUBLIC VIDOUT{
public:
    VIDOUT(){}
    ~VIDOUT(){release();}
    VIDOUT(const VIDOUT &obj)=delete;
    VIDOUT& operator=(const VIDOUT & rhs)=delete;
    unsigned char* bgrbuf=nullptr;
    unsigned char* yuvbuf=nullptr;
    int w,h,s;//yuv
    int _w,_h;//bgr
    void release(){
        if(bgrbuf!=nullptr) free(bgrbuf);
        if(yuvbuf!=nullptr) free(yuvbuf);
    }
};
class UCLOUD_API_PUBLIC vidReader{
public:
    vidReader(){}
    ~vidReader(){release();}
    bool init(std::string filename);
    unsigned char* getbgrImg(int &width, int &height);
    unsigned char* getyuvImg(int &width, int &height, int &stride);
    VIDOUT* getImg();
    int len(){return m_len;}
    int width();
    int height();
    int fps();
private:
    void release();
    void* handle_t=nullptr;
    int m_len = 0;
};
class UCLOUD_API_PUBLIC vidWriter{
public:
    vidWriter(){};
    ~vidWriter(){release();}
    bool init(std::string filename, int width, int height, int fps);
    void writeImg(unsigned char* buf, int bufw, int bufh);
private:
    void release();
    void* handle_t=nullptr;
    int m_width;
    int m_height;
    int m_fps;
};

/*******************************************************************************
运行时间测试
chaffee.chen@2022-10-10
*******************************************************************************/
class UCLOUD_API_PUBLIC Clocker{
public:
    Clocker();
    ~Clocker();
    void start();
    double end(std::string title, bool display=true);//return seconds
private:
    void* ctx;
};

/*******************************************************************************
输入argc argv的解析 仅支持float/int/string
chaffee.chen@2022-10-20
*******************************************************************************/
class UCLOUD_API_PUBLIC ArgParser{
public: 
    ArgParser(){}
    ~ArgParser(){}
    bool add_argument(const std::string &keyword, int default_value , const std::string &helpword);
    bool add_argument(const std::string &keyword, float default_value , const std::string &helpword);
    // bool add_argument(const std::string &keyword, bool default_value , const std::string &helpword);
    bool add_argument(const std::string &keyword, const std::string &default_value , const std::string &helpword);
    bool parser(int argc, char* argv[]);
    void print_help();
protected:
    std::map<std::string,float> m_cmd_float;
    std::map<std::string,int> m_cmd_int;
    std::map<std::string,bool> m_cmd_bool;
    std::map<std::string, std::string> m_cmd_str;
    std::map<std::string, std::string> m_cmd_help;
public:
    float get_value_float(const std::string &keyword);
    int get_value_int(const std::string &keyword);
    // bool get_value_bool(const std::string &keyword);
    std::string get_value_string(const std::string &keyword);
    bool is_value_exist(const std::string &keyword);
};

////////////////////////////////////////////////////////////////////////////////////////////////////////

// };

// ////////////////////////////////////////////////////////////////////////////////////////////////////////
// //mlu环境类
// ////////////////////////////////////////////////////////////////////////////////////////////////////////
// class UCLOUD_API_PUBLIC PrivateContext{
// public:
//     PrivateContext();
//     virtual ~PrivateContext();
// protected:
// bool status_=false;
// void* env_;
// };

// //FOR FUTURE, play for fun
// template<class DerivedClass>
// class UCLOUD_API_PUBLIC AlgoFutureAPI {
//     RET_CODE init(std::map<InitParam, std::string> &modelpath){
//         return static_cast<DerivedClass*>(this)->init(modelpath);
//     }
//     RET_CODE run(BatchImageIN &batch_tvimages, VecObjBBox &bboxes){
//         return static_cast<DerivedClass*>(this)->run(batch_tvimages, bboxes);
//     }
//     RET_CODE get_class_type(std::vector<CLS_TYPE> &valid_clss){
//         return static_cast<DerivedClass*>(this)->get_class_type(valid_clss);
//     }
// };

// class UCLOUD_API_PUBLIC SpecDerviedClass: public AlgoFutureAPI<SpecDerviedClass> {
// };



/**
 * 20210917
 * 以下是历史遗留产物, 后期不再更新维护.
 */
////////////////////////////////////////////////////////////////////////////////////////////////////////
//特征比对, 可改用blas加速计算
////////////////////////////////////////////////////////////////////////////////////////////////////////
inline float _calcSimilarity(float* fA, float* fB, int dims){
    float val = 0;
    for(int i=0 ; i < dims ; i++ )
        val += (*fA++) * (*fB++);
    return val;
};


inline void calcSimilarity(VecFeat& featA,VecFeat& featB, std::vector<std::vector<float>>& result){
    std::vector<std::vector<float>>().swap(result);
    result.clear();
    for( int a = 0 ; a < featA.size() ; a++ ){
        float* fA = reinterpret_cast<float*>(featA[a].pFeature);
        int dims = featA[a].featureLen/sizeof(float);
        std::vector<float> inner_reuslt;
        for ( int b = 0; b < featB.size() ; b++ ){
            float* fB = reinterpret_cast<float*>(featB[b].pFeature);
            float val = _calcSimilarity(fA, fB, dims);
            inner_reuslt.push_back(val);
        }
        result.push_back(inner_reuslt);
    }
};

};//namespace ucloud



#endif