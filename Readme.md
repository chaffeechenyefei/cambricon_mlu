# C++ SDK for cambricon mlu270/mlu220edge
## 0. 接口说明
005-001 思元220边缘盒子算法API说明: https://ushare.ucloudadmin.com/pages/viewpage.action?pageId=96537552

具体使用请参考: test_one_new.cpp or test_case_vid_new.cpp

### 0.1 HEADER FILE
```
libai_core.hpp
libai_core.so
```

### 0.2 工厂类枚举方式实例化方法
算法功能的使用通过```AICoreFactory::getAlgoAPI(AlgoAPIName apiName)```枚举AlgoAPIName类型获得:
```
    typedef enum _AlgoAPIName{
        FACE_DETECTOR       = 0,//人脸检测
        FACE_EXTRACTOR      = 1,//人脸特征提取
        GENERAL_DETECTOR    = 2,//通用物体检测器即yolodetector, 可用于人车非
        ACTION_CLASSIFIER   = 3,//行为识别, 目前支持打斗 [需要数据更新模型]
        MOD_DETECTOR        = 4,//高空抛物, Moving Object Detection(MOD)[需要改善后处理]
        PED_DETECTOR        = 5,//行人检测加强版, 针对摔倒进行数据增强, mAP高于人车非中的人
        FIRE_DETECTOR       = 6,//火焰检测
        FIRE_DETECTOR_X     = 7,//火焰检测加强版, 带火焰分类器
        WATER_DETECTOR      = 8,//积水检测
        PED_FALL_DETECTOR   = 9,//行人摔倒检测, 只检测摔倒的行人
        SKELETON_DETECTOR   = 10,//人体骨架/关键点检测器--后续对接可用于摔倒检测等业务
        SAFETY_HAT_DETECTOR = 11,//安全帽检测
        TRASH_BAG_DETECTOR  = 12,//垃圾袋检测
        BANNER_DETECTOR     = 13,//横幅检测
        NONCAR_DETECTOR     = 14,//非机动车检测加强版, 针对非机动车进电梯开发
        //=========内部使用======================================================================
        GENERAL_TRACKOR     = 50,//通用跟踪模块, 不能实例化, 但可以在内部使用
        MOD_MOG2_DETECTOR   = 51,//高空抛物, Moving Object Detection(MOD)[MoG2版本]
        BATCH_GENERAL_DETECTOR    = 100,//测试用
        FIRE_CLASSIFIER                ,//火焰分类, 内部测试用
        WATER_DETECTOR_OLD      = 1008,//积水检测(旧版unet,与新版之间存在后处理的逻辑差异)
    }AlgoAPIName;
```

### 0.3 每个方法使用说明
具体案例可见test_one.cpp

一般使用方法:
```
1. 通过AICoreFactory获得对应的方法对象
AlgoAPISPtr apiHandle = AICoreFactory::getAlgoAPI(AlgoAPIName::GENERAL_DETECTOR);
2. 通过init函数, 初始化方法, 主要负责载入模型文件
RET_CODE ret_status = apiHandle->init("xxx.cambricon");
3. 通过run函数, 处理每一帧输入数据RGB/YUVNV21, 并得到返回结果
ret_status = apiHandle->run(TvaiImage tvimage, VecObjBBox &bboxes);
4. 如果需要, 则释放返回结果中的指针
releaseVecObjBBox(bboxes);
```

在MLU220的编译时, 增加了USE_STATIC_MODEL开关, 开启时, 各个方法将在/cambricon/model目录下自行推导模型文件. init(std::map<InitParam, std::string> &modelpath)将执行推导出的模型文件.

算法API接口:
接口实现采用继承统一的基类AlgoAPI实现, 该基类中仅包含虚函数, 用于多态.

| 类型 | 接口作用 | 对应函数 |
| :-- | :-- | :-- |
| 模型导入及初始化 | 单模型导入 | virtual RET_CODE init(const std::string &modelpath) |
|    -       | 单模型+跟踪模型导入 | virtual RET_CODE init(const std::string &modelpath, const std::string &trackmodelpath) |
| - | 检测模型+判别模型+跟踪模型导入 (适用: 火焰检测加强版)| virtual RET_CODE init( const std::string &detect_modelpath, const std::string &classify_modelpath, const std::string &track_modelpath) |
| - | 新的导入方式, 合并上述接口(部分算法API功能已支持) | virtual RET_CODE init(std::map<InitParam, std::string> &modelpath) |
| 固定参数设定 | 设定固定参数. 阈值、非极大抑制阈值、最大(小)检测结果、有效检测(识别)区域 | virtual RET_CODE set_param(float threshold, float nms_threshold, TvaiResolution maxTargetSize, TvaiResolution minTargetSize, std::vector<TvaiRect> &pAoiRect) |
| 模型推理 | 单帧推理, 支持大部分算法API功能 | virtual RET_CODE run(TvaiImage& tvimage, VecObjBBox &bboxes) |
| - | 多帧推理 (适用: 高空抛物、行为识别(全图或设定的检测区域)) | virtual RET_CODE run(BatchImageIN &batch_tvimages, VecObjBBox &bboxes) |
| - | 多帧推理 (适用: 人体检测后的行为识别) | virtual RET_CODE run(BatchImageIN &batch_tvimages, BatchBBoxIN &batch_bboxes, VecObjBBox &bboxes) | 

算法API功能表:
<table class="wrapped relative-table confluenceTable" style="letter-spacing: 0px;"><colgroup><col style="width: 13.5112%;" /><col style="width: 13.4592%;" /><col style="width: 32.9835%;" /><col style="width: 6.35979%;" /><col style="width: 13.3333%;" /><col style="width: 6.43376%;" /><col style="width: 13.9192%;" /></colgroup><tbody><tr><th class="confluenceTh">算法API功能</th><th class="confluenceTh"><p>初始化方式</p><p>对应init函数</p></th><th class="confluenceTh"><p>推理方式</p><p>对应run函数</p></th><th class="confluenceTh"><p>是否支持RGB/BGR输入</p><p>(此时跟踪器无效)</p></th><th class="confluenceTh">是否过滤特定类别</th><th class="confluenceTh" colspan="1">推荐阈值</th><th class="confluenceTh" colspan="1">底层实现(继承关系 子-父)</th></tr><tr><td class="confluenceTd"><p>FACE_DETECTOR</p><p>人脸检测器</p></td><td class="confluenceTd"><p>支持单模型导入</p><p>支持单模型+跟踪模型导入</p></td><td class="confluenceTd"><p>单帧推理</p><p>输入TvaiImage格式图像</p><p>输出VecObjBBox结构数据, 包含单个画面中检测到的人脸位置、5个关键点坐标(</p><p>LandMark结构)、置信度以及质量.</p><p>目标包含Face类</p></td><td class="confluenceTd">是</td><td class="confluenceTd">--</td><td class="confluenceTd" colspan="1">0.8</td><td class="confluenceTd" colspan="1"><ul><li>FaceDetectionV2<ul><li>BaseModel<ul><li>PrivateContext</li><li>AlgoAPI</li></ul></li></ul></li></ul><p style="margin-left: 30.0px;"><br /></p><p><br /></p></td></tr><tr><td class="confluenceTd"><p>FACE_EXTRACTOR</p><p>人脸特征提取器</p></td><td class="confluenceTd"><p>仅支持单模型导入</p></td><td class="confluenceTd"><p>单帧推理,&nbsp;依赖人脸检测器</p><p>输入TvaiImage格式图像,&nbsp;VecObjBBox结构数据(依赖人脸检测器给出的人脸位置信息)</p><p>输出共用输入的VecObjBBox结构数据, 增加TvaiFeature结构的人脸特征数据</p></td><td class="confluenceTd">是</td><td class="confluenceTd">是, 仅处理CLS_TYPE::<span style="letter-spacing: 0.0px;">Face的BBox数据, 其余类型自动跳过</span></td><td class="confluenceTd" colspan="1">0.55(相似度)</td><td class="confluenceTd" colspan="1"><ul><li>FaceExtractionV2<ul><li>BaseModel<ul><li>PrivateContext</li><li>AlgoAPI</li></ul></li></ul></li></ul></td></tr><tr><td class="confluenceTd"><p>GENERAL_DETECTOR</p><p>人车非检测器</p></td><td class="confluenceTd"><p>支持单模型导入</p><p>支持单模型+跟踪模型导入</p></td><td class="confluenceTd"><p>单帧推理</p><p>输入TvaiImage格式图像</p><p>输出VecObjBBox结构数据, 包含单个画面中检测到的目标位置、置信度以及质量. 质量复用objectness/confidence.</p><p>目标包含CAR、NONCAR、PEDESTRIAN类</p></td><td class="confluenceTd">是</td><td class="confluenceTd">--</td><td class="confluenceTd" colspan="1">0.6</td><td class="confluenceTd" colspan="1"><ul><li>YoloDetectionV2<ul><li>BaseModel<ul><li>PrivateContext</li><li>AlgoAPI</li></ul></li></ul></li></ul></td></tr><tr><td class="confluenceTd"><p>PED_DETECTOR</p><p>行人检测器</p></td><td class="confluenceTd"><p>支持单模型导入</p><p>支持单模型+跟踪模型导入</p></td><td class="confluenceTd"><p>单帧推理</p><p>输入TvaiImage格式图像</p><p>输出VecObjBBox结构数据, 包含单个画面中检测到的目标位置、置信度以及质量. 质量复用objectness/confidence.</p><p>目标包含PEDESTRIAN类</p></td><td class="confluenceTd">是</td><td class="confluenceTd">--</td><td class="confluenceTd" colspan="1">0.3</td><td class="confluenceTd" colspan="1"><ul><li>YoloDetectionV2<ul><li>BaseModel<ul><li>PrivateContext</li><li>AlgoAPI</li></ul></li></ul></li></ul></td></tr><tr><td class="confluenceTd"><p>FIRE_DETECTOR</p><p>火焰检测器</p></td><td class="confluenceTd"><p>支持单模型导入</p><p>支持单模型+跟踪模型导入</p></td><td class="confluenceTd"><p>单帧推理</p><p>输入TvaiImage格式图像</p><p>输出VecObjBBox结构数据, 包含单个画面中检测到的目标位置、置信度以及质量. 质量复用objectness/confidence.</p><p>目标包含FIRE类</p></td><td class="confluenceTd">是</td><td class="confluenceTd">--</td><td class="confluenceTd" colspan="1">0.7</td><td class="confluenceTd" colspan="1"><ul><li>YoloDetectionV2<ul><li>BaseModel<ul><li>PrivateContext</li><li>AlgoAPI</li></ul></li></ul></li></ul></td></tr><tr><td class="confluenceTd" colspan="1"><p>FIRE_DETECTOR_X</p><p>火焰检测器加强版</p></td><td class="confluenceTd" colspan="1">仅支持多模型导入, 即火焰检测模型+火焰判别模型+跟踪模型导入, 不使用跟踪模型时, 传入string类型&ldquo;&rdquo; (空字符串).</td><td class="confluenceTd" colspan="1"><p>单帧推理</p><p>输入TvaiImage格式图像</p><p>输出VecObjBBox结构数据, 包含单个画面中检测到的目标位置、置信度以及质量. 质量复用objectness/confidence.</p><p>目标包含FIRE类</p></td><td class="confluenceTd" colspan="1">是</td><td class="confluenceTd" colspan="1">--</td><td class="confluenceTd" colspan="1">0.2</td><td class="confluenceTd" colspan="1"><ul><li>CascadeDetection<ul><li><p>AlgoAPI</p></li></ul></li></ul><ol><li>YoloDetectionV2</li><li>BinaryClassification</li></ol></td></tr><tr><td class="confluenceTd" colspan="1"><p>WATER_DETECTOR</p><p>积水检测器</p></td><td class="confluenceTd" colspan="1">仅支持单模型导入</td><td class="confluenceTd" colspan="1"><p>单帧推理</p><p>需要通过set_param函数设定警戒区域</p><p>输入TvaiImage格式图像</p><p>输出VecObjBBox结构数据, 对应每个警戒区域的积水程度(积水像素占比). confidence/objectness数值一致, 表示积水程度.</p><p>目标包含WATER_PUDDLE类</p></td><td class="confluenceTd" colspan="1">是</td><td class="confluenceTd" colspan="1">--</td><td class="confluenceTd" colspan="1">0.5</td><td class="confluenceTd" colspan="1"><ul><li>PSPNetWaterSegment<ul><li>BaseModel<ul><li>PrivateContext</li><li>AlgoAPI</li></ul></li></ul></li></ul></td></tr><tr><td class="confluenceTd" colspan="1"><p>PED_FALL_DETECTOR</p><p>行人摔倒检测</p></td><td class="confluenceTd" colspan="1"><p>支持单模型导入</p><p>支持单模型+跟踪模型导入</p></td><td class="confluenceTd" colspan="1"><p>单帧推理</p><p>输入TvaiImage格式图像</p><p>输出VecObjBBox结构数据, 包含单个画面中检测到的目标位置、置信度以及质量. 质量复用objectness/confidence.</p><p>目标包含PEDESTRIAN_FALL类</p></td><td class="confluenceTd" colspan="1">是</td><td class="confluenceTd" colspan="1">--</td><td class="confluenceTd" colspan="1">0.5</td><td class="confluenceTd" colspan="1"><ul><li>YoloDetectionV2<ul><li>BaseModel<ul><li>PrivateContext</li><li>AlgoAPI</li></ul></li></ul></li></ul></td></tr><tr><td class="confluenceTd" colspan="1"><p>SAFETY_HAT_DETECTOR</p><p>安全帽检测</p></td><td class="confluenceTd" colspan="1"><p>支持单模型导入</p><p>支持单模型+跟踪模型导入</p></td><td class="confluenceTd" colspan="1"><p>单帧推理</p><p>输入TvaiImage格式图像</p><p>输出VecObjBBox结构数据, 包含单个画面中检测到的目标位置、置信度以及质量. 质量复用objectness/confidence.</p><p>目标包含PED_HEAD、PED_SAFETY_HAT类</p></td><td class="confluenceTd" colspan="1">是</td><td class="confluenceTd" colspan="1">--</td><td class="confluenceTd" colspan="1">0.5</td><td class="confluenceTd" colspan="1"><ul><li>YoloDetectionV2<ul><li>BaseModel<ul><li>PrivateContext</li><li>AlgoAPI</li></ul></li></ul></li></ul></td></tr><tr><td class="confluenceTd" colspan="1"><p>TRASH_BAG_DETECTOR</p><p>垃圾袋检测</p></td><td class="confluenceTd" colspan="1"><p>支持单模型导入</p><p>支持单模型+跟踪模型导入[不建议使用]</p></td><td class="confluenceTd" colspan="1"><p>单帧推理</p><p>输入TvaiImage格式图像</p><p>输出VecObjBBox结构数据, 包含单个画面中检测到的目标位置、置信度以及质量. 质量复用objectness/confidence.</p><p>目标包含TRASH_BAG类</p></td><td class="confluenceTd" colspan="1">是</td><td class="confluenceTd" colspan="1">--</td><td class="confluenceTd" colspan="1">0.3</td><td class="confluenceTd" colspan="1"><ul><li>YoloDetectionV2<ul><li>BaseModel<ul><li>PrivateContext</li><li>AlgoAPI</li></ul></li></ul></li></ul></td></tr><tr><td class="confluenceTd" colspan="1"><p>BANNER_DETECTOR</p><p>横幅检测</p></td><td class="confluenceTd" colspan="1"><p>支持单模型导入</p><p>支持单模型+跟踪模型导入[不建议使用]</p></td><td class="confluenceTd" colspan="1"><p>单帧推理</p><p>输入TvaiImage格式图像</p><p>输出VecObjBBox结构数据, 包含单个画面中检测到的目标位置、置信度以及质量. 质量复用objectness/confidence.</p><p>目标包含BANNER类</p></td><td class="confluenceTd" colspan="1">是</td><td class="confluenceTd" colspan="1">--</td><td class="confluenceTd" colspan="1">0.5</td><td class="confluenceTd" colspan="1"><ul><li>YoloDetectionV2<ul><li>BaseModel<ul><li>PrivateContext</li><li>AlgoAPI</li></ul></li></ul></li></ul></td></tr><tr><td class="confluenceTd" colspan="1"><p>NONCAR_DETECTOR</p><p>非机动车检测</p></td><td class="confluenceTd" colspan="1"><p>支持单模型导入</p><p>支持单模型+跟踪模型导入</p></td><td class="confluenceTd" colspan="1"><p>单帧推理</p><p>输入TvaiImage格式图像</p><p>输出VecObjBBox结构数据, 包含单个画面中检测到的目标位置、置信度以及质量. 质量复用objectness/confidence.</p><p>目标包含BYCYCLE、EBYCYCLE类</p></td><td class="confluenceTd" colspan="1">是</td><td class="confluenceTd" colspan="1">--</td><td class="confluenceTd" colspan="1">0.6</td><td class="confluenceTd" colspan="1"><ul><li>YoloDetectionV2<ul><li>BaseModel<ul><li>PrivateContext</li><li>AlgoAPI</li></ul></li></ul></li></ul></td></tr><tr><td class="confluenceTd" colspan="1"><p>ACTION_CLASSIFIER</p><p>行为识别打斗检测器</p></td><td class="confluenceTd" colspan="1">仅支持单模型导入</td><td class="confluenceTd" colspan="1"><p>多帧推理(8帧), 针对整个画面进行判断, 或通过set_param函数设定检测区域, 仅能支持一个矩形检测区域</p><p>输入BatchImageIN即包含TvaiImage结构的vector序列图像</p><p>输出VecObjBBox结构数据, 如果存在打架行为, 则返回整个画面框或检测区域的的概率</p></td><td class="confluenceTd" colspan="1">否</td><td class="confluenceTd" colspan="1">--</td><td class="confluenceTd" colspan="1">0.8</td><td class="confluenceTd" colspan="1"><ul><li>TSNActionClassify<ul><li>BaseModel<ul><li>PrivateContext</li><li>AlgoAPI</li></ul></li></ul></li></ul></td></tr><tr><td class="confluenceTd" colspan="1"><p>ACTION_CLASSIFIER</p><p>行为识别打斗检测器</p></td><td class="confluenceTd" colspan="1">仅支持单模型导入</td><td class="confluenceTd" colspan="1"><p>多帧推理(8帧), 依赖人体检测器, 对多人聚集的位置进行判断</p><p>输入BatchImageIN结构数据, 即包含TvaiImage结构的vector序列图像</p><p>输入BatchBBoxIN结构数据, 即前序人体检测器给出的人体位置信息. 每个画面的人体结果以VecObjBBox结构存储.</p><p>输出VecObjBBox结构数据, 对多人聚集的位置进行判断, 如果存在打斗行为则输出位置、打斗置信度.</p></td><td class="confluenceTd" colspan="1">否</td><td class="confluenceTd" colspan="1">是, 仅处理CLS_TYPE::PEDESTRIAN的BBox数据, 其余类型自动跳过</td><td class="confluenceTd" colspan="1">0.8</td><td class="confluenceTd" colspan="1"><ul><li>TSNActionClassify<ul><li>BaseModel<ul><li>PrivateContext</li><li>AlgoAPI</li></ul></li></ul></li></ul></td></tr><tr><td class="confluenceTd" colspan="1"><p>MOD_DETECTOR</p><p>高空抛物检测器</p><p>移动物体检测器</p></td><td class="confluenceTd" colspan="1">仅支持单模型导入</td><td class="confluenceTd" colspan="1"><p>多帧推理(2帧)</p><p>输入BatchImageIN即包含TvaiImage结构的vector序列图像</p><p>输出VecObjBBox结构数据, 如果存在移动目标, 则返回目标框, 无概率输出</p></td><td class="confluenceTd" colspan="1">否</td><td class="confluenceTd" colspan="1">--</td><td class="confluenceTd" colspan="1">0.4</td><td class="confluenceTd" colspan="1"><ul><li>UNet2DShiftSegment<ul><li>BaseModel<ul><li>PrivateContext</li><li>AlgoAPI</li></ul></li></ul></li></ul></td></tr></tbody></table>

重要Class介绍:
<table class="wrapped relative-table confluenceTable"><colgroup><col style="width: 7.33793%;" /><col style="width: 43.9185%;" /><col style="width: 48.7436%;" /></colgroup><tbody><tr><th class="confluenceTh">对象</th><th class="confluenceTh">函数</th><th class="confluenceTh">功能</th></tr><tr><td class="confluenceTd"><p>BaseModel</p><p>原则:</p><p>涵盖大部分任务需求, 简化上层功能, 剥离mlu处理部分的代码</p></td><td class="confluenceTd"><p>RET_CODE <span style="color: #800080;"><strong>base_init</strong></span>(const std::string &amp;modelpath, BASE_CONFIG config)</p></td><td class="confluenceTd"><ul><li>模型加载, edk<span style="letter-spacing: 0.0px;">::</span><span style="letter-spacing: 0.0px;">EasyInfer初始化</span></li><li>模型输入输出格式配置(如RGB/YUVNV21/UINT8/NHWC等)</li><li><p>edk::MluResizeConvertOp初始化, 用于图像scale和crop</p></li><li>mlu上输入输出的内存开辟</li></ul></td></tr><tr><td class="confluenceTd"><br /></td><td class="confluenceTd"><p>RET_CODE <span style="color: #800080;"><strong>general_preprocess_yuv_on_mlu_phyAddr</strong></span>(TvaiImage &amp;tvimage, float &amp;aspect_ratio, float &amp;aX, float &amp;aY);</p></td><td class="confluenceTd"><ul><li><strong>单输入图像, 整个图像区域</strong></li><li>仅支持<strong>yuvnv21</strong>的<strong>物理地址</strong>数据</li><li>实现图像与模型输入的对接</li><li>速度最快: mlu上yuvnv21 uint8数据&nbsp;&rarr; 推理</li></ul></td></tr><tr><td class="confluenceTd" colspan="1"><br /></td><td class="confluenceTd" colspan="1"><p>RET_CODE <span style="color: #800080;"><strong>general_preprocess_yuv_on_mlu</strong></span>(TvaiImage &amp;tvimage, float &amp;aspect_ratio, float &amp;aX, float &amp;aY);</p></td><td class="confluenceTd" colspan="1"><ul><li><strong>单输入图像, 整个图像区域</strong></li><li>仅支持<strong>yuvnv21</strong>的<strong>虚拟址</strong>数据</li><li>实现图像与模型输入的对接</li><li>速度快: cpu上yuvnv21 uint8数据&nbsp;&rarr; mlu上yuvnv21 uint8数据&nbsp;&rarr; 推理</li></ul></td></tr><tr><td class="confluenceTd" colspan="1"><br /></td><td class="confluenceTd" colspan="1"><p>RET_CODE <span style="color: #800080;"><strong>general_preprocess_bgr_on_cpu</strong></span>(TvaiImage &amp;tvimage, float &amp;aspect_ratio, float &amp;aX, float &amp;aY);</p></td><td class="confluenceTd" colspan="1"><ul><li><strong>单输入图像, 整个图像区域</strong></li><li>仅支持<strong>rgb</strong>的<strong>虚拟地址</strong>数据</li><li>实现图像与模型输入的对接</li><li>速度较慢, 因为经过几次转换: cpu上rgb uint8数据&rarr;cpu上rgb float32数据&nbsp;&rarr; mlu上rgb float32数据&nbsp;&rarr; mlu上rgb uint8数据 &rarr; 推理</li></ul></td></tr><tr><td class="confluenceTd" colspan="1"><br /></td><td class="confluenceTd" colspan="1"><p>RET_CODE <span style="color: #800080;"><strong>general_preprocess_yuv_on_mlu_phyAddr</strong></span>(TvaiImage &amp;tvimage, TvaiRect roiRect, float &amp;aspect_ratio, float &amp;aX, float &amp;aY);</p></td><td class="confluenceTd" colspan="1"><ul><li><strong>单输入图像, 单roi区域</strong></li><li>仅支持<strong>yuvnv21的物理地址</strong>数据</li><li>实现图像与模型输入的对接</li></ul></td></tr><tr><td class="confluenceTd" colspan="1"><br /></td><td class="confluenceTd" colspan="1"><p>RET_CODE <span style="color: #800080;"><strong>general_preprocess_yuv_on_mlu</strong></span>(TvaiImage &amp;tvimage, TvaiRect roiRect, float &amp;aspect_ratio, float &amp;aX, float &amp;aY);</p></td><td class="confluenceTd" colspan="1"><ul><li><strong>单输入图像, 单roi区域</strong></li><li>仅支持<strong>yuvnv21的虚拟地址</strong>数据</li><li>实现图像与模型输入的对接</li></ul></td></tr><tr><td class="confluenceTd" colspan="1"><br /></td><td class="confluenceTd" colspan="1"><p>RET_CODE <span style="color: #800080;"><strong>general_preprocess_infer_bgr_on_cpu</strong></span>(TvaiImage &amp;tvimage, std::vector&lt;TvaiRect&gt;&amp; roiRects, std::vector&lt;float*&gt; &amp;model_output, std::vector&lt;float&gt; &amp;aspect_ratios);</p></td><td class="confluenceTd" colspan="1"><ul><li><strong>单输入图像, 多roi区域</strong></li><li>仅支持<strong>rgb的虚拟地址</strong></li><li>实现图像与模型输入的对接</li></ul><p>不同于mlu上的前处理, mlu上函数使用受限, 但处理速度快, 因此多roi区域的循环处理结构包在外侧. 而cpu上处理速度慢, 只能在内部实现.</p></td></tr><tr><td class="confluenceTd" colspan="1"><br /></td><td class="confluenceTd" colspan="1"><p>RET_CODE <span style="color: #800080;"><strong>general_preprocess_infer_bgr_on_cpu</strong></span>(TvaiImage &amp;tvimage, VecObjBBox&amp; bboxes, std::vector&lt;float*&gt; &amp;model_output, std::vector&lt;float&gt; &amp;aspect_ratios, std::vector&lt;CLS_TYPE&gt; &amp;valid_class);</p></td><td class="confluenceTd" colspan="1"><ul><li><strong>单输入图像, 多roi区域</strong></li><li>仅支持<strong>rgb的虚拟地址</strong></li><li>实现图像与模型输入的对接</li></ul></td></tr><tr><td class="confluenceTd" colspan="1"><br /></td><td class="confluenceTd" colspan="1"><p><span style="color: #000000;">RET_CODE <span style="color: #800080;"><strong>general_batch_preprocess_yuv_on_mlu</strong></span>(TvaiImage &amp;tvimage, VecObjBBox&amp; bboxes,std::vector&lt;float&gt; &amp;batch_aspect_ratio, int offset);</span></p></td><td class="confluenceTd" colspan="1"><ul><li><strong>单输入图像, 多个roi区域</strong></li><li><strong>仅支持yuvnv21的物理/虚拟地址</strong></li><li>实现图像与模型输入的对接</li></ul></td></tr><tr><td class="confluenceTd" colspan="1"><br /></td><td class="confluenceTd" colspan="1"><p><span style="color: #000000;">RET_CODE <span style="color: #800080;"><strong>general_batch_preprocess_yuv_on_mlu</strong></span>(BatchImageIN &amp;batch_tvimage, std::vector&lt;TvaiRect&gt; &amp;batch_roiRect,std::vector&lt;float&gt; &amp;batch_aspect_ratio);</span></p></td><td class="confluenceTd" colspan="1"><ul><li><strong>多输入图像, 多roi区域,&nbsp;每个图像仅对应一个roi区域</strong></li><li><strong>仅支持yuvnv21的物理/虚拟地址</strong></li><li>实现图像与模型输入的对接</li></ul></td></tr><tr><td class="confluenceTd" colspan="1"><br /></td><td class="confluenceTd" colspan="1"><p><span style="color: #000000;">float* <span style="color: #800080;"><strong>general_mlu_infer</strong></span>();</span></p></td><td class="confluenceTd" colspan="1"><ul><li>模型通用推理</li><li>将通用前处理得到的输入进行推理, 得到mlu上的输出</li></ul></td></tr></tbody></table>


每个方法说明:
```
人脸检测
* AlgoAPIName::FACE_DETECTOR:
    - RET_CODE init(const std::string &modelpath)
        + 输入人脸检测模型路径, 不带跟踪模块
    - RET_CODE init(const std::string &modelpath, const std::string &trackmodelpath)
        + 输入人脸检测模型路径以及通用跟踪模型路径, 在给定跟踪模型的情况下, 返回检测结果时, track_id将做标识.
        + modelpath 采用retinaface模型检测人脸
        + trackmodelpath 目前采用cambricon提供的通用特征提取器
    - RET_CODE init(std::map<InitParam, std::string> &modelpath)
        + 新的初始化方式, 通过枚举的方式入参, 避免接口过多的问题
    - RET_CODE run(TvaiImage tvimage, VecObjBBox &bboxes)
        + 输入RGB/BGR/NV21/NV12类型图像数据, 以及空的变量bboxes, 检测到的人脸数据将返回到变量bboxes中.
        + BBox.rect是图像坐标系下人脸框的位置
        + BBox.objectness 人脸概率, (0,1)之间, 人脸检测下objectness=confidence
        + BBox.confidence 人脸概率, (0,1)之间
        + BBox.Pts 存储了人脸五点关键坐标
        + BBox.feat 用于存储后续的特征, 需要关注指针的释放
        + BBox.objtype = CLS_TYPE::FACE
    - RET_CODE set_param(...)
        + 设置人脸检测参数, 与先前一致
```

```
人脸特征提取
* AlgoAPIName::FACE_EXTRACTOR:
    - 初始化方法同上
    - RET_CODE run(TvaiImage tvimage, VecObjBBox &bboxes)
        + 输入RGB/BGR/NV21/NV12类型图像数据, 以及先前检测到的人脸框, 根据人脸框返回人脸特征, 并将特征数据指针返回给BBox.feat
        + 仅当BBox.objtype = CLS_TYPE::FACE 时提取特征. 如果是其它类别, 则该检测框将被跳过.
        + BBox.feat 接收抽取出的特征指针
```

```
人车非检测
* AlgoAPIName::GENERAL_DETECTOR:
    - RET_CODE init(const std::string &modelpath)
        + 输入通用物体检测模型路径, 不带跟踪模块
    - RET_CODE init(const std::string &modelpath, const std::string &trackmodelpath)
        + 输入通用物体检测模型路径以及通用跟踪模型路径, 在给定跟踪模型的情况下, 返回检测结果时, track_id将做标识.
        + modelpath 采用yolov5s-conv模型检测
        + trackmodelpath 目前采用cambricon提供的通用特征提取器
    - RET_CODE init(std::map<InitParam, std::string> &modelpath)
        + 新的初始化方式, 通过枚举的方式入参, 避免接口过多的问题
    - RET_CODE run(TvaiImage tvimage, VecObjBBox &bboxes)
        + 输入RGB/BGR/NV21/NV12类型图像数据, 以及空的变量bboxes, 检测到的人车非数据将返回到变量bboxes中.
        + BBox.rect是图像坐标系下目标框的位置
        + BBox.objectness 是目标物体的概率, 算法内部使用, 无需关注
        + BBox.confidence 具体到某一类别的概率, 实际检测目标的概率/得分, (0,1)之间
        + BBox.Pts 可用于存储人体骨架检测的信息
        + BBox.feat 可用于存储物体的特征, 人车非阶段目前没有用, 后续Re-ID行人识别会用到.
        + BBox.objtype 由set_output_cls_order接口设定的类别.
    - RET_CODE set_param(...)
        + 设置的检测参数, 与先前一致, 与人脸检测一样
    - RET_CODE set_output_cls_order(CLS_TYPE* output_clss, int len_output_clss)
        + 设置模型的输出与实际需要类别的映射关系, 比如:yolov5s-conv-9_736x416_mlu220_bs1c1_fp16.cambricon 输出的类别是: [ 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor' ], 为了与人车非对应, 那么设定 CLS_TYPE yolov5s_conv_9[] = {CLS_TYPE::PEDESTRIAN, CLS_TYPE::NONCAR, CLS_TYPE::CAR, CLS_TYPE::CAR, CLS_TYPE::CAR, CLS_TYPE::NONCAR, CLS_TYPE::NONCAR, CLS_TYPE::CAR, CLS_TYPE::NONCAR}; 这样比较灵活, 不同模型(同框架)可以复用一个接口, 同时, 便于模型训练, 不用替换现有训练标签.
```

```
骨架检测
* AlgoAPIName::SKELETON_DETECTOR:
    - RET_CODE run(TvaiImage tvimage, VecObjBBox &bboxes)
        + 输入RGB/BGR/NV21/NV12类型图像数据, 以及先前检测到的人体框, 根据人脸框返回人脸特征, 并将特征数据指针返回给BBox.feat
        + 仅当BBox.objtype = CLS_TYPE::PEDESTRIAN 时检测骨架点位. 如果是其它类别, 则该检测框将被跳过.
        + BBox.Pts 存储人体骨架检测的信息(共15个点), 可用于将来摔倒检查及姿态检测业务.
```

```
打斗行为检测, 需要输入序列图像, 因此仅支持nv21/12格式, 不支持RGB/BGR
* AlgoAPIName::ACTION_CLASSIFIER:
    - RET_CODE run(BatchImageIN &batch_tvimages, BatchBBoxIN &batch_bboxes, VecObjBBox &bboxes)
        + 本函数依赖检测模型的到的检测结果, 对画面中人员聚集的区域进行判断, 判断是否存在打斗的概率(只支持NV21/12格式, 不支持RGB/BGR).
        + batch_tvimages 输入时序图像(可以非连续), vector of TvaiImage, 只支持NV21/12格式, 不支持RGB/BGR
        + batch_bboxes 输入检测模型的到的检测结果, 每个图像对应一组vector of BBox
        + bboxes 输出分类结果, 打斗的区域以及打斗的概率
    - RET_CODE run(BatchImageIN &batch_tvimages, VecObjBBox &bboxes)
        + 本函数将整个画面当成整体, 判断是否存在打斗的概率, (只支持NV21/12格式, 不支持RGB/BGR).
        + batch_tvimages 输入时序图像(可以非连续), vector of TvaiImage, 只支持NV21/12格式, 不支持RGB/BGR
        + bboxes 输出分类结果即打斗的概率, vector中仅一个结果, 即代表整个画面
```

```
高空抛物, 需要输入序列图像, 因此仅支持nv21/12格式, 不支持RGB/BGR
* AlgoAPIName::MOD_DETECTOR:
    - RET_CODE run(BatchImageIN &batch_tvimages, VecObjBBox &bboxes)
        + 本函数将整个画面当成整体, 标记高空抛物物体所在, 即快速移动物体(只支持NV21/12格式, 不支持RGB/BGR).
        + batch_tvimages 输入时序图像(可以非连续), vector of TvaiImage, 只支持NV21/12格式, 不支持RGB/BGR
        + bboxes 输出一系列BBox, 每个BBox即找到的疑似下落(移动)物体, bboxes中BBox.objtype = CLS_TYPE::UNKNOWN
```

```
火焰检测
* AlgoAPIName::FIRE_DETECTOR:
    - RET_CODE run(TvaiImage &tvimage, VecObjBBox &bboxes)
        + 本函数将检测画面中的火焰位置, 并通过bboxes返回结果, bboxes中BBox.objtype = CLS_TYPE::FIRE
```

```
积水检测
* AlgoAPIName::FIRE_DETECTOR:
    - RET_CODE run(TvaiImage &tvimage, VecObjBBox &bboxes)
        + 本函数将检测画面中的积水区域, 并通过bboxes返回结果, bboxes中BBox.objtype = CLS_TYPE::WATER_PUDDLE
        + 由于积水不规则, 因此内部实现通过分割算法实现, 而非检测算法. 
```
        
## 1.REFERENCE
See https://ushare.ucloudadmin.com/pages/viewpage.action?pageId=82384346 for more detail.

## 2.Compile
### mlu270
```
cd build
cmake ..
make -j12
```

### mlu220
需要修改```cmake/cross-compile.cmake```中的交叉编译工具链地址, 以及```CMakeLists.txt```中各个第三方库文件的位置.
当前设置:
```
/opt/make_tool/aarch64/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu/bin
```
然后运行脚本文件自动编译:
```
./run_mlu220.sh
```

## 3.LIB LINK
如果采用x86系统, 则使用如下命令安装即可.
```
sudo apt-get install libgflags-dev libgoogle-glog-dev cmake
sudo apt-get install libfreetype6 ttf-wqy-zenhei libsdl2-dev curl libcurl4-openssl-dev
```
原先采用apt-get install方式安装libopencv-dev, 但是, 由于清华源提供版本是2.4.9与mlu220上的3.4.6版本不一致, 因此采用github下载源码进行编译.
如果采用交叉编译, 则参考: http://forum.cambricon.com/index.php?m=content&c=index&a=show&catid=47&id=357, 并按照下面的步骤进行逐个编译.

### 3.1 ffmpeg
download:  http://ffmpeg.org/download.html latest
```
./configure --prefix=/usr/local/mlu220_1.7.0_3rdparty_static/ffmpeg/ \
--cross-prefix=/opt/make_tool/aarch64/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu- --enable-cross-compile --arch=arm64 --target-os=linux --enable-static --enable-gpl --enable-nonfree  --disable-debug --disable-doc  --enable-pic --enable-ffmpeg --enable-decoder=h264
make -j42 install
```
编译出的库将install到/usr/local/mlu220_1.7.0_3rdparty_static/ffmpeg/

如果需要在嵌入式上使用opencv的视频流读取, 则ffmpeg不能太新, 否则opencv编译失败, 因为ffmpeg更新了某个结构体:
(https://forum.opencv.org/t/error-avstream-aka-struct-avstream-has-no-member-named-codec/3506/2)
(https://github.com/opencv/opencv/issues/20147)
```
git clone https://github.com/FFmpeg/FFmpeg.git
git checkout n4.3.4
./configure --prefix=/usr/local/mlu220_1.7.0_3rdparty_static/ffmpeg_n4.3.4/ \
--cross-prefix=/opt/make_tool/aarch64/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu- --enable-cross-compile --arch=arm64 --target-os=linux --enable-static --enable-gpl --enable-nonfree  --disable-debug --disable-doc  --enable-pic --enable-ffmpeg --enable-decoder=h264
make -j42 install
```
编译出的库将install到/usr/local/mlu220_1.7.0_3rdparty_static/ffmpeg_n4.3.4/中


### 3.2 glog
download: https://github.com/google/glog/releases/tag/v0.4.0
```
cmake .. -DCMAKE_TOOLCHAIN_FILE=/opt/make_tool/aarch64/cross-compile.cmake -DCMAKE_INSTALL_PREFIX=/usr/local/mlu220_1.7.0_3rdparty_static/glog/ \
-DBUILD_SHARED_LIBS=OFF
make clean
make -j42 install
```
编译出的库将install到/usr/local/mlu220_1.7.0_3rdparty_static/glog/, 其中cross-compile.cmake即${PROJECT_DIR}/cmake/cross-compile.cmake.

### 3.3 gflags
download: https://github.com/gflags/gflags/releases
```
cmake .. -DCMAKE_TOOLCHAIN_FILE=/opt/make_tool/aarch64/cross-compile.cmake \
    -DCMAKE_INSTALL_PREFIX=/usr/local/mlu220_1.7.0_3rdparty_static/gflags/ \
    -DCMAKE_CXX_FLAGS="-fPIC" -DBUILD_SHARED_LIBS=on \
    -DBUILD_STATIC_LIBS=on \
    -DBUILD_gflags_LIB=on \
    -DINSTALL_STATIC_LIBS=on \
    -DINSTALL_SHARED_LIBS=on \
    -DREGISTER_INSTALL_PREFIX=off
make clean
make -j42 install
```
编译出的库将install到/usr/local/mlu220_1.7.0_3rdparty_static/gflags/

### 3.4 opencv
download: https://opencv.org/releases/  v3.4.6
```
cd build_aarch64/
cmake -DCMAKE_TOOLCHAIN_FILE=../platforms/linux/aarch64-gnu.toolchain.cmake \
-DCMAKE_INSTALL_PREFIX=/usr/local/mlu220_1.7.0_3rdparty_static/opencv/ \
-DBUILD_SHARED_LIBS=OFF -D BUILD_PNG=ON -D BUILD_JASPER=ON -D BUILD_JPEG=ON -D BUILD_TIFF=ON -D BUILD_ZLIB=ON -D WITH_JPEG=ON -D WITH_PNG=ON -D WITH_JASPER=ON -D WITH_TIFF=ON -DWITH_1394=OFF -DWITH_GTK=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -DWITH_FFMPEG=OFF ..
# -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-master/modules
make -j42 install
```
如果使用contrib module, 会需要额外编译freetype库, 暂时搁置.
编译出的库将install到/usr/local/mlu220_1.7.0_3rdparty_static/opencv/

如果需要在嵌入式上使用ffmpeg进行编解码读取视频流, 则需要进行如下设置:

启用OPENCV_ENABLE_PKG_CONFIG, 否则无法通过pkg-config找到ffmeg库; 在交叉编译工具链中加入ffmpeg的pkgconfig
(https://blog.csdn.net/woainannanta/article/details/78260419)
```
/opt/make_tool/aarch64/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu/lib/pkgconfig
```

```
cd build_aarch64/
cmake -DCMAKE_TOOLCHAIN_FILE=../platforms/linux/aarch64-gnu.toolchain.cmake \
-DCMAKE_INSTALL_PREFIX=/usr/local/mlu220_1.7.0_3rdparty_static/opencv_ffmpeg/ \
-DBUILD_SHARED_LIBS=OFF -D BUILD_PNG=ON -D BUILD_JASPER=ON -D BUILD_JPEG=ON -D BUILD_TIFF=ON -D BUILD_ZLIB=ON -D WITH_JPEG=ON -D WITH_PNG=ON -D WITH_JASPER=ON -D WITH_TIFF=ON -DWITH_1394=OFF -DWITH_GTK=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -DWITH_FFMPEG=ON -DOPENCV_ENABLE_PKG_CONFIG=ON -DBUILD_opencv_apps=OFF ..
```
编译apps会报找不到ffmpeg的动态库, 估计是CMakeLists.txt配置的问题, 没有深究.

对于x86系统, 同样进行手动编译:
```
cd build_x86
cmake \
-DCMAKE_INSTALL_PREFIX=/usr/local/mlu270_3rdparty/opencv/ \
-DWITH_CUDA=OFF \
-DBUILD_EXAMPLES=ON \
-DBUILD_SHARED_LIBS=ON -D BUILD_PNG=ON -D BUILD_JASPER=ON -D BUILD_JPEG=ON -D BUILD_TIFF=ON -D BUILD_ZLIB=ON -D WITH_JPEG=ON -D WITH_PNG=ON -D WITH_JASPER=ON -D WITH_TIFF=ON -DWITH_1394=ON -DWITH_GTK=ON -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=ON -DWITH_FFMPEG=ON -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.6/modules ..
```
使用extra modules时, 需要git下载opencv_contrib, 同时
```
git clone https://github.com/opencv/opencv_contrib.git
git checkout 3.4.6
```


#### 关于zlib的问题
由于opencv静态编译出来的静态库xxx.a依然需要依赖libzlib.a, 因为编译的时候采用了隐式接口(visibility=hidden), 这就导致后续libai_core.so<-opencv.a<-libzlib.a, 所以, 如果要消除libzlib.a的依赖, 就需要:
1) libzlib.a编译时, 采用visibility=default, 首先将compress和uncompress接口暴露出来.
2) libai_core.so在链接libzlib.a时, 载入所有的未使用的symbol: 
```
 -L. -Wl,--whole-archive libzlib.a -Wl,--no-whole-archive
```

### 3.5 easydk
download: https://github.com/Cambricon/easydk
寒武纪官方提供的简化推理的库
由于内容比较多, 详见: https://ushare.ucloudadmin.com/pages/viewpage.action?pageId=79059385
注意, 需要修改CMakeList.txt, 将easydk的生成变为静态库.
```
export CNSTREAM_MLU220EDGE_DIR=/project/workspace/samples/CNStream220edge/
>> cd ${CNSTREAM_MLU220EDGE_DIR}
>> mkdir build
>> cd build
交叉编译:
>> cmake ${CNSTREAM_MLU220EDGE_DIR} -DCMAKE_TOOLCHAIN_FILE=${CNSTREAM_MLU220EDGE_DIR}/cmake/cross-compile.cmake  -DCNIS_WITH_CURL=OFF -Dbuild_display=OFF -DMLU=MLU220EDGE -Dbuild_samples=OFF
```





