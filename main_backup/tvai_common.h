/******************************************************************************
*Copyright(C),2021-2022, TVT. All rights reserved.
*Description: 通用数据类型和接口定义
******************************************************************************/
#ifndef TVAI_COMMON_H
#define TVAI_COMMON_H

#ifdef __cplusplus
extern "C" {
#endif

#ifndef NULL
#define NULL (0)
#endif

#ifndef IN
    #define IN
#endif
#ifndef OUT
    #define OUT
#endif

// 错误码
typedef enum {
    TVAI_SUCCESS                                   = 0,    /* 操作成功 */
    TVAI_FAILED                                    = 1,    /* 未知异常 */

    /* 通用错误 */
    TVAI_ERR_COMMON_INVALID_PARA                   = 100, /* 无效的入参 */
    TVAI_ERR_COMMON_NOT_REG_CALLBACK               = 101, /* 未注册回调函数   */
    TVAI_ERR_COMMON_NOT_INIT                       = 102, /* 未初始化   */
    TVAI_ERR_COMMON_NOT_SUPPORTED                  = 103, /* 不支持的操作   */
    TVAI_ERR_COMMON_BUSY                           = 104, /* 系统忙，过载 */
    TVAI_ERR_COMMON_OUT_OF_MEMORY                  = 105, /* 内存资源不足 */
    TVAI_ERR_COMMON_TIME_OUT                       = 106, /* 处理超时 */
    TVAI_ERR_COMMON_FEATURE_VERSION_IMCOMPATIBLE   = 107, /* 特征数据版本不兼容 */

    /* 算法库内部错误 */
    TVAI_ERR_ALGORITHM_MODEL_FILE_INVALID          = 200,  /* 算法模型文件异常 */
    TVAI_ERR_ALGORITHM_CONFIG_FILE_INVALID         = 201,  /* 算法配置文件异常 */
    TVAI_ERR_ALGORITHM_SAVE_DATA_FAILED            = 202,  /* 算法持久化操作失败 */

    /* 特征库操作错误 */
    TVAI_ERR_FEATURE_REPO_DOES_EXIST               = 300, /* 特征库已存在 */
    TVAI_ERR_FEATURE_REPO_NOT_EXIST                = 301, /* 特征库不存在 */
    TVAI_ERR_FEATURE_REPO_REACHED_MAX_CAPACITY     = 302, /* 特征库已达到最大规格 */

    /* 特征操作错误 */
    TVAI_ERR_FEATURE_ID_DOES_EXIST                 = 400, /* 特征ID已存在 */
    TVAI_ERR_FEATURE_ID_NOT_EXIST                  = 401, /* 特征ID不存在 */
    TVAI_ERR_FEATURE_ID_REACHED_REPO_MAX_CAPACITY  = 402, /* 特征ID已达到单库最大规格 */
    TVAI_ERR_FEATURE_ID_REACHED_SYS_MAX_CAPACITY   = 403, /* 特征ID已达到系统最大规格 */
    TVAI_ERR_FEATURE_ID_TARGET_TYPE_NOT_MATCH      = 404, /* 特征对应的目标类型与库的目标类型不相同 */
    TVAI_ERR_FEATURE_ID_TOO_MUCH_TARGET            = 405, /* 录入特征时，图片存在多目标 */

    /* License错误 */
    TVAI_ERR_LICENSE_FILE_INVALID                  = 500, /* 无效的License文件 */
    TVAI_ERR_LICENSE_EXCEED_VALID_DATE             = 501, /* License文件已过期 */

    /* 视频流处理错误 */
    TVAI_ERR_VIDEO_OVERLOAD                        = 600, /* 视频帧过多，无法处理 */

    /* 图片流处理错误 */
    TVAI_ERR_IMAGE_OVERLOAD                        = 700, /* 图片帧过多，无法处理 */


    /* 智能业务处理错误 */
    TVAI_ERR_AITOOL_TOO_MUCH_TARGET                = 800, /* 搜索、比对，检测出图片中大于1个目标 */
    TVAI_ERR_AITOOL_NO_TARGET                      = 801, /* 搜索、比对、目标检测时，图片中没有目标 */

    /* 参数配置 */
    TVAI_ERR_CONFIG_EVENT_RULE_DOES_EXIST          = 900, /* 新增侦测事件规则，规则ID已存在 */
    TVAI_ERR_CONFIG_EVENT_RULE_NOT_EXIST           = 901, /* 删除/查询侦测事件规则，规则ID不存在 */
    TVAI_ERR_CONFIG_EVENT_RULE_TOO_MUCH            = 902, /* 规则过多，超过上限 */


    /* 检测，提取错误 */
    TVAI_ERR_EXTRACT_FEATURE_FAILED                = 1000, /* 提取特征失败 */
    TVAI_ERR_DETECT_NO_TARGET                      = 1001, /* 未检测到指定目标 */
    TVAI_ERR_EXTRACT_ATTR_FAILED                   = 1002  /* 提取结构属性失败 */

}TvaiErrorCode;

//日志级别定义
typedef enum
{
    TVAI_LOG_LEVEL_DEBUG = 1,                   /* Debug日志 */
    TVAI_LOG_LEVEL_RUN,                         /* 运行日志 */
    TVAI_LOG_LEVEL_WARN,                        /* 告警日志 */
    TVAI_LOG_LEVEL_ERROR                        /* 错误日志 */
}TvaiLogLevel;

// 目标类型
typedef enum {
    TVAI_TARGET_TYPE_UNDEFINED         = 0,     /* 未定义  */
    TVAI_TARGET_TYPE_FACE,                      /* 人脸  */
    TVAI_TARGET_TYPE_PED,                       /* 人体  */
    TVAI_TARGET_TYPE_VEHICLE,                   /* 车辆  */
    TVAI_TARGET_TYPE_NONMOTOR                   /* 非机动车 */
}TvaiTargetType;

// 图像格式
typedef enum {
    TVAI_IMAGE_FORMAT_NULL    = 0,              /* 格式为空 */
    TVAI_IMAGE_FORMAT_GRAY,                     /* 单通道灰度图像 */
    TVAI_IMAGE_FORMAT_NV12,                     /* YUV420SP_NV12：YYYYYYYY UV UV */
    TVAI_IMAGE_FORMAT_NV21,                     /* YVU420SP_NV21：YYYYYYYY VU VU */
    TVAI_IMAGE_FORMAT_RGB,                      /* 3通道，RGBRGBRGBRGB */
    TVAI_IMAGE_FORMAT_BGR,                      /* 3通道，BGRBGRBGRBGR */
    TVAI_IMAGE_FORMAT_I420,                     /* YUV420p_I420 ：YYYYYYYY UU VV */
    TVAI_IMAGE_FORMAT_YUV_PHY = 0x10            /* 以物理地址形式缓存的YUV帧数据 */
}TvaiImageFormat;

// 关联状态
typedef enum {
    TVAI_RELATION_UNKNOWN = 0,                 /* 未知 */
    TVAI_RELATION_MASTER,                      /* master (主) */
    TVAI_RELATION_SLAVE                        /* slave  (从) */
}TvaiRelation;

//规则类型定义
typedef enum {
    TVAI_EVENT_RULE_TYPE_AOI_BEGIN            = 0,      /* 区域类规则起始值，用于分界，实际并不使用 */
    TVAI_EVENT_RULE_TYPE_AOI_ENTERS,                    /* 区域进入,检测到目标从区域外进入指定区域 */
    TVAI_EVENT_RULE_TYPE_AOI_EXITS,                     /* 区域离开,检测到目标离开指定区域 */
    TVAI_EVENT_RULE_TYPE_AOI_INVADE,                    /* 区域入侵，区域内检测到目标（持续一定周期)。 */

    TVAI_EVENT_RULE_TYPE_TRIPWIRE_BEGIN       = 1000,   /* 单绊线类规则起始值，用于分界，实际并不使用 */
    TVAI_EVENT_TYPE_TRIPWIRE,                           /* 单绊线事件, 越过指定的警戒线 */

    TVAI_EVENT_RULE_TYPE_CUSTOM_BEGIN         = 100000  /* 自定义规则类型起始值,用于分界，实际并不使用 */
}TvaiEventDetectType;

//单绊线方向定义
typedef enum
{
    TVAI_LINE_CROSS_DIRECTION_ANY             = 0,     /* 任意方向绊线,用于规则配置时使用 */
    TVAI_LINE_CROSS_DIRECTION_LEFT_TO_RIGHT,           /* 从左到右绊线 */
    TVAI_LINE_CROSS_DIRECTION_RIGHT_TO_LEFT            /* 从右到左绊线 */
}TvaiLineCrossDirection;

//数据类型:图片or特征
typedef enum
{
    TVAI_DATA_TYPE_IMAGE             = 0,             /* 图片数据 */
    TVAI_DATA_TYPE_FEATURE                            /* 特征数据 */
}TvaiDataType;

//抓拍策略
typedef enum
{
    TVAI_CAPTURE_STRATEGY_BEST             = 0,      /* 选择质量分数最高的图 */
    TVAI_CAPTURE_STRATEGY_FIRST,                     /* 选择第一张图 */
    TVAI_CAPTURE_STRATEGY_LOOP                       /* 定时抓拍，选择时间间隔内质量分数最高的图 */
}TvaiCaptureStrategy;

//属性枚举定义
typedef enum
{
    /* 人脸结构属性 */
    TVAI_FACE_ATTR_AGE                            = 0,                   /* 年龄 */
    TVAI_FACE_ATTR_GENDER,                                               /* 性别 */
    TVAI_FACE_ATTR_GLASS,                                                /* 眼镜 */
    TVAI_FACE_ATTR_HAT   ,                                               /* 帽子 */
    TVAI_FACE_ATTR_MASK,                                                 /* 口罩 */
    TVAI_FACE_ATTR_BEARD,                                                /* 胡子 */
    TVAI_FACE_ATTR_EXPRESSION,                                           /* 表情   */
    TVAI_FACE_ATTR_PITCH,                                                /* 俯仰姿态 */
    TVAI_FACE_ATTR_YAW,                                                  /* 偏摆姿态 */
    TVAI_FACE_ATTR_ROLL,                                                 /* 翻滚姿态 */

    /* 人体结构属性 */
    TVAI_PED_ATTR_GENDER                         = 1000,                 /* 性别 */
    TVAI_PED_ATTR_AGE_STAGE,                                             /* 年龄段 */
    TVAI_PED_ATTR_BAG_TYPE,                                              /* 背包类型 */
    TVAI_PED_ATTR_FRONT_HOLD,                                            /* 正面是否抱东西或者小孩 */
    TVAI_PED_ATTR_GLASS,                                                 /* 眼镜类型 */
    TVAI_PED_ATTR_HAIR_COLOR,                                            /* 头发颜色 */
    TVAI_PED_ATTR_HAIR_LENGTH,                                           /* 头发长度 */
    TVAI_PED_ATTR_HEADWEAR,                                              /* 头戴类型 */
    TVAI_PED_ATTR_TROUSERS_COLOR,                                        /* 下衣颜色 */
    TVAI_PED_ATTR_TROUSERS_TYPE,                                         /* 下衣类型 */
    TVAI_PED_ATTR_TROUSERS_TEXTURE,                                      /* 下衣花案 */
    TVAI_PED_ATTR_COAT_COLOR,                                            /* 上衣颜色 */
    TVAI_PED_ATTR_COAT_TYPE,                                             /* 上衣类型 */
    TVAI_PED_ATTR_COAT_TEXTURE,                                          /* 上衣花案 */
    TVAI_PED_ATTR_MASK,                                                  /* 是否戴口罩 */
    TVAI_PED_ATTR_HAT_TYPE,                                              /* 帽子类型 */
    TVAI_PED_ATTR_ORIENTATION,                                           /* 朝向 */
    TVAI_PED_ATTR_OVERCOAT,                                              /* 是否穿外套 */
    TVAI_PED_ATTR_TROLLEY_CASE,                                          /* 是否携带拉杆箱 */
    TVAI_PED_ATTR_BEHAVIOR,                                              /* 行为动作 */
    TVAI_PED_ATTR_UNIFORM,                                               /* 制服 */
    TVAI_PED_ATTR_SHOES_TYPE,                                            /* 鞋子类型 */
    TVAI_PED_ATTR_SHOES_COLOR,                                           /* 鞋子颜色 */
    TVAI_PED_ATTR_UMBRELLA_COLOR,                                        /* 雨伞颜色 */

    /* 车辆结构属性 */
    TVAI_VEHICLE_ATTR_COLOR                      = 2000,                 /* 车辆颜色 */
    TVAI_VEHICLE_ATTR_CLASSIFY,                                          /* 车辆分类 */
    TVAI_VEHICLE_ATTR_BRAND,                                             /* 车辆品牌 */

    /* 非机动车结构属性 */
    TVAI_NONMOTOR_ATTR_COLOR                     = 3000                  /* 非机动车颜色 */

}TvaiTargetAttributeType;

//人脸属性取值定义
//人脸属性性别
typedef enum {
    TVAI_FACE_ATTR_GENDER_FEMALE                  = 0,                   /* 女 */
    TVAI_FACE_ATTR_GENDER_MALE                    = 1                    /* 男 */
}TvaiFaceAttrGender;

//人脸属性眼镜
typedef enum {
    TVAI_FACE_ATTR_GLASS_NO_GLASSES               = 0,                   /* 不戴眼镜 */
    TVAI_FACE_ATTR_GLASS_ORDINARY_GLASSES         = 1,                   /* 普通眼镜 */
    TVAI_FACE_ATTR_GLASS_SUNGLASSES               = 2,                   /* 太阳眼镜/墨镜       */
}TvaiFaceAttrGlasses;

//人脸属性帽子
typedef enum {
    TVAI_FACE_ATTR_HAT_NO_HAT                      = 0,                  /* 不戴帽子 */
    TVAI_FACE_ATTR_HAT_HAS_HAT                     = 1                   /* 戴帽子 */
}TvaiFaceAttrHat;

//人脸属性口罩
typedef enum {
    TVAI_FACE_ATTR_MASK_NO_MASK                    = 0,                  /* 不戴口罩 */
    TVAI_FACE_ATTR_MASK_HAS_MASK                   = 1                   /* 戴口罩 */
}TvaiFaceAttrMask;

//人脸属性胡子
typedef enum {
    TVAI_FACE_ATTR_MASK_NO_BEARD                    = 0,                 /* 没有胡子 */
    TVAI_FACE_ATTR_MASK_HAS_BEARD                   = 1                  /* 有胡子 */
}TvaiFaceAttrBeard;

//人脸属性表情
typedef enum {
    TVAI_FACE_ATTR_EXPRESSION_ANGRY                 = 0,                 /* 生气 */
    TVAI_FACE_ATTR_EXPRESSION_HAPPY                 = 1,                 /* 快乐 */
    TVAI_FACE_ATTR_EXPRESSION_SORROW                = 2,                 /* 悲伤 */
    TVAI_FACE_ATTR_EXPRESSION_CALM                  = 3,                 /* 平静 */
    TVAI_FACE_ATTR_EXPRESSION_SUPPRISED             = 4,                 /* 惊讶 */
    TVAI_FACE_ATTR_EXPRESSION_SCARED                = 5,                 /* 害怕 */
    TVAI_FACE_ATTR_EXPRESSION_DISGUST               = 6,                 /* 厌恶 */
    // TVAI_FACE_ATTR_EXPRESSION_DISGUST               = 7,                 /* 打哈欠 */
}TvaiFaceAttrExpression;

//人体属性取值定义
//人体属性性别
typedef enum {
    TVAI_PED_ATTR_GENDER_FEMALE                     = 0,                 /* 女 */
    TVAI_PED_ATTR_GENDER_MALE                       = 1                  /* 男 */
}TvaiPedAttrGender;

//人体属性年龄段
typedef enum {
    TVAI_PED_ATTR_AGE_STAGE_OLD                     = 0,                 /* 老人 */
    TVAI_PED_ATTR_AGE_STAGE_ADULT                   = 1,                 /* 成年人 */
    TVAI_PED_ATTR_AGE_STAGE_CHILDREN                = 2                  /* 儿童 */
}TvaiPedAttrAgeStage;

//人体属性包包类型
typedef enum {
    TVAI_PED_ATTR_BAG_NONE                          = 0,                 /* 没有背包 */
    TVAI_PED_ATTR_BAG_HAND_BAG                      = 1,                 /* 手提包 */
    TVAI_PED_ATTR_BAG_SHOULDER_BAG                  = 2,                 /* 双肩包 */
    TVAI_PED_ATTR_BAG_BACKPACK                      = 3,                 /* 背包 */
    TVAI_PED_ATTR_BAG_WAIST_PACK                    = 4                  /* 腰包 */
}TvaiPedAttrBagType;

//人体属性是否抱东西或小孩
typedef enum {
    TVAI_PED_ATTR_FRONT_HOLD_NONE                   = 0,                 /* 没有抱 */
    TVAI_PED_ATTR_FRONT_HOLD_HAS_HOLD               = 1                  /* 有抱 */
}TvaiPedAttrFrontHold;

//人体属性眼镜
typedef enum {
    TVAI_PED_ATTR_GLASS_NO_GLASSES                  = 0,                 /* 不戴眼镜 */
    TVAI_PED_ATTR_GLASS_ORDINARY_GLASSES            = 1,                 /* 普通眼镜 */
    TVAI_PED_ATTR_GLASS_SUNGLASSES                  = 2                  /* 太阳眼镜/墨镜       */
}TvaiPedAttrGlasses;

//人体属性头发颜色
typedef enum {
    TVAI_PED_ATTR_HAIR_COLOR_NONE                   = -1,                /* 光头 */
    TVAI_PED_ATTR_HAIR_COLOR_BLACK                  = 0,                 /* 黑色 */
    TVAI_PED_ATTR_HAIR_COLOR_BROWN                  = 1,                 /* 棕色 */
    TVAI_PED_ATTR_HAIR_COLOR_GRAY                   = 2,                 /* 灰色   */
    TVAI_PED_ATTR_HAIR_COLOR_WHITE                  = 3,                 /* 白色   */
    TVAI_PED_ATTR_HAIR_COLOR_YELLOW                 = 4                  /* 黄色 */
}TvaiPedAttrHairColor;

//人体属性头发长度
typedef enum {
    TVAI_PED_ATTR_HAIR_LENGTH_BALD                  = 0,                 /* 秃顶 */
    TVAI_PED_ATTR_HAIR_LENGTH_LONGS                 = 1,                 /* 长头发 */
    TVAI_PED_ATTR_HAIR_LENGTH_SHORTS                = 2                  /* 短头发   */
}TvaiPedAttrHairLength;

//人体属性头戴类型
typedef enum {
    TVAI_PED_ATTR_HEAD_WEAR_NONE                    = 0,                 /* 无头戴 */
    TVAI_PED_ATTR_HEAD_WEAR_HAT                     = 1,                 /* 戴帽子 */
    TVAI_PED_ATTR_HEAD_WEAR_HELMET                  = 2,                 /* 戴头盔    */
    TVAI_PED_ATTR_HEAD_WEAR_OTHERS                  = 99                 /* 其它    */
}TvaiPedAttrHeadWear;

//人体属性上衣 or 下衣颜色 or 鞋子颜色 or 雨伞颜色等颜色相关的枚举
typedef enum {
    TVAI_PED_ATTR_COLOR_OTHERS                      = 0,                 /* 其它   */
    TVAI_PED_ATTR_COLOR_BLUE                        = 1,                 /* 蓝色 */
    TVAI_PED_ATTR_COLOR_GRAY                        = 2,                 /* 灰色   */
    TVAI_PED_ATTR_COLOR_GREEN                       = 3,                 /* 绿色   */
    TVAI_PED_ATTR_COLOR_ORANGE                      = 4,                 /* 橙色   */
    TVAI_PED_ATTR_COLOR_PURPLE                      = 5,                 /* 紫色   */
    TVAI_PED_ATTR_COLOR_RED                         = 6,                 /* 红色   */
    TVAI_PED_ATTR_COLOR_WHITE                       = 7,                 /* 白色   */
    TVAI_PED_ATTR_COLOR_YELLOW                      = 8,                 /* 黄色   */
    TVAI_PED_ATTR_COLOR_PINK                        = 9,                 /* 粉色   */
    TVAI_PED_ATTR_COLOR_BROWN                       = 10,                /* 棕色   */
    TVAI_PED_ATTR_COLOR_BLACK                       = 11                 /* 黑色 */
}TvaiPedAttrColor;

//人体属性上衣 or 下衣花案
typedef enum {
    TVAI_PED_ATTR_CLOTHES_TEXTURE_OTHERS            = 0,                 /* 其它   */
    TVAI_PED_ATTR_CLOTHES_TEXTURE_PATTERN           = 1,                 /* 图案 */
    TVAI_PED_ATTR_CLOTHES_TEXTURE_PURE              = 2,                 /* 纯色   */
    TVAI_PED_ATTR_CLOTHES_TEXTURE_STRIPE            = 3,                 /* 条纹   */
    TVAI_PED_ATTR_CLOTHES_TEXTURE_UNIFORM           = 4,                 /* 制服   */
    TVAI_PED_ATTR_CLOTHES_TEXTURE_GRID              = 5                  /* 格子 */

}TvaiPedAttrClothesTexture;

//人体属性下衣类型
typedef enum {
    TVAI_PED_ATTR_TROUSERS_TYPE_OTHERS              = 0,                 /* 其它   */
    TVAI_PED_ATTR_TROUSERS_TYPE_SHORTS              = 1,                 /* 短裤 */
    TVAI_PED_ATTR_TROUSERS_TYPE_DRESS               = 2,                 /* 短裙   */
    TVAI_PED_ATTR_TROUSERS_TYPE_PANTS               = 3                  /* 长裤 */
}TvaiPedAttrTrousersType;

//人体属性上衣类型
typedef enum {
    TVAI_PED_ATTR_COAT_TYPE_NO_SLEEVE               = 0,                 /* 赤膊 */
    TVAI_PED_ATTR_COAT_TYPE_SHORT_SLEEVE            = 1,                 /* 短袖 */
    TVAI_PED_ATTR_COAT_TYPE_LONG_SLEEVE             = 2                  /* 长袖   */
}TvaiPedAttrCoatType;

//人体属性口罩
typedef enum {
    TVAI_PED_ATTR_MASK_NO_MASK                      = 0,                 /* 不戴口罩 */
    TVAI_PED_ATTR_MASK_HAS_MASK                     = 1                  /* 戴口罩 */
}TvaiPedAttrMask;

//人体属性帽子
typedef enum {
    TVAI_PED_ATTR_HAT_TYPE_NO_HAT                   = 0,                 /* 不戴帽子 */
    TVAI_PED_ATTR_HAT_TYPE_OTHERS                   = 1,                 /* 其它   */
    TVAI_PED_ATTR_HAT_TYPE_BONNET                   = 2,                 /* 无檐帽 */
    TVAI_PED_ATTR_HAT_TYPE_CAP                      = 3,                 /* 鸭舌帽 */
    TVAI_PED_ATTR_HAT_TYPE_BUCKET_HAT               = 4,                 /* 渔夫帽 */
    TVAI_PED_ATTR_HAT_TYPE_HARD_HAT                 = 5                  /* 安全帽 */
}TvaiPedAttrHatType;

//人体属性朝向
typedef enum {
    TVAI_PED_ATTR_ORIENTATION_FRONT                 = 0,                 /* 正面 */
    TVAI_PED_ATTR_ORIENTATION_SIDE                  = 1,                 /* 侧面   */
    TVAI_PED_ATTR_ORIENTATION_BACK                  = 2                  /* 背面 */
}TvaiPedAttrOrientation;

//人体属性是否穿外套
typedef enum {
    TVAI_PED_ATTR_OVERCOAT_NO_OVERCOAT              = 0,                 /* 没有穿外套 */
    TVAI_PED_ATTR_OVERCOAT_HAS_OVERCOAT             = 1                  /* 穿了外套     */
}TvaiPedAttrOvercoat;

//人体属性是否携带拉杆箱
typedef enum {
    TVAI_PED_ATTR_TROLLEY_CASE_NO_TROLLEY_CASE      = 0,                 /* 没有拉杆箱 */
    TVAI_PED_ATTR_TROLLEY_CASE_HAS_TROLLEY_CASE     = 1                  /* 有拉杆箱     */
}TvaiPedAttrTrolleyCase;

//人体属性行为
typedef enum {
    TVAI_PED_ATTR_BEHAVIOR_OTHERS                   = 0,                 /* 其它 */
    TVAI_PED_ATTR_BEHAVIOR_NORMAL                   = 1,                 /* 无 */
    TVAI_PED_ATTR_BEHAVIOR_CALL                     = 2,                 /* 打电话    */
    TVAI_PED_ATTR_BEHAVIOR_SMOKE                    = 3,                 /* 抽烟   */
    TVAI_PED_ATTR_BEHAVIOR_PLAYPHONE                = 4                  /* 玩手机    */
}TvaiPedAttrBehavior;

//人体属性制服
typedef enum {
    TVAI_PED_ATTR_UNIFORM_OTHERS                    = 0,                 /* 其它 */
    TVAI_PED_ATTR_UNIFORM_COMMON                    = 1,                 /* 普通服装 */
    TVAI_PED_ATTR_UNIFORM_OFFICE                    = 2,                 /* 办公室制服      */
    TVAI_PED_ATTR_UNIFORM_WORKER                    = 3,                 /* 工人制服     */
    TVAI_PED_ATTR_UNIFORM_CHEF                      = 4,                 /* 厨师制服     */
    TVAI_PED_ATTR_UNIFORM_MEDICAL                   = 5,                 /* 医护制服     */
    TVAI_PED_ATTR_UNIFORM_POLICE                    = 6,                 /* 警察制服     */
    TVAI_PED_ATTR_UNIFORM_FIREFIGHTER               = 7                  /* 消防员制服      */
}TvaiPedAttrUniform;

//人体属性鞋子类型
typedef enum {
    TVAI_PED_ATTR_SHOES_TYPE_OTHERS                  = 0,                /* 其它 */
    TVAI_PED_ATTR_SHOES_TYPE_LEATHER_SHOES           = 1,                /* 皮鞋 */
    TVAI_PED_ATTR_SHOES_TYPE_BOOTS                   = 2,                /* 靴子    */
    TVAI_PED_ATTR_SHOES_TYPE_WALKING_SHOES           = 3,                /* 休闲鞋    */
    TVAI_PED_ATTR_SHOES_TYPE_SANDAL                  = 4                 /* 凉鞋   */
}TvaiPedAttrShoesType;

//机动车属性取值定义
// 车体属性颜色特征 等相关颜色类的取值。
typedef enum {
    TVAI_VEHICLE_ATTR_COLOR_OTHERS                   = 0,                /* 其它   */
    TVAI_VEHICLE_ATTR_COLOR_BLUE                     = 1,                /* 蓝色 */
    TVAI_VEHICLE_ATTR_COLOR_GRAY                     = 2,                /* 灰色   */
    TVAI_VEHICLE_ATTR_COLOR_GREEN                    = 3,                /* 绿色   */
    TVAI_VEHICLE_ATTR_COLOR_ORANGE                   = 4,                /* 橙色   */
    TVAI_VEHICLE_ATTR_COLOR_PURPLE                   = 5,                /* 紫色   */
    TVAI_VEHICLE_ATTR_COLOR_RED                      = 6,                /* 红色   */
    TVAI_VEHICLE_ATTR_COLOR_WHITE                    = 7,                /* 白色   */
    TVAI_VEHICLE_ATTR_COLOR_YELLOW                   = 8,                /* 黄色   */
    TVAI_VEHICLE_ATTR_COLOR_PINK                     = 9,                /* 粉色   */
    TVAI_VEHICLE_ATTR_COLOR_BROWN                    = 10,               /* 棕色   */
    TVAI_VEHICLE_ATTR_COLOR_BLACK                    = 11                /* 黑色 */
}TvaiVehicleAttrColor;

// 车体属性车辆分类
typedef enum {
    TVAI_VEHICLE_ATTR_CLASSIFY_OTHERS                = 0,                /* 其它   */
    TVAI_VEHICLE_ATTR_CLASSIFY_CAR                   = 1,                /* 轿车 */
    TVAI_VEHICLE_ATTR_CLASSIFY_SUV                   = 2,                /* SUV */
    TVAI_VEHICLE_ATTR_CLASSIFY_SMALL_TRUCK           = 3,                /* 小卡车    */
    TVAI_VEHICLE_ATTR_CLASSIFY_BIG_TRUCK             = 4,                /* 大卡车    */
    TVAI_VEHICLE_ATTR_CLASSIFY_BUS                   = 5,                /* 公交车    */
    TVAI_VEHICLE_ATTR_CLASSIFY_BIG_BUS               = 6,                /* 大巴车    */
    TVAI_VEHICLE_ATTR_CLASSIFY_MIDDLE_BUS            = 7,                /* 中巴车    */
    TVAI_VEHICLE_ATTR_CLASSIFY_SCHOOL_BUS            = 8,                /* 校车   */
    TVAI_VEHICLE_ATTR_CLASSIFY_TAXI                  = 9,                /* 出租车    */
    TVAI_VEHICLE_ATTR_CLASSIFY_FIRE_ENGINE           = 10,               /* 消防车    */
    TVAI_VEHICLE_ATTR_CLASSIFY_AMBULANCE             = 11,               /* 救护车    */
    TVAI_VEHICLE_ATTR_CLASSIFY_POLICE_CAR            = 12,               /* 警车   */
    TVAI_VEHICLE_ATTR_CLASSIFY_WATER_CAR             = 13                /* 洒水车   */
}TvaiVehicleAttrClassify;

// 车体属性车辆品牌
typedef enum {
    TVAI_VEHICLE_ATTR_BRAND_OTHERS                   = 0,                /* 其它   */
    TVAI_VEHICLE_ATTR_BRAND_VOLKWAGEN                = 1,                /* 大众 */
    TVAI_VEHICLE_ATTR_BRAND_BUICK                    = 2,                /* 别克 */
    TVAI_VEHICLE_ATTR_BRAND_BMW                      = 3,                /* 宝马    */
    TVAI_VEHICLE_ATTR_BRAND_HONDA                    = 4,                /* 本田    */
    TVAI_VEHICLE_ATTR_BRAND_PEUGEOT                  = 5,                /* 标致    */
    TVAI_VEHICLE_ATTR_BRAND_TOYOTA                   = 6,                /* 丰田    */
    TVAI_VEHICLE_ATTR_BRAND_FORD                     = 7,                /* 福特    */
    TVAI_VEHICLE_ATTR_BRAND_NISSAN                   = 8,                /* 日产   */
    TVAI_VEHICLE_ATTR_BRAND_AUDI                     = 9,                /* 奥迪    */
    TVAI_VEHICLE_ATTR_BRAND_MAZDA                    = 10,               /* 马自达    */
    TVAI_VEHICLE_ATTR_BRAND_CHEVROLET                = 11,               /* 雪佛兰    */
    TVAI_VEHICLE_ATTR_BRAND_CITROEN                  = 12,               /* 雪铁龙    */
    TVAI_VEHICLE_ATTR_BRAND_HYUNDAI                  = 13,               /* 现代   */
    TVAI_VEHICLE_ATTR_BRAND_CHERY                    = 14,               /* 奇瑞   */
    TVAI_VEHICLE_ATTR_BRAND_KIA                      = 15,               /* 起亚   */
    TVAI_VEHICLE_ATTR_BRAND_ROEWE                    = 16,               /* 荣威   */
    TVAI_VEHICLE_ATTR_BRAND_MITSUBISHI               = 17,               /* 三菱   */
    TVAI_VEHICLE_ATTR_BRAND_SKODA                    = 18,               /* 斯柯达    */
    TVAI_VEHICLE_ATTR_BRAND_GEELY                    = 19,               /* 吉利   */
    TVAI_VEHICLE_ATTR_BRAND_ZHONGHUA                 = 20,               /* 中华   */
    TVAI_VEHICLE_ATTR_BRAND_VOLVO                    = 21,               /* 沃尔沃    */
    TVAI_VEHICLE_ATTR_BRAND_LEXUS                    = 22,               /* 雷克萨斯     */
    TVAI_VEHICLE_ATTR_BRAND_FIAT                     = 23,               /* 菲亚特    */
    TVAI_VEHICLE_ATTR_BRAND_DONGFENG                 = 25,               /* 东风   */
    TVAI_VEHICLE_ATTR_BRAND_BYD                      = 26,               /* 比亚迪    */
    TVAI_VEHICLE_ATTR_BRAND_CHANGAN                  = 38,               /* 长安   */
    TVAI_VEHICLE_ATTR_BRAND_BENZ                     = 41,               /* 奔驰   */
    TVAI_VEHICLE_ATTR_BRAND_PORSCHE                  = 65,               /* 保时捷   */
    TVAI_VEHICLE_ATTR_BRAND_LAND_ROVER               = 70,               /* 路虎   */
    TVAI_VEHICLE_ATTR_BRAND_BENTLEY                  = 191,              /* 宾利   */
    TVAI_VEHICLE_ATTR_BRAND_RED_FLAG                 = 230               /* 红旗   */
}TvaiVehicleAttrBrand;


//非机动车属性取值定义
// 车体属性颜色特征 等相关颜色类的取值。
typedef enum {
    TVAI_NONMOTOR_ATTR_COLOR_OTHERS                  = 0,                /* 其它   */
    TVAI_NONMOTOR_ATTR_COLOR_BLUE                    = 1,                /* 蓝色 */
    TVAI_NONMOTOR_ATTR_COLOR_GRAY                    = 2,                /* 灰色   */
    TVAI_NONMOTOR_ATTR_COLOR_GREEN                   = 3,                /* 绿色   */
    TVAI_NONMOTOR_ATTR_COLOR_ORANGE                  = 4,                /* 橙色   */
    TVAI_NONMOTOR_ATTR_COLOR_PURPLE                  = 5,                /* 紫色   */
    TVAI_NONMOTOR_ATTR_COLOR_RED                     = 6,                /* 红色   */
    TVAI_NONMOTOR_ATTR_COLOR_WHITE                   = 7,                /* 白色   */
    TVAI_NONMOTOR_ATTR_COLOR_YELLOW                  = 8,                /* 黄色   */
    TVAI_NONMOTOR_ATTR_COLOR_PINK                    = 9,                /* 粉色   */
    TVAI_NONMOTOR_ATTR_COLOR_BROWN                   = 10,               /* 棕色   */
    TVAI_NONMOTOR_ATTR_COLOR_BLACK                   = 11                /* 黑色 */
}TvaiNonmotorAttrColor;

//目标属性定义
typedef struct TvaiTargetAttribute_S
{
    unsigned int                  attrType;                 /* 属性类型      TvaiTargetAttributeType  */
    float                         confidence;               /* 置信度 [0, 1.0] */
    int                           value;                    /* 属性值。根据属性类型不同，取值不同*/
}TvaiTargetAttribute;

//目标属性定义列表
typedef struct TvaiTargetAttributeList_S
{
    unsigned int                  number;                   /* 属性个数 */
    TvaiTargetAttribute           *pTargetAttr;             /* 属性 */
}TvaiTargetAttributeList;

// 矩形框
typedef struct TvaiRect_S
{
    int     x;          /* 左上角X坐标 */
    int     y;          /* 左上角Y坐标 */
    int     width;      /* 区域宽度 */
    int     height;     /* 区域高度 */
}TvaiRect;

// 目标关联信息
typedef struct TvaiTargetRelation_S
{
    TvaiRelation                relation;    /* 与关联目标的关系 */
    TvaiDetectTarget            *pTarget;    /* 关联的目标 */
}TvaiTargetRelation;

// 关联集合
typedef struct TvaiTargetRelationList_S
{
    unsigned int             number;                 /* 关联个数， 为0表示无关联目标 */
    TvaiTargetRelation       *pTargetRelation;       /* 关联的目标详情 */
}TvaiTargetRelationList;

//检测出的目标
typedef struct TvaiDetectTarget_S
{
    unsigned int                targetId;            /* 目标ID */
    unsigned int                targetType;          /* 目标类型 TvaiTargetType */
    float                       confidence;          /* 目标检测置信度[0, 1.0]   */
    TvaiRect                    rect;                /* 目标坐标 */
    float                       qualityScore;        /* 质量分[0, 1.0]    */
    TvaiTargetRelationList      targetRelationList;  /* 相关联的目标 */
}TvaiDetectTarget;

// yuv
typedef struct TvaiImageRawDataInfo_S
{
    /*! 图像格式。 */
    unsigned int u32Width;        /* 图像宽度 */
    unsigned int u32Height;       /* 图像高度 */
    unsigned int enField;

    unsigned int enPixelFormat;
    unsigned int u32Stride[3];    /* 图像水平跨度，取第0个 */
    union
    {
        struct
        {
            unsigned long long	u64PhyAddr[3]; /* 数据的物理地址 */
            unsigned long long  u64VirAddr[3]; /* 数据的虚拟地址 */
        }st64;
        struct
        {
            unsigned int  u32PhyAddr[3];       /* 数据的物理地址 */
            unsigned int  u32VirAddr[3];       /* 数据的虚拟地址 */
        }st32;
    }Addr;

    short   s16OffsetTop;          /* top offset of show area */
    short   s16OffsetBottom;       /* bottom offset of show area */
    short   s16OffsetLeft;         /* left offset of show area */
    short   s16OffsetRight;        /* right offset of show area */

    unsigned long long  u64PTS;

    int chip_index;                /* 芯片编号 */
    int decode_index;              /* 解码通道编码 */
    int channel_num;               /* 通道号，唯一的 */
    unsigned int serial_number;
}TvaiImageRawDataInfo;

typedef struct TvaiImageRawData_S {
    TvaiImageRawDataInfo      *pOriginRawData; /* 原始视频数据 YUV格式 */
    TvaiImageRawDataInfo      *pResizeRawData; /* 压缩过的视频数据         */
}TvaiImageRawData;

// 图像
typedef struct TvaiImage_S
{
    TvaiImageFormat      format;      /* 图像像素格式 */
    int                  width;       /* 图像宽度 */
    int                  height;      /* 图像高度 */
    int                  stride;      /* 图像水平跨度 */
    unsigned char        *pData;      /* 图像数据。当format为TVAI_IMAGE_FORMAT_YUV_PHY时，指针结构为TvaiImageRawData*/
    int                  dataSize;    /* 图像数据的长度 */
}TvaiImage;

//userData结构
typedef struct TvaiUserData_S {
    unsigned char             *pData;        /* AP自定义数据指针 */
    unsigned int               dataSize;     /* AP自定义数据长度 */
}TvaiUserData;

//引用计数定义
typedef unsigned char TvaiReference;

//算法处理句柄，用于视频流和图片流通道标识
typedef unsigned char TvaiHandle;

// 帧数据
typedef struct TvaiFrame_S
{
    TvaiImage         *pImg;         /* 图像数据 */
    TvaiUserData      *pUserData;    /* AP自定义数据 */
    TvaiReference     *pFrameRef;    /* 帧数据的引用计数 */
}TvaiFrame;

//特征
typedef struct TvaiFeature_S
{
    unsigned int          featureLen;        /* 特征值长度, 字节数 */
    unsigned char         *pFeature;         /* 特征值指针 */
}TvaiFeature;

// 特征库
typedef struct TvaiFeatureRepo_S
{
    unsigned int     repoId;          /* 特征库ID */
    unsigned int     repoCapacity;    /* 特征库容量 */
    unsigned int     repoSize;        /* 特征库已录入的特征数量 */
}TvaiFeatureRepo;

// 特征库列表
typedef struct TvaiFeatureRepoList_S
{
    unsigned int               number;       /* 特征库个数 */
    TvaiFeatureRepo            *pRepo;       /* 特征库详情 */
}TvaiFeatureRepoList;

// 特征ID列表
typedef struct TvaiFeatureIdList_S
{
    unsigned int               number;         /* 特征Id个数 */
    unsigned long long         *pFeatureIds;    /* 特征ID */
}TvaiFeatureIdList;


//识别结果
typedef struct TvaiVerifyResult_S {
    unsigned int                repoId;                /* 特征库ID */
    unsigned long long          featureId;             /* 特征ID */
    float                       score;                 /* 相似度 */
}TvaiVerifyResult;

//识别结果Top K列表
typedef struct TvaiVerifyResultList_S {
    unsigned int                number;                 /* 个数, 0表示没有识别结果,或没有进行识别 */
    TvaiVerifyResult            *pVerifyResult;         /* 识别结果详情 */
}TvaiVerifyResultList;

// 精准防范分析结果，抓拍+识别。如人脸识别、识别。
typedef struct TvaiPreciseResult_S
{
    TvaiDetectTarget            target;                    /* 检测出的目标信息 */
    TvaiTargetAttributeList     targetAttributeList;       /* 目标对应的属性, 未提取属性时,个数为0,内部指针为空 */
    TvaiFeature                 feature;                   /* 提取的目标特征数据,   未提取特征时,内部指针为空，长度为0*/
    TvaiVerifyResultList        topkVerifyResult;          /* 识别TopK结果 */
}TvaiPreciseResult;

typedef struct TvaiPreciseResultList_S
{
    unsigned int                number;            /* 结果个数，0表示没有结果。 */
    TvaiPreciseResult           *pPreciseResult;   /* 结果详情, number个pPreciseResult  */
}TvaiPreciseResultList;

//单绊线事件描述信息
typedef struct TvaiTripwireEventAdditionalInfo_S
{
    TvaiLineCrossDirection    direction;       /*绊线方向*/
}TvaiTripwireEventAdditionalInfo;

//AOI类事件的描述信息
typedef struct TvaiAoiEventAdditionalInfo_S
{
    /* 暂无，有需要时补充。 */
}TvaiAoiEventAdditionalInfo;
//自定义（扩展）规则描述
typedef struct TvaiCustomEventEventAdditionalInfo_S {
    unsigned char       *pData;        /* 数据指针 */
    unsigned int        dataLength;    /* 数据长度 */
}TvaiCustomEventAdditionalInfo;

//事件附加信息union，根据事件类型取结构
typedef union TvaiEventAdditionalInfo_U
{
    TvaiAoiEventAdditionalInfo             aoiEventAdditionalInfo;        /* 对应AOI相关的事件 */
    TvaiTripwireEventAdditionalInfo        tripwireEventAdditionalInfo;   /* 单绊线 */
    TvaiCustomEventAdditionalInfo          customEventAdditionalInfo;     /* 自定义（扩展）规则匹配后的事件附加信息 */
}TvaiEventAdditionalInfo;

// 侦测事件结果（如人车非)
typedef struct TvaiEventDetectResult_S
{
    unsigned int              ruleId;                  /* 触发事件对应的规则ID, 事件触发后填回给AP */
    TvaiDetectTarget          *pTarget;                /* 事件对应的目标信息,事件不一定有目标 */
    TvaiTargetAttributeList   targetAttributeList;     /* 目标对应的属性, pTarget不为空有效，为空时个数填0，内部指为赋空。 */
    TvaiFeature               feature;                 /* 目标对应的属性, pTarget不为空有效，为空时长度填0，内部指为赋空。 */
    unsigned int              ruleType;                /* 规则类型 TvaiEventDetectType 用于方便解析pEventAdditionalInfo */
    TvaiEventAdditionalInfo   *pEventAdditionalInfo;   /* 事件匹配的附加信息。可能有，没有时指针为空 */
}TvaiEventDetectResult;

//周界防范分析结果，特定区域的特定事件，如区域入侵。
typedef struct TvaiEventDetectResultList_S
{
    unsigned int              number;                 /* 侦测的事件结果个数， 0表示没有结果。 */
    TvaiEventDetectResult     *pEventDetectResult;     /* 侦测事件结果（如人车非侦测) */
}TvaiEventDetectResultList;

// 视频帧处理结果
typedef struct TvaiVideoFrameProcResult_S {
    TvaiPreciseResultList         *pPreciseResultList;           /* 精准防范分析结果列表，抓拍+识别。如人脸识别、识别。 */
    TvaiEventDetectResultList     *pEventDetectResultList;       /* 周界事件侦测分析结果，特定区域的特定事件，如区域入侵。 */
}TvaiVideoFrameProcResult;

// 图片流处理结果
typedef struct TvaiImageFrameProcResult_S {
    TvaiPreciseResultList       *pPreciseResultList;             /* 精准防范分析结果列表，抓拍+识别。如人脸识别、识别。 */
}TvaiImageFrameProcResult;

//图片或特征数据
typedef union TvaiImageOrFeatureData_U {
    TvaiFeature        feature;                /* 特征数据 */
    TvaiImage          image;                  /* 图片数据 */
}TvaiImageOrFeatureData;

// 1：N搜索参数
typedef struct TvaiSearchParam_S
{
    TvaiDataType              dataType;            /* 搜索的数据类型，图片或特征数据 */
    TvaiImageOrFeatureData    data;                /* 图片或特征数据，union，根据dataType取值 */ 
    unsigned int              topk;                /* 相似度最高的前k个 */
    unsigned int              repoIdNum;           /* 特征库个数， 0 表示全部特征库 */
    unsigned int              *pRepoIds;           /* 特征库IDs指针，repoIdNum个特征库。*/
}TvaiSearchParam;

// 1：1比对参数
typedef struct TvaiCompareData_S
{
    TvaiDataType              dataType;            /* 搜索的数据类型，图片或特征数据 */
    TvaiImageOrFeatureData    data;                /* 图片或特征数据，union，根据dataType取值 */ 
}TvaiCompareData;

// 分辨率
typedef struct TvaiResolution_S {
    unsigned int     width;                 /* 宽度 */
    unsigned int     height;                /* 高度 */
}TvaiResolution;

// 目标检测参数 
typedef struct TvaiTargetParam_S {
    unsigned int            targetType;                 /* 目标类型， TvaiTargetType      TVAI_TARGET_TYPE_UNDEFINED表示任意支持的目标*/
    TvaiResolution          maxTargetSize;              /* 目标最大值。 0表示不限制 */
    TvaiResolution          minTargetSize;              /* 目标最小值。 0表示不限制 */
    TvaiRect                *pAoiRect;                  /* 只检测区域内的目标，为空表示全部区域 */
    bool                    isExtractFeature;           /* 检测出目标后，是否同时提取特征 */
    bool                    isExtractAttr;              /* 检测出目标后，是否同时提取结构属性 */
}TvaiTargetParam;

// 图片检测、提取属性、特征的参数设置
typedef struct TvaiDetectParam_S
{
    unsigned int              number;                   /* 需要检测的目标类别数量 */
    TvaiTargetParam           *pTargetParam;            /* 各目标的参数设置，number个pTargetParam */
}TvaiDetectParam;

// 图片检测、提取属性、特征的结果
typedef struct TvaiDetectResult_S {
    TvaiDetectTarget            target;                     /* 检测出的目标信息 */
    TvaiTargetAttributeList     targetAttributeList;        /* 目标对应的属性, 未提取属性时,内部指针为空，个数为0 */
    TvaiFeature                 feature;                    /* 提取的目标特征数据,   未提取特征时,内部指针为空，长度为0 */
}TvaiDetectResult;

// 图片检测、提取属性、特征的结果列表
typedef struct TvaiDetectResultList_S {
    unsigned int            number;                     /* 结果个数， 0表示没有结果。*/
    TvaiDetectResult        *pDetectResult;             /* 检测结果 */ 
}TvaiDetectResultList;

// 点 结构体
typedef struct TvaiPoint_S {
    int               x;                   /* x坐标 */
    int               y;                   /* y坐标 */
}TvaiPoint;

// 线 结构体
typedef struct TvaiLine_S
{
    TvaiPoint                startPoint;        /*线起点 */
    TvaiPoint                endPoint;          /*线终点 */
}TvaiLine;

// 多边形 结构体
typedef struct TvaiPolygon_S
{
    unsigned int          number;               /* 点的个数 */
    TvaiPoint             *pPoint;              /* 点的坐标 */
}TvaiPolygon;

// 触发规则类型
typedef enum
{
    TVAI_EVENT_DETECT_TRIGGER_CENTER,              /* 以目标中心点计算。*/
    TVAI_EVENT_DETECT_TRIGGER_VERTEX,              /* 以目标四个顶点计算。*/
}TvaiEventDetectTriggerType;

// 绊线触发模式
typedef enum
{
    TVAI_LINE_TRIGGER_MODE_TRIP_LINE,     /* 伴线触发报警 */
    TVAI_LINE_TRIGGER_MODE_CROSS_LINE,    /* 过线触发报警 */
}TvaiLineTriggerMode;

//单绊线规则的详细描述
typedef struct TvaiTripwireEventDetectRuleDescription_S
{
    TvaiLine                 line;             /* 线的坐标，起点和终点 */
    TvaiLineCrossDirection   direction;        /*绊线方向*/
    TvaiLineTriggerMode      triggerMode;      /*触发模式*/	 
} TvaiTripwireEventDetectRuleDescription;

//区域事件规则的详细描述
typedef struct TvaiAoiEventDetectRuleDescription_S {
    unsigned int             duration;        /* 持续时间(S) */
    TvaiPolygon              polygon;         /* 侦察区域 */
}TvaiAoiEventDetectRuleDescription;

//自定义（扩展）规则描述
typedef struct TvaiCustomEventDetectRuleDescription_S {
    unsigned char               *pRuleData;       /* 规则描述,自定义规则 */
    unsigned int                ruleDataLength;   /* 规则描述的长度 */
}TvaiCustomEventDetectRuleDescription;


//事件规则的详细描述union
typedef union TvaiDetectRuleDescription_U {
    TvaiAoiEventDetectRuleDescription            aoiEventRuleDescription;        /* 区域类事件规则描述 */
    TvaiTripwireEventDetectRuleDescription       tripwireEventRuleDescription;   /* 单绊线（越界）事件规则描述。 */
    TvaiCustomEventDetectRuleDescription         customEventRuleDescription;     /* 自定义（扩展）规则描述 */
}TvaiEventDetectRuleDescription;

// 需要侦测的目标
typedef struct TvaiEventDetectTargetInfo_S {
    unsigned int                targetType;                 /* 目标类型，      TvaiTargetType  TVAI_TARGET_TYPE_UNDEFINED表示任意支持的目标 */
    TvaiResolution              maxTargetSize;              /* 目标最大值。 0表示不限制 */
    TvaiResolution              minTargetSize;              /* 目标最小值。 0表示不限制 */
    TvaiEventDetectTriggerType  triggerType;                /* 触发规则类型。 */
    unsigned int                sensitivity;                /* 灵敏度 */
    unsigned char               *pExtendParam;              /* 扩展参数,默认为空，根据项目情况适配 */
    unsigned int                extendParamLength;          /* 扩展参数长度 默认为0 */
}TvaiEventDetectTargetInfo;

// 需要侦测的目标列表
typedef struct TvaiEventDetectTargetInfoList_S
{
    unsigned int                  number;                   /* 需要侦测的目标列表数量 */
    TvaiEventDetectTargetInfo     *pEventTarget;            /* 需要侦测的目标列表 */
}TvaiEventDetectTargetInfoList;

// 需要检测的事件规则
typedef struct TvaiEventDetectRule_S {
    unsigned int                       ruleId;              /* 规则ID */
    unsigned int                       ruleType;            /* 规则类型 TvaiEventDetectType */
    TvaiEventDetectRuleDescription     ruleDescription;     /* 规则描述 */
    TvaiEventDetectTriggerType         triggerType;         /* 触发规则类型。 */
    TvaiEventDetectTargetInfoList      targetList;          /* 需要侦测的目标列表及参数 */
}TvaiEventDetectRule;

// 需要检测的事件规则列表
typedef struct TvaiEventDetectRuleList_S {
    unsigned int                       number;             /* 需要检测的事件规则个数 */
    TvaiEventDetectRule                *pEventDetectRule;  /* 需要检测的事件规则 */
}TvaiEventDetectRuleList;

typedef struct TvaiTargetTypeList_S {
    unsigned int                  number;                   /* 目标类型的数量 */
    unsigned int                  *pTargetTypes;            /* 目标类型。number个         TvaiTargetType    */
}TvaiTargetTypeList;

#ifdef __cplusplus
} //extern "C"
#endif

#endif //TVAI_COMMON_H

// I0909 15:44:11.808130 17637 test_forward_offline.cpp:158] in_n 4 in_c 3 in_h 128 in_w 64
// I0909 15:44:11.808143 17637 test_forward_offline.cpp:192] out_n 4 out_c 512 out_h 1 out_w 1
