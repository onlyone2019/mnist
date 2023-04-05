//
// Created by wangjie on 23-4-5.
//

#ifndef AIGES_MNIST_TYPE_H
#define AIGES_MNIST_TYPE_H

#define WrapperAPI __attribute__ ((visibility("default")))
typedef enum{
    CTMeterCustom =   0,      // 自定义计量接口
    CTMetricsLog  =   1,      // 自定义metrics日志接口
    CTTraceLog    =   2,      // 自定义trace日志接口

} CtrlType;

typedef enum{
    DataText    =   0,      // 文本数据
    DataAudio   =   1,      // 音频数据
    DataImage   =   2,      // 图像数据
    DataVideo   =   3,      // 视频数据
    DataPer     =   4,      // 个性化数据
} DataType;

typedef enum{
    DataBegin   =   0,      // 首数据
    DataContinue =  1,      // 中间数据
    DataEnd     =   2,      // 尾数据
    DataOnce    =   3,      // 非会话单次输入输出
} DataStatus;

typedef struct ParamList{
    char* key;
    char* value;
    unsigned int vlen;
    struct ParamList* next;
}* pParamList, *pConfig, *pDescList;     // 配置对复用该结构定义

typedef struct DataList{
    char*   key;            // 数据标识
    void*   data;           // 数据实体
    unsigned int len;       // 数据长度
    DataType    type;       // 数据类型
    DataStatus status;      // 数据状态
    pDescList desc;         // 数据描述
    struct DataList* next;  // 链表指针
}*  pDataList;



#endif //AIGES_MNIST_TYPE_H
