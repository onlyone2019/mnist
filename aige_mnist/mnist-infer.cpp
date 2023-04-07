//
// Created by wangjie on 23-4-5.
//
#include "include/type.h"
#include "include/utils.h"
#include "include/wrapper.h"
//#include "base64.h"
#include <torch/torch.h>


pDataList wrapperInnerRslt = NULL;
pDataList wrapperOnceRslt = NULL;
std::string model_path = "model/mnist.pth";
std::shared_ptr<LeNet> mnist = std::make_shared<LeNet>();
const void* wrapperInnerHdl = "wrapperTestHandle";
const char* defResult = "Erorr! ";
torch::Device device = select_device();

/*
 * Mat 编码成 base64
 * */
//static std::string Mat2Base64(const cv::Mat &img, std::string imgType)
//{
//    std::string img_data;
//    std::vector<uchar> vecImg;
//    std::vector<int> vecCompression_params;
//    vecCompression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
//    vecCompression_params.push_back(90);
//    imgType = "." + imgType;
//    cv::imencode(imgType, img, vecImg, vecCompression_params);
//    img_data = base64Encode(vecImg.data(), vecImg.size());
//    return img_data;
//}

/*
 * stream 转 Mat
 * */
static cv::Mat  streamTomat(void *data, int length)
{
    cv::Mat image;
    const unsigned char *charBuffer = (unsigned char *) data;
    std::vector<unsigned char> vec_data(charBuffer, charBuffer + length);
    image = cv::imdecode(vec_data, cv::IMREAD_COLOR);
    if(!image.data)
    {
        throw"图片非法!";
        //return "{\"msg\":\"图片非法!\"}";
    }
    return image;
}

/*
 * 初始化
 * */
int WrapperAPI wrapperInit(pConfig cfg){
    std::cout <<"===========>wrapperInit\n";
    // 打印输入配置项
    while (cfg != NULL) {
        printf("key=%s, value=%s\n", cfg->key, cfg->value);
        cfg = cfg->next;
    }
    //加载模型
    torch::load(mnist, model_path);
    if (mnist == NULL) {
        printf("Error Allocate memory\n");
        exit(-1);
    }
    mnist->to(device);
    mnist->eval();

    // 构建内部测试read值
    wrapperInnerRslt = (struct DataList*)malloc(sizeof(struct DataList));
    wrapperInnerRslt->key = (char*)"result";
    wrapperInnerRslt->data = (void*)defResult;
    wrapperInnerRslt->len = strlen(defResult);
    wrapperInnerRslt->desc = NULL;
    wrapperInnerRslt->type = DataText;
    wrapperInnerRslt->status = DataEnd;
    wrapperInnerRslt->next = NULL;

    wrapperOnceRslt = (struct DataList*)malloc(sizeof(struct DataList));
    wrapperOnceRslt->key = (char*)"result";
    wrapperOnceRslt->data = (void*)defResult;
    wrapperOnceRslt->len = strlen(defResult);
    wrapperOnceRslt->desc = NULL;
    wrapperOnceRslt->type = DataText;
    wrapperOnceRslt->status = DataOnce;
    wrapperOnceRslt->next = NULL;
    return 0;
}


/*
 * 推理
 * */
int WrapperAPI wrapperExec(const char* usrTag, pParamList params, pDataList reqData, pDataList* respData, unsigned int psrIds[], int psrCnt){
    printf("=========>wrapperExec\n");
    *respData = wrapperOnceRslt;
    while (reqData!=NULL && reqData->len > 0) {
        std::cout<<"data input"<<std::endl;
        bool Legel_Iamge = true;
        cv::Mat img;
        DataList* wrapperRslt = (struct DataList*)malloc(sizeof(struct DataList));
        std::string imageResult;
        try{
            std::cout<<"begin streamTomat"<<std::endl;
            // 预处理
            img = streamTomat(reqData->data,reqData->len);
            cv::Mat gray_image;
            cv::cvtColor(img, gray_image, cv::COLOR_BGR2GRAY);
            cv::resize(gray_image, gray_image, cv::Size(28, 28));
            torch::Tensor input_tensor = torch::from_blob(gray_image.data, {1, 1, 28, 28}, torch::kByte).to(torch::kFloat).div_(255);
            input_tensor = input_tensor.to(device);
            auto output = mnist->forward(input_tensor).to(device);
            auto max_result = output.max(1, true);
            auto classes = std::get<1>(max_result).item<int>();
//            res = int(classes)
            std::cout << "Detected number is: " << int(classes) << std::endl;
            imageResult = "{ 'result' : " + std::to_string(classes) + "}";
        }catch (const char* msg) {
            printf(msg);
            Legel_Iamge = false;
            imageResult = "{\"code\": 5000,\"error_msg\": \"Illegal Picture!\"}";
        }
        std::string res_json = imageResult;
        std::cout << res_json << std::endl;
        wrapperRslt->key = (char*)"result";
        char * temp = (char*) malloc(res_json.length());
        memset(temp, 0, res_json.length());
        strcpy(temp,res_json.c_str());
        wrapperRslt->data = temp;
        wrapperRslt->len = (unsigned int )res_json.length();
        wrapperRslt->desc = NULL;
        wrapperRslt->type = DataText;
        wrapperRslt->status = DataOnce;
        wrapperRslt->next = NULL;
        *respData = wrapperRslt;
        reqData = reqData->next;
    }
}


/*
 * 释放内存
 * */
int WrapperAPI wrapperExecFree(const char* usrTag, pDataList* respData){
    printf("========>wrapperExecFree\n");
    if (*respData != wrapperOnceRslt && respData!=NULL) {
        free((*respData)->data);
        free(*respData);
        respData=NULL;
    }
}
int WrapperAPI wrapperFini(){
    printf("=========>wrapperFini\n");
//    delete mnist;
    return 0;
}
