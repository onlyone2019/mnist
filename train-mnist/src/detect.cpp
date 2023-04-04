//
// Created by wangjie on 23-4-4.
//
#include "../include/utils.h"
#include <torch/torch.h>

using namespace std;

int main(int argc, const char *argv[]) {
    if (argc < 2) {
        cerr << "Usage: ./detect <path-to-image>\n";
        cerr << "Example: ./detect ../data/image.png ../checkpoint/model_best.pth\n";
        return 0;
    }

    // choice GPU or CPU
    torch::Device device = select_device();
    // init model
    auto model = std::make_shared<LeNet>();
    if (argv[2])
        torch::load(model, argv[2]);
    else
        torch::load(model, "../checkpoint/model_best.pth");
    // check model is loaded.
    AT_ASSERT(model != nullptr);

    // move to GPU
    model->to(device);

    // load image
    cv::Mat image = cv::imread(argv[1]);
    if (image.empty()) {
        std::cerr << "Cant't load image.\n";
        return -1;
    }
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    cv::resize(gray_image, gray_image, cv::Size(28, 28));


    torch::Tensor input_tensor = torch::from_blob(gray_image.data, {1, 1, 28, 28}, torch::kByte).to(torch::kFloat).div_(255);
    input_tensor = input_tensor.to(device); // 将张量移动到GPU上
    // 进行预测
    model->eval(); // 将模型设置为评估模式
    auto output = model->forward(input_tensor).to(device);
    auto max_result = output.max(1, true);
    auto classes = std::get<1>(max_result).item<int>();
    cout << int(classes) << endl;
    return 0;
}