#include <iostream>
#include <torch/torch.h>
// 定义卷积神经网络模型
class Net : public torch::nn::Module {
public:
    Net() {
        // 定义卷积层和池化层
        conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 5).stride(1).padding(2));
        pool1 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2));
        conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 5).stride(1).padding(2));
        pool2 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2));
        // 定义全连接层
        fc1 = torch::nn::Linear(7 * 7 * 64, 128);
        fc2 = torch::nn::Linear(128, 10);
        // 注册所有层
        register_module("conv1", conv1);
        register_module("pool1", pool1);
        register_module("conv2", conv2);
        register_module("pool2", pool2);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }
    // 前向传播函数
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(pool1(conv1(x)));
        x = torch::relu(pool2(conv2(x)));
        x = x.view({-1, 7 * 7 * 64});
        x = torch::relu(fc1(x));
        x = fc2(x);
        return torch::log_softmax(x, /*dim=*/1);
    }
private:
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::MaxPool2d pool1{nullptr}, pool2{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};
// 训练模型
void train(Net& model, torch::data::DataLoader& train_loader, torch::data::DataLoader& test_loader,
           size_t batch_size, size_t epochs, double learning_rate) {
    // 将模型和数据转移到 GPU 设备上
    torch::Device device(torch::kCUDA);
    model.to(device);
    auto criterion = torch::nn::NLLLoss();
    auto optimizer = torch::optim::SGD(model.parameters(), learning_rate);
    // 训练模型
    for (size_t epoch = 1; epoch <= epochs; ++epoch) {
        double running_loss = 0.0;
        size_t num_samples = 0;
        model.train();
        for (auto& batch : train_loader) {
            auto data_batch = batch.data.to(device);
            auto label_batch = batch.target.to(device);
            optimizer.zero_grad();
            auto output = model.forward(data_batch);
            auto loss = criterion(output, label_batch);
            loss.backward();
            optimizer.step();
            running_loss += loss.item<double>() * data_batch.size(0);
            num_samples += data_batch.size(0);
        }
        double train_loss = running_loss / num_samples;
        double train_acc = 0.0;
        model.eval();
        size_t num_correct = 0;
        for (auto& batch : test_loader) {
            auto data_batch = batch.data.to(device);
            auto label_batch = batch.target.to(device);
            auto output = model.forward(data_batch);
            auto predicted_labels = output.argmax(1);
            num_correct += predicted_labels.eq(label_batch).sum().item<int64_t>();
        }
        double test_acc = static_cast<double>(num_correct) / test_loader.dataset().size().value();
        std::cout << "Epoch " << epoch << ": Train Loss = " << train_loss << ", Test Accuracy = " << test_acc << std::endl;
    }
    // 保存模型
    torch::save(model, "mnist_cnn_gpu.pt");
}
int main() {
    // 加载 MNIST 数据集
    auto train_dataset = torch::data::datasets::MNIST("./data").map(torch::data::transforms::Stack<>());
    auto test_dataset = torch::data::datasets::MNIST("./data", torch::data::datasets::MNIST::Mode::kTest).map(torch::data::transforms::Stack<>());
    const size_t batch_size = 64;
    // 创建数据加载器
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(train_dataset), torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));
    // 创建模型并训练
    Net model;
    size_t epochs = 10;
    double learning_rate = 0.01;
    train(model, train_loader, test_loader, batch_size, epochs, learning_rate);
    return 0;
}
