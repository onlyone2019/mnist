//
// Created by wangjie on 23-4-4.
//
#include <torch/cuda.h>
#include <torch/data.h>
#include "../include/model.h"
#include "../include/utils.h"


auto main(int argc, const char *argv[]) -> int {
    if (argc < 2) {
        std::cerr << "Usage: ./train <data-to-path>\n";
        std::cerr << "Example: ./train ../data\n";
        return 0;
    }

    const char *checkpoint_path = "../checkpoint";
    if (access(checkpoint_path, 0))
        mkdir(checkpoint_path, 0755);

    std::string model_path = checkpoint_path;
    model_path.append("/");
    model_path.append("checkpoint.pth");
    std::string model_best_path = checkpoint_path;
    model_best_path.append("/");
    model_best_path.append("model_best.pth");

    // Where to find the MNIST dataset.
    const char *data_root = argv[1];

    // The batch size for training.
    const int64_t batch_size = 64;

    // The number of epochs to train.
    const int64_t epochs = 10;

    std::vector<double> result;
    double best_acc = 0.;
    double accuracy;

    torch::manual_seed(1);

    // choice GPU or CPU
    torch::Device device = select_device();

    auto model = std::make_shared<LeNet>();
    model->to(device);

    // Load dataset
    auto train_dataset =
            torch::data::datasets::MNIST(data_root)
                    .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                    .map(torch::data::transforms::Stack<>());
    const size_t train_dataset_size = train_dataset.size().value();
    auto train_loader =
            torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
                    std::move(train_dataset), batch_size);

    auto test_dataset =
            torch::data::datasets::MNIST(data_root,
                                         torch::data::datasets::MNIST::Mode::kTest)
                    .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
                    .map(torch::data::transforms::Stack<>());
    const size_t test_dataset_size = test_dataset.size().value();
    auto test_loader =
            torch::data::make_data_loader(std::move(test_dataset), batch_size * 20);

    torch::optim::SGD optimizer(model->parameters(),
                                torch::optim::SGDOptions(0.01).momentum(0.5));

    for (size_t epoch = 1; epoch <= epochs; ++epoch) {
        train(epoch, model, device, *train_loader, optimizer, train_dataset_size);
        result = evaluate(model, device, *test_loader, test_dataset_size);
        accuracy = result[1];

        // save model
        torch::save(model, model_path);

        // save best model
        if (accuracy >= best_acc) {
            torch::save(model, model_best_path);
            best_acc = accuracy;
        }
    }
}
