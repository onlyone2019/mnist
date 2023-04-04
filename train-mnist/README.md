# 使用 gpu 版 LibTorch 训练并测试 mnist

## 依赖：
- cuda 11.1
- libtorch 1.9.0
- opencv

(具体版本影响不大，opencv装的是最新版，libtorch可以写一个简单的测试程序，能用即可)

## 训练

### 编译
```bash
git clone https://github.com/onlyone2019/mnist.git
cd mnist/train-mnist
mkdir build
cd build
cmake ..
make
```

### 下载数据集
```bash
cd <repo>/data
chmod +x download.sh
./download.sh
```

### 训练
```bash
cd <repo>/build
./train ../data
```

## 推理
```bash
cd <repo>/build
./detect ../data/image.png
or
./detect ../data/image.png ../checkpoint/model_best.pth
```
