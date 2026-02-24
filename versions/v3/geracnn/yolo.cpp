#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

int main() {
    cv::dnn::Net net = cv::dnn::readNetFromONNX("best.onnx");

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    cv::Mat img = cv::imread("test.jpg");
    if (img.empty()) {
        std::cerr << "Erro ao carregar imagem\n";
        return -1;
    }

    cv::Mat blob = cv::dnn::blobFromImage(
        img, 1/255.0, cv::Size(320, 320),
        cv::Scalar(), true, false
    );

    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    net.forward(outputs);

    std::cout << "Inferência OK, outputs: " << outputs.size() << "\n";
    std::cout << outputs[0].size << "\n";

    return 0;
}
