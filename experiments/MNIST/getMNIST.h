#pragma once
#include <filesystem>
#include "../../data/Extra/download.h"
#include <string>
#include <vector>

using namespace std::literals;

namespace Forge::Extra{
    void getMNIST(const std::string& output);
}

inline void Forge::Extra::getMNIST(const std::string& output) {
    if (!std::filesystem::is_directory(output)) std::filesystem::create_directory(output);
    std::vector<std::string> data {"https://raw.githubusercontent.com/lessLakshay/MNSIT-Dataset/main/MNIST.zip"};
    const auto output_ {std::filesystem::absolute(std::filesystem::path(output)/"MNIST.zip").string()};
    const auto unzip_output {std::filesystem::absolute(std::filesystem::path(output)).string()};
    download_with_fallback(data, output_);
    const auto command {"cmake -E tar xf \"" + output_ + "\" --format=zip"};
    const auto previous_path {std::filesystem::current_path()};
    std::filesystem::current_path(unzip_output);
    const int status {std::system(command.c_str())};
    std::filesystem::current_path(previous_path);
    if (status != 0) throw std::runtime_error("Failed to extract MNIST.zip");
}
