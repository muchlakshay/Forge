#pragma once
#include <filesystem>
#include <zlib.h>
#include "download.h"
#include <string>
#include <vector>

using namespace std::literals;

namespace Forge::Extra::Scripts{
    void getMNIST(const std::string& output);
}

inline void Forge::Extra::Scripts::getMNIST(const std::string& output) {
    std::vector<std::string> data {"https://raw.githubusercontent.com/lessLakshay/MNSIT-Dataset/main/MNIST.zip"};
    const auto output_ {(std::filesystem::path(output)/"MNIST.zip").string()};
    const auto unzip_output {(std::filesystem::path(output)).string()};
    download_with_fallback(data, output_);
    std::system(("tar -xf " + output_ + " -C " + unzip_output).c_str());
}
