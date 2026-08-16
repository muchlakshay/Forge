#pragma once
#include <fstream>
#include <stdexcept>
#include "../../include/Forge.h"

static uint32_t read_uint32(std::ifstream& f) {
    uint32_t val;
    f.read(reinterpret_cast<char*>(&val), 4);
    return __builtin_bswap32(val);
}

struct MNISTData {
    Forge::Tensor images;
    Forge::Tensor labels;
};

inline MNISTData load_mnist(const std::string& images_path, const std::string& labels_path) {

    std::ifstream img_f(images_path, std::ios::binary);
    if (!img_f) throw std::runtime_error("Cannot open images file: " + images_path);

    uint32_t magic = read_uint32(img_f);
    if (magic != 0x00000803) throw std::runtime_error("Invalid image file magic number");

    uint32_t n_images = read_uint32(img_f);
    uint32_t rows     = read_uint32(img_f);
    uint32_t cols     = read_uint32(img_f);
    uint32_t n_pixels = rows * cols;

    std::vector<uint8_t> raw_images(n_images * n_pixels);
    img_f.read(reinterpret_cast<char*>(raw_images.data()), raw_images.size());

    std::ifstream lbl_f(labels_path, std::ios::binary);
    if (!lbl_f) throw std::runtime_error("Cannot open labels file: " + labels_path);

    uint32_t lbl_magic = read_uint32(lbl_f);
    if (lbl_magic != 0x00000801) throw std::runtime_error("Invalid label file magic number");

    uint32_t n_labels = read_uint32(lbl_f);
    if (n_labels != n_images) throw std::runtime_error("Image/label count mismatch");

    std::vector<uint8_t> raw_labels(n_labels);
    lbl_f.read(reinterpret_cast<char*>(raw_labels.data()), raw_labels.size());

    Forge::Tensor images = Forge::Tensor::Zeros({1, 1, n_images, n_pixels}, false);
    {
        auto img_map = images.as_eigen<float>();
        for (uint32_t i = 0; i < n_images; ++i)
            for (uint32_t p = 0; p < n_pixels; ++p)
                img_map(0, 0, i, p) = static_cast<float>(raw_images[i * n_pixels + p]) / 255.0f;
    }

    Forge::Tensor labels = Forge::Tensor::Zeros({1, 1, n_images, 10}, false);
    {
        auto lbl_map = labels.as_eigen<float>();
        for (uint32_t i = 0; i < n_labels; ++i)
            lbl_map(0, 0, i, raw_labels[i]) = 1.0f;
    }

    return {std::move(images), std::move(labels)};
}
