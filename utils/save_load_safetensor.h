#pragma once
#include "tensor.h"
#include "parameter.h"
#include <rfl.hpp>
#include <rfl/json.hpp>
#include <fstream>

namespace Forge {
    std::string get_dtype(Dtype dtype);
    std::size_t get_dtype_size(Dtype dtype);

    template <typename T>
    void save(T& data_members, const std::string& filename);

    struct TensorMetadata {
        std::string dtype;
        std::vector<std::size_t> shape;
        std::vector<std::size_t> data_offsets;
    };

    template <typename T>
    std::map<std::string, Tensor*> get_state_dict(T& data_members);
}

inline std::string Forge::get_dtype(const Dtype dtype) {
    switch (dtype) {
        case Dtype::float16: return "F16";
        case Dtype::float32: return "F32";
        case Dtype::float64: return "F64";
        default: throw std::invalid_argument("Unsupported dtype");
    }
}

inline std::size_t Forge::get_dtype_size(Dtype dtype) {
    switch (dtype) {
        case Dtype::float16: return 2;
        case Dtype::float32: return 4;
        case Dtype::float64: return 8;
        default: throw std::invalid_argument("Unsupported dtype");
    }
}

template<typename T>
std::map<std::string, Forge::Tensor*> Forge::get_state_dict(T &data_members) {
    auto view {rfl::to_view(data_members)};
    std::map<std::string, Tensor*> state_dict;
    view.apply(
        [&state_dict](auto& member) {
            auto mem_name {member.name()};
            if constexpr (auto value {member.value()}; HasParameters<std::remove_pointer_t<decltype(value)>>) {
                for (std::vector<Parameter> parameters {value->parameters()}; auto p : parameters) {
                    auto param_name {std::string{mem_name}+"."+p.name};
                    state_dict[param_name] = p.m_param_ptr;
                }
            }
        }
    );
    return state_dict;
}

template <typename T>
void Forge::save(T& data_members, const std::string& filename) {
    auto view {rfl::to_view(data_members)};
    std::map<std::string, TensorMetadata> header_map;
    std::map<std::string, Tensor*> params;
    std::size_t offset{};

    view.apply([&](auto& member) {
        if constexpr (auto value {member.value()}; HasParameters<std::remove_pointer_t<decltype(value)>>) {
            auto name {member.name()};
            for (std::vector<Parameter> parameters {value->parameters()}; auto p : parameters) {
                TensorMetadata metadata;
                metadata.dtype = get_dtype(p.m_param_ptr->dtype());
                metadata.shape = p.m_param_ptr->shape();
                metadata.data_offsets = std::vector{offset, offset+p.m_param_ptr->size()*get_dtype_size(p.m_param_ptr->dtype())};
                offset = metadata.data_offsets.back();

                auto param_name {std::string(name)+"."+p.name};
                header_map[param_name] = metadata;
                params[param_name] = p.m_param_ptr;
            }
        }
    });

    auto json_header {rfl::json::write(header_map)};
    std::size_t header_size {json_header.size()};
    auto total_header_size {header_size+8};
    auto rem {total_header_size%8};

    if (rem!=0) {
        auto pad_size {8-rem};
        json_header.append(pad_size, ' ');
        header_size = json_header.size();
    }

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error(std::format("Couldn't open file '{}'", filename));
    uint64_t  header_size_uint {static_cast<uint64_t>(header_size)};
    file.write(reinterpret_cast<const char*>(&header_size_uint), sizeof(header_size_uint));
    file.write(json_header.data(), header_size);
    for (auto& hm : header_map) {
        auto tensor {*params[hm.first]};
        file.write(static_cast<const char*>(tensor.data()), tensor.size()*get_dtype_size(tensor.dtype()));
    }
    file.close();
}

