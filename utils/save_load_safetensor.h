#pragma once
#include "tensor.h"
#include "parameter.h"
#include <rfl.hpp>
#include <rfl/json.hpp>
#include <fstream>

namespace Forge {
    std::string get_dtype(Dtype dtype);
    std::size_t get_dtype_size(Dtype dtype);
    Dtype str_to_dtype(std::string str);

    template <typename T>
    void save(T& data_members, const std::string& filename);
    template <typename T>
    void load(T& data_members, const std::string& filename);

    std::map<std::string, Tensor> load_safetensors(const std::string& filename);

    struct TensorMetadata {
        std::string dtype;
        std::vector<std::size_t> shape;
        std::vector<std::size_t> data_offsets;
    };

    template <typename T>
    std::map<std::string, Tensor*> get_state_dict(T& data_members);

    using Generic = std::variant<std::map<std::string, std::string>, TensorMetadata>;
    template <typename T>
    concept ReflectableWithView = requires(const T& x) {rfl::to_view(x);};
}

inline std::string Forge::get_dtype(const Dtype dtype) {
    switch (dtype) {
        case Dtype::float32: return "F32";
        default: throw std::invalid_argument("Unsupported dtype");
    }
}

inline std::size_t Forge::get_dtype_size(Dtype dtype) {
    switch (dtype) {
        case Dtype::float32: return 4;
        default: throw std::invalid_argument("Unsupported dtype");
    }
}

inline Forge::Dtype Forge::str_to_dtype(std::string str) {
    if (str == "F32") return Dtype::float32;
    throw std::invalid_argument("Unsupported dtype");
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
                    state_dict[p.name] = p.m_param_ptr;
                }
            }
            if constexpr (auto value {member.value()}; std::is_aggregate_v<std::remove_pointer_t<decltype(value)>>) {
                for (auto& p : get_state_dict(value))state_dict[p.first] = p.second;
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

inline std::map<std::string, Forge::Tensor> Forge::load_safetensors(const std::string &filename) {
    std::ifstream file{filename, std::ios::binary};
    if (!file.is_open()) throw std::runtime_error(std::format("Couldn't open file '{}'", filename));

    std::map<std::string, TensorMetadata> metadata{};
    std::map<std::string, Tensor> state;
    uint64_t header_size {};
    std::string json_header;

    file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    json_header.resize(header_size);
    file.read(json_header.data(), static_cast<std::streamsize>(header_size));
    std::size_t fptr {file.tellg()};

    auto generic {rfl::json::read<rfl::Object<Generic>>(json_header).value()};
    for (auto& entry : generic) {
        if (entry.first=="__metadata__") continue;
        auto md {std::get<TensorMetadata>(entry.second)};
        Tensor data {{md.shape}, str_to_dtype(md.dtype), false, Device::CPU};

        file.clear();
        file.seekg(fptr+md.data_offsets[0]);
        file.read(static_cast<char*>(data.data()), data.size()*get_dtype_size(data.dtype()));
        if (!file) {
            throw std::runtime_error(std::format("Failed to read data block for tensor: {}", entry.first));
        }
        state[entry.first] = data;
    }
    return std::move(state);
}

template<typename T>
void Forge::load(T &data_members, const std::string &filename) {
    auto loaded_state {load_safetensors(filename)};
    auto state_dict {get_state_dict(data_members)};

    if (loaded_state.size() != state_dict.size()) throw std::runtime_error(
        std::format("Couldn't load '{}, Architecture Differs'", filename));

    for (auto& entry : loaded_state) *state_dict[entry.first] = entry.second;
}
