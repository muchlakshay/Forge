#pragma once
#include "data/Extra/download.h"
#include <filesystem>

namespace Forge {
    namespace Extra {
        void download_GPT2_124M(const std::string& opt);
    }
}

inline void Forge::Extra::download_GPT2_124M(const std::string &opt) {
    if (std::filesystem::exists(opt)) {
        std::cout<<"file already exists\n";
        return;
    }
    std::vector<std::string> urls{
        "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"
    };
    download_with_fallback(urls, opt);
}
