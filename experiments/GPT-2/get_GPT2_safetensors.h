#pragma once
#include "data/Extra/download.h"
#include <filesystem>

namespace Forge {
    namespace Extra {
        void download_GPT2_124M(const std::string& opt);
        void donwload_GPT2_merge_rules(const std::string& opt);
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

inline void Forge::Extra::donwload_GPT2_merge_rules(const std::string &opt) {
    if (std::filesystem::exists(opt)) {
        std::cout<<"file already exists\n";
        return;
    }
    std::vector<std::string> urls{
        "https://github.com/lessLakshay/GPT-2_tokenizer_merge_rules/blob/main/gpt2MergeRules.tk"
    };
    download_with_fallback(urls, opt);
}
