#pragma once
#include <vector>
#include <string>
#include <iostream>

namespace Forge::Extra {
    bool download_file(const std::string& url, const std::string& output);
    bool download_with_fallback(const std::vector<std::string>& urls, const std::string& output, bool verbose=true);
}

inline bool Forge::Extra::download_file(const std::string& url, const std::string& output) {
    const std::string command {"curl -L " + url + " -o " + output};
    return std::system(command.c_str());
}

inline bool Forge::Extra::download_with_fallback(const std::vector<std::string>& urls, const std::string& output, bool verbose) {
    if (verbose) std::cout<<"Downloading ...\n";
    for (const auto& url : urls) {
        if (verbose) std::cout<<"Trying: "<<url<<"\n";
        if (!download_file(url, output)) {
            if (verbose) std::cout<<"Download Complete!!\n";
            return true;
        }
    }
    if (verbose) std::cout<<"Download Failed!!\n";
    return false;
}
