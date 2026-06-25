#pragma once

#include <map>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <set>
#include <vector>
#include <utility>
#include <regex>
#include "Eigen/Dense"
#include "tensor.h"

using EigenMatrixf = Eigen::MatrixXf;
using EigenMatrixi = Eigen::MatrixXi;
using EigenVectorf = Eigen::VectorXf;
using EigenVectori = Eigen::VectorXi;
using vec3d = std::vector<EigenMatrixf>;
using vec4d = std::vector<vec3d>;
using string_pair = std::pair<std::string, std::string>;
using StringVec = std::vector<std::string>;
using StringVec2D = std::vector <StringVec>;

using namespace std::string_literals;

class SimpleTokenizer {
public:
    enum class Type { BPE };
private:

    static constexpr int UNKNOWN_ID{ 1 };
    static constexpr int DEFAULT_MAX_VOCAB{ 500 };
    static constexpr int MIN_VOCAB_SIZE{ 300 };
    static constexpr int INT_INF{ std::numeric_limits<int>::max() };

    std::map<std::string, int> m_ids;
    StringVec m_tokens;
    int m_vocabulary_size{};
    Type m_type;

    void clean_content(std::string& content) const;
    [[nodiscard]] StringVec2D split_str(const std::string& str) const;
    [[nodiscard]] string_pair get_frequent_pair(const StringVec2D& tokens, bool isEncoding = false) const;
    [[nodiscard]] StringVec2D merge_pair(const StringVec2D& tokens, string_pair pair) const;
    void BoW(const std::string& filename);
    void BPE(const std::string& filename, int max_vocab);
    [[nodiscard]] StringVec submerge_subvec_BPE(StringVec sub_vec) const;
    [[nodiscard]] EigenVectori encode_str_BPE(const std::string&) const;
    [[nodiscard]] EigenMatrixi encode_file_BPE(const std::string& filename) const;

public:
    SimpleTokenizer(Type type) : m_type{ type } {};
    void on_file(const std::string& filename, int max_vocab = DEFAULT_MAX_VOCAB);
    [[nodiscard]] Forge::Tensor encode(std::string str) const;
    [[nodiscard]] EigenMatrixi encode_file(const std::string& filename, int max_words_len = -1) const;
    [[nodiscard]] StringVec decode(const EigenVectori& token_ids) const;
    void save(const std::string& filename) const;
    void load(const std::string& filename);

    [[nodiscard]] auto getIds() const { return m_ids; }
    [[nodiscard]] int vocabularySize() const { return m_vocabulary_size; }
};
