#include "tokenizer.h"

#include <utility>

void print2(StringVec vec) { for (const auto& s : vec)std::cout << s << " "; std::cout << "\n"; }

StringVec2D SimpleTokenizer::split_str(const std::string& str, std::function<void(std::string&)> transformation) const {
	StringVec2D tokens;
	StringVec word_vec;

	std::regex pattern(R"('s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9_]+|\s+)");
	auto word_start{ std::sregex_iterator(str.begin(), str.end(), pattern) };
	auto word_end{ std::sregex_iterator() };

	for (auto i{ word_start }; i != word_end; ++i) {
		std::string word{ i->str() };
		if (transformation!=nullptr) transformation(word);
		for (const auto& c : word) word_vec.push_back(std::string(1, c));
		tokens.push_back(word_vec);
		word_vec.clear();
	}
	return tokens;
}

StringVec2D SimpleTokenizer::merge_pair(const StringVec2D& tokens, string_pair pair) const {
	std::string& first{ pair.first };
	std::string& second{ pair.second };

	StringVec2D new_tokens;

	for (const auto& word : tokens) {
		std::vector<std::string> new_word;
		size_t i{};
		while (i < word.size()) {
			if (i < word.size() - 1 && word[i] == first && word[i + 1] == second) {
				new_word.push_back(first + second);
				i += 2;
			}
			else {
				new_word.push_back(word[i]);
				i++;
			}
		}
		if (!new_word.empty()) new_tokens.push_back(new_word);
	}

	return new_tokens;
}

void SimpleTokenizer::clean_content(std::string& content) const {
	std::transform(content.begin(), content.end(),
		content.begin(), [](unsigned char c) {return std::tolower(c); });
	std::replace_if(content.begin(), content.end(),
		[](unsigned char c) {return !std::isalnum(c); }, ' ');
}

string_pair SimpleTokenizer::get_frequent_pair(const StringVec2D& tokens, bool isEncoding) const {

	if (tokens.size() == 0) return string_pair{ "", "" };

	std::map<string_pair, int> m_token_pairs;
	for (const auto& word : tokens) {
		for (int i{}; i < word.size() - 1; ++i) {
			m_token_pairs[{word[i], word[i + 1]}]++;
		}
	}

	auto it{ std::max_element(m_token_pairs.begin(), m_token_pairs.end(),
			[](const auto& p1, const auto& p2) {
				return p1.second < p2.second;
			}
	) };

	return it->second > 1 ? it->first : string_pair{ "", "" };
}


void SimpleTokenizer::BPE(const std::string& filename, int max_vocab) {
	m_ids.clear();
	int id{};
	m_ids["<pad>"] = id++;
	m_ids["<bos>"] = id++;
	m_ids["<eos>"] = id++;

	for (int c{}; c <= 255; ++c) m_ids[std::string(1, c)] = id++;

	std::ifstream file(filename, std::ios::binary);

	if (!file) {
		std::cerr << "Couldn't Open The File\n";
		return;
	}

	std::ostringstream buffer;
	buffer << file.rdbuf();

	std::string content{ buffer.str() };
	file.close();

	StringVec2D tokens{ split_str(content) };
	while (m_ids.size() < max_vocab) {
		string_pair frequent_pair{ get_frequent_pair(tokens) };
		auto& first{ frequent_pair.first };
		auto& second{ frequent_pair.second };
		if (first.empty() || second.empty()) break;
		m_ids[first + second] = id++;
		tokens = std::move(merge_pair(tokens, frequent_pair));
		//std::cout <<"Vocab Size: " << m_ids.size()<<"\n";
	}

	m_vocabulary_size = m_ids.size();

	m_tokens.resize(m_ids.size());
	for (const auto& p : m_ids)m_tokens[p.second] = p.first;

}

void SimpleTokenizer::on_file(const std::string& filename, int max_vocab) {
	if (max_vocab < MIN_VOCAB_SIZE)
		throw std::invalid_argument("Vocabulary Size Can not be less then 600");

	BPE(filename, max_vocab);
};


StringVec SimpleTokenizer::submerge_subvec_BPE(StringVec sub_vec) const {
	while (true) {
		int best_merge_id{ INT_INF };
		int merge_idx{ -1 };
		for (int j{}; j < sub_vec.size() - 1; ++j) {
			std::string pair{ sub_vec[j] + sub_vec[j + 1] };
			auto it{ m_ids.find(pair) };
			if (it != m_ids.end()) {
				if (it->second < best_merge_id) {
					best_merge_id = { it->second };
					merge_idx = j;
				};
			}
		}
		if (merge_idx == -1) {
			break;
		}
		StringVec new_sub_vec;
		for (int j{}; j < sub_vec.size(); ++j) {
			if (j == merge_idx) {
				new_sub_vec.push_back(sub_vec[j] + sub_vec[j + 1]);
				j++;
			}
			else new_sub_vec.push_back(sub_vec[j]);
		}
		sub_vec = std::move(new_sub_vec);
	}
	return sub_vec;
}

EigenVectori SimpleTokenizer::encode_str_BPE(const std::string& str,
	std::function<void(std::string&)> transformation) const {

	StringVec2D tokens{ split_str(str, std::move(transformation)) };
	// print2(tokens[0]);
	// print2(tokens[1]);
	std::vector<int> final_tokens_ids;
	for (int i{}; i < tokens.size(); ++i) {
		auto& sub_vec{ tokens[i] };
		if (sub_vec.size() < 2) {
			if (sub_vec.size() > 0)
				final_tokens_ids.push_back(m_ids.find(sub_vec[0])->second);
			continue;
		}
		for (const auto& token : submerge_subvec_BPE(sub_vec)) {
			auto it{ m_ids.find(token) };
			if (it != m_ids.end()) final_tokens_ids.push_back(it->second);
			else {
				for (const auto& c : token)
					final_tokens_ids.push_back(m_ids.find(std::string(1, c))->second);
			}
		}
	}

	if (final_tokens_ids.empty()) return EigenVectori(0);

	EigenVectori encoded_vec{ Eigen::Map<const EigenVectori>(final_tokens_ids.data(), final_tokens_ids.size()) };
	return encoded_vec;
}

Forge::Tensor SimpleTokenizer::encode(std::string str, std::function<void(std::string&)> transformation) const {
	auto encoded {encode_str_BPE(str, std::move(transformation))};
	return Forge::Tensor::FromHostPtr(encoded.data(), {static_cast<std::size_t>(encoded.size())}, false).clone();
}

StringVec SimpleTokenizer::decode(const Forge::Tensor& ids) const {
	EigenVectori token_ids { Eigen::Map<Eigen::VectorXi>(static_cast<int*>(ids.data()), ids.size())};
	StringVec str_vec;
	for (int i{}; i < token_ids.size(); ++i) {
		if (token_ids[i] > m_tokens.size())
			throw std::invalid_argument(
				"Invalid Token ID Found (Token-" + std::to_string(token_ids[i]) + ")");

		str_vec.push_back(m_tokens[token_ids(i)]);
	}
	return str_vec;
}

EigenMatrixi SimpleTokenizer::encode_file(const std::string& filename, int max_words_len) const {
	std::ifstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Couldn't Open The File\n";
		return {};
	}

	std::string line{ "" };
	int max_len{};
	int num_lines{};
	std::vector<EigenVectori> tokens_vec;
	while (std::getline(file, line)) {
		if (line.empty()) continue;

		auto encoded_vec{encode_str_BPE(line) };
		tokens_vec.push_back(encoded_vec);

		//std::cout << encoded_vec.transpose() << "\n";
		max_len = std::max(static_cast<int>(encoded_vec.size()), max_len);
		num_lines++;
	}

	EigenMatrixi token_ids{ EigenMatrixi::Zero(num_lines, max_len) };
	for (int i{}; i < tokens_vec.size(); ++i) {
		auto& line_tokens{ tokens_vec[i] };
		line_tokens.conservativeResizeLike(EigenVectori::Zero(max_len));
		token_ids.row(i) = line_tokens;
	}
	return token_ids;
}

void SimpleTokenizer::save(const std::string& filename) const {
	std::ofstream file(filename, std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Couldnt Open File " << filename << "\n";
		return;
	}
	file.write(reinterpret_cast<const char*>(&m_vocabulary_size), sizeof(m_vocabulary_size));
	for (const auto& vocab : m_ids) {
		std::size_t vocab_str_len{ vocab.first.size() };
		file.write(reinterpret_cast<const char*>(&vocab_str_len), sizeof(vocab_str_len));
		file.write(vocab.first.data(), vocab_str_len);
		file.write(reinterpret_cast<const char*>(&(vocab.second)), sizeof(vocab.second));
	}
	file.close();
}

void SimpleTokenizer::load(const std::string& filename) {
	std::ifstream file(filename, std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Couldnt Open File " << filename << "\n";
		return;
	}
	file.read(reinterpret_cast<char*>(&m_vocabulary_size), sizeof(m_vocabulary_size));

	if (m_vocabulary_size > 0) m_tokens.resize(m_vocabulary_size);
	for (int i{}; i < m_vocabulary_size; ++i) {
		std::string vocab;
		std::size_t vocab_str_len{};
		int token_id{};
		file.read(reinterpret_cast<char*>(&vocab_str_len), sizeof(vocab_str_len));
		vocab.resize(vocab_str_len);
		file.read(vocab.data(), vocab_str_len);
		file.read(reinterpret_cast<char*>(&token_id), sizeof(token_id));

		m_ids[vocab] = token_id;
		m_tokens[token_id] = vocab;
	}

	file.close();
}

void SimpleTokenizer::fill_reverse() {for (auto& e : m_ids) m_ids_reverse[e.second] = e.first;}