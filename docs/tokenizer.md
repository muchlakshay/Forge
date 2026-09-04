# Tokenizer

this class, `SimpleTokenizer` is a from-scratch byte pair encoding (BPE) tokenizer with GPT-2-style pre-tokenization.
it first splits text into chunks using the same general regex pattern as GPT-2.
### Training

```c++
void on_file(const std::string& filename, int max_vocab = DEFAULT_MAX_VOCAB);
```

this function trains the tokenizer from a text file that contains the corpus, the initial vocabulary contains 3 special
tokens (`<pad>`, `<bos>`, `<eos>`) and all 256 possible chars/byte values.

`max_vocab` must be at least 300 (`MIN_VOCAB_SIZE`).

**example**
```c++
SimpleTokenizer tok(SimpleTokenizer::Type::BPE);
tok.on_file("corpus.txt", /*max_vocab=*/8000);
```

### Encoding and decoding

```c++
//to encode
Forge::Tensor encode(std::string str, std::function<void(std::string&)> transformation=nullptr)

//to decode
StringVec decode(const Forge::Tensor& token_ids) const
```
**example**
```c++
Tensor ids {tok.encode("he went to the")};   // returns a 1-D int32 token ids Tensor
std::vector<std::string> tokens {tok.decode(ids)}; // token strings
```

`encode` returns a 1-D `Forge::Tensor` of `int32` token ids

`decode` converts the token ids back into their corresponding token strings and throws `std::invalid_argument` for IDs outside the vocabulary

`encode` can optionally take a `transformation` callback to modify each pretokenized chunk before the BPE merging, such as converting text to lowercase

### Saving And Loading

```c++
    void save(const std::string& filename) const;
    void load(const std::string& filename);
```

`save()` saves to a binary file, and `load()` loads from it

**example**

```c++
//saving a trained tokenizer
tok.save("merge_rules.tk"); 

//loading from the saved binary file
SimpleTokenizer tok2 {SimpleTokenizer::Type::BPE};
tok2.load("merge_rules.tk");
```

### Member Functions

| Member Function                   | Usage                                                         |
|-----------------------------------|---------------------------------------------------------------|
| `getIds()`                        | returns a std::map<std::string, int> contains the merge rules |
| `vocabularySize()`                | returns the vocabulary size                                   |
