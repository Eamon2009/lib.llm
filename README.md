<h1 align="center"> lib.llm - Open Source</h1>

### The Repository
This repository is where the lib.llm project is developed. It is LLM training in C++17 with no frameworks. The core is roughly 1,000 lines of dependency-free C++ spread across `main.cpp`, `config/config.h`, and `include/*.h`.

If you want to understand what `loss.backward()` actually does without PyTorch hiding the details, this is the place. There is no need for PyTorch or Python to train a transformer locally.

The implementation is a decoder-only GPT with custom tensors, embeddings, multi-head causal self-attention, layer normalization, cross-entropy loss, and an analytical backward pass with an Adam-style optimizer. A token-level BPE tokenizer lives in [`include/tokenizer.h`](include/tokenizer.h). There is no autograd engine and no external framework, so every gradient is explicitly derived and written out.

This is not a framework. It is a reference implementation: the kind of thing you build once to prove to yourself that you understand every operation from the matrix multiplications up to the cross-entropy loss, and then keep around because it is genuinely useful for training small models on a laptop CPU without fighting a Python environment.

This source code is available to everyone under the GPL-3.0 license.

## lib.llm

lib.llm takes a text file, tokenizes it, and trains a transformer to predict the next token. The code is organized the way you would think about the problem:

- A `GPTLanguageModel` class that holds the parameters
- A forward pass that computes logits and loss
- A backward pass that walks the computation graph in reverse
- An `AdamW` optimizer that updates the weights

Everything is explicit. If you want to know how gradient accumulation works, how the causal mask is applied, or how the repetition penalty modifies the logits during sampling, read the code and it is right there.

You can:

- Train a model from scratch, saving the best checkpoint based on validation loss
- Load a checkpoint and generate text indefinitely
- Start an interactive chat session where a system prompt is prepended to every turn and tokens stream back in real time

The architecture is fully configurable through a single header file: embedding dimension, number of layers, attention heads, context length, and learning rate schedule.

## Getting Started

No CMake required. Prepare the data, then compile with `g++` directly:

```bash
cd data
python data_set.py
```

```bash
# Compile
g++ -std=c++17 -O3 -march=native -fopenmp -I. -Iinclude -o llm.exe main.cpp

# Train
./llm.exe data/input.txt
```

This is a single training loop packed into a binary that runs on Linux, macOS, and Windows. It trains from scratch on `data/input.txt` and writes the best checkpoint to `best_model.bin`.

Once you have a checkpoint:

```bash
./llm.exe data/input.txt --generate
./llm.exe data/input.txt --chat --chat-tokens 300
```

Debugging tip: swap `-O3` for `-g` when compiling if you want to step through `include/backward.h` or `include/gpt.h` in a debugger. The manual backward pass is much easier to follow one breakpoint at a time.

### Example Output

```text
[DATA]  Total tokens : 3521179
[DATA]  Train tokens : 3169061
[DATA]  Val tokens   : 352118
  +------------------------------------------+------------------------------------------+
  | LLM Architecture                                                                    |
  +------------------------------------------+------------------------------------------+
  | Max Context Length   : 24                | Vocab Size (BPE)     : 2056              |
  | Number of Layers     : 6                 | Attention Heads      : 6                 |
  | Embedding Channels   : 128               | Total Parameters     : 1712904           |
  | Repetition Penalty   : 500               | Repetition Window    : 500               |
  +------------------------------------------+------------------------------------------+

  +-------------------------------------------------------------------------------------+
  | Host Hardware Specs                                                                 |
  +-------------------------------------------------------------------------------------+
  | Host CPU Device      : AMD Ryzen 5 PRO 3500U w/ Radeon...                           |
  | Host RAM (Total)     : 6045 MB                                                      |
  +-------------------------------------------------------------------------------------+

step 1/20000 (0.01%) | train loss 7.647731 | val loss 7.663259 | lr 3.00e-07 | 960.52 ms | 199 tok/s | ram 70.9 MB
step 2/20000 (0.01%) | train loss 7.637784 | val loss 7.663259 | lr 6.00e-07 | 1243.27 ms | 154 tok/s | ram 70.9 MB
step 3/20000 (0.01%) | train loss 7.658248 | val loss 7.663259 | lr 9.00e-07 | 994.39 ms | 193 tok/s | ram 70.9 MB
step 4/20000 (0.02%) | train loss 7.643033 | val loss 7.663259 | lr 1.20e-06 | 1025.49 ms | 187 tok/s | ram 70.9 MB
step 5/20000 (0.03%) | train loss 7.623671 | val loss 7.663259 | lr 1.50e-06 | 1013.43 ms | 189 tok/s | ram 71.0 MB
[SAVE]  Weights written to best_model.bin
step 6/20000 (0.03%) | train loss 7.665278 | val loss 7.654684* | lr 1.80e-06 | 1042.40 ms | 184 tok/s | ram 71.0 MB
generating:
What sall I hae a sight of the king's crow
...
```

### Memory Tracking

RAM is measured with `get_ram_usage_mb()` and printed after every training step, next to the loss and tokens per second.

- Linux: reads `VmRSS` from `/proc/self/status`
- macOS: calls `task_info` for `resident_size`
- Windows: reads `WorkingSetSize` from `K32GetProcessMemoryInfo`

If you are training on a laptop with 16 GB of RAM and the printed value climbs past 12 GB, you know immediately. You can watch the resident set size jump when the model initializes, hold steady through the forward and backward passes, and tick up during the periodic validation run when a second forward graph is alive. If the number is too high, reduce `BATCH_SIZE` or `N_LAYER` in `config.h` and recompile. The memory footprint is predictable because every byte is accounted for in the code.

### Runtime Arguments

```
llm.exe [data_path] [--generate] [--chat] [--chat-tokens N]
```

Environment variables:

- `GPT_DATA_PATH` (default `data/input.txt`): override the default training corpus
- `GPT_MODEL_PATH` (default `best_model.bin`): override the checkpoint path

## Architecture

The model is a decoder-only GPT with pre-layer-norm residual blocks. Everything is a compile-time constant in `config/config.h`:

```cpp
static const unsigned int SEED = 1337;
static const double TRAIN_SPLIT = 0.9;
static const int BATCH_SIZE = 32;
static const int BLOCK_SIZE = 64;
static const int MAX_ITERS = 5000;
static const int EVAL_INTERVAL = 500;
static const float LEARNING_RATE = 5e-4f;
static const int EVAL_ITERS = 25;
static const int N_EMBD = 128;
static const int N_HEAD = 2;
static const int N_LAYER = 4;
static const float DROPOUT = 0.05f;
static const int BPE_VOCAB_SIZE = 2048;
```

What is in the box:

- Token and positional embeddings
- Multi-head causal self-attention with explicit Q/K/V projections
- Feed-forward MLP (ReLU)
- LayerNorm
- Cross-entropy loss
- Fully analytical backward pass in `include/backward.h`
- AdamW optimizer (first and second moment estimates)
- Checkpoint save and load
- Autoregressive generation and terminal chat mode

### The Backward Pass

No autograd and no `.backward()` magic, just C++ loops that do exactly what the math says.

`SavedForward` caches every intermediate from the forward pass: pre-softmax attention scores, post-softmax weights, dropout masks, ReLU inputs, and layer norm means and inverse standard deviations. The `backward()` function then walks the model in reverse:

- `backward_cross_entropy` produces `dlogits`
- LM head: `backward_linear`, then `backward_layernorm`
- For each block, in reverse:
  - FFN branch: dropout, linear, ReLU, linear, layernorm
  - Residual add
  - MHA branch: dropout, linear, split heads, per-head softmax and QKV backprop
  - Residual add
  - LayerNorm backward
- Embeddings

`Grads` holds accumulators for every parameter (`GradLinear`, `GradEmbedding`, `GradLayerNorm`, and so on). Every gradient is accumulated rather than overwritten, so gradient accumulation across mini-batches works.

`AdamWState` tracks `m` and `v` for every parameter. `apply_grads()` computes bias-corrected moments, then applies `param -= lr * m_hat / (sqrt(v_hat) + eps)`. There is no weight decay in this version, so it behaves as vanilla Adam.

## Tokenizer

The tokenizer lives in `include/tokenizer.h` and has two modes, selected automatically by `load()`:

- TEXT: for small and medium datasets that fit in RAM. Reads a `.txt` file, trains BPE, and stores the result in a `std::vector<int>`.
- SHARDED: for large datasets (billions of tokens). Memory-maps binary shards of `uint16_t` token IDs and streams them on demand.

### TEXT Mode

BPE from scratch. Every unique character starts as its own token, then merges are applied iteratively: find the most common adjacent pair, merge it, repeat. A linked-list structure (`BPEIndex`) tracks active tokens, and a hash map (`pair_pos`) provides fast pair frequency lookup. The merge table is cached to `tokenizer.bin` so you do not retrain on every run.

- Encoding: `base_encode()` followed by `apply_merges()`
- Decoding: vocab lookup and concatenation

### SHARDED Mode

Data is pre-tokenized into binary shards. Each shard is a flat stream of `uint16_t` token IDs with a small header. `uint16_t` caps the vocab at 65,536 but halves disk I/O and memory bandwidth compared to `int32`.

- `MMapShard` handles platform-specific memory mapping (`mmap` on POSIX, `MapViewOfFile` on Windows). Move-only semantics prevent accidental copies of file descriptors.
- `ShardedSplit` builds a prefix-sum index over token counts, so `token_at(global_idx)` is O(1) pointer arithmetic.
- `get_batch()` samples random starting positions and extracts `block_size` consecutive tokens. OpenMP parallelizes across the batch dimension, and each thread gets its own RNG seed for determinism.

## Benchmarks

These are small character-level models on TinyStories. Do not expect GPT-2 quality. The point is to see the pipeline work end to end.

- 0.83M params: 4 layers, dim 128, 4 heads, ctx 64, 105 char vocab, 3,000 iters, val loss 1.6371, 76 min, CPU (AMD Ryzen)
- 2.00M params: 4 layers, dim 200, 4 heads, ctx 200, 110 char vocab, 5,000 iters, val loss 0.9301, 86 min, CPU x64

## Repository Layout

- `main.cpp`: application entry point
- `CMakeLists.txt`: optional CMake build configuration
- `Dockerfile`: container definition for isolated environments
- `config/`
  - `config.h`: hyperparameters and global model settings
- `data/`
  - `data_set.py`: prepares specific dataset formats
  - `dataset.py`: loads and processes training data
  - `export.py`: exports PyTorch weights to the C++ binary format
- `include/`
  - `attention.h`: self-attention
  - `backward.h`: backward pass
  - `block.h`: transformer block
  - `embedding.h`: token and positional embeddings
  - `feedforward.h`: feed-forward network
  - `gpt.h`: core GPT model assembly
  - `layernorm.h`: layer normalization
  - `linear.h`: linear (dense) layer
  - `llm-cpp.hpp`: primary library interface
  - `lm.h`: language modeling head
  - `sampler.h`: token sampling (temperature, top-k, top-p)
  - `tensor.h`: custom CPU tensor
  - `tokenizer.h`: BPE tokenizer
  - `torch_bridge.h`: utilities for bridging with PyTorch tensors
- `scripts/`
  - `build.sh`: builds the project via CMake and Make

## What This Is and Isn't

This is a readable C++ reference for how transformer training works under the hood. If you have read Karpathy's `llm.c` and want the same concepts in C++ with an explicit backward pass, this is it.

This is not a production training framework. Models are tiny (sub-20M parameters), and there is no distributed training, gradient checkpointing, model parallelism, or quantization. If you want to train something useful, use `llm.c`, nanoGPT, or a real framework.

## References

- [Vaswani et al., "Attention Is All You Need", 2017](https://arxiv.org/abs/1706.03762)
- [Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2), 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Karpathy, A., llm.c repository](https://github.com/karpathy/llm.c): LLM training in simple, raw C
- [Karpathy, A., "Let's reproduce GPT-2 (124M)", 2024](https://youtu.be/l8pRSuU81PU): the multi-head attention structure, learning rate schedule, and binary token shard loading were implemented using this walkthrough (see the [build-nanogpt README](https://github.com/karpathy/build-nanogpt/blob/master/README.md))

## License

Copyright (c) lib.llm contributors. Licensed under the [GPL-3.0](LICENSE) license.
