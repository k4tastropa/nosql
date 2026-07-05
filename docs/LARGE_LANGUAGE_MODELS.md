A large language model, or LLM, is a neural network trained to predict and generate sequences of tokens. In text models, those tokens usually represent words, word fragments, punctuation, or other text pieces. The model takes an input sequence, converts it into vectors, processes those vectors through many transformer layers, and produces probabilities for what token should come next. The same broad idea can also be extended to images, audio, or video, but a language model specifically is centered around token sequences.

At a mechanical level, an LLM is not a separate elaborate reasoning machine. It is a transformer-based neural network: embeddings, repeated attention and MLP blocks, normalization, residual connections, and an output head. Its reasoning-like behavior comes from learned weights and large-scale training. By learning to predict tokens across huge amounts of data, the model develops internal representations of grammar, facts, code, patterns, and problem-solving behavior, but these are emergent properties of the neural network rather than hand-written reasoning rules.
# 1. Embedding

When text input is received, the text is split into **tokens**. A token may be a whole word, part of a word, punctuation, or another text fragment, depending on the tokenizer. For English, one token averages about **4 characters** or **0.75 words**.

Model then maps token to an integer ID, then performs a lookup in a learned embedding matrix. Each row in the matrix is arbitrary vector whose values are learned during training.

                             Input Text
                                  │
                                  ▼
                   "The cat sat on the mat."
                                  │
                                  ▼
                        ┌───────────────────┐
                        │   Tokenization    │
                        └───────────────────┘
                                  │
                                  ▼
        ["The", "cat", "sat", "on", "the", "mat", "."]
                                  │
                                  ▼
                        ┌───────────────────┐
                        │     Token IDs     │
                        └───────────────────┘

        "The" →  412
        "cat" → 9821
        "sat" → 3754
        "on"  →  526
        "the" →  279
        "mat" → 6418
        "."   →   13

                                  │
                                  ▼
                   ┌─────────────────────────┐
                   │    Embedding Lookup     │
                   │ (Learned Matrix E)      │
                   └─────────────────────────┘
                                  │
                                  ▼

        412  → [ 0.12, -0.44,  0.91, ... ]
        9821 → [-0.82,  1.37,  0.05, ... ]
        3754 → [ 0.33,  0.08, -1.14, ... ]
        ...

                                  │
                                  ▼
                    Sequence of Embedding Vectors

        [
          [ 0.12, -0.44,  0.91, ... ],
          [-0.82,  1.37,  0.05, ... ],
          [ 0.33,  0.08, -1.14, ... ],
          ...
        ]

                                  │
                                  ▼
                         ┌────────────────┐
                         │  Transformer   │
                         └────────────────┘

If we map those vectors to a multidimensional axis, some words that are contextually close will map closer to each other.

                 Embedding Space
        simplified 3D view of many-dimensional vectors


                         meaning axis 2
                              ▲
                              │
                              │              ● "tiger"
                              │            ↗
                              │          ↗
                              │      ● "cat"
                              │    ↗
                              │  ↗
                              │
                              │
                              │                         ● "car"
                              │                      ↗
                              │                   ↗
                              │               ● "truck"
                              │
                              │
                              └────────────────────────────────► meaning axis 1
                             ╱
                            ╱
                           ╱
                          ╱
                         ▼
                  meaning axis 3

*The axes are not literally named “meaning axis 1/2/3” in a real model. Real embeddings may have hundreds or thousands of dimensions, and individual dimensions usually are not clean human concepts. This is just a simplified projection.*

## 2. Transformer

The sequence of embedding vectors is passed through a stack of **transformer blocks**. A transformer is not just one self-attention step. It is usually many repeated layers, and each layer updates the token representations a little more.

A simplified transformer looks like this:

```txt
embedding vectors
        │
        ▼
transformer block 1
        │
        ▼
transformer block 2
        │
        ▼
transformer block 3
        │
        ▼
...
        │
        ▼
final token representations
```

Each transformer block usually contains two main parts:

```txt
self-attention
MLP / feed-forward network
```

The **self-attention** part lets tokens exchange information with other tokens. It asks which tokens are relevant to each other and how much information should flow between them.

The **MLP** part then processes each token’s updated representation internally. It does not compare tokens to each other. It transforms each token vector independently using learned weights.

So the transformer alternates between these two operations:

```txt
attention = mix information between tokens
MLP       = process each token's features internally
```

This repeats many times. That repetition is important. The first attention layer may detect simple relationships. Later layers can build on those relationships and form more abstract representations.

A simplified transformer block looks like this:

```txt
token representations
        │
        ▼
self-attention
        │
        ▼
updated token representations
        │
        ▼
MLP
        │
        ▼
updated token representations
```

*In real LLMs, there are also residual connections and normalization layers around these parts*

### 2.1 Self-Attention 

Self-attention is the part of the transformer where every token can look at other tokens in the sequence and decide how much information to pull from them.

Suppose the input is:

```txt
The animal didn't cross the street because it was tired
```

When the model processes `"it"`, it has to represent what `"it"` refers to. Is it `"animal"` or `"street"`?

At the beginning, every token is represented as a vector:

```txt
"The"     → v₁
"animal"  → v₂
"didn't"  → v₃
...
"it"      → v₇
...
```

The attention layer asks, for every token:

```txt
which other tokens should this token pay attention to?
```

For `"it"`, the model computes attention scores against the other tokens. After normalization, these become **attention weights**. For example:

```txt
"The"      0.02
"animal"   0.45  ← important
"didn't"   0.03
"cross"    0.03
"street"   0.05
"because"  0.04
"it"       0.08
"was"      0.10
"tired"    0.20  ← also relevant
```

These weights show how much information `"it"` should pull from each token. The weights for one token’s attention distribution sum to `1.0`.

Now instead of using only its previous vector, `"it"` creates a new representation by taking a weighted combination of the other token value vectors:

```txt
new_it =
0.45 × animal +
0.20 × tired +
0.10 × was +
0.08 × it +
...
```

That produces a new vector for `"it"` that contains more context than the original embedding did.

Each other token does the same thing from its own point of view. `"animal"` computes what it should attend to. `"street"` computes what it should attend to. `"tired"` computes what it should attend to. The result is a new sequence of token vectors, where every token has been updated using the context most relevant to it.

Then this updated sequence continues into the MLP, and after that into the next transformer block, where attention happens again on the improved token representations.

#### 2.1.1 Query, Key, Value

Query, Key, and Value are how attention weights are produced.

For a given layer, we start with hidden states `X`, where `X` is the matrix of current token representations.

The model computes:

```txt
Q = X · WQ
K = X · WK
V = X · WV
```

Where:

```txt
WQ = learned query projection
WK = learned key projection
WV = learned value projection
```

These weight matrices start as mostly random values and are learned during training.

Conceptually:

```txt
Q = what this token is looking for
K = what this token offers as something others can match against
V = what information this token actually contributes if attended to
```

The model compares queries against keys:

```txt
scores = Q · Kᵀ
```

This gives a score for how much each token should attend to each other token.

Then the scores are normalized with softmax:

```txt
attention_weights = softmax(scores)
```

Finally, those weights are used to mix the value vectors:

```txt
attention_output = attention_weights · V
```

So the simplified flow is:

```txt
X
│
├── X · WQ = Q
├── X · WK = K
└── X · WV = V

Q · Kᵀ
│
▼
attention scores
│
▼
softmax
│
▼
attention weights
│
▼
attention weights · V
│
▼
new token representations
```

This happens separately in every attention layer. Since a transformer has many layers, Q, K, and V are recalculated again and again from the current hidden states at that layer.

### 2.2 Multilayer Perceptron

After the attention block, the MLP takes each token vector and transforms it through a normal feed-forward neural network.

While attention mixes information between tokens, the MLP processes each token’s updated representation internally. It is applied to all token vectors in parallel, using the same learned weights for each token.

A simplified MLP looks like this:

```txt
MLP(x) = W₂ · activation(W₁ · x + b₁) + b₂
```

Where:

```txt
x  = one token's current representation
W₁ = learned weight matrix
b₁ = learned bias
W₂ = learned weight matrix
b₂ = learned bias
```

Mechanically, this means:

```txt
token vector
    │
    ▼
linear projection
    │
    ▼
nonlinear activation
    │
    ▼
linear projection
    │
    ▼
updated token vector
```

For example, if the model has hidden size `768`, the MLP may expand the vector into a larger internal space and then compress it back:

```txt
[768]
  │
  ▼
Linear: 768 → 3072
  │
  ▼
Activation
  │
  ▼
Linear: 3072 → 768
```

The expansion gives the model more room to compute features. The activation function adds nonlinearity, which allows the MLP to model more complex patterns than a single matrix multiplication.

The MLP does not decide which other tokens matter. That is attention’s job. The MLP transforms the features that are already inside the token vector.

For the earlier sentence:

```txt
The animal didn't cross the street because it was tired
```

Attention may update `"it"` so that its vector contains information from `"animal"` and `"tired"`. The MLP then processes that vector internally and reshapes it into a representation that is more useful for the next transformer block.

So the flow is:

```txt
attention:
"it" gathers context from "animal", "tired", "was", ...

MLP:
the updated "it" vector is transformed internally

next attention layer:
the improved "it" vector can again interact with other tokens
```

This alternating process is why transformers can build complex representations over many layers. Attention moves information between tokens. The MLP processes that information inside each token. Then the next attention layer works with the improved representations.