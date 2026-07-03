An LLM is a transformer, a specific kind of machine learning model, that takes some text, video, or audio and predicts what comes next in the passage. 

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

# 2. Transformer

The sequence of embedding vectors is then passed into the **self-attention** layer, where every token can attend to every other token in the sequence. Each token gathers information from other relevant tokens to build a richer, context-aware representation.

The **self-attention** mechanism determines which tokens are most relevant to one another and how much information should flow between them. This allows a token's representation to change depending on its surrounding context, giving the model a contextual understanding of the input rather than treating each token independently.

## 2.1 Self-Attention in practice

This is a core idea of transformers, and it's actually simpler than it first appears. 

Suppose the input is: 
`The animal didn't cross the street because it was tired`

When the model reaches "it", it has to determine what "it" refers to. Is it "animal" or "street"?

Since initially every token is just an embedding vector:
```
"The"     → v₁
"animal"  → v₂
"didn't"  → v₃
...
"it"      → v₇
...
```

**The attention layer asks, for every token "which other tokens should i pay attention to?".**

For "it", the model computes a scope against every other token that become **attention weight**
```
"The"      0.02
"animal"   0.83  ← important
"didn't"   0.05
"cross"    0.04
"street"   0.03
"because"  0.06
"it"       0.10
"was"      0.12
"tired"    0.75  ← also important
```

Now instead of using only its original embedding, "it" creates a new representation by taking a weighted combination of the other token vectors:
```
new_it =  
0.83 × animal +  
0.75 × tired +  
0.12 × was +  
...
```

That produces a new vector for "it" that now encodes the fact that "it" refers to the animal and each other token also computes attention from **its own point of view**, producing its own weighted sum of the other tokens' value vectors. This creates a new representation that incorporates the context most relevant to that token.

### 2.1.1 (Q)uery, (K)ey, (V)alues

This is how attention weights are produced and it's stupidly simple as well.