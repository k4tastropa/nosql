In machine learning, **knowledge distillation**, or **model distillation**, is a technique for transferring useful behavior from a larger, more capable model into a smaller one.

A large _teacher_ model is used to guide the training of a smaller _student_ model. Instead of learning only from hard labels in a dataset, the student can learn from the teacher’s outputs, such as predicted probabilities, logits, generated answers, explanations, or corrected examples. These teacher outputs often contain richer information than simple labels, because they show how the teacher generalizes across inputs.

The goal is to **make the smaller model approximate the teacher’s performance while using fewer computational resources.** This can make the student model faster, cheaper to run, and easier to deploy on limited hardware such as CPUs, mobile devices, or edge systems.

**Distillation does not usually preserve all of the teacher’s knowledge perfectly.** Some capability loss is expected, especially for broad or complex tasks. However, in a narrow domain, a well-distilled student model can retain much of the teacher’s useful behavior, and in some cases even outperform the teacher on a specific benchmark if the training data is carefully filtered and optimized.

## 1. Logits and Softmax

Before describing specific distillation methods, it is useful to understand **logits** and **softmax**, because they are part of how neural networks produce outputs.

### 1.1 Logits

A **logit** is the raw score a model produces before softmax.

For an LLM, at each generation step, the model outputs one score for every possible token in the vocabulary. These scores are called logits.

For example, if the model is generating SQL and the current partial output is:

```text
SELECT name
```

the model may produce logits for possible next tokens:

```text
FROM:   7.2
WHERE:  2.1
name:   1.4
users:  0.8
SELECT: 0.2
```

These values are **not probabilities yet**. They can be positive, negative, larger than 1, smaller than 0, and they do not need to sum to anything meaningful.

Logits are simply the model’s raw preference scores.

In generation, logits answer the question:

```text
How strongly does the model prefer each possible next token right now?
```

### 1.2 Softmax

Softmax converts raw model scores called **logits** into a probability-like distribution.

For a vector of logits:

$$  
z = [z_1, z_2, ..., z_n]  
$$

the softmax value for class/token `i` is:

$$  
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}  
$$

Where:

- $z_i$ = the logit for token/class `i`
    
- $e^{z_i}$ = exponentiated logit
    
- $\sum_{j=1}^{n} e^{z_j}$ = sum of exponentiated logits for all possible tokens/classes

The output values are between `0` and `1`, and all outputs sum to `1`.

## 2. Methods

There are several ways to perform model distillation, depending on what kind of signal the student model receives from the teacher.

### 2.1 Response Distillation

**Response distillation is one of the simplest distillation methods. The teacher model is given an input, and its final generated answer is used as the training target for the smaller student model.**

For example, if the task is text-to-SQL, the teacher receives a natural language question and database schema, then generates an SQL query.

Response distillation is simple because we do not need access to the teacher model’s internal parameters, hidden states, logits, or probability distributions. We only need input examples and teacher-generated outputs, which makes the method practical even when the teacher is a closed-source or API-based model.

The main weakness is that the student can inherit the teacher’s mistakes, quirks, formatting habits, verbosity, and hallucinations. If the teacher produces bad SQL and we train on it directly, we are teaching the student bad SQL, so teacher outputs should be filtered and validated before training.

### 2.1.1 How Logits and Softmax Are Used in Response Distillation

In response distillation, the teacher still uses logits and softmax internally when generating the response.

The teacher generation process looks like this:

```text
input
→ teacher forward pass
→ teacher logits
→ teacher softmax
→ chosen/sampled tokens
→ final teacher response
```

However, in basic response distillation, we usually do **not** save or train directly on the teacher’s logits or full softmax probability distribution.

Instead, we save only the final generated response.

So the teacher may internally produce a probability distribution like:

```text
FROM:   0.98
WHERE:  0.01
name:   0.006
users:  0.003
SELECT: 0.001
```

But the response distillation dataset only keeps the selected token:

```text
FROM
```

After the full sequence is generated, the stored training example becomes:

```text
input:
"Show users older than 18"

target:
SELECT name FROM users WHERE age > 18;
```

The student sees the target as text, which is then converted into token IDs.

During training, the student learns token by token:

```text
input + ""                         → predict "SELECT"
input + "SELECT"                   → predict "name"
input + "SELECT name"              → predict "FROM"
input + "SELECT name FROM"         → predict "users"
input + "SELECT name FROM users"   → predict "WHERE"
...
```

At each step, the student produces its own logits and softmax distribution. The loss function compares the student’s predicted distribution against the correct next token from the teacher-generated response.

## 2.2 Logit Distillation 

Another common method is the **logit distillation**. Instead of only copying the teacher's final answer, the student learns from the teacher's probability distribution over possible outputs.

These soft targets contain more information than hard labels because they show not only what answer the teacher chose, but also which alternatives it considered plausible. 

### 2.2.1 How Logit values are used in Logit Distillation

**Contrary to the name, in logit distillation the softmax values, rather than raw logits**

The student also produces logits, which are passed through softmax, and then the training loss compares the teacher’s distribution with the student’s distribution, usually using **KL divergence** or cross-entropy.

## 2.3 Feature Distillation

Much stronger variant of distillation: **The student is trained to match some of the teacher's internal representations, not just the final output**. This method is more common when both student and the teacher architectures are accessible compatible. It can be powerful, but it is harder to apply with closed-source teacher models because their internal activations are not available.

When a model processes an input, it doesn't immediately jump from text to answer. The input first passes through embedding layers and transformer layers. Each layer produces hidden representations, aka **features** or **hidden states**. These states are simply vector representations of what the model has learned about the input at that point.

The same input is passed through both teacher and the student model. Then both models produce their own hidden states and eventually these hidden states are compared and added as extra training loss.

`total loss = output loss + feature matching loss`

So the student is still trained to generate the correct final answer, but it is also pushed to build internal representation similar to the teacher's.












