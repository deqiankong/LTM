---
layout: my_distill
title: "Latent Thought Models"
pretty_title: "Latent Thought Models <br><span style='font-size: 0.6em; font-weight: bold; display: block; margin-top: 0.2em;'>with Variational Bayes Inference-Time Computation</span>"
full_title: "Latent Thought Models with Variational Bayes Inference-Time Computation"
permalink: /ltm/
description: We propose Latent Thought Models (LTMs), a novel class of language models that incorporate explicit latent thought vectors to guide autoregressive generation. Through dual-rate optimization and inference-time computation, LTMs achieve superior efficiency and emergent reasoning capabilities at significantly smaller scales than traditional models.
date: 2025-06-03
future: true
htmlwidgets: true
hidden: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Deqian Kong
    url: "https://sites.google.com/view/deqiankong/"
    affiliations:
      name: UCLA

# must be the exact same name as your blogpost
bibliography: ltm.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: The Problem with Current AI
  - name: Images and Figures
    subsections:
    - name: Interactive Figures
  - name: Citations
  - name: Footnotes
  - name: Code Blocks
  - name: Diagrams
  - name: Tweets
  - name: Layouts
  - name: Other Typography?

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---


## The Problem with Current AI

Think about how you write. Before putting pen to paper (or fingers to keyboard), your mind forms an abstract understanding of what you want to express. You might think about the main themes, the emotional tone, or the logical structure. Only then do you translate these abstract thoughts into concrete words.

Current language models like GPT work differently. They generate text token by token, word by word, without any higher-level planning or abstract representation. It's like speaking without thinking—impressive, but ultimately limited.

<div class="key-insight">
<strong>Key Insight:</strong> Traditional language models lack explicit mechanisms for abstract reasoning and planning, limiting their ability to perform complex cognitive tasks efficiently.
</div>

## Our Solution: Latent Thought Models

We propose **Latent Thought Models (LTMs)**—a new class of language models that explicitly learn to form abstract "thoughts" before generating text. These thoughts are represented as latent vectors that capture the essence of what the model wants to express, before translating it into actual words.

### Core Architecture

Imagine an AI that works more like a human writer:

1. **Think first**: Form abstract thoughts about the content, themes, and structure
2. **Write second**: Use these thoughts to guide the actual text generation

Our LTMs do exactly this through a two-stage process:

$$\text{Abstract Thoughts } (\mathbf{z}) \rightarrow \text{Concrete Words } (\mathbf{x})$$

The latent thought vectors $\mathbf{z}$ serve as a compressed, abstract representation that guides the generation of each token in the sequence.

{% include figure.html path="assets/img/ltm/ltm_architecture.png" class="img-fluid rounded z-depth-1" %}
<div class="caption">
    <b>LTM Architecture:</b> Latent thought vectors $\mathbf{z}$ are sampled from a prior distribution and guide autoregressive generation through cross-attention at each layer.
</div>

#### Layered Thought Vectors

Instead of having just one set of thoughts, our models use **layered thought vectors**—different abstract representations for different layers of the neural network. This creates a hierarchy of abstraction:

- **Lower layers**: Basic linguistic patterns and syntax
- **Middle layers**: Semantic relationships and concepts  
- **Higher layers**: High-level themes and narrative structure

We assume $\mathbf{z} = (\mathbf{z}_1, ..., \mathbf{z}_L)$, where $\mathbf{z}_l$ consists of thought vectors cross-attending to layer $l$ of the Transformer decoder.

#### Thought-Guided Generation

The key component is a thought-conditioned autoregressive generator $p_{\beta}(\mathbf{x}|\mathbf{z})$:

<div class="equation-highlight">
$$p_{\beta}(\mathbf{x}|\mathbf{z}) = \prod_{n=1}^N p_{\beta}(x^{(n)}|\mathbf{z}, \mathbf{x}^{(<n)})$$
</div>

Unlike standard autoregressive models that only condition on previous tokens, our model incorporates the thought vectors $\mathbf{z}$ at each generation step through cross-attention.

### Dual-Rate Learning Algorithm

Our training process mirrors human learning with a **dual-rate optimization**:
- **Fast learning**: Quick adaptation to specific examples (like episodic memory)
- **Slow learning**: Gradual accumulation of general knowledge (like procedural memory)

<div class="algorithm-box">
<strong>Algorithm: Fast-Slow Learning of LTMs</strong>

For each training batch:
1. **Fast Learning (Inference-time computation)**:
   - Initialize variational parameters $(\boldsymbol{\mu}_i, \boldsymbol{\sigma}^2_i)$ for each sequence
   - For $t = 1$ to $T_{\text{fast}}$ steps:
     - Sample $\mathbf{z} \sim q_{\boldsymbol{\mu}_i, \boldsymbol{\sigma}^2_i}(\mathbf{z}|\mathbf{x}_i)$
     - Compute ELBO: $\mathcal{L}_i = \mathbb{E}_{q}[\log p_{\beta}(\mathbf{x}_i|\mathbf{z})] - \text{KL}(q(\mathbf{z}|\mathbf{x}_i) || p(\mathbf{z}))$
     - Update $(\boldsymbol{\mu}_i, \boldsymbol{\sigma}^2_i)$ with high learning rate (0.3)

2. **Slow Learning**:
   - Update global decoder parameters $\beta$ with low learning rate (0.0004)
</div>

This reflects the **declarative-procedural framework** from cognitive science—our latent thoughts act like declarative memory while the text generator represents procedural knowledge.

## Key Insights and Breakthroughs

### New Scaling Dimensions

While traditional language models scale along two main axes (model size and training data), LTMs introduce a third crucial dimension: **inference steps**. More thinking time leads to better performance—you can trade off model size for more deliberate reasoning.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/ltm/ppl_val_2.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <b>Scaling behaviors over training tokens and compute.</b> Models with more inference steps demonstrate improved sample efficiency and become compute-efficient beyond certain training compute thresholds.
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/ltm/scaling_tokens_new.png" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/ltm/scaling_flops_new.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <b>Scaling behaviors over training tokens and compute.</b> We plot the performance of LTM training runs ($N_{z}=24$) across inference steps ($T_\mathrm{fast}=$16-64) and model sizes (38M-76M). Models with more inference steps demonstrate improved sample efficiency and become compute-efficient beyond certain training compute thresholds.
</div>

### Emergent Few-Shot Learning at Small Scale

Something remarkable happens with LTMs: they develop few-shot learning abilities (like GPT-3's in-context learning) but with **dramatically fewer parameters**. Our smallest model achieves this with just 38M parameters—a fraction of what's typically needed.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/ltm/gsm8k.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <b>Arithmetic reasoning on GSM8K:</b> LTMs with few-shot demonstrations outperform much larger GPT-2 models across various settings.
</div>

### Superior Efficiency

The results speak for themselves:
- **76% fewer training tokens** needed compared to similar-sized models
- **93% fewer parameters** than GPT-2-Large while matching its performance
- **5× faster generation** compared to some diffusion-based alternatives

## Technical Deep Dive

### Model Formulation

We formulate LTMs within the classical variational Bayes framework. The model assumes latent thought vectors $\mathbf{z}$ follow a prior $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ and generate text $\mathbf{x}$ via a Transformer decoder.

We introduce a sequence-specific variational posterior $q(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$ and maximize the evidence lower bound (ELBO):

<div class="equation-highlight">
$$\mathcal{L}(\beta, \boldsymbol{\mu}, \boldsymbol{\sigma}^2) = \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}[\log p_{\beta}(\mathbf{x}|\mathbf{z})] - \text{KL}(q(\mathbf{z}|\mathbf{x})||p(\mathbf{z}))$$
</div>

Crucially, $(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$ are **local parameters** specific to each sequence, while $\beta$ represents **global parameters** shared across all samples.

### Inference-Time Computation

LTMs introduce a distinct computational cost: **inference-time computation** stemming from the fast learning of latent thought vectors. This occurs in both training and testing.

For a model with $L$ layers, $N_{\mathbf{z}}$ latent vectors per layer, and $T_{\text{fast}}$ inference steps, the computational complexity scales as:

$$\mathcal{O}(T_{\text{fast}} \cdot L \cdot (N^2H + NN_{\mathbf{z}}H + NH^2))$$

When $T_{\text{fast}} \gg 1$, the inference computation dominates, making thinking time the primary computational factor.

## Experimental Results

We conducted extensive experiments at GPT-2 scale using the OpenWebText dataset. Our results demonstrate:

**Zero-shot Language Modeling Performance:**

| Model | Parameters | Training FLOPs/tok | PTB | WikiText | LM1B |
|-------|------------|-------------------|-----|----------|------|
| GPT-2-Large | 762M | 5.32G | 161.33 | 30.09 | 45.61 |
| LTM-Medium | 51M | 5.52G | ≤32.06 | ≤17.39 | ≤25.16 |
| LTM-Large | 76M | 32.2G | ≤**4.43** | ≤**3.66** | ≤**3.92** |

**Text Generation Quality:**

| Model | Sampling | MAUVE ↑ |
|-------|----------|---------|
| GPT-2-Medium | Nucleus-0.95 | 0.955 |
| GPT-2-Medium | Multinomial | 0.802 |
| LTM-Large | Multinomial | **0.974** |
| LTM-Large | Greedy | 0.972 |

### Probing the Latent Thoughts

We investigated how semantic information is distributed across layers through progressive reconstruction experiments. The results reveal that LTMs process information hierarchically:

- **6-layer models**: Gradual improvement (~55% accuracy) followed by sharp synthesis in the final layer
- **12-layer models**: Distributed processing with steady increases through layers 1-8 (~65%) and crucial integration at layers 9-10 (>95% accuracy)

This demonstrates distinctive "synthesis layers" that integrate information from earlier representations.

## What This Means for AI

### The Language of Thought Hypothesis

Our approach connects to a deep idea in cognitive science: that thinking happens in an internal "language of thought" that's distinct from the language we speak. The latent thought vectors can be seen as "words" in this internal cognitive language.

### Inference-Time Computation as a New Paradigm

Perhaps most importantly, LTMs demonstrate that **thinking time** can be as valuable as model size or training data. This opens up new possibilities:

- **Adaptive computation**: Thinking harder on difficult problems
- **Resource allocation**: Trading model size for inference time based on computational budgets
- **Reasoning capabilities**: Using iterative refinement for complex problem-solving

## Looking Forward

This work opens several exciting directions:

1. **Structured Prior Models**: Moving beyond simple Gaussian priors to more sophisticated reasoning structures
2. **Reward-Guided Thinking**: Using reward models to guide the thinking process toward better outcomes  
3. **Hierarchical Abstraction**: Developing even more sophisticated multi-level thought representations

### Current Limitations

We acknowledge important areas for future work:

- **Learnable Prior Models**: Our current Gaussian prior is simple—more structured priors could enable even richer reasoning
- **Reward Models in Latent Space**: Incorporating verifier models $p_\gamma(r|\mathbf{z})$ to guide optimization for reasoning tasks

## Conclusion

Latent Thought Models represent a fundamental shift in how we think about language generation. Instead of immediate word-by-word generation, they introduce a more human-like process of abstract thinking followed by linguistic expression.

The key insight is simple but profound: **giving AI systems explicit space to think makes them more efficient, more capable, and more aligned with how humans actually process language**.

As we continue to push the boundaries of AI capabilities, approaches like LTMs suggest that the future lies not just in bigger models or more data, but in architectures that more closely mirror the sophisticated cognitive processes that make human intelligence so remarkable.


## Images and Figures

Its generally a better idea to avoid linking to images hosted elsewhere - links can break and you
might face losing important information in your blog post.
To include images in your submission in this way, you must do something like the following:

```markdown
{% raw %}{% include figure.html path="assets/img/2025-04-28-distill-example/iclr.png" class="img-fluid" %}{% endraw %}
```

which results in the following image:

{% include figure.html path="assets/img/2025-04-28-distill-example/iclr.png" class="img-fluid" %}

To ensure that there are no namespace conflicts, you must save your asset to your unique directory
`/assets/img/2025-04-28-[SUBMISSION NAME]` within your submission.

Please avoid using the direct markdown method of embedding images; they may not be properly resized.
Some more complex ways to load images (note the different styles of the shapes/shadows):



