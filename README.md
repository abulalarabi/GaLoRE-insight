

# Tech Notes: GaLoRE (Gradient Low-Rank Projection)
A technical note on how GaLoRE minimizes memory overhead during large language model training, specifically optimized for memory-constrained **Continued Pre-Training (CPT)**.

---

## 1. High-Level Intuition & Core Problem
### The Bottleneck: Optimizer States
When training a massive AI model (e.g., a 7B parameter LLM), the model weights themselves are not the primary VRAM bottleneck. The true memory hog is the **optimizer** (e.g., AdamW).

For every single weight parameter, AdamW must track **two floating-point values** (momentum and variance) to calculate steps accurately.
* **Weights (16-bit Precision):** A 7B model requires $\approx$ **14 GB** of VRAM.
* **AdamW Optimizer States:** Tracking states for that same model require $\approx$ **28 GB** of VRAM!

### The Analogy
* **Full Fine-Tuning:** 
The GPU maps out and tracks the entire, massive gradient blueprint on a huge canvas. Highly accurate, but requires massive memory infrastructure.
* **LoRA (Low-Rank Adaptation):** Freezes the base architecture and hooks up a small, temporary scaffolding system on the side. This is cheap but lacks the capacity to structuralize dense, foundational knowledge.
* **GaLoRE:** Compresses the massive gradient blueprint into a compact pocket card (via low-rank projection). The optimizer tracks metrics *only* on this pocket card. When applying updates, the card is scaled back up to full size to modify the foundation directly.

---

## 2. Core Architecture Loop
GaLoRE achieves full-parameter training efficiency by applying **Singular Value Decomposition (SVD)** to the *gradients*, rather than freezing weights or using adapters.

```mermaid
graph TD
    A[Forward/Backward Pass] --> B(Full Gradient G<br/>Dim: m × n)

    B -->|SVD / Low-Rank Approximation<br/>Rank: r| C[Projection Matrices P, Q<br/>P: m × r | Q: n × r]

    C -->|Down-Projection| D(Compressed Gradient G̃ = PᵀGQ<br/>Dim: r × r)

    D -->|AdamW<br/>Track m, v in Compressed Space| E[Compressed Update ΔG̃<br/>Dim: r × r]

    E -->|Up-Projection| F(Full-Space Update ΔG = P ΔG̃ Qᵀ<br/>Dim: m × n)

    F -->|Apply Update| G[Weights W ← W − ηΔG<br/>Dim: m × n]

    G -.->|Every T Steps<br/>update_proj_gap| B
```


1. **The Backward Pass:** The model performs a standard pass, generating a full-sized gradient matrix G (size m × n) for a given weight layer.
2. **Low-Rank Projection:** GaLoRE computes the SVD of G to extract a left projection matrix P and a right projection matrix Q, capturing the directions of highest variance.
3. **Down-Projection Optimization:** The gradient is squished down into a tiny core matrix: 
   \[\tilde{G} = P^T G Q\]
   The heavy AdamW tracking tensors are instantiated **only** for this small G̃ matrix, dropping optimizer memory overhead by **up to 65% to 80%**.
4. **Direct Weight Modification:** When stepping, the low-rank update is scaled back to full size (\(P \tilde{G} Q^T\)) and applied **directly to the original base weights**.

---

## 3. SVD Mechanics: A Toy Matrix Example

Any real matrix A can be factored cleanly into three distinct matrices:
\[A = U \cdot \Sigma \cdot V^T\]

### The Setup
Consider a simplified 2 × 2 gradient matrix G representing a layer with 4 parameters:
\[G = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}\]

Performing SVD decomposes G into:

#### 1. Left-Singular Vectors (U) — Row Subspace
\[U = \begin{pmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \end{pmatrix} \approx \begin{pmatrix} 0.707 & 0.707 \\ 0.707 & -0.707 \end{pmatrix}\]

#### 2. Singular Values (Σ) — Sorting & Importance
The diagonal elements are automatically sorted from largest to smallest, representing feature importance.
\[\Sigma = \begin{pmatrix} 4 & 0 \\ 0 & 2 \end{pmatrix}\]
*The first direction (magnitude 4) captures twice as much gradient information as the second direction (magnitude 2).*

#### 3. Right-Singular Vectors (\(V^T\)) — Column Subspace
\[V^T = \begin{pmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \end{pmatrix} \approx \begin{pmatrix} 0.707 & 0.707 \\ 0.707 & -0.707 \end{pmatrix}\]

---

### Truncation & Compression (Rank r = 1)
To compress this gradient down to a **Rank r = 1**, we slice out the components belonging exclusively to the highest singular value (4), abandoning the rest:

* **Projection Matrix P (First column of U):**
  \[P = \begin{pmatrix} 0.707 \\ 0.707 \end{pmatrix}\]
* **Projection Matrix Q (First column of V / First row of \(V^T\)):**
  \[Q = \begin{pmatrix} 0.707 \\ 0.707 \end{pmatrix}\]

### The Squish (Down-Projection Steps)
We pass the full matrix G into our low-rank subspace via \(\tilde{G} = P^T \cdot G \cdot Q\):

1. **Left Multiply (\(P^T \cdot G\)):**
   \[\begin{pmatrix} 0.707 & 0.707 \end{pmatrix} \cdot \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix} = \begin{pmatrix} 2.828 & 2.828 \end{pmatrix}\]

2. **Right Multiply (Result ⋅ Q):**
   \[\begin{pmatrix} 2.828 & 2.828 \end{pmatrix} \cdot \begin{pmatrix} 0.707 \\ 0.707 \end{pmatrix} = (2 + 2) = 4\]

**Outcome:** A 2 × 2 gradient matrix (4 parameters) has been compressed into a single scalar matrix G̃ = (4) (1 parameter). The optimizer tracks memory for only this one cell, throwing away negligible geometric noise to conserve VRAM.

---

## 4. Subspace Switching (Achieving Full-Rank Capability)

If P and Q remained static, the model would get trapped optimizing within a narrow, low-rank trajectory—limiting its capacity exactly like standard LoRA. GaLoRE solves this through **Periodic Subspace Switching**:

1. **Step Interval:** Training occurs within the compressed G̃ space for a fixed step interval, configured via the `update_proj_gap` hyperparameter (typically every **200 to 500 steps**).
2. **Subspace Rotation:** When the step threshold is met, GaLoRE pauses compression, calculates a fresh full-scale gradient G, and runs a fresh SVD.
3. **Mapping Trajectories:** This produces a **completely new set of P and Q projection matrices**, rotating the low-rank projection window to face a completely different angle of the global loss landscape.
4. **Full Parameter Updates:** Over the duration of training, shifting through these distinct subspaces allows the model to learn and assemble **full-rank updates** directly inside the base weights.

---

## 5. Reference Matrix: GaLoRE vs. LoRA

| Feature | Standard LoRA | GaLoRE |
| :--- | :--- | :--- |
| **Base Weights** | Completely frozen (100%). | Modified and updated directly. |
| **Target Storage** | Temporary low-rank adapter matrices (A and B). | Direct memory rewrite of base model layers. |
| **Rank Capability** | **Strictly Low-Rank.** Restricted permanently to adapter size. | **Full-Rank.** Aggregates alternating low-rank paths into full parameters. |
| **Primary Use-Case** | Supervised Fine-Tuning (SFT), alignment, task customization. | **Continued Pre-Training (CPT)**, deep domain/language injection. |
| **Inference Overhead** | Requires programmatic merging to prevent latency. | **Zero overhead natively.** Base weights are already altered. |

---

## 6. Hyperparameter Engineering Notes

* **Rank Selection (r):** Set r = 128 or r = 256. This is structurally higher than typical LoRA setups (r=8 or r=16) because GaLoRE can comfortably handle the parameter capacity without triggering optimizer VRAM bloat.
* **`update_proj_gap` Sweet Spot:** Keep between **200 and 500**. Updating too frequently (e.g., < 50 steps) degrades step throughput due to intensive SVD calculation bottlenecks. Updating too rarely (e.g., > 2000 steps) locks the model into an artificial low-rank convergence ceiling, stalling knowledge absorption.
