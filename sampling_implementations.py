import numpy as np
import matplotlib.pyplot as plt


vocabulary = ["the", "cat", "sat", "on", "a", "mat", "dog", "ran", "fast", "slowly"]
logits = np.array([2.5, 1.2, 0.8, 3.1, -0.5, 1.9, 0.3, -1.2, 2.8, 0.1])



'____Task 1: Softmax with Temperature____'

def softmax_with_temperature(logits, temperature=1.0):
    scaled_logits = logits / temperature
    scaled_logits = scaled_logits - np.max(scaled_logits)
    exp_values = np.exp(scaled_logits)
    probs = exp_values / np.sum(exp_values)
    return probs


'___Visualization 1___'
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
temperatures = [0.1, 0.5, 1.0, 2.0]

for ax, T in zip(axes, temperatures):
    probs = softmax_with_temperature(logits, T)
    ax.bar(vocabulary, probs)
    ax.set_title(f"Temperature = {T}")
    ax.set_xlabel("Token")
    ax.set_ylabel("Probability")
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("temperature_effect.png", dpi=150)
plt.show()

# Test 1
for T in [0.1, 0.5, 1.0, 2.0]:
    probs = softmax_with_temperature(logits, T)
    print(f"\nTemperature = {T}")
    for word, p in zip(vocabulary, probs):
        print(f" {word:8s}: {p:.4f}")

"""
The logits are divided by a relatively small number at low temperatures 
like 0.1, which significantly enhances the differences between them. 
Even slight variations in scaled logits become incredibly huge after 
exponentiation since softmax uses the exponential function. This leads 
to near-deterministic behavior since the greatest logit dominates the denominator, 
making its probability approach 1 while others approach 0.

At high temperature like 0.2, the logits are divided by a larger number,
which compresses the differences between them. The ratios between logits become
closer together before exponentiation, so their exponentials are more similar.
As a result, the probabilities become more evenly distributed, pushing the
distribution toward a more uniform shape.
"""

'____Task 2: Naive Top-K____'

def top_k_sampling(logits, k, temperature=1.0):
    masked_logits = logits.copy()
    sorted_indices = np.argsort(masked_logits)
    indices_to_remove = sorted_indices[:-k]
    masked_logits[indices_to_remove] = -np.inf
    return softmax_with_temperature(masked_logits, temperature)

# Test 1
for k in [1, 2, 3, 5]:
    probs = top_k_sampling(logits, k)
    nonzero = [(vocabulary[i], f"{probs[i]:.4f}")
               for i in range(len(probs)) if probs[i] > 0]
    print(f"top_k={k}: {nonzero}")

'____Task 3: Efficient Top-K____'

def top_k_sampling_efficient(logits, k, temperature=1.0):
    V = len(logits)
    top_k_indices = np.argpartition(logits, -k)[-k:]
    top_k_logits = logits[top_k_indices]
    top_k_probs = softmax_with_temperature(top_k_logits, temperature)
    probs = np.zeros(V)
    probs[top_k_indices] = top_k_probs
    return probs

# Test 1
print("Verifying naive == efficient for all k values:")
for k in [1, 2, 3, 5, 10]:
    naive = top_k_sampling(logits, k)
    efficient = top_k_sampling_efficient(logits, k)
    match = np.allclose(naive, efficient)
    print(f" k={k}: {'MATCH' if match else 'MISMATCH'}")
    assert match, f"Implementations disagree at k={k}!"

print("All checks passed.")

'____Task 4: Top-P (Nucleus)____'

def top_p_sampling(logits, p, temperature=1.0):
    probs = softmax_with_temperature(logits, 1.0)

    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative = np.cumsum(sorted_probs)

    cutoff_index = np.where(cumulative >= p)[0][0]

    nucleus_indices = sorted_indices[:cutoff_index + 1]

    masked_logits = logits.copy()
    for i in range(len(logits)):
        if i not in nucleus_indices:
            masked_logits[i] = -np.inf

    return softmax_with_temperature(masked_logits, temperature)

# Test 1
for p in [0.5, 0.75, 0.9, 0.95, 1.0]:
    probs = top_p_sampling(logits, p)
    nonzero = [(vocabulary[i], f"{probs[i]:.4f}")
               for i in range(len(probs)) if probs[i] > 0]
    print(f"top_p={p}: {nonzero}")

'____Task 5: Logit Bias____'

def logit_bias_sampling(logits, bias_dict, temperature=1.0):
    biased_logits = logits.copy()
    for idx, bias in bias_dict.items():
        biased_logits[idx] += bias
    return softmax_with_temperature(biased_logits, temperature)

# Test 1
probs1 = logit_bias_sampling(logits, {7: 5.0})
print(f"Boosted 'ran': {probs1[7]:.4f}")

# Test 2
probs2 = logit_bias_sampling(logits, {3: -100.0})
print(f"Banned 'on': {probs2[3]:.6f}")

# Test 3
probs3 = logit_bias_sampling(logits, {1: 3.0, 8: -100.0})
print(f"Boosted 'cat': {probs3[1]:.4f}, Banned 'fast': {probs3[8]:.6f}")

'____Task 6: Combine All Parameters____'

def sample_with_all_parameters(logits, temperature=1.0, top_k=None, top_p=None, logit_bias=None):
    working_logits = logits.copy()

    if logit_bias is not None:
        for idx, bias in logit_bias.items():
            working_logits[idx] += bias

    if top_k is not None:
        indices = np.argpartition(working_logits, -top_k)[-top_k:]
        mask = np.ones(len(working_logits), dtype=bool)
        mask[indices] = False
        working_logits[mask] = -np.inf

    if top_p is not None:
        probs = softmax_with_temperature(working_logits, 1.0)
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative = np.cumsum(sorted_probs)
        cutoff_index = np.where(cumulative >= top_p)[0][0]
        nucleus_indices = sorted_indices[:cutoff_index + 1]
        mask = np.ones(len(working_logits), dtype=bool)
        mask[nucleus_indices] = False
        working_logits[mask] = -np.inf

    return softmax_with_temperature(working_logits, temperature)



'____Visualization 2: Sampling Comparison____'


fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

configs = [
    ("Baseline (T=1.0)", softmax_with_temperature(logits, 1.0)),
    ("Low Temperature (T=0.2)", softmax_with_temperature(logits, 0.2)),
    ("Top-k=3 (T=1.0)", top_k_sampling_efficient(logits, 3, 1.0)),
    ("Top-p=0.8 (T=1.0)", top_p_sampling(logits, 0.8, 1.0)),
    ("Logit bias: ban 'on'", logit_bias_sampling(logits, {3: -100.0}, 1.0)),
    ("Combined: T=0.7, k=4, p=0.9, ban 'on'",
     sample_with_all_parameters(logits, 0.7, 4, 0.9, {3: -100.0}))
]

for ax, (title, probs) in zip(axes, configs):
    ax.bar(vocabulary, probs)
    ax.set_title(title)
    ax.set_xlabel("Token")
    ax.set_ylabel("Probability")
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("sampling_comparison.png", dpi=150)
plt.show()

"""
Every sampling parameter modifies the probability distribution differently 
across the six panels. The natural softmax distribution of the logits is 
reflected in the baseline (T=1.0), where higher-scoring tokens like "on" 
and "fast" predominate but others maintain nonzero probability.
The distribution becomes much sharper when the temperature is lowered (T=0.2), 
with the highest logit dominating nearly completely and the others being suppressed. 
Top-k sampling produces a hard cutoff effect by limiting the distribution to the 
k highest-scoring tokens and giving all other tokens a zero probability. 
On the other hand, top-p sampling creates a flexible boundary that adjusts to the 
form of the distribution by dynamically choosing tokens based on cumulative probability.

Since imposing a huge negative bias (such as prohibiting "on") totally eliminates the 
highest-probability token and drastically alters the distribution, logit bias has the 
most dramatic effect in this scenario.
Combining factors has compounding effects: temperature regulates sharpness, 
bias alters relative preference, and filtering narrows the candidate pool. 
When combined, they provide exact control over token selection behavior and unpredictability.

"""