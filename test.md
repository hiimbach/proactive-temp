# MoE Routing-k as a Probe for Quantity vs Implicature Accuracy Tradeoff

## Problem

When evaluating a MoE model (128 experts, trained with num_experts_per_tok=4) on utterance-only inputs for Maxim of Quantity + Implicature prediction, Quantity accuracy is disproportionately low. One hypothesis: the fixed routing-k=4 is too narrow to capture representations that support Quantity judgments.

## Key Insight

*Quantity is underdetermined from utterance alone.* Maxim of Quantity ("too little" vs "too much" vs "adequate") is inherently relative to a latent Question Under Discussion (QUD) or common ground. With utterance-only inputs, the model must infer this context—just like annotators implicitly did when labeling. This creates a noise floor that inference-time knobs (routing-k, sampling) can only partially address.

Implicature prediction may appear easier because gold implicature text encodes the "intended point," giving a stronger surface pattern to match.

## Experimental Approach

### 1. Metrics That Distinguish Capability from Reliability

| Metric | What It Measures |
|--------|------------------|
| pass@N(quantity) | Any of N samples matches Quantity gold |
| pass@N(implicature) | Any of N samples matches implicature gold |
| joint_pass@N | Any sample gets both correct |
| majority@N | Consistency without reranking |

joint_pass@N directly answers: "Does a configuration exist that can yield both?"

### 2. Staged Routing-k Sweep

Route to k experts: {1, 2, 4, 8, 16, 32, 64} with fixed decoding, N=16 on dev slice. Model was trained at k=4, so k>>4 may degrade. Increase N (64-256) only on best ks.

### 3. Prompting Modes as a Factor

- *Separate*: Quantity label → Implicature (separate calls)
- *Sequential (implicature-first)*: Generate implicature → label Quantity given that implicature
- *Sequential (Quantity-first)*: Control

If implicature-first boosts Quantity, the label is effectively "Quantity relative to intended meaning."

### 4. Structured Output

Force JSON with direction, not just binary:
{
  "quantity_state": "too_little|adequate|too_much",
  "quantity_label": "observed|flouted",
  "implicature": "..."
}

### 5. Implementation via Model Variants

Create lightweight variant directories that share all weights but override config.json:num_experts_per_tok. Restart vLLM per variant. Use n=N for efficient multi-sample generation.

## Expected Results

| Outcome                                          | Interpretation                                                         |
| ------------------------------------------------ | ---------------------------------------------------------------------- |
| joint_pass@N rises with k>4, majority@N flat | Routing expands diversity; needs reranking                             |
| Neither rises                                    | Quantity labels too context-dependent; add QUD to inputs               |
| Implicature-first > Separate                     | Quantity is "relative to meaning"; program matters more than routing-k |

## References

- [[1772732753-moe-routing-k-sweep-quantity-implicature]] - Capture note
Phong

# MoE Routing-k Sweep for Quantity vs Implicature Accuracy

## Context
- Model: openai/gpt-oss-20b (128 experts, default top_k=4)
- Running locally via vLLM
- Using GPT-5 as judge to compare generated implicature with ground truth
- Task: utterance → Maxim Quantity (flouted/observed) + Implicature
- Problem: Quantity accuracy is especially low; want to test if top_k=4 is too small

## Data Characteristics
- Input: utterance only (no explicit QUD/context)
- Labels: Quantity (flouted/observed), Implicature (text)
- Both under-informativeness and over-informativeness exist in "flouted" labels

## Discussion Summary

### Key Insight: Quantity is Underdetermined from Utterance Alone
Quantity ("too little" vs "too much" vs "adequate") is inherently relative to a Question Under Discussion (QUD) or common ground. With utterance-only inputs, the model (and judge) must infer the latent context that annotators assumed. This creates a hard noise floor. Implicature may appear easier because the gold implicature text often encodes the "intended point," giving a stronger pattern to match.

### Hypothesis to Test
- Smaller top_k (1, 2, 4) may under-represent the representation space needed for Quantity judgments
- Larger top_k (8, 16, 32, 64) might capture a region where both Quantity and Implicature can be predicted well
- However: model was trained at top_k=4, so k>>4 may degrade performance

### Proposed Experiment Design

1. *Metrics* (distinguish "possible" from "reliable"):
   - pass@N(quantity), pass@N(implicature): any of N samples matches gold
   - joint_pass@N: any sample gets both correct
   - majority@N: reliability without reranking

2. *Sweep Strategy* (staged, not brute-force):
   - Stage A: routing k ∈ {1, 2, 4, 8, 16, 32, 64}, temp ∈ {1.0, 1.3, 1.6}, N=16 on dev slice
   - Stage B: refine around best ks, increase N (64-256)

3. *Prompting Modes* (important confounder):
   - Separate: Quantity label → Implicature (separate calls)
   - Sequential (implicature-first): Generate implicature → label Quantity given implicature
   - Sequential (Quantity-first): Control

4. *Model Variants* (avoid duplicating weights):
   - Create /models/variants/gpt-oss-20b-ept{k}/ directories
   - Symlink all files except config.json
   - Edit only num_experts_per_tok in each variant

5. *Output Schema* (reduce ambiguity):
   
   {
     "quantity_state": "too_little|adequate|too_much",
     "quantity_label": "observed|flouted",
     "implicature": "..."
   }
   

### Expected Outcomes
- If joint_pass@N rises with larger k but majority@N stays flat: routing expands diversity but needs reranking
- If nothing helps: Quantity labels are too context-dependent; need QUD/context in inputs
- If "implicature-first" beats "separate": Quantity is effectively "relative to intended meaning"

### Unresolved
- Whether GPT-5 judge should see the gold implicature when scoring Quantity (to avoid double-guessing)
- Exact vLLM command for multi-sample (n=N) batching

## Distilled into
- [[1772732900-moe-routing-k-quantity-implicature-evaluation]]