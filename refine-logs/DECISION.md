# Project Evaluation: Should this research direction continue?

**Date:** 2026-04-18
**Asker:** user ("我只想评估有没有往下做的潜力")
**Source proposal:** `lwe in transformer.docx`

This document is a deliberately unvarnished assessment written to help decide whether to keep investing in this direction. I will not soften conclusions to spare effort already spent.

---

## 1. The question we set out to answer

> Can a weight-sparse Transformer learn the Regev LWE decryption circuit and the SWIFFT lattice hash as *mechanistically interpretable subgraphs*, in a form that unlocks CPA-security reasoning and new post-quantum primitives?

Original proposal breaks this into 5 phases / 5 claims:

| Claim | Name | Novelty-report rating |
|---|---|---|
| C1 | LWE decryption as weight-sparse interpretable Transformer circuit | MEDIUM |
| C2 | Secret `s` hidden in weights via superposition (non-extractable) | LOW–MEDIUM |
| C3 | Formal CPA reduction under black/gray/white-box access | MEDIUM |
| C4 | SWIFFT as neural-native hash (FFT ≅ Transformer linear layer) | **HIGH** |
| C5 | Neural-LWE attack-surface taxonomy | LOW–MEDIUM |

Overall novelty score: **5.5 / 10. PROCEED WITH CAUTION.**

---

## 2. What we actually have after 1 day of compute

### Empirical results

| Experiment | Scale | Outcome |
|---|---|---|
| LWE dense, Mode B | n=4, q=7, h=2 | ✅ 100% @ step 2.6k |
| LWE dense, Mode B | n=8, q=7, h=2 | ✅ 100% @ step 12k |
| LWE dense, Mode B | n=16, q=7, h=4 | ❌ plateau at 80k steps (also with wd=0.01, with 4L d=256) |
| LWE dense, **Mode A** | any n ≥ 4 | ❌ never grokked in attempted budgets |
| LWE sparse + L0, Mode B | n=4 | ✅ soft=hard=1.000, density=0.917 |
| LWE sparse + L0, Mode B | n=8 | ✅ soft=1.000, hard=0.991, **density=0.570** |
| LWE mechanistic probe, Mode B | n=4, n=8 | ✅ circuit localizes at L1 post-attn at both scales |
| SWIFFT dense | m=2, n=4, p=17 | ✅ 100% |
| SWIFFT sparse + L0 | m=2, n=4, p=17 | ✅ soft=0.991, hard=0.964 |
| SWIFFT | m≥2, n≥8 | ❌ stuck at chance |
| CPA security reduction | — | ⌀ no work |
| Neural-LWE attacks | — | ⌀ no work |

### Claim verdicts

- **C1 at toy scale**: ✅ Supported, but only Mode B (fixed secret). The *algorithmic-learning* version (Mode A) never worked.
- **C2**: ✗ Not evidenced at all. Mode B fixed-secret doesn't count — the secret is trivially hardcoded, not "hidden via superposition." Without Mode A, this claim has zero empirical footing.
- **C3**: ⌀ Purely theoretical; we didn't attempt. Would need a paper of its own.
- **C4**: Weakly supported (m=2, n=4 vs real SWIFFT m=16, n=64 — 128× gap in n).
- **C5**: ⌀ Nothing done.

---

## 3. Honest assessment per domain

### 3a. As an ML / interpretability contribution

**Verdict: small increment, not novel enough for a top venue.**

Gao et al. 2025 (arXiv 2511.13653, OpenAI) already showed weight-sparse Transformers learn interpretable circuits for Python tasks. Quirke et al. (arXiv 2402.02619) showed the same for modular addition. Africa et al. (arXiv 2506.23679) showed it for modular exponentiation. **LWE decryption is just another task on the same shelf.** Our probe progression at n=4/n=8 matches their findings almost 1-to-1.

The only thing that would make this ML-wise novel is if we'd shown Mode A at a non-trivial n — demonstrating that the Transformer learns the *algorithm* rather than memorizing a specific key. We did not.

### 3b. As a cryptography contribution

**Verdict: essentially zero.**

Cryptographers do not care whether a Transformer can compute Regev's decryption, any more than they care that a Python interpreter can. The interesting crypto questions are:

- Is the embedded scheme **provably** secure under realistic attack? → We have no proof.
- Does the neural embedding create **new** attack surfaces or resistances? → We have no analysis.
- Does this enable a **new primitive** that wasn't possible before? → No.

Shamir, Gerault, Hambitzer, Ronen (IACR ePrint 2025/288, EUROCRYPT 2026, "Deep Neural Cryptography") have already laid the theoretical foundation for "how to securely implement crypto in a DNN." Any CPA reduction we produce will be seen as a special case of their framework, with LWE as a specific instantiation. Our key differentiator — the LWE-hardness reduction chain — requires heavy theory work we haven't started.

### 3c. As a security-analysis / mechanistic-attack contribution

**Verdict: preempted.**

The promise was "extract the LWE secret from weights via mechanistic analysis, and show it's hard." But:

- Carlini et al. (Crypto '20; arXiv 2003.04884; ePrint 2024/1580; ePrint 2026/253) already show **polynomial-time** cryptanalytic extraction of multi-layer ReLU networks. This directly threatens the "secret-as-weight" thesis.
- We would need to empirically demonstrate that superposition-obfuscated secrets survive Carlini's attack. We did not attempt this.
- The "non-standard floating-point input" attack angle (proposal Phase 5) is already in Shamir 2025/288 on AES-DNN. Not novel if we redo it for LWE.

### 3d. As an applied / downstream contribution

**Verdict: no clear downstream.** There is no obvious application that benefits from "LWE embedded in a Transformer." Candidates like zkML care about **computation graph size**, which is smaller with a handwritten circuit, not with a trained Transformer. Neural watermarking already works with simpler schemes. FHE-friendly Transformers are being developed independently without this angle.

---

## 4. What's genuinely unique here (if anything)

**Claim C4: SWIFFT's NTT butterflies are architecturally isomorphic to Transformer FFN linear layers.**

This is the one angle the novelty report rated HIGH. It has two parts:

1. **Theoretical observation** (no training required): write out the NTT matrix, show it factors into exactly the operations a Transformer FFN applies. Compare to SHA-256 which needs bitwise XOR/shifts and is thus a natural mismatch.
2. **Empirical demonstration**: train a sparse Transformer to compute SWIFFT, verify the sparse weights reproduce the NTT structure.

We have (1) partially in the proposal text (informal argument) and (2) only at m=2, n=4.

**Honest ceiling for C4**: a 4–6 page theory + small-experiment paper, perhaps at a crypto–ML workshop. Not a top-venue headline.

---

## 5. What would revive full promise

To salvage the **original** 5-claim vision, minimum investment is roughly:

| Gap | Fix | Effort |
|---|---|---|
| Mode A never works | Curriculum n=4→8→16, muP init, finite-dataset overfit-then-grok | 2-3 weeks GPU tuning |
| C3 CPA reduction | Formal theory work connecting Carlini extraction hardness to LWE hardness | 1-2 months math (ideally with a crypto co-author) |
| C4 SWIFFT at real params (m=16, n=64, p=257) | Architectural redesign + scale | 2-4 weeks GPU + tuning |
| C5 attack taxonomy | Empirical attack suite (float subnormals, gradient leakage, activation side-channels) | 2-3 weeks |
| Differentiation from Shamir 2025/288 | Paper-framing pivot: reposition as "LWE-specific structural isomorphism" not "another general crypto-in-DNN" | Requires thinking not compute |

Minimum path to a **mid-tier venue paper**: 2-3 months of focused work. Minimum path to a **top-tier venue paper**: 4-6 months plus an experienced crypto collaborator.

---

## 6. Decision framework

**Continue investing iff at least one of the following is true:**

1. You have a specific **downstream application** that benefits from neural-crypto (zkML pipeline, on-chain AI, a product idea you're building). I don't see evidence you do.
2. You have a **crypto collaborator** (senior cryptographer or theory-track grad student) who will carry C3 through on their own timeline. Without this, C3 remains a weakness reviewers will hammer.
3. You personally find the **SWIFFT–Transformer isomorphism** (C4) intrinsically exciting enough to spend 1-2 months on a short theory-flavored paper that isn't a career milestone.

**Stop investing iff:**

- You were exploring the proposal speculatively and want a realistic reality check (→ this doc is it).
- Your time has competing uses with higher expected value.
- You don't have the crypto-theory background or collaborator to credibly land C3.

---

## 7. My direct recommendation

**Stop here for this direction. Archive the work cleanly.**

Not because the work is wrong or bad — it's clean, the results are real, and the code is reusable. But the **ceiling** of what this direction can become is visible now:

- A workshop paper (writable in ~1 week with current material)
- A blog post / preprint demonstrating the toy-scale circuit
- A foundation you *could* return to later if crypto-ML becomes your specialty

None of those is a great ROI if your goal is impactful research. Investigating whether Mode A can be made to work at n=16-64 would be more scientifically interesting, but that's a **different project** about grokking dynamics, not about neural crypto per se — and it's been studied by the grokking literature already.

If you want to keep doing AI-crypto research, I'd recommend pivoting to questions the field actually cares about:
- **Mechanistic interpretability of existing models** trained on crypto-relevant data (e.g., "what does a Transformer trained on Kyber ciphertexts learn?")
- **FHE-friendly architectures** (polynomial Transformers) — currently active
- **zkML compilation** — immediate practical value, different set of skills
- **Neural cryptanalysis** (the SALSA direction, not embedding) — well-funded, active

None of these directly extend the current proposal, but any of them would use the skills you've built.

---

## 8. What to do right now

If you accept the recommendation to stop:

- [ ] Commit current code and results as a tagged snapshot
- [ ] (Optional, 1 week) Write a short preprint / blog post with the n=4 + n=8 evidence + SWIFFT toy result
- [ ] Close the project in your notes
- [ ] Decide the next direction

If you want to push one more lever before deciding:

- [ ] Try Mode A at n=8 with weight_decay=0.01 + curriculum warm-start from Mode B checkpoint. If this works → Mode A might be reachable; project has more legs than I thought. If it doesn't → confirm the stop decision. ~1 day GPU.

---

This is my honest read. Take 24–48 hours before deciding.
