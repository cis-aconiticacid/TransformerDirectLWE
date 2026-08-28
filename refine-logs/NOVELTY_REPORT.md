# Novelty Check Report — LWE in Transformer

**Date:** 2026-04-18
**Source:** `lwe in transformer.docx`

## Proposed Method
Embed the Regev/R-LWE decryption circuit into a weight-sparse Transformer so the LWE secret `s` lives in weights (protected by superposition), the decryption path is a mechanistically-interpretable subgraph, CPA security is formally reduced to network-extraction hardness, and a SWIFFT-based neural hash replaces SHA-like FO redundancy. Five-phase plan. New Neural-LWE attack taxonomy proposed.

## Core Claims & Novelty
| # | Claim | Novelty | Closest prior |
|---|---|---|---|
| C1 | LWE decryption as weight-sparse interpretable Transformer circuit | MEDIUM | Gao et al. 2025 (arXiv 2511.13653) — methodology, Python tasks only. SALSA/Verde/Picante — attack not embed. |
| C2 | Secret `s` hidden in weights via superposition | LOW–MEDIUM | ModelObfuscator 2306.06112; antagonized by Carlini cryptanalytic extraction 2024/1580, 2026/253 |
| C3 | Formal CPA reduction under black/gray/white-box | MEDIUM | Shamir et al. "Deep Neural Cryptography" ePrint 2025/288 (EUROCRYPT 2026) defines security for crypto-in-DNN but not LWE-specific |
| C4 | SWIFFT as neural-native hash inside Transformer FFN | HIGH | No prior FFT-lattice hash + Transformer pairing found |
| C5 | Neural-LWE attack taxonomy (float subnormals, gradient leakage, side channels) | LOW–MEDIUM | Shamir 2025/288 pre-empts float-subnormal; gradient leakage & side channels well-known |

## Overall
- **Score: 5.5/10**
- **Recommendation: PROCEED WITH CAUTION**
- **Unique angle:** SWIFFT-Transformer isomorphism (C4) + LWE-hardness-specific reduction (C3).
- **Biggest risk:** Shamir 2025/288 pre-empts much of the threat-model novelty. Carlini-style extraction threatens the secret-as-weight story.

## Repositioning suggestions
1. Lead with SWIFFT, not generic LWE decryption.
2. Make LWE-hardness reduction the theoretical centerpiece.
3. Defend against Carlini extraction upfront, or scope paper to black-box only.
4. Drop/demote float-subnormal material (pre-empted by Shamir).
5. Frame Phase 1 as "first mechanistically-verified crypto circuit", not "first embedded LWE".

## Key sources
- arXiv 2511.13653 — Weight-sparse transformers (Gao, OpenAI, 2025)
- IACR ePrint 2025/288 — Deep Neural Cryptography (Shamir et al.)
- arXiv 2207.04785 / 2303.04178 — SALSA / Picante
- NDSS 2025 — TensorCrypt (Jin, Ma, Lin)
- Crypto'20, ePrint 2024/1580, 2026/253 — Cryptanalytic extraction (Carlini et al.)
- arXiv 2506.23679 — Modular exponentiation in transformers
- arXiv 2301.05217 — Progress measures for grokking
- arXiv 2501.00684 — IGC (Dietz & Klakow)
- SWIFFT (Lyubashevsky et al., FSE 2008)

*Phase C (Codex cross-review) skipped — MCP was not available when this report was generated.*
