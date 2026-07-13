---
name: golden-falsifiable-testing
description: How to design a FALSIFIABLE golden test for quant-matmul (full-code coverage, dual rtol+cosine gate, exact-accum scale, negative control); what counts as weak vs strong evidence
metadata:
  type: reference
---

# Designing a falsifiable golden test (quant-matmul / fp4 example)

General methodology distilled from validating the fp4 MX matmul golden on board. Goal: a PASS that actually means something, not a vacuous green.

## Ingredients of a falsifiable test
1. **Full-code coverage.** Emit fully random bytes (`randint(0,256)`) so every nibble uniformly hits all 16 e2m1 codes (incl. the easy-to-miss ±2/±3/±4/±6). Golden decodes the SAME bytes via the same LUT + nibble order, so golden ≡ NPU by construction (modulo HW LUT/nibble unknowns). Random bytes serve CODE COVERAGE, not nibble-order verification.
2. **A precision-sensitive output dtype + dual gate.** Use bf16 output (mantissa 8b → rel-err ≤ 2^-8 ≈ 0.0039) and require BOTH `allclose(rtol≈0.008, atol≈0.02)` AND `cosine >= 0.999`. fp16/fp32-only or "cos≈1" alone is a WEAK gate.
3. **Make accumulation exact so HW ≡ golden is order-independent.** Tighten scale (e.g. e8m0 ∈ {0.5,1,2}) so every product is an integer multiple of a small power of two and the K-sum fits without overflow → fp32 cube accum is EXACT regardless of summation order → max_diff can legitimately reach 0.
4. **A negative control.** Add a switch (default off) that corrupts the golden in a FALSIFIABLE direction (e.g. a magnitude error: code7 6.0→7.0, symmetric code15 −6.0→−7.0). On board: correct → all PASS; corrupted → the affected cases FAIL (both gates break), unaffected dtype (fp8) stays PASS. This proves the harness has DISCRIMINATING POWER. Keep it in the script as a re-verifiable meta-test asset.

## What max_diff=0 proves (and doesn't)
- Under full random bytes, `max_diff=0` across all cases is direct falsifiable evidence the **HW decode LUT ≡ reference, code-by-code** (any per-code divergence would amplify across the K dot-product → max_diff ≫ 0).
- It says NOTHING about structural blind spots that are math-invariant (nibble order, global sign flip — see [[fp4-e2m1-packing]]). A negative control built on a magnitude error does NOT cover those either; they're confirmable only by spec + kernel read.

## Weak vs strong evidence ledger
- **WEAK** (do not cite as proof): "cos≈1" under restricted coverage (few codes, small-int values, 2^k scales that bypass rounding); a local pure-Python fp64 probe (no cube, non-fp32/bf16 arithmetic — NOT board evidence).
- **STRONG** (board, multi-batch / full-code / independent path): full-16-code max_diff=0; multi-batch PASS for a per-batch offset; transpose cases; true N=1 boundary; the negative control demonstrating discrimination.

See [[fp4-e2m1-packing]], [[device-print-verification]], [[qbmm-batch-status]].
