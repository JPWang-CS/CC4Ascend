---
name: device-print-verification
description: AscendC::PRINTF on device can confirm "code was reached" but cannot falsify "not reached"; what actually counts as load-bearing on-board evidence
metadata:
  type: reference
---

# Device-side verification: what AscendC::PRINTF can and cannot prove

General rule for verifying kernel execution on Ascend hardware.

- A temporary device `AscendC::PRINTF` in a kernel branch can **confirm "this code was reached"** (if you see the output, it ran).
- It **CANNOT falsify "not reached"**: buffering, flush behavior, and multi-core scheduling do not guarantee complete/ordered output. Absence of a print is NOT proof the branch didn't execute.
- Therefore PRINTF is a positive probe only. The **load-bearing evidence is numeric**: e.g. for a per-batch offset, a multi-batch test that PASSES is the real proof (a wrong offset would numerically FAIL batch>0), not the print.
- A device PRINTF probe is a TEMP debugging asset — delete it from formal code before finalizing.

See [[golden-falsifiable-testing]].
