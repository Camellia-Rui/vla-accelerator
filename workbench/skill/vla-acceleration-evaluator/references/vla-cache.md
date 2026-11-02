# VLA-Cache quarantine status

This release keeps VLA-Cache as historical evidence only. It does not bundle an overlay and must not generate a deployment adapter from the archived implementation.

Verified on the recorded RTX 4090/OpenVLA/LIBERO Object/Transformers 4.47 environment:

- Decoder prefill compute and FLOPs were genuinely reduced; real skip/memo events occurred.
- A clean fixed-observation `cache_once_memo` screen showed about 1.025x query speed with a positive paired mean-saved confidence interval.
- The original paired closed loop produced native 14/20 successes versus cache 0/20, despite about 1.036x median per-query speed.
- Correcting the SDPA/manual-attention mismatch restored the full-refresh control, but actual cross-frame reuse diverged on the first reused frame.
- Small reuse budgets, refresh scans, and late-layer reuse did not yield a speed/correctness intersection.

Therefore the tested implementation is **REJECTED for closed-loop deployment**. Do not repeat refresh interval, reuse budget, threshold, or start-layer micro-scans. Reopening VLA-Cache requires an algorithmic redesign, a fresh source identity, a retain-all control, first/second-action equivalence checks, and the full paired closed-loop protocol.

Archived raw evidence is outside the Skill under `archive/excluded-vla-cache-evidence/` in the release bundle. GRIDs/OpenPI evidence is unrelated and must not be mixed into this conclusion.

