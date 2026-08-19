# Design decisions

A record of the non-obvious choices in this project, what the alternatives were,
and what each one costs. Entries marked **open** have not been settled by
measurement yet.

---

## 1. Qwen2-VL-2B rather than a larger VLM

**Chose:** `Qwen2-VL-2B-Instruct`.

The binding constraint is a single free-tier GPU with ~15 GB and a session that
can be killed at any time. In 4-bit, a 2B model leaves enough headroom for
activations at a useful image resolution; a 7B model fits in memory but only by
cutting `max_pixels` far enough that receipt small print stops being legible,
which trades away the thing being measured.

Qwen2-VL specifically because its native dynamic resolution handling suits
receipts, which are tall and narrow and get badly distorted by architectures that
assume a fixed square input.

**Cost:** absolute accuracy is below what a 7B model would reach. Since the
headline claim is a *delta* — base vs fine-tuned, clean vs corrupted — measured
under a fixed budget, that is acceptable. It would not be if the goal were a
leaderboard number.

## 2. QLoRA rather than full fine-tuning

**Chose:** 4-bit NF4 base weights, LoRA adapters at r=16, double quantisation,
paged 8-bit AdamW.

Full fine-tuning of even a 2B model needs optimiser state and gradients for every
parameter — far past the budget. LoRA trains well under 1% of parameters, so
optimiser state is negligible and the adapter is a few tens of MB, which also
makes checkpoint-and-resume cheap enough to do every 100 steps.

NF4 over int4 because it is information-theoretically matched to the roughly
normal distribution of pretrained weights. Double quantisation buys ~0.4 GB by
quantising the quantisation constants; at this scale it is free in both quality
and speed.

**Cost:** 4-bit quantisation of the *base* model means the frozen backbone is
slightly degraded before training starts, so the base-model row in the results
table is a 4-bit baseline, not an fp16 one. This is stated in the README because
comparing a 4-bit fine-tune against an fp16 baseline would flatter the fine-tune.

## 3. Vision tower frozen; adapters on the language tower only

**Chose:** LoRA on `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
in the language model. Vision encoder untouched.

The failure mode being fixed is not "cannot read the text" — the base model reads
receipts adequately. It is "reads the text, then emits prose or malformed JSON
instead of the requested schema." That is a language-side behaviour, so that is
where the capacity goes.

Attention and MLP projections both, rather than attention only: the MLP blocks
carry most of the format-following behaviour, and excluding them was measurably
worse in the QLoRA paper's ablations for the equivalent decision.

**Cost:** if the vision encoder turns out to be the bottleneck under heavy
corruption, this choice caps how much fine-tuning can help there. That is exactly
what the robustness sweep should reveal, and it is the most interesting negative
result this project could produce.

**Open:** whether unfreezing the vision merger layer alone recovers corrupted-input
accuracy at acceptable memory cost.

## 4. Field-level F1 rather than string similarity

**Chose:** flatten both prediction and target to `(dotted.path, value)` pairs,
normalise, compare as multisets.

BLEU and edit distance on serialised JSON measure the wrong thing in both
directions: a model that emits every field correctly but orders keys differently
is punished, and a model that emits beautifully formed JSON with the wrong total
is rewarded. Neither corresponds to whether the output is usable.

List indices are included in the flattened path, which is a deliberate
strictness — for receipts, line-item order is meaningful, and the right items in
the wrong order is a real error.

**Cost:** the metric is unforgiving about schema drift. A model that invents a
sensible-but-different key structure scores near zero even if a human would call
the extraction correct. Acceptable, because a downstream consumer parsing these
outputs would break in exactly the same way.

## 5. Locale-aware numeric normalisation

**Chose:** infer the role of `.` and `,` from digit grouping rather than assuming
a locale.

CORD is Indonesian: `15.000` is fifteen thousand. An implementation that reads
the dot as a decimal point turns that into 15.0 and marks a correct prediction
wrong. The first version of `metrics.py` in this repo had exactly that bug, and
it was caught by a unit test rather than by a training run — which is the reason
the numeric cases are pinned in `_self_test()`.

The one genuinely ambiguous case is a single separator with exactly three
trailing digits. It resolves to thousands grouping, which is correct for this
corpus and wrong for a corpus of prices under ten. Documented rather than hidden.

## 6. Sorted, compact JSON targets

**Chose:** `sort_keys=True`, `separators=(",", ":")`, `ensure_ascii=False`.

Without sorted keys, the model is asked to reproduce whatever key order happened
to land in the annotation file. That is unlearnable noise and shows up as a
phantom accuracy ceiling that looks like a capacity problem. Compact separators
save output tokens on every example. `ensure_ascii=False` keeps Indonesian text
intact instead of expanding it into escape sequences that waste the token budget.

## 7. Prompt tokens masked out of the loss

**Chose:** labels set to `-100` for the system prompt, user turn, image
placeholders, and padding.

Only the assistant's JSON should contribute gradient. Training on the prompt as
well spends capacity learning to reproduce a system prompt that is byte-identical
on every example.

The prompt length has to be measured per sample with that sample's image
attached, because Qwen2-VL expands the image placeholder into a variable number
of visual tokens depending on resolution. Using a fixed offset here silently
misaligns the mask and is close to invisible in the loss curve.

## 8. Deterministic, seeded corruptions

**Chose:** each corruption seeded from `(name, severity, sample_index)`.

Base and fine-tuned models must be compared on byte-identical inputs, or the
robustness delta is partly measuring which random noise draw each model happened
to get. Seeding also makes reruns reproducible, so a regression is
distinguishable from sampling noise.

Rotation alternates sign per sample rather than always tilting one way, so the
evaluation set does not contain a learnable directional bias.

## 9. Greedy decoding at evaluation

**Chose:** `do_sample=False`.

The task has one correct answer. Sampling would add variance to every cell of a
19-condition grid for no upside, and the robustness deltas are the entire point.

## 10. Corruption severity capped at "still human-legible"

**Chose:** severity 3 is the worst condition, calibrated so a human can still
read the receipt.

Pushing to genuinely illegible inputs produces a dramatic accuracy collapse that
means nothing — no model should extract fields from an image with no information
in it. Capping at human legibility means every point of accuracy lost is a point
the model should have kept, which makes the numbers interpretable rather than
merely large.

## 11. Evaluation subset of 50 test samples

**Chose:** `eval_limit: 50` (revised down from 100 during implementation).

The sweep is 19 conditions per tag and two tags, so the sample count multiplies
by 38 into total generation cost. At 100 samples that is 1,900 generations per
tag at up to 768 new tokens each, which on the actual hardware below does not
fit in a single sitting. 50 halves it.

**Cost:** this is the decision in the file with the worst power-to-cost ratio,
and it is a real loss. At n=50 a per-condition F1 difference of a few points is
not distinguishable from sampling noise. What survives at this n is the shape of
the degradation curve — monotonic decline across severities 1/2/3 within a
corruption, and the ordering of corruptions by how much they hurt — not any
individual cell. The README limitations section says this rather than burying it.

**Changed from:** the original `eval_limit: 100`. Recorded here rather than
silently edited because it weakens every claim the results table makes, and an
interviewer should be able to see the tradeoff was deliberate.

## 12. Hardware is an 8 GB RTX 4060, not a 16 GB T4/P100

**Chose:** run on the machine that exists.

Decision #1 sizes the whole project against "a single free-tier GPU with ~15 GB."
The actual device is an RTX 4060 Laptop with 8.0 GB — roughly half the assumed
budget, on Windows rather than Linux. The 2B-in-4-bit choice survives this
comfortably; it is the `max_pixels: 401408` headroom that does not, and that is
the knob most likely to move.

**Consequence to watch:** `optim: paged_adamw_8bit` relies on `cudaMallocManaged`,
whose behaviour under Windows WDDM is not the same as under Linux. If paging
misbehaves, the fallback is plain `adamw_8bit` — the paging exists to survive
memory spikes, not because the optimiser state is large.

**Settled by measurement (Phase 1/2).** `max_pixels: 401408` fits unchanged; it
never needed halving. Measured on this card:

| | |
| --- | --- |
| Weights, 4-bit NF4 | 2.43 GB |
| Peak allocated, fwd+bwd, batch_size=1 | 4.99 GB |
| Peak reserved | 5.73 GB |
| Free after a training step | 1.16 GB of 8.00 GB |
| Peak during generation (eval) | 1.53 GB |

Evaluation is far cheaper than training because there are no activations to
retain. Training throughput is ~6.5 s per example (forward+backward, gradient
checkpointing on); base-model generation is ~7.0 s per sample at ~75 new tokens.

`paged_adamw_8bit` ran without incident on Windows, so the `cudaMallocManaged`
concern above did not materialise and the fallback was not needed.

## 13. Corrupted images are re-normalised to the clean image's pixel area

**Chose:** `resize_for_budget` -> `apply_corruption` -> resize back to the clean
image's own area, in `src/evaluate.py`.

**Changed from:** a single resize before corruption. That was wrong, and it was
caught by measurement in Phase 1, not by reading the code.

`rotation` uses `expand=True`, so the canvas grows to contain the tilted image.
Measured against the 401,408-pixel budget:

| severity | pixels | vs budget |
| --- | --- | --- |
| 1 (+/-3 deg)  | 446,157 | 1.11x |
| 2 (+/-7 deg)  | 507,297 | 1.26x |
| 3 (+/-13 deg) | 592,767 | 1.48x |

Qwen2-VL emits one visual token per 28x28 patch, so uncapped this hands
`rotation` up to ~48% more visual tokens than any other condition. Its headline
number - "delta vs clean" - would then be a mix of a tilt effect and a
resolution/compute effect, with no way to separate them. That is exactly the
confound the fixed pixel budget exists to prevent, and rotation was the single
corruption escaping it.

Normalising to *the clean image's own area* rather than to the global cap
matters: many CORD receipts resize to well under 401,408 px, and a global cap
would still let rotation gain pixels on those. Measured after the fix, visual
token counts are identical to clean for all five size-preserving corruptions and
within +/-5% for rotation, with the residual non-monotonic across severity
(+23/+7/+12 tokens on one sample, -20/+5 on another) - i.e. rounding to the
28-pixel patch grid, not a systematic advantage.

Re-normalising is also the more physically faithful choice. A real phone photo of
a tilted receipt is taken with the same sensor, so the receipt occupies *fewer*
pixels of the frame, not more. The original code was implicitly giving the tilted
document a larger camera.

**Cost:** rotation now carries a small resolution loss on top of the tilt
(~0.82x linear at severity 3), so it is not a pure geometric transform and the
README should not describe it as one. Driving the residual to exactly zero would
mean cropping the rotated image back to the original aspect ratio, which cuts off
receipt corners - a worse trade than a few percent of token count.

**Not fixed in `corruptions.py`**, which is the verified measurement contract.
The correction belongs to how evaluation feeds the model, not to what a rotation
is.

## 14. A schema-free value-recall metric reported alongside field F1

**Chose:** add `src/value_metrics.py` - flatten both records to leaf values,
discard the keys, compare as multisets. Field F1 remains the headline metric and
`metrics.py` is unmodified.

**Why it was added, stated plainly:** field F1 collapses two different failures
onto the same number - "cannot read the receipt" and "reads it perfectly, names
the keys differently". Measured in Phase 2, base Qwen2-VL-2B emits well-formed
JSON (parse rate 1.000) with correct values under its own key names, e.g.
`items[].name` where CORD wants `menu[].nm`, and scores field F1 exactly 0.000.

The consequence is not cosmetic. A baseline pinned at 0.000 on clean input also
scores 0.000 under all eighteen corruption conditions, so the base half of the
robustness sweep - half the deliverable - would have been nineteen zeros and
could not have shown degradation at all. You cannot measure a decline in a
quantity already at the floor. Measured on two test samples, base value recall is
0.263, which is off the floor and therefore has somewhere to fall.

**The credibility cost, which is real.** This metric was added *after* observing
that the primary metric returned zero for the baseline. That is the shape of
metric-shopping and should be treated with suspicion by default. Three things are
offered against that, and they are checkable rather than assertions:

1. Field F1 is unchanged and still leads every table. `metrics.py` was not
   touched; the new module imports from it so the locale-aware numeric handling
   in decision #5 has exactly one definition.
2. The definition was committed *before* the baseline evaluation was run, so git
   history shows it could not have been tuned against results that did not yet
   exist.
3. It is reported for both models. It is not a metric that only ever flatters the
   baseline - fine-tuning is expected to win on both views.

**Cost:** value recall cannot distinguish a correctly structured record from a
bag of right-looking strings. A model emitting every correct value in a flat list
scores identically to one emitting the exact target schema. That is precisely the
weakness `metrics.py` exists to cover, which is why this is reported *alongside*
and never *instead*. Read alone it would overstate how usable the output is.

**Also settled here:** evaluation runs with `--save-predictions`, and those files
are committed rather than gitignored. Raw model output is the audit trail - it
makes every number in the README recomputable offline, including by a metric that
does not exist yet, without re-running a two-hour sweep.

## 15. Evaluation results are written per condition, and resumable

**Chose:** `src/evaluate.py` writes `results/<tag>.json` after every condition and
accepts `--resume` to skip conditions already on disk.

**Changed from:** a single write after all 19 conditions finished. That was a
defect, and it defeated the stated purpose of running the baseline first. Phase 3
of BUILD_PLAN.md exists so that "if the full training run eats the session, a
baseline already on disk means the session was not wasted" - but the baseline
never reached disk until the final condition completed, so any interruption lost
the entire sweep.

It stopped being theoretical during the Phase 3 run. Measured per-condition
wall-clock on the base model, 50 samples each:

| condition | seconds |
| --- | --- |
| clean | 550.7 |
| gaussian_blur s1 | 435.6 |
| gaussian_blur s2 | 585.1 |
| gaussian_blur s3 | **10819.6** |

Three hours for one condition, ~20x the clean case. The cause is visible in the
scores: at severity 3 the model stops emitting a compact JSON object
(parse rate 0.780, value recall 0.073) and instead degenerates, running to the
`max_new_tokens` cap on most samples. Generation cost scales with tokens emitted,
so the worst conditions are also the slowest, and a full sweep has a long and
unpredictable tail.

**Cost:** one JSON write per condition, which is negligible against a run measured
in hours. The payload gains a `complete` flag so a partial file is never mistaken
for a finished sweep.

## 16. `max_new_tokens` stays at 768 - a wrong diagnosis, corrected

**Chose:** 768. Briefly lowered to 512 during Phase 3, then reverted. Both the
change and the revert are recorded because the reasoning behind the change was
wrong, and the way it was caught is the useful part.

**What was observed.** During the first baseline sweep, `gaussian_blur s3` took
10819.6 s against 550.7 s for clean - roughly 20x. Extrapolating, a 19-condition
sweep looked like 15-20 h per tag.

**What it was blamed on.** Degenerate generation: the theory was that under heavy
corruption the model stops emitting compact JSON and runs to the `max_new_tokens`
cap on every sample, so cost tracks the cap. `max_new_tokens` was lowered to 512
and the sweep restarted, discarding six completed conditions.

**What actually disproved it.** On the rerun, all four completed conditions
reproduced their scores *exactly* - `blur s3` again at value recall 0.073, parse
rate 0.780 - while `blur s3` fell to 410.9 s. A 26x speedup cannot come from a
768 -> 512 cap, and identical scores mean identical output. Measuring the saved
predictions directly:

| condition | samples at the cap | mean tokens |
| --- | --- | --- |
| clean | 2 / 50 | 130 |
| gaussian_blur s1 | 2 / 50 | 134 |
| gaussian_blur s2 | 4 / 50 | 144 |
| gaussian_blur s3 | 11 / 50 | 181 |

Eleven capped samples at `blur s3`, and parse rate 0.780 is exactly eleven
failures - the capped samples *are* the parse failures. Eleven samples times the
256-token difference is roughly 400 s of extra generation, predicting ~800 s at
the 768 cap. Not 10819 s.

**The real cause was environmental.** The GPU was observed in performance state
P3 drawing 15 W of a 140 W budget with clock event reason `Idle: Active`, no
thermal or power throttling. Every condition in that first run was also ~1.7x
slower than its rerun (550/435/585 s versus 301/308/335 s), which is background
contention, not workload. The slow run was a machine state, not a property of the
evaluation.

**Why 512 was reverted rather than kept.** The cap provably did not change base
model scores, so keeping it would have been harmless *for the baseline*. It would
not have been harmless for the fine-tuned model. Target serialisations run to 573
tokens at p100, and a fine-tuned model emits well-formed CORD JSON of exactly that
length - so a 512 cap truncates legitimate long answers and counts them as
misses. It would bias the headline delta downward on precisely the long receipts
the fine-tune is trained to produce, while leaving the base model untouched
because its long outputs are degenerate anyway. An asymmetric ceiling on the
comparison is worse than a slow sweep.

**Cost of the mistake:** about four hours of GPU time, and six conditions
discarded on a diagnosis that the evidence did not support. The corrective was
cheap and should have come first - comparing scores across the two caps, and
measuring generated token lengths from saved predictions, both took under a
minute and both were available before the restart.

---

## Rejected

**TrOCR / Donut as the backbone.** Purpose-built for document understanding and
would likely score higher on clean CORD. Rejected because the interesting
question is how a *general-purpose* VLM degrades — that is the class of model
people are actually deploying for document tasks now, and its failure modes are
less well characterised.

**Full-resolution images.** OOMs, and the fix (a bigger GPU) is not available.
Capping total pixels rather than a fixed side length preserves receipt aspect
ratio, which a fixed square resize destroys.

**TRL `SFTTrainer`.** Would remove some collator boilerplate. Rejected because
the label-masking behaviour for interleaved image tokens is version-sensitive and
easy to get subtly wrong; an explicit collator makes the masking auditable, and
masking bugs here are nearly invisible in the loss curve.

**Reporting only clean-set accuracy.** The default in this space, and the reason
the project exists.
