# ACT policy parameters — full reference

Every field on `ACTConfig`, what it does, how to set it from an experiment YAML in
this repo, and where it comes from academically. Source of truth for the field
list: `cyclo_brain/policy/lerobot/lerobot/src/lerobot/policies/act/configuration_act.py`.

ACT ("Action Chunking with Transformers") was introduced in:

> Zhao, Tony Z., Kumar, Vikash, Levine, Sergey, Finn, Chelsea. **"Learning
> Fine-Grained Bimanual Manipulation with Low-Cost Hardware."** arXiv:2304.13705
> (2023). [huggingface.co/papers/2304.13705](https://huggingface.co/papers/2304.13705) ·
> [project page](https://tonyzhaozh.github.io/aloha) ·
> [original code](https://github.com/tonyzhaozh/act)

Unless a field's entry says otherwise, that paper is the source. lerobot's ACT
also borrows two specific architectural pieces from **DETR** (learned decoder
queries, dilated-backbone option) and one from the **VAE** literature (the KL
term); those are cited individually below.

**How to change any of these in this repo:** every key under `policy:` in an
experiment YAML (`cyclo_brain/train/experiments/<name>.yaml`) is forwarded
verbatim as `--policy.<key>=<value>` to `lerobot-train`. See
`_base/act.yaml` for the current defaults and any `peanut_act_*.yaml` for how a
child experiment overrides just one or two keys. `./docker/container.sh
train-lerobot <experiment> --dry-run` prints the exact resolved command
without launching anything — always sanity-check a new value this way first.

---

## Contents

1. [Input / output structure](#input--output-structure)
2. [Vision backbone](#vision-backbone)
3. [Transformer architecture](#transformer-architecture)
4. [VAE (CVAE) objective](#vae-cvae-objective)
5. [Inference-time behavior](#inference-time-behavior)
6. [Training & loss](#training--loss)
7. [Optimizer preset](#optimizer-preset)
8. [Adjacent knobs (not ACT-specific)](#adjacent-knobs-not-act-specific)
9. [Recommended changes](#recommended-changes)

---

## Input / output structure

### `n_obs_steps`
- **Default:** `1` (and, for ACT specifically, the *only* legal value)
- **What it affects:** how many past environment steps of observation the
  policy is allowed to condition on.
- **How to change it:** `policy.n_obs_steps` — but `ACTConfig.__post_init__`
  raises `ValueError` if it's anything other than `1`. It exists on the shared
  `PreTrainedConfig` base (other policies like Diffusion Policy or π₀ use
  multi-step history); ACT's paper/architecture only ever conditions on the
  current step, so this repo will never actually use a different value.
- **Source:** general policy-config field, not ACT-paper-specific.

### `chunk_size`
- **Default:** `100` (lerobot) / **`30`** (this repo's `_base/act.yaml`, with a
  comment flagging `100, 150?` as untested alternatives — see
  [Recommended changes](#recommended-changes))
- **What it affects:** the action-chunking horizon — how many future timesteps
  the transformer decoder predicts in one forward pass. This is *the* idea the
  paper introduces: instead of predicting one action at a time, predict a
  whole chunk, which reduces compounding error and lets the policy express
  multi-modal, temporally-consistent behavior.
- **How to change it:** `policy.chunk_size`. Also sets
  `action_delta_indices = range(chunk_size)`, which determines how many future
  action rows get pulled from the dataset per training sample.
- **Constraint:** `n_action_steps <= chunk_size` (enforced).
- **Source:** ACT paper, §III (action chunking).

### `n_action_steps`
- **Default:** `100` (lerobot) / **`16`** (this repo)
- **What it affects:** how many actions from a predicted chunk are actually
  executed on the robot before the policy is queried again for a fresh chunk.
  Small values replan more often (more reactive, more inference calls);
  values close to `chunk_size` commit to a longer open-loop segment per
  inference call.
- **How to change it:** `policy.n_action_steps`. Must be `1` if
  `temporal_ensemble_coeff` is set (see below) and must not exceed
  `chunk_size`.
- **Source:** ACT paper, §III.

### `input_features` / `output_features`
- **Default:** `{}` (auto-inferred)
- **What it affects:** the literal dict of `{feature_name: PolicyFeature(type,
  shape)}` for everything the model reads (`observation.state`,
  `observation.images.*`) and writes (`action`). This is the schema the model
  is built against.
- **How to change it:** **you don't, in this repo.** `cyclo_train` derives
  these straight from the dataset's `meta/info.json` at run time (that's why
  `manifest.json` records `action_shape`/`observation_state_shape`/
  `camera_keys` — it's recording what got auto-inferred). Hand-setting these
  is only relevant if you're calling `lerobot-train` directly against a
  dataset outside this pipeline.
- **Source:** general policy-config field.

### `normalization_mapping`
- **Default:** `{VISUAL: MEAN_STD, STATE: MEAN_STD, ACTION: MEAN_STD}`
- **What it affects:** which normalization is applied per modality before it
  hits the network — `MEAN_STD` z-scores using dataset statistics
  (`meta/stats.json`), `MIN_MAX` rescales to `[-1, 1]` (or `[0, 1]`) using
  dataset min/max instead.
- **How to change it:** `policy.normalization_mapping.VISUAL=...` etc. Not
  exposed in any experiment YAML here — all runs use the default (MEAN_STD
  everywhere), which is standard for ACT.
- **Source:** general policy-config field; MEAN_STD vs MIN_MAX is a standard
  imitation-learning preprocessing choice, not something the ACT paper itself
  prescribes.

---

## Vision backbone

### `vision_backbone`
- **Default:** `"resnet18"`
- **What it affects:** which torchvision ResNet variant encodes each camera
  frame into a feature map. Resolved as `getattr(torchvision.models,
  vision_backbone)`, so any ResNet name torchvision exposes works
  (`resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`) — validated
  to require the string start with `"resnet"`.
- **How to change it:** `policy.vision_backbone=resnet34` (etc). Every run in
  this repo uses the default `resnet18` — no backbone-size ablation has been
  run yet.
- **Source:** He, Kaiming, et al. **"Deep Residual Learning for Image
  Recognition."** arXiv:1512.03385 (2015). ACT's paper itself uses ResNet-18
  for all its real-robot experiments.

### `pretrained_backbone_weights`
- **Default:** `"ResNet18_Weights.IMAGENET1K_V1"`
- **What it affects:** whether the backbone starts from ImageNet-pretrained
  weights (torchvision's typed-weights string) or from scratch (`None`).
  Pretrained weights are why 23–63k frames of training data is enough at all
  — the backbone isn't learning "what an edge looks like" from your robot
  data, only fine-tuning.
- **How to change it:** `policy.pretrained_backbone_weights=null` for random
  init, or e.g. `ResNet34_Weights.IMAGENET1K_V1` if you also change
  `vision_backbone` to `resnet34`. Must match the chosen `vision_backbone`'s
  weights enum.
- **Source:** ImageNet pretraining is standard practice; not from the ACT
  paper specifically (their released code also defaults to ImageNet-init
  ResNet-18).

### `replace_final_stride_with_dilation`
- **Default:** `False`
- **What it affects:** swaps the ResNet's last stage's 2× spatial downsampling
  stride for a dilated (atrous) convolution instead — same receptive field,
  but the final feature map comes out **2× higher resolution** in each spatial
  dim. Costs more compute per image, may help on tasks needing finer spatial
  precision (small objects, tight insertion tasks).
- **How to change it:** `policy.replace_final_stride_with_dilation=true`.
  Passed straight through to torchvision's ResNet constructor as
  `replace_stride_with_dilation=[False, False, <this value>]` (only the last
  of the three stages is affected).
- **Source:** the dilated-final-stage trick is the "DC5" variant from Carion,
  Nicolas, et al. **"End-to-End Object Detection with Transformers"**
  (DETR). arXiv:2005.12872 (2020) — ACT's transformer head design borrows
  directly from DETR, and this option is a direct port of DETR's `--dilation`
  flag.

---

## Transformer architecture

These control the encoder/decoder transformer that sits on top of the vision
backbone (fuses image features + robot state + latent style variable `z`,
and decodes the action chunk via cross-attention). Design follows DETR's
encoder-decoder structure with learned decoder queries standing in for DETR's
object queries — here, one query per chunk timestep instead of one per object.

### `dim_model`
- **Default:** `512`
- **What it affects:** the transformer's hidden width — every attention block
  and residual stream is this size. Bigger = more capacity, more memory,
  slower.
- **How to change it:** `policy.dim_model`. Unchanged from default in every
  run so far.
- **Source:** Vaswani, Ashish, et al. **"Attention Is All You Need."**
  arXiv:1706.03762 (2017) — the base Transformer architecture. 512 specifically
  matches the ACT paper's own setting.

### `n_heads`
- **Default:** `8`
- **What it affects:** number of attention heads in every multi-head
  attention block (`dim_model` is split evenly across heads — must divide
  `dim_model`).
- **How to change it:** `policy.n_heads`.
- **Source:** Vaswani et al. 2017 (multi-head attention).

### `dim_feedforward`
- **Default:** `3200`
- **What it affects:** the width of the feed-forward (MLP) sublayer inside
  each transformer block — the usual "expand then project back down"
  bottleneck. `3200 ≈ 6.25× dim_model`, wider than the Transformer paper's
  original 4× rule of thumb.
- **How to change it:** `policy.dim_feedforward`.
- **Source:** Vaswani et al. 2017; the specific 3200 value matches the ACT
  paper/codebase's setting.

### `feedforward_activation`
- **Default:** `"relu"`
- **What it affects:** the nonlinearity inside that feed-forward sublayer.
  Accepts `"relu"`, `"gelu"`, or `"glu"` (anything else raises at runtime).
- **How to change it:** `policy.feedforward_activation=gelu`.
- **Source:** ReLU is the ACT paper's default; GELU/GLU are standard
  transformer alternatives available in lerobot's implementation but not
  ablated by the paper or by any run here.

### `n_encoder_layers`
- **Default:** `4`
- **What it affects:** depth of the transformer **encoder** (the part that
  fuses backbone image tokens + state token + latent `z` token before
  cross-attention). More layers = more mixing of the different modalities
  before decoding.
- **How to change it:** `policy.n_encoder_layers`.
- **Source:** ACT paper architecture; DETR-style encoder-decoder split.

### `n_decoder_layers`
- **Default:** `1`
- **What it affects:** depth of the transformer **decoder** (cross-attends
  from the chunk-length query embeddings into the encoder's output to produce
  the action sequence).
- **How to change it:** `policy.n_decoder_layers`. **Note the code comment
  here is important**: the original ACT implementation set this to 7, but a
  bug in that code meant only the first decoder layer ever actually
  contributed (see
  [tonyzhaozh/act#25](https://github.com/tonyzhaozh/act/issues/25)). lerobot
  pins this to `1` to *faithfully reproduce* the original's actual (buggy)
  behavior rather than "fix" it into a materially different model. Setting
  this above 1 gives you a real, deeper decoder that the original paper's
  reported numbers don't reflect — untested territory, not a bug fix.
- **Source:** ACT paper + the linked GitHub issue documenting the discrepancy.

### `pre_norm`
- **Default:** `False` (i.e. post-norm, LayerNorm *after* each sublayer)
- **What it affects:** whether LayerNorm is applied before (`pre_norm=True`)
  or after (`False`) each attention/feed-forward sublayer. Pre-norm generally
  trains more stably at higher learning rates / greater depth; post-norm (the
  original Transformer's choice, and ACT's default) can perform slightly
  better once it does converge.
- **How to change it:** `policy.pre_norm=true`.
- **Source:** Xiong, Ruibin, et al. **"On Layer Normalization in the
  Transformer Architecture."** arXiv:2002.04745 (2020) — the paper that
  formalized and compared pre-LN vs. post-LN.

---

## VAE (CVAE) objective

ACT trains a conditional VAE: during training, a small transformer encoder
("VAE encoder") looks at the ground-truth action chunk + robot state and
produces a latent style variable `z`; the main encoder-decoder then predicts
the action chunk *conditioned on* `z`. At inference time there's no
ground-truth chunk to encode, so `z` is just set to zero.

### `use_vae`
- **Default:** `True`
- **What it affects:** whether the CVAE machinery (encoder + KL loss) is used
  at all. `False` collapses ACT to a plain deterministic chunked-action
  predictor (no `z`, no KL term).
- **How to change it:** `policy.use_vae=false`. Every run so far keeps this
  on (the default, and what the paper reports its main numbers with).
- **Source:** ACT paper §III-B, "Modeling human data with a conditional VAE";
  the conditional-VAE formalism itself is from Sohn, Kihyuk, Lee, Honglak, and
  Yan, Xinchen. **"Learning Structured Output Representation using Deep
  Conditional Generative Models."** NeurIPS 2015.

### `latent_dim`
- **Default:** `32`
- **What it affects:** dimensionality of the latent style variable `z`. Small
  by VAE standards on purpose — `z` is meant to capture *stylistic* variation
  in how a demonstrator performed the task (speed, exact trajectory), not
  a large generative code.
- **How to change it:** `policy.latent_dim`.
- **Source:** ACT paper §III-B; VAE latent-space concept from Kingma &
  Welling (below).

### `n_vae_encoder_layers`
- **Default:** `4`
- **What it affects:** depth of the VAE encoder transformer (processes
  `[cls, robot_state, *action_sequence]` down to the latent distribution's
  mean/log-variance). Only matters when `use_vae=True`.
- **How to change it:** `policy.n_vae_encoder_layers`.
- **Source:** ACT paper architecture.

---

## Inference-time behavior

### `temporal_ensemble_coeff`
- **Default:** `None` (disabled)
- **What it affects:** an alternative to executing a whole `n_action_steps`
  chunk open-loop. With this set, the policy is queried at **every** step
  (forces `n_action_steps=1`), and the action actually executed is an
  exponentially-weighted average across the overlapping chunk predictions
  from several recent queries — newer predictions weighted by
  `exp(-coeff * age)`. Smooths jitter between chunk boundaries at the cost of
  one inference call per environment step instead of one per
  `n_action_steps`.
- **How to change it:** `policy.temporal_ensemble_coeff=0.01` — the paper's
  own default value when this feature is used. Implemented as
  `ACTTemporalEnsembler`.
- **Source:** ACT paper, Algorithm 2 ("temporal ensembling"). Not used by any
  run in this repo — all runs use the fixed-chunk open-loop mode
  (`n_action_steps=16`) instead.

---

## Training & loss

### `dropout`
- **Default:** `0.1`
- **What it affects:** dropout probability applied throughout the
  transformer's sublayers during training (regularization against
  overfitting). Unchanged from default in every run here.
- **How to change it:** `policy.dropout`.
- **Source:** Srivastava, Nitish, et al. **"Dropout: A Simple Way to Prevent
  Neural Networks from Overfitting."** JMLR 15 (2014). Generic regularization
  technique, not ACT-paper-specific; ACT's own code just uses a conventional
  0.1 value.

### `kl_weight`
- **Default:** `10.0`
- **What it affects:** the weight on the KL-divergence term in the VAE loss:
  `loss = reconstruction_loss + kl_weight * kl_divergence(latent_pdf ||
  standard_normal)`. Higher values push `z` harder toward a standard normal
  (more regularized, less informative latent, more deterministic-feeling
  policy); lower values let `z` carry more information about demonstration
  style (closer to plain reconstruction, but the latent can "leak" behavior
  that isn't available at inference time since `z=0` then).
- **How to change it:** `policy.kl_weight`. **This repo has ablated it**:
  `kl_weight=10` (default) vs `kl_weight=5` on both the F2 stage-2 and SG2
  stage-3 datasets — see [Recommended changes](#recommended-changes) for why
  the result was inconclusive.
- **Source:** the KL term itself is the standard VAE loss from Kingma,
  Diederik P., and Welling, Max. **"Auto-Encoding Variational Bayes."**
  arXiv:1312.6114 (2013) (lerobot's KL computation cites this directly,
  App. B). The specific `kl_weight` *hyperparameter* — i.e., beta-weighting
  that KL term rather than using it at weight 1 — follows the beta-VAE
  framing from Higgins, Irina, et al. **"β-VAE: Learning Basic Visual
  Concepts with a Constrained Variational Framework."** ICLR 2017. ACT's own
  paper (Table III) reports using a single fixed value, **β = 10**, uniformly
  across all its experiments — it does not report sweeping or needing
  different values per task. This repo's default of `10.0` matches that
  directly; the `5.0` variant tested here is this project's own ablation, not
  something the paper itself motivates.

---

## Optimizer preset

ACT defines its own copy of these three (`get_optimizer_preset()` builds an
`AdamWConfig` from them) rather than relying on the top-level training
config's optimizer block — so they live under `policy:` in your YAML, not
under `train:`.

### `optimizer_lr`
- **Default:** `1e-5`
- **What it affects:** learning rate for every parameter *except* the vision
  backbone.
- **How to change it:** `policy.optimizer_lr`.
- **Source:** AdamW optimizer itself — Loshchilov, Ilya, and Hutter, Frank.
  **"Decoupled Weight Decay Regularization."** arXiv:1711.05101 (2019). The
  `1e-5` value matches the ACT paper's setting.

### `optimizer_lr_backbone`
- **Default:** `1e-5`
- **What it affects:** a **separate** learning rate specifically for the
  ResNet backbone's parameters (`get_optim_params()` splits backbone vs.
  everything-else into two optimizer param groups). The idea (standard in
  detection/DETR-style transfer learning) is that a pretrained CNN backbone
  usually wants a smaller LR than a randomly-initialized transformer head
  trained from scratch on top of it.
- **How to change it:** `policy.optimizer_lr_backbone`. **Currently a
  dead knob in practice**: the code has a standing `TODO(aliberts, rcadene):
  As of now, lr_backbone == lr` and indeed both default to `1e-5` — no run in
  this repo (or lerobot's own defaults) actually differentiates them yet.
- **Source:** differential backbone/head learning rates are DETR's convention
  (Carion et al. 2020); AdamW as above.

### `optimizer_weight_decay`
- **Default:** `1e-4`
- **What it affects:** L2-style weight decay (AdamW's decoupled variant) on
  the non-backbone parameter group.
- **How to change it:** `policy.optimizer_weight_decay`.
- **Source:** Loshchilov & Hutter 2019 (decoupled weight decay).

---

## Adjacent knobs (not ACT-specific)

These live on the generic `TrainPipelineConfig` (`train:` in the YAML), not on
`ACTConfig` — every policy in lerobot shares them. Listed here because every
run in this repo touches them and they materially change how ACT trains, but
they aren't part of the "ACT parameters" list above.

| Key | Default | What it does |
|---|---|---|
| `train.batch_size` | 8 | Samples per gradient step. |
| `train.steps` | 100,000 | Total optimizer steps. Combined with `batch_size` and dataset frame count, this determines **epochs** — see [Recommended changes](#recommended-changes). |
| `train.save_freq` | 10,000 (this repo) | How often a checkpoint is written under `output/checkpoints/`. |
| `train.log_freq` | 200 | How often loss is logged to stdout/wandb. |
| `train.num_workers` | 4 | Dataloader worker processes (video decode is the bottleneck, not CPU-bound Python). |
| `train.seed` | 1000 | Global RNG seed. |
| `policy.device` | `cuda` | Also on `PreTrainedConfig` — `cuda`, `cuda:N`, `cpu`, `mps`. On this box (single RTX 5090) `cuda:N` is meaningless; `accelerate` picks the device. |
| `policy.use_amp` | `False` | Also on `PreTrainedConfig` — enables mixed-precision training. Not tried on any run yet; a plausible throughput win, especially for the slower 4-camera SG2 runs. |

---

## Recommended changes

Grounded in the actual runs done in this repo so far (5 completed baseline/KL
runs + the 6-run chunk-size ablation), not just paper defaults:

1. **Don't compare `chunk_size` settings by raw training loss.** The
   just-completed ablation shows loss going *up* monotonically with
   `chunk_size` on both datasets (SG2 stage-3: 0.026 → 0.029 → 0.034 → 0.037
   for chunk 30/60/100/150; F2 stage-2 similarly increases). That's expected
   and not a regression signal — predicting further into the future is a
   strictly harder reconstruction target, so the loss scale isn't comparable
   across chunk sizes the way it is across, say, a `kl_weight` sweep at fixed
   `chunk_size`. **The only way to actually pick a winner here is closed-loop
   rollout success rate on hardware**, not the `train.log` numbers.

2. **`kl_weight=10` vs `kl_weight=5` was inconclusive by design of the test.**
   Both landed within 0.001 of each other in final loss on both robots. That's
   consistent with the fact that `kl_weight` trades off latent
   regularization vs. reconstruction fidelity — a tradeoff that shows up in
   *closed-loop behavior* (does the policy commit confidently to one mode of
   the demonstrations, or hedge/average across styles?), not in the training
   loss. Re-running more `kl_weight` values won't resolve this; an on-robot
   A/B between the two existing checkpoints will.

3. **F2 stage-2 is overfitting risk, independent of any hyperparameter
   choice.** 53 episodes / 23.4k frames at 100k steps = 34 epochs, roughly
   2.5× the ~12–14 epoch range the larger SG2 datasets (221–297 episodes)
   land in at their own 80–100k-step runs. This is *why* the chunk-size
   ablation for F2 in this repo intentionally uses 40k steps (~13.7 epochs)
   instead of repeating 100k — don't let future F2 experiments default back
   to 100k without deliberately deciding to re-accept that overfitting risk.
   The highest-leverage next step for F2 is **more episodes**, not more
   hyperparameter search.

4. **`use_amp` has never been tried.** SG2 runs are the slow ones (4 cameras,
   ~4.2 step/s vs. F2's ~9.5 step/s) — mixed precision is a plausible free
   throughput win there specifically, worth a short smoke test
   (`--dry-run` first, then a short `--set train.steps=2000` run) to check it
   doesn't destabilize the KL loss term before trusting it on a full run.

5. **`vision_backbone` has never been ablated.** Every run uses `resnet18`,
   which is also the ACT paper's own choice and is unlikely to be the
   bottleneck for these tasks — deprioritize unless a specific failure mode
   (e.g. missing small objects) suggests more backbone capacity would help.

6. **`n_decoder_layers` should probably stay at `1`.** It's pinned there to
   match the original ACT paper's actual (buggy-but-reported) behavior, not
   because 1 layer is architecturally recommended. If exploring beyond it,
   treat it as "a materially different model than the paper," not a bugfix —
   worth its own isolated ablation with its own eval, not a casual bump.

7. **`optimizer_lr_backbone` is currently a no-op relative to `optimizer_lr`**
   (both `1e-5`). If a future run wants to protect the pretrained ImageNet
   backbone from drifting as fast as the from-scratch transformer head
   (standard practice when fine-tuning on a small dataset — relevant
   especially for F2's 53 episodes), this is the knob to separate out, e.g.
   `optimizer_lr_backbone=1e-6` while leaving `optimizer_lr=1e-5`. Untested
   here.
