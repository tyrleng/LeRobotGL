# Relative Actions for ACT and Multi-Task DiT

This document describes the addition of **relative action** support to the **ACT**
and **Multi-Task DiT** policies, why it was added, how to compute the statistics it
needs, and the exact CLI commands (with all relevant options) for training each
policy.

For the conceptual background on absolute vs. relative vs. delta actions, see
[`action_representations.mdx`](./source/action_representations.mdx).

---

## 1. What changed and why

### Background

A **relative action** is a predicted joint target expressed as an offset from the
robot's *current* state rather than as an absolute position:

```
relative = absolute - current_state          # training (preprocessing)
absolute = relative + current_state           # inference (postprocessing)
```

Training on relative actions centers the targets around zero, which is easier to
normalize and yields more stable training. Because every predicted chunk references
the same current state, there is no error accumulation across chunks.

Relative actions were already supported by the pi family (`pi0`, `pi0.5`,
`pi0_fast`). **ACT and Multi-Task DiT did not support them**, so they could not be
trained on relative targets. This change brings both policies to parity.

### Changes made

| Area | File | Change |
|------|------|--------|
| Shared processor | `src/lerobot/processor/relative_action_processor.py` | `RelativeActionsProcessorStep` now collapses a multi-observation-step state `(B, n_obs_steps, state_dim)` to its **most recent frame** before computing the offset. This is required by DiT (`n_obs_steps=2`); it is a no-op for single-step policies (ACT, pi0). |
| ACT config | `src/lerobot/policies/act/configuration_act.py` | Added `use_relative_actions`, `relative_exclude_joints`, `action_feature_names`. |
| ACT processor | `src/lerobot/policies/act/processor_act.py` | Wired `RelativeActionsProcessorStep` (before normalization) and `AbsoluteActionsProcessorStep` (after unnormalization). |
| DiT config | `src/lerobot/policies/multi_task_dit/configuration_multi_task_dit.py` | Added the same three fields. |
| DiT processor | `src/lerobot/policies/multi_task_dit/processor_multi_task_dit.py` | Same processor wiring as ACT. |

The pipeline order matches the pi family exactly:

```
Training :  raw absolute action → RelativeActionsProcessorStep → normalize → model
Inference:  model output → unnormalize → AbsoluteActionsProcessorStep → robot
```

`AbsoluteActionsProcessorStep` reads the current state cached by its paired
`RelativeActionsProcessorStep`; the two are wired together automatically by the
policy factory (including after loading a checkpoint).

### Why the DiT-specific fix was needed

DiT uses `n_obs_steps = 2`, so at training time its batches carry a temporal
dimension on the state: `observation.state` has shape `(B, n_obs_steps, state_dim)`,
while the action chunk has shape `(B, horizon, action_dim)`. The relative processor
assumed a 2-D state and could not broadcast the 3-D state across the chunk. The fix
selects the **current** observation frame (the most recent step, `state[:, -1]`) as
the single reference point, which is the correct anchor for the whole chunk. ACT
keeps `n_obs_steps = 1`, so its state is always 2-D and the fix never triggers.

### Configuration fields

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `use_relative_actions` | `bool` | `false` | Master switch. When `true`, actions are converted to relative offsets for training and back to absolute at inference. |
| `relative_exclude_joints` | `list[str]` | `["gripper"]` | Joint names that stay **absolute** (matched case-insensitively as substrings of the dataset's action feature names). Gripper commands are typically binary and don't benefit from relative encoding. |
| `action_feature_names` | `list[str] \| None` | `null` | **Auto-populated** from the dataset metadata by the policy factory. Not a user-facing CLI flag — used to build the exclude-joints mask. |

---

## 2. Computing relative-action statistics (required before training)

### Why it is needed

LeRobot normalizes actions before they reach the model. The relative conversion runs
**before** normalization, so the normalizer (and the saved `stats.json`) must describe
the distribution of **relative** actions — *not* absolute ones.

If you train with `use_relative_actions=true` but normalize with the dataset's default
**absolute** action statistics, the offsets (which cluster near zero with a much
smaller spread) will be mis-scaled. The model then sees badly normalized targets and
training quality degrades. Relative-action stats therefore need to be computed once and
stored in the dataset before training.

### What the stats are used for

The computed `mean`, `std`, `min`, `max`, and quantiles (`q01`…`q99`) of the relative
actions are written to the dataset's `stats.json` under the `action` key. At training and
inference, `NormalizerProcessorStep` / `UnnormalizerProcessorStep` use them to normalize
the relative offsets to a well-behaved range:

- ACT uses **`MEAN_STD`** normalization for actions → uses `mean`/`std`.
- DiT uses **`MIN_MAX`** normalization for actions → uses `min`/`max`.

(The pi family uses `MEAN_STD` or `QUANTILES`.) The stats computation emits all of these,
so the same recomputed dataset works for any of these policies.

### How it works

The stats are computed over **all valid action chunks** within each episode:
for every chunk it subtracts the chunk's starting state (`action[t+k] - state[t]`),
keeping the `relative_exclude_joints` dimensions absolute, then aggregates per-dimension
statistics. The `chunk_size` you pass should match the action chunk length the policy
trains on:

- **ACT** → use the policy's `chunk_size` (default `100`).
- **DiT** → use the policy's `horizon` (default `32`).

### Command

```bash
lerobot-edit-dataset \
  --repo_id=${HF_USER}/my_task \
  --operation.type=recompute_stats \
  --operation.relative_action=true \
  --operation.chunk_size=100 \
  --operation.relative_exclude_joints="['gripper']" \
  --operation.num_workers=4 \
  --new_repo_id=${HF_USER}/my_task_relative
```

`recompute_stats` options:

| Option | Default | Meaning |
|--------|---------|---------|
| `--operation.relative_action` | `false` | Compute action stats in relative space. |
| `--operation.chunk_size` | `50` | Chunk length for relative-stats sampling. Set to ACT `chunk_size` or DiT `horizon`. |
| `--operation.relative_exclude_joints` | `null` (→ `['gripper']`) | Joints kept absolute. **Must match** the value you train with. |
| `--operation.num_workers` | `0` | Parallel threads for stats computation. |
| `--operation.skip_image_video` | `true` | Only recompute numeric features (action/state). |
| `--operation.overwrite` | `false` | Recompute in-place (creates a backup). |
| `--new_repo_id` | — | Write the recomputed dataset to a new id (recommended; keeps the original intact). |
| `--push_to_hub` | `false` | Push the recomputed dataset to the Hub. |

> Keep `relative_exclude_joints` identical between this command and training. A mismatch
> means the stored stats and the runtime conversion disagree about which dimensions are
> relative.

---

## 3. Training ACT with relative actions

ACT enforces `n_obs_steps = 1`.

### Minimal command

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/my_task_relative \
  --policy.type=act \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints='["gripper"]' \
  --policy.device=cuda \
  --output_dir=outputs/train/act_relative \
  --job_name=act_relative \
  --batch_size=8 \
  --steps=100000 \
  --policy.repo_id=${HF_USER}/act_relative
```

### Full set of relevant options

```bash
lerobot-train \
  # --- data / run ---
  --dataset.repo_id=${HF_USER}/my_task_relative \
  --output_dir=outputs/train/act_relative \
  --job_name=act_relative \
  --batch_size=8 \
  --steps=100000 \
  --num_workers=4 \
  --save_freq=20000 \
  --eval_freq=20000 \
  --seed=1000 \
  --wandb.enable=true \
  # --- policy selection / relative actions ---
  --policy.type=act \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints='["gripper"]' \
  --policy.device=cuda \
  --policy.push_to_hub=true \
  --policy.repo_id=${HF_USER}/act_relative \
  # --- ACT chunking ---
  --policy.n_obs_steps=1 \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --policy.temporal_ensemble_coeff=null \
  # --- ACT architecture ---
  --policy.vision_backbone=resnet18 \
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
  --policy.dim_model=512 \
  --policy.n_heads=8 \
  --policy.dim_feedforward=3200 \
  --policy.n_encoder_layers=4 \
  --policy.n_decoder_layers=1 \
  --policy.pre_norm=false \
  --policy.feedforward_activation=relu \
  --policy.dropout=0.1 \
  # --- VAE ---
  --policy.use_vae=true \
  --policy.latent_dim=32 \
  --policy.n_vae_encoder_layers=4 \
  --policy.kl_weight=10.0 \
  # --- optimizer ---
  --policy.optimizer_lr=1e-5 \
  --policy.optimizer_weight_decay=1e-4 \
  --policy.optimizer_lr_backbone=1e-5
```

> If using temporal ensembling (`--policy.temporal_ensemble_coeff=0.01`), set
> `--policy.n_action_steps=1`.

---

## 4. Training Multi-Task DiT with relative actions

DiT uses `n_obs_steps = 2` and supports two objectives (`diffusion`, `flow_matching`).
It builds CLIP vision/text encoders, so install its extra: `uv sync --extra multi_task_dit`.

### Minimal command

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/my_task_relative \
  --policy.type=multi_task_dit \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints='["gripper"]' \
  --policy.device=cuda \
  --output_dir=outputs/train/dit_relative \
  --job_name=dit_relative \
  --batch_size=64 \
  --steps=200000 \
  --policy.repo_id=${HF_USER}/dit_relative
```

> When computing stats for DiT, set `--operation.chunk_size` to the policy's
> `horizon` (default `32`), not ACT's `chunk_size`.

### Full set of relevant options

```bash
lerobot-train \
  # --- data / run ---
  --dataset.repo_id=${HF_USER}/my_task_relative \
  --output_dir=outputs/train/dit_relative \
  --job_name=dit_relative \
  --batch_size=64 \
  --steps=200000 \
  --num_workers=4 \
  --save_freq=20000 \
  --wandb.enable=true \
  # --- policy selection / relative actions ---
  --policy.type=multi_task_dit \
  --policy.use_relative_actions=true \
  --policy.relative_exclude_joints='["gripper"]' \
  --policy.device=cuda \
  --policy.push_to_hub=true \
  --policy.repo_id=${HF_USER}/dit_relative \
  # --- horizon / chunking ---
  --policy.n_obs_steps=2 \
  --policy.horizon=32 \
  --policy.n_action_steps=24 \
  # --- objective: "diffusion" or "flow_matching" ---
  --policy.objective=diffusion \
  # --- diffusion-specific (objective=diffusion) ---
  --policy.noise_scheduler_type=DDPM \
  --policy.num_train_timesteps=100 \
  --policy.beta_schedule=squaredcos_cap_v2 \
  --policy.beta_start=0.0001 \
  --policy.beta_end=0.02 \
  --policy.prediction_type=epsilon \
  --policy.clip_sample=true \
  --policy.clip_sample_range=1.0 \
  --policy.num_inference_steps=null \
  # --- flow-matching-specific (objective=flow_matching) ---
  --policy.num_integration_steps=100 \
  --policy.integration_method=euler \
  --policy.timestep_sampling_strategy=beta \
  --policy.sigma_min=0.0 \
  # --- transformer ---
  --policy.hidden_dim=512 \
  --policy.num_layers=6 \
  --policy.num_heads=8 \
  --policy.dropout=0.1 \
  --policy.use_rope=true \
  --policy.timestep_embed_dim=256 \
  # --- vision / text encoders (CLIP) ---
  --policy.vision_encoder_name=openai/clip-vit-base-patch16 \
  --policy.text_encoder_name=openai/clip-vit-base-patch16 \
  --policy.image_crop_shape="(224, 224)" \
  --policy.image_crop_is_random=true \
  --policy.tokenizer_max_length=77 \
  # --- optimizer / scheduler ---
  --policy.optimizer_lr=2e-5 \
  --policy.optimizer_weight_decay=0.0 \
  --policy.scheduler_name=cosine \
  --policy.scheduler_warmup_steps=0
```

---

## 5. Inference

Once trained, relative actions are converted back to absolute automatically by the
policy's postprocessor — no extra flags are needed for the standard robot control loop
(e.g. `lerobot-record --policy.path=...`).

For the rollout/eval engines:

- **`--inference.type=rtc`** (real-time chunking) fully supports relative actions and is
  the recommended mode for relative-action policies.
- **`--inference.type=sync`** raises `NotImplementedError` for *any* relative-action
  policy (this is a global limitation shared with the pi family, not specific to
  ACT/DiT). Use RTC instead.

---

## 6. End-to-end checklist

1. **Recompute stats** in relative space (§2), with `chunk_size` = ACT `chunk_size` or
   DiT `horizon`, and the same `relative_exclude_joints` you will train with.
2. **Train** with `--policy.use_relative_actions=true` on the recomputed dataset (§3/§4).
3. **Run inference** with `--inference.type=rtc` or the standard control loop (§5).
