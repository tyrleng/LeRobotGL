#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Relative-action support for the ACT and Multi-Task DiT policies.

Covers:
  - the shared ``RelativeActionsProcessorStep`` handling of multi-obs-step
    (3-D) state, which is what makes DiT (``n_obs_steps=2``) work at training time;
  - that each policy's processor factory wires ``RelativeActionsProcessorStep``
    (before normalization) and ``AbsoluteActionsProcessorStep`` (after
    unnormalization, paired to the relative step) when ``use_relative_actions=True``.
"""

import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.processor import TransitionKey, batch_to_transition
from lerobot.processor.relative_action_processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
    to_absolute_actions,
)
from lerobot.utils.constants import ACTION, OBS_STATE

ACTION_DIM = 6


def _identity_stats(dim=ACTION_DIM):
    z = np.zeros(dim, dtype=np.float32)
    o = np.ones(dim, dtype=np.float32)
    base = {"mean": z, "std": o, "min": -3 * o, "max": 3 * o, "q01": -2 * o, "q99": 2 * o}
    return {OBS_STATE: dict(base), ACTION: dict(base)}


# --- Shared step: multi-obs-step (3-D) state handling (the DiT-specific path) ---


def test_relative_step_collapses_multistep_state():
    """DiT passes state as (B, n_obs_steps, state_dim); the step must anchor on the current frame."""
    n_obs, horizon = 2, 8
    state = torch.randn(4, n_obs, ACTION_DIM)
    actions = torch.randn(4, horizon, ACTION_DIM)

    step = RelativeActionsProcessorStep(enabled=True)
    out = step(batch_to_transition({ACTION: actions, OBS_STATE: state}))

    # Reference is the most recent obs step (delta=0), broadcast across the chunk.
    expected = actions - state[:, -1].unsqueeze(1)
    torch.testing.assert_close(out[TransitionKey.ACTION], expected)

    # Cached state (used by the postprocessor) is collapsed to the current frame.
    torch.testing.assert_close(step.get_cached_state(), state[:, -1])

    recovered = to_absolute_actions(out[TransitionKey.ACTION], state[:, -1], [True] * ACTION_DIM)
    torch.testing.assert_close(recovered, actions)


def test_relative_step_keeps_single_step_state():
    """Single-obs-step (ACT/Pi0) state stays (B, state_dim) — collapse is a no-op."""
    state = torch.randn(4, ACTION_DIM)
    actions = torch.randn(4, 8, ACTION_DIM)

    step = RelativeActionsProcessorStep(enabled=True)
    out = step(batch_to_transition({ACTION: actions, OBS_STATE: state}))

    torch.testing.assert_close(step.get_cached_state(), state)
    torch.testing.assert_close(out[TransitionKey.ACTION], actions - state.unsqueeze(1))


# --- Config fields ---


def test_act_config_exposes_relative_fields():
    from lerobot.policies.act.configuration_act import ACTConfig

    cfg = ACTConfig(device="cpu")
    assert cfg.use_relative_actions is False
    assert cfg.relative_exclude_joints == ["gripper"]
    assert cfg.action_feature_names is None


# --- ACT processor wiring (no tokenizer -> CI-safe) ---


def _act_processors(use_relative=True):
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.processor_act import make_act_pre_post_processors

    cfg = ACTConfig(use_relative_actions=use_relative, device="cpu")
    cfg.input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(ACTION_DIM,))}
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
    pre, post = make_act_pre_post_processors(cfg, _identity_stats())
    return cfg, pre, post


def test_act_factory_wires_relative_absolute_steps():
    _, pre, post = _act_processors(use_relative=True)
    pre_names = [type(s).__name__ for s in pre.steps]
    post_names = [type(s).__name__ for s in post.steps]

    # Relative runs on raw absolute actions, before normalization.
    assert pre_names.index("RelativeActionsProcessorStep") < pre_names.index("NormalizerProcessorStep")
    # Absolute runs after unnormalization.
    assert post_names.index("UnnormalizerProcessorStep") < post_names.index("AbsoluteActionsProcessorStep")

    rel = next(s for s in pre.steps if isinstance(s, RelativeActionsProcessorStep))
    abs_step = next(s for s in post.steps if isinstance(s, AbsoluteActionsProcessorStep))
    assert rel.enabled and abs_step.enabled
    assert abs_step.relative_step is rel  # paired so the postprocessor reads the cached state
    assert rel.exclude_joints == ["gripper"]


def test_act_factory_relative_roundtrip():
    cfg, pre, post = _act_processors(use_relative=True)
    rel = next(s for s in pre.steps if isinstance(s, RelativeActionsProcessorStep))
    abs_step = next(s for s in post.steps if isinstance(s, AbsoluteActionsProcessorStep))

    state = torch.randn(2, ACTION_DIM)
    actions = torch.randn(2, cfg.chunk_size, ACTION_DIM)

    rel_t = rel(batch_to_transition({ACTION: actions, OBS_STATE: state}))
    assert not torch.allclose(rel_t[TransitionKey.ACTION], actions)

    abs_t = abs_step(batch_to_transition({ACTION: rel_t[TransitionKey.ACTION]}))
    torch.testing.assert_close(abs_t[TransitionKey.ACTION], actions)


def test_act_factory_disabled_is_noop():
    cfg, pre, post = _act_processors(use_relative=False)
    rel = next(s for s in pre.steps if isinstance(s, RelativeActionsProcessorStep))
    abs_step = next(s for s in post.steps if isinstance(s, AbsoluteActionsProcessorStep))
    assert not rel.enabled and not abs_step.enabled

    state = torch.randn(2, ACTION_DIM)
    actions = torch.randn(2, cfg.chunk_size, ACTION_DIM)
    out = rel(batch_to_transition({ACTION: actions, OBS_STATE: state}))
    torch.testing.assert_close(out[TransitionKey.ACTION], actions)


# --- DiT processor wiring (needs transformers; CLIP tokenizer stubbed so it stays offline) ---


def test_dit_config_and_factory_wire_relative_absolute_steps(monkeypatch):
    pytest.importorskip("transformers")
    from unittest.mock import MagicMock

    import lerobot.processor.tokenizer_processor as tokenizer_processor

    # The CLIP tokenizer is irrelevant to relative actions; stub the loader so the
    # factory builds without hitting the HF hub.
    monkeypatch.setattr(tokenizer_processor, "AutoTokenizer", MagicMock())

    from lerobot.policies.multi_task_dit.configuration_multi_task_dit import MultiTaskDiTConfig
    from lerobot.policies.multi_task_dit.processor_multi_task_dit import (
        make_multi_task_dit_pre_post_processors,
    )

    cfg = MultiTaskDiTConfig(use_relative_actions=True, device="cpu")
    assert cfg.relative_exclude_joints == ["gripper"]
    assert cfg.action_feature_names is None
    cfg.input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(ACTION_DIM,))}
    cfg.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}

    pre, post = make_multi_task_dit_pre_post_processors(cfg, _identity_stats())
    pre_names = [type(s).__name__ for s in pre.steps]
    post_names = [type(s).__name__ for s in post.steps]
    assert pre_names.index("RelativeActionsProcessorStep") < pre_names.index("NormalizerProcessorStep")
    assert post_names.index("UnnormalizerProcessorStep") < post_names.index("AbsoluteActionsProcessorStep")

    rel = next(s for s in pre.steps if isinstance(s, RelativeActionsProcessorStep))
    abs_step = next(s for s in post.steps if isinstance(s, AbsoluteActionsProcessorStep))
    assert rel.enabled and abs_step.enabled and abs_step.relative_step is rel

    # Training-shaped batch: state (B, n_obs_steps, D), action chunk (B, horizon, D).
    state = torch.randn(2, cfg.n_obs_steps, ACTION_DIM)
    actions = torch.randn(2, cfg.horizon, ACTION_DIM)
    rel_t = rel(batch_to_transition({ACTION: actions, OBS_STATE: state}))
    abs_t = abs_step(batch_to_transition({ACTION: rel_t[TransitionKey.ACTION]}))
    torch.testing.assert_close(abs_t[TransitionKey.ACTION], actions)
