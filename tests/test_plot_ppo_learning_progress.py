# Copyright 2026 Mezael Docoy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import pytest

from scripts.replay_ppo_learning import (
    checkpoint_step,
    find_checkpoints,
)


def select_checkpoints(directory, max_steps):
    if max_steps <= 0:
        raise ValueError(
            "Maximum training steps must be positive."
        )

    return [
        (
            checkpoint_step(path),
            path,
        )
        for path in find_checkpoints(directory)
        if checkpoint_step(path) <= max_steps
    ]


def test_learning_progress_checkpoint_selection(
    tmp_path,
):
    for name in (
        "checkpoint_0_steps.zip",
        "checkpoint_5000_steps.zip",
        "checkpoint_10000_steps.zip",
        "checkpoint_wrong_steps.zip",
        "best_model.zip",
        "final_model.zip",
    ):
        (tmp_path / name).touch()

    selected = select_checkpoints(
        tmp_path,
        5000,
    )

    assert [
        (step, path.name)
        for step, path in selected
    ] == [
        (0, "checkpoint_0_steps.zip"),
        (5000, "checkpoint_5000_steps.zip"),
    ]


@pytest.mark.parametrize(
    "max_steps",
    [
        0,
        -1,
    ],
)
def test_learning_progress_rejects_invalid_max_steps(
    max_steps,
):
    with pytest.raises(
        ValueError,
        match="must be positive",
    ):
        select_checkpoints(
            Path("."),
            max_steps,
        )
