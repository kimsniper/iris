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


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("checkpoint_0_steps.zip", 0),
        ("checkpoint_5000_steps.zip", 5000),
        ("checkpoint_150000_steps.zip", 150000),
        ("best_model.zip", 10**18),
        ("final_model.zip", 10**18 + 1),
        ("other.zip", 10**18 + 2),
    ],
)
def test_checkpoint_step(name, expected):
    assert checkpoint_step(
        Path(name)
    ) == expected


def test_find_checkpoints_orders_models(tmp_path):
    names = [
        "checkpoint_10000_steps.zip",
        "final_model.zip",
        "checkpoint_0_steps.zip",
        "best_model.zip",
        "checkpoint_5000_steps.zip",
    ]

    for name in names:
        (tmp_path / name).touch()

    checkpoints = find_checkpoints(
        tmp_path
    )

    assert [
        path.name
        for path in checkpoints
    ] == [
        "checkpoint_0_steps.zip",
        "checkpoint_5000_steps.zip",
        "checkpoint_10000_steps.zip",
        "best_model.zip",
        "final_model.zip",
    ]


def test_find_checkpoints_ignores_unrelated_files(
    tmp_path,
):
    (tmp_path / "checkpoint_wrong_steps.zip").touch()
    (tmp_path / "notes.txt").touch()
    (tmp_path / "checkpoint_50_steps.zip").touch()

    checkpoints = find_checkpoints(
        tmp_path
    )

    assert [
        path.name
        for path in checkpoints
    ] == [
        "checkpoint_50_steps.zip",
    ]


def test_find_checkpoints_requires_matching_models(
    tmp_path,
):
    with pytest.raises(
        FileNotFoundError,
        match="No PPO checkpoints found",
    ):
        find_checkpoints(tmp_path)
