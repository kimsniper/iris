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

import json
import socket

import numpy as np
import pytest

from scripts.telemetry_publisher import TelemetryPublisher


@pytest.mark.parametrize(
    "host",
    [
        "",
        "   ",
        None,
        123,
    ],
)
def test_publisher_rejects_invalid_host(host):
    with pytest.raises(
        ValueError,
        match="non-empty string",
    ):
        TelemetryPublisher(
            host=host,
            port=8765,
        )


@pytest.mark.parametrize(
    "port",
    [
        0,
        -1,
        65536,
        True,
        8765.0,
        "8765",
    ],
)
def test_publisher_rejects_invalid_port(port):
    with pytest.raises(
        ValueError,
        match="between 1 and 65535",
    ):
        TelemetryPublisher(
            port=port
        )


def test_publisher_sends_compact_json():
    receiver = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM,
    )
    receiver.bind(
        ("127.0.0.1", 0)
    )
    receiver.settimeout(1.0)

    host, port = receiver.getsockname()
    publisher = TelemetryPublisher(
        host=host,
        port=port,
    )

    try:
        message = {
            "type": "sample",
            "time_s": 0.02,
            "reward": 1.0,
        }

        publisher.publish(message)

        payload, _ = receiver.recvfrom(
            65535
        )

        assert json.loads(
            payload.decode("utf-8")
        ) == message
        assert b" " not in payload
    finally:
        publisher.close()
        receiver.close()


@pytest.mark.parametrize(
    "message",
    [
        [],
        "sample",
        None,
    ],
)
def test_publisher_rejects_non_dictionary_message(
    message,
):
    publisher = TelemetryPublisher()

    try:
        with pytest.raises(
            ValueError,
            match="must be a dictionary",
        ):
            publisher.publish(message)
    finally:
        publisher.close()


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
        object(),
    ],
)
def test_publisher_rejects_invalid_message_value(
    invalid_value,
):
    publisher = TelemetryPublisher()

    try:
        with pytest.raises(
            ValueError,
            match="JSON-compatible finite values",
        ):
            publisher.publish(
                {
                    "value": invalid_value,
                }
            )
    finally:
        publisher.close()


def test_publisher_rejects_publish_after_close():
    publisher = TelemetryPublisher()
    publisher.close()

    with pytest.raises(
        RuntimeError,
        match="publisher is closed",
    ):
        publisher.publish(
            {
                "type": "sample",
            }
        )


def test_publisher_close_is_idempotent():
    publisher = TelemetryPublisher()

    publisher.close()
    publisher.close()

    assert publisher.closed
