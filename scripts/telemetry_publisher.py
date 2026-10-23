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


class TelemetryPublisher:
    def __init__(
        self,
        host="127.0.0.1",
        port=8765,
    ):
        if not isinstance(host, str) or not host.strip():
            raise ValueError(
                "Telemetry host must be a non-empty string."
            )

        if (
            isinstance(port, bool)
            or not isinstance(port, int)
            or not 1 <= port <= 65535
        ):
            raise ValueError(
                "Telemetry port must be between 1 and 65535."
            )

        self.address = (
            host,
            port,
        )
        self.socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )
        self.closed = False

    def publish(self, message):
        if self.closed:
            raise RuntimeError(
                "Telemetry publisher is closed."
            )

        if not isinstance(message, dict):
            raise ValueError(
                "Telemetry message must be a dictionary."
            )

        try:
            payload = json.dumps(
                message,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                "Telemetry message must contain "
                "JSON-compatible finite values."
            ) from error

        self.socket.sendto(
            payload,
            self.address,
        )

    def close(self):
        if self.closed:
            return

        self.socket.close()
        self.closed = True
