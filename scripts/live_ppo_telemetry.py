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

import argparse
import json
import math
import socket

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Display live IRIS PPO replay telemetry."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=10.0,
    )
    return parser.parse_args()


class TelemetryViewer:
    def __init__(
        self,
        host,
        port,
        window_seconds,
    ):
        if not isinstance(host, str) or not host.strip():
            raise ValueError(
                "Telemetry host must be a non-empty string."
            )

        if (
            isinstance(port, bool)
            or not isinstance(port, int)
            or not 0 <= port <= 65535
        ):
            raise ValueError(
                "Telemetry port must be between 0 and 65535."
            )

        self.window_seconds = float(
            window_seconds
        )

        if (
            not math.isfinite(self.window_seconds)
            or self.window_seconds <= 0.0
        ):
            raise ValueError(
                "Telemetry window must be finite "
                "and greater than zero."
            )

        self.socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )
        self.socket.bind(
            (host, int(port))
        )
        self.socket.setblocking(False)

        self.title = (
            "Waiting for PPO replay telemetry"
        )
        self.clear_data()

        self.figure, self.axes = plt.subplots(
            2,
            2,
            figsize=(11, 7),
            sharex=True,
        )

        self.truth_line, = (
            self.axes[0, 0].plot(
                [],
                [],
                label="Truth pitch",
            )
        )
        self.estimated_line, = (
            self.axes[0, 0].plot(
                [],
                [],
                label="Estimated pitch",
            )
        )
        self.left_velocity_line, = (
            self.axes[0, 1].plot(
                [],
                [],
                label="Left wheel",
            )
        )
        self.right_velocity_line, = (
            self.axes[0, 1].plot(
                [],
                [],
                label="Right wheel",
            )
        )
        self.position_line, = (
            self.axes[1, 0].plot(
                [],
                [],
                label="Average position",
            )
        )
        self.action_line, = (
            self.axes[1, 1].plot(
                [],
                [],
                label="Normalized action",
            )
        )
        self.torque_line, = (
            self.axes[1, 1].plot(
                [],
                [],
                label="Torque (N m)",
            )
        )

        self.axes[0, 0].set_title(
            "Body pitch"
        )
        self.axes[0, 0].set_ylabel(
            "Pitch (rad)"
        )
        self.axes[0, 0].legend()

        self.axes[0, 1].set_title(
            "Wheel velocity"
        )
        self.axes[0, 1].set_ylabel(
            "Velocity (rad/s)"
        )
        self.axes[0, 1].legend()

        self.axes[1, 0].set_title(
            "Relative wheel position"
        )
        self.axes[1, 0].set_xlabel(
            "Time (s)"
        )
        self.axes[1, 0].set_ylabel(
            "Position (rad)"
        )

        self.axes[1, 1].set_title(
            "PPO action and torque"
        )
        self.axes[1, 1].set_xlabel(
            "Time (s)"
        )
        self.axes[1, 1].set_ylabel(
            "Action / torque"
        )
        self.axes[1, 1].legend()

        for axis in self.axes.flat:
            axis.axhline(
                0.0,
                color="black",
                linewidth=0.7,
            )
            axis.grid(
                True,
                alpha=0.3,
            )

        self.figure.tight_layout(
            rect=[0.0, 0.0, 1.0, 0.95]
        )

    def clear_data(self):
        self.times = []
        self.truth_pitch = []
        self.estimated_pitch = []
        self.left_velocity = []
        self.right_velocity = []
        self.average_position = []
        self.normalized_action = []
        self.motor_torque = []

    def process_message(self, message):
        if not isinstance(message, dict):
            return False

        message_type = message.get("type")

        if message_type == "episode_start":
            required = (
                "checkpoint",
                "episode",
                "seed",
            )

            if not all(
                key in message
                for key in required
            ):
                return False

            self.clear_data()
            self.title = (
                f"{message['checkpoint']} | "
                f"episode {message['episode']} | "
                f"seed {message['seed']}"
            )
            return True

        if message_type == "episode_end":
            required = (
                "status",
                "seconds",
                "return",
            )

            if not all(
                key in message
                for key in required
            ):
                return False

            self.title = (
                f"{message['status']} | "
                f"{float(message['seconds']):.2f} s | "
                f"return={float(message['return']):.2f}"
            )
            return True

        if message_type != "sample":
            return False

        fields = (
            "time_s",
            "truth_pitch_rad",
            "estimated_pitch_rad",
            "left_wheel_velocity_rad_s",
            "right_wheel_velocity_rad_s",
            "average_wheel_position_rad",
            "normalized_action",
            "motor_torque_nm",
        )

        if not all(
            field in message
            for field in fields
        ):
            return False

        try:
            values = [
                float(message[field])
                for field in fields
            ]
        except (
            TypeError,
            ValueError,
        ):
            return False

        if not all(
            math.isfinite(value)
            for value in values
        ):
            return False

        (
            time_s,
            truth_pitch,
            estimated_pitch,
            left_velocity,
            right_velocity,
            average_position,
            normalized_action,
            motor_torque,
        ) = values

        self.times.append(time_s)
        self.truth_pitch.append(truth_pitch)
        self.estimated_pitch.append(
            estimated_pitch
        )
        self.left_velocity.append(
            left_velocity
        )
        self.right_velocity.append(
            right_velocity
        )
        self.average_position.append(
            average_position
        )
        self.normalized_action.append(
            normalized_action
        )
        self.motor_torque.append(
            motor_torque
        )

        minimum = max(
            0.0,
            time_s - self.window_seconds,
        )

        while (
            self.times
            and self.times[0] < minimum
        ):
            self.times.pop(0)
            self.truth_pitch.pop(0)
            self.estimated_pitch.pop(0)
            self.left_velocity.pop(0)
            self.right_velocity.pop(0)
            self.average_position.pop(0)
            self.normalized_action.pop(0)
            self.motor_torque.pop(0)

        return True

    def receive(self):
        while True:
            try:
                payload, _ = (
                    self.socket.recvfrom(65535)
                )
            except BlockingIOError:
                break

            try:
                message = json.loads(
                    payload.decode("utf-8")
                )
            except (
                UnicodeDecodeError,
                json.JSONDecodeError,
            ):
                continue

            self.process_message(message)

    def update(self, _):
        self.receive()
        self.figure.suptitle(self.title)

        if not self.times:
            return ()

        latest = self.times[-1]
        minimum = max(
            0.0,
            latest - self.window_seconds,
        )

        start = 0

        for index, time_s in enumerate(
            self.times
        ):
            if time_s >= minimum:
                start = index
                break

        visible_time = self.times[start:]

        line_data = [
            (
                self.truth_line,
                self.truth_pitch[start:],
            ),
            (
                self.estimated_line,
                self.estimated_pitch[start:],
            ),
            (
                self.left_velocity_line,
                self.left_velocity[start:],
            ),
            (
                self.right_velocity_line,
                self.right_velocity[start:],
            ),
            (
                self.position_line,
                self.average_position[start:],
            ),
            (
                self.action_line,
                self.normalized_action[start:],
            ),
            (
                self.torque_line,
                self.motor_torque[start:],
            ),
        ]

        for line, values in line_data:
            line.set_data(
                visible_time,
                values,
            )

        for axis in self.axes.flat:
            axis.set_xlim(
                minimum,
                max(
                    self.window_seconds,
                    latest,
                ),
            )
            axis.relim()
            axis.autoscale_view(
                scalex=False,
                scaley=True,
            )

        return tuple(
            line
            for line, _ in line_data
        )

    def run(self):
        self.animation = FuncAnimation(
            self.figure,
            self.update,
            interval=100,
            blit=False,
            cache_frame_data=False,
        )
        plt.show()


def main():
    args = parse_arguments()

    viewer = TelemetryViewer(
        host=args.host,
        port=args.port,
        window_seconds=args.window_seconds,
    )

    print(
        f"Listening for telemetry on "
        f"{args.host}:{args.port}"
    )

    try:
        viewer.run()
    finally:
        viewer.socket.close()


if __name__ == "__main__":
    main()
