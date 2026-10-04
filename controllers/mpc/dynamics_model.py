import numpy as np

def linear_model(dt: float):
    # Initial reduced model: [pitch, pitch_rate, position, velocity].
    # Replace coefficients after system identification or analytical derivation.
    A = np.array([[1.0, dt, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, dt], [0.0, 0.0, 0.0, 1.0]])
    B = np.array([[0.0], [dt], [0.0], [dt]])
    return A, B
