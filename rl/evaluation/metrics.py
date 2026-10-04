import numpy as np

def summarize(pitches, actions, survived):
    return {'survived': bool(survived), 'rms_pitch': float(np.sqrt(np.mean(np.square(pitches)))), 'peak_abs_pitch': float(np.max(np.abs(pitches))), 'rms_action': float(np.sqrt(np.mean(np.square(actions))))}
