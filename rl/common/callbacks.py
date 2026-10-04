from stable_baselines3.common.callbacks import CheckpointCallback

def checkpoint_callback(save_freq, save_path, name_prefix):
    return CheckpointCallback(save_freq=save_freq, save_path=save_path, name_prefix=name_prefix)
