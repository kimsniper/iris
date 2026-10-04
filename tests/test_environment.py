from rl.environments.balancing_env import BalancingEnv

def test_environment_reset_and_step():
    env=BalancingEnv(max_seconds=0.1); obs,_=env.reset(seed=0); assert obs.shape==(7,)
    obs,reward,terminated,truncated,info=env.step([0.0]); assert obs.shape==(7,) and isinstance(reward,float)


def test_reset_applies_requested_pitch_range():
    env = BalancingEnv(max_initial_pitch=0.08)
    env.reset(seed=0)

    assert abs(env._truth_pitch()) <= 0.08 + 1e-6
