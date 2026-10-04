from controllers.ppo.observation_builder import build_observation

def test_observation_shape(): assert build_observation(0,0,0,0,0,9.81,0).shape == (7,)
