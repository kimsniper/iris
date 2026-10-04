def balancing_reward(pitch, pitch_rate, wheel_velocity, action, previous_action):
    return 1.0 - 8.0*pitch*pitch - 0.2*pitch_rate*pitch_rate - 0.02*wheel_velocity*wheel_velocity - 0.01*action*action - 0.02*(action-previous_action)**2
