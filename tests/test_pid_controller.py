from controllers.pid.pid_controller import PIDController

def test_pid_opposes_positive_pitch():
    u=PIDController(20,0,1).compute([0.1,0,0,0,0,9.81,0],0.02)
    assert u[0] < 0 and u[0] == u[1]
