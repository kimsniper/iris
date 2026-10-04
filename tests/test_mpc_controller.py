from controllers.mpc.mpc_controller import MPCController

def test_mpc_returns_bounded_pair():
    u=MPCController().compute([0.1,0,0,0,0,9.81,0],0.02)
    assert len(u)==2 and max(abs(u))<=2.0
