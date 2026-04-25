def test_t_star():
    assert abs(552.0 - engine.t_star_theorem()["C_P1_Tstar"]) < 1e-6

def test_tau_matches():
    assert len(engine.tau_dist) == 7
