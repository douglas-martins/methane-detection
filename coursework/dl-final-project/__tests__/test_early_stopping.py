from early_stopping import EarlyStopper


class TestEarlyStopper:
    def test_does_not_stop_while_improving(self):
        stopper = EarlyStopper(patience=2, mode="min")
        assert stopper.step(1.0) is False
        assert stopper.step(0.9) is False
        assert stopper.step(0.8) is False

    def test_stops_after_patience_epochs_without_improvement(self):
        stopper = EarlyStopper(patience=2, mode="min")
        stopper.step(1.0)  # best so far
        assert stopper.step(1.1) is False  # 1 bad epoch
        assert stopper.step(1.2) is True  # 2 bad epochs -- patience exhausted

    def test_a_new_best_resets_the_counter(self):
        stopper = EarlyStopper(patience=2, mode="min")
        stopper.step(1.0)
        stopper.step(1.1)  # 1 bad epoch
        stopper.step(0.5)  # new best -- resets
        assert stopper.step(0.6) is False  # only 1 bad epoch since the reset

    def test_mode_max_treats_higher_as_better(self):
        stopper = EarlyStopper(patience=1, mode="max")
        stopper.step(0.5)  # best so far
        assert stopper.step(0.4) is True  # worse, patience=1 exhausted immediately

    def test_tracks_whether_the_current_step_is_a_new_best(self):
        stopper = EarlyStopper(patience=3, mode="min")
        stopper.step(1.0)
        assert stopper.is_best is True
        stopper.step(1.5)
        assert stopper.is_best is False

    def test_mode_defaults_to_min(self):
        stopper = EarlyStopper(patience=1)
        stopper.step(0.5)  # best so far
        assert stopper.step(0.6) is True  # higher is worse under min-mode default

    def test_initial_state_before_any_step(self):
        stopper = EarlyStopper(patience=2, mode="min")
        assert stopper.counter == 0
        assert stopper.is_best is False

    def test_equal_value_does_not_count_as_improvement_in_min_mode(self):
        stopper = EarlyStopper(patience=1, mode="min")
        stopper.step(1.0)  # best so far
        assert stopper.step(1.0) is True  # tie is not an improvement, patience exhausted

    def test_equal_value_does_not_count_as_improvement_in_max_mode(self):
        stopper = EarlyStopper(patience=1, mode="max")
        stopper.step(1.0)  # best so far
        assert stopper.step(1.0) is True  # tie is not an improvement, patience exhausted
