from pr_curve_plots import curve_points_from_artifact


class TestCurvePointsFromArtifact:
    def test_converts_json_lists_to_float_tuples(self):
        # JSON round-trips a (recall, precision) tuple as a two-element list --
        # plotting code (zip(*points)) needs real tuples back.
        artifact = {"precision_recall_curve": [[1.0, 0.5], [0.0, 0.0]]}

        points = curve_points_from_artifact(artifact)

        assert points == [(0.0, 0.0), (1.0, 0.5)]
        assert all(isinstance(point, tuple) for point in points)

    def test_sorts_ascending_by_recall(self):
        # Logged in threshold order (high threshold / low recall first) --
        # a plot line needs ascending recall, matching
        # average_precision_from_sweep's own convention.
        artifact = {"precision_recall_curve": [[1.0, 0.5], [0.5, 0.5], [0.0, 0.0]]}

        points = curve_points_from_artifact(artifact)

        assert points == [(0.0, 0.0), (0.5, 0.5), (1.0, 0.5)]
