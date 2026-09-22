import re
from pathlib import Path

import pytest
from paper_reference import REFERENCES, Reference

STARCOP_KEYS = {"strong_f1", "weak_f1", "tile_fpr", "auprc"}
HEREC_KEYS = {"recall", "precision", "f1", "f1_strong"}


class TestStarcopTable2:
    def test_the_hyperstarcop_mag1c_plus_rgb_row(self):
        row = REFERENCES["starcop_mag1c_rgb"]

        assert row.metrics == {
            "strong_f1": (81.96, 3.71),
            "weak_f1": (43.42, 5.72),
            "tile_fpr": (43.66, 7.36),
            "auprc": (51.99, 2.76),
        }

    def test_the_hyperstarcop_mag1c_only_row(self):
        assert REFERENCES["starcop_mag1c_only"].metrics == {
            "strong_f1": (74.15, 6.10),
            "weak_f1": (47.57, 4.17),
            "tile_fpr": (52.11, 10.98),
            "auprc": (49.41, 5.49),
        }

    def test_the_matched_filter_baseline_has_no_spread_and_no_auprc(self):
        # One deterministic run of mag1c + morphology: no std, and the paper prints N/A for AUPRC.
        assert REFERENCES["starcop_baseline"].metrics == {
            "strong_f1": (67.45, None),
            "weak_f1": (39.95, None),
            "tile_fpr": (75.43, None),
            "auprc": (None, None),
        }

    def test_the_text_and_the_table_disagree_on_the_fpr_and_that_is_recorded(self):
        # The paper's prose says 43.79, its Table 2 says 43.66.
        caveats = " ".join(REFERENCES["starcop_mag1c_rgb"].caveats)

        assert "43.79" in caveats
        assert "43.66" in caveats


class TestHerecTableI:
    def test_the_linknet_mag1c_sas_row(self):
        assert REFERENCES["herec_linknet_mag1c_sas"].metrics == {
            "recall": (51.11, 7.2),
            "precision": (40.43, 6.4),
            "f1": (44.44, 3.9),
            "f1_strong": (60.37, 5.1),
        }

    def test_the_unet_mag1c_sas_row(self):
        assert REFERENCES["herec_unet_mag1c_sas"].metrics == {
            "recall": (56.41, 7.0),
            "precision": (34.62, 7.4),
            "f1": (42.54, 6.7),
            "f1_strong": (61.38, 7.7),
        }

    def test_the_original_mag1c_baseline_has_no_spread(self):
        assert REFERENCES["herec_mag1c_column_wise"].metrics == {
            "recall": (58.42, None),
            "precision": (30.57, None),
            "f1": (40.14, None),
            "f1_strong": (67.50, None),
        }

    def test_the_comparability_caveats_are_recorded_with_every_row(self):
        for name in ("herec_linknet_mag1c_sas", "herec_unet_mag1c_sas"):
            caveats = " ".join(REFERENCES[name].caveats)
            assert "label size" in caveats  # weak/strong split differs from STARCOP's kg/h
            assert "assumption" in caveats  # what 'F1' pools is not stated
            assert "Mag1c-SAS" in caveats  # a different input product than our mag1c

    def test_herec_rows_cite_page_8_of_the_verified_2026_pdf(self):
        # The 2025 reference note guessed p. 5-7; the 2026 PDF (Table I) is actually on p. 8.
        for name in ("herec_mag1c_column_wise", "herec_unet_mag1c_sas", "herec_linknet_mag1c_sas"):
            assert "p. 8" in REFERENCES[name].citation, name

    def test_the_run_count_and_threshold_are_now_verified_not_assumed(self):
        # 2026-09-21: read directly from the PDF (Table I caption + Methodology, p. 6 and 8),
        # instead of the 2025 reference note, which could not confirm either fact.
        for name in ("herec_unet_mag1c_sas", "herec_linknet_mag1c_sas"):
            caveats = " ".join(REFERENCES[name].caveats)
            assert "5 repeated training runs" in caveats
            assert "threshold is 0.5" in caveats
            assert "not stated" not in caveats.lower()

    def test_the_f1_aggregation_caveat_still_flags_an_assumption(self):
        # The paper calls this column F1 "for all plumes" (p. 6, 8) but never says whether pixels
        # are pooled or averaged per scene -- that part is still an assumption, verified or not.
        for name in ("herec_unet_mag1c_sas", "herec_linknet_mag1c_sas"):
            caveats = " ".join(REFERENCES[name].caveats)
            assert "for all plumes" in caveats
            assert "pooled" in caveats


class TestEveryRow:
    def test_rows_are_reference_records_with_the_expected_metric_keys(self):
        for name, row in REFERENCES.items():
            assert isinstance(row, Reference)
            expected = STARCOP_KEYS if name.startswith("starcop") else HEREC_KEYS
            assert set(row.metrics) == expected, name

    def test_every_row_cites_its_table_and_source(self):
        for name, row in REFERENCES.items():
            assert re.search(r"Table (2|I)\b", row.citation), name
            assert row.source in ("starcop", "herec"), name
            assert row.label, name

    def test_starcop_rows_cite_a_page(self):
        for name, row in REFERENCES.items():
            if name.startswith("starcop"):
                assert "page 10" in row.citation, name


def _repo_reference_file():
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "internal-docs" / "plans" / "paper_reference_metrics.md"
        if candidate.is_file():
            return candidate
    return None


@pytest.mark.skipif(_repo_reference_file() is None, reason="repo reference table not available")
class TestCrossCheckAgainstTheRepoReferenceTable:
    # `internal-docs/plans/paper_reference_metrics.md` was entered by hand, independently, with a
    # machine-readable YAML block in fractions (0-1); ours are percentages as printed.
    def _yaml_values(self, key):
        text = _repo_reference_file().read_text()
        block = re.search(rf"^{key}:\n((?:  .*\n)+)", text, re.MULTILINE).group(1)
        values = {}
        for metric, mean, std in re.findall(r"  (\w+): \{mean: ([\d.]+), std: ([\d.]+)\}", block):
            values[metric] = (round(float(mean) * 100, 2), round(float(std) * 100, 2))
        return values

    @pytest.mark.parametrize(
        ("ours", "theirs"),
        [("starcop_mag1c_rgb", "mag1c_rgb"), ("starcop_mag1c_only", "mag1c_only")],
    )
    def test_the_hyperstarcop_rows_match(self, ours, theirs):
        expected = self._yaml_values(theirs)
        row = REFERENCES[ours].metrics

        assert row["strong_f1"] == expected["strong_f1score"]
        assert row["weak_f1"] == expected["weak_f1score"]
        assert row["tile_fpr"] == expected["no_plume_FPR"]
        assert row["auprc"] == expected["auprc"]
