import torch
from architectures import build_e1, build_e2, build_e3

_BATCH = torch.randn(2, 4, 128, 128)


def _assert_forward_backward_works(model: torch.nn.Module) -> torch.Tensor:
    model.train()
    logits = model(_BATCH)
    target = torch.randint(0, 2, logits.shape, dtype=torch.float32)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for p in model.parameters())
    return logits


class TestE1TinyUNet:
    def test_output_shape_matches_the_documented_contract(self):
        logits = _assert_forward_backward_works(build_e1())
        assert logits.shape == (2, 1, 128, 128)

    def test_param_count_lands_in_the_designed_range(self):
        # base=8 -- chosen for a clean three-way size spread against E2
        # (~6.6M) and E3 (~0.857M): E1 unpretrained-small vs. E3
        # pretrained-small isolates the pretraining effect at matched
        # scale. Exact count locks the architecture against silent drift.
        n_params = sum(p.numel() for p in build_e1().parameters())
        assert n_params == 487_361

    def test_has_no_pretrained_weights(self):
        # Two fresh instances must differ -- proof there's no shared
        # pretrained initialization being loaded from anywhere.
        first = list(build_e1().parameters())[0]
        second = list(build_e1().parameters())[0]
        assert not torch.equal(first, second)


class TestE2UnetMobileNetV2:
    def test_output_shape_matches_the_documented_contract(self):
        logits = _assert_forward_backward_works(build_e2(pretrained=False))
        assert logits.shape == (2, 1, 128, 128)

    def test_param_count_matches_the_plan_estimate(self):
        n_params = sum(p.numel() for p in build_e2(pretrained=False).parameters())
        assert n_params == 6_629_233


class TestE3LinknetMobileNetV3SmallMinimal:
    def test_output_shape_matches_the_documented_contract(self):
        logits = _assert_forward_backward_works(build_e3(pretrained=False))
        assert logits.shape == (2, 1, 128, 128)

    def test_param_count_matches_the_plan_estimate(self):
        n_params = sum(p.numel() for p in build_e3(pretrained=False).parameters())
        assert n_params == 856_635


class TestInputOutputContractIsIdenticalAcrossConfigurations:
    def test_all_three_accept_four_channels_and_produce_one_channel(self):
        for build in (build_e1, build_e2, build_e3):
            model = build() if build is build_e1 else build(pretrained=False)
            logits = model(_BATCH)
            assert logits.shape == (2, 1, 128, 128)
