"""Tests for the U-Net (smp.Unet) forward pass reached via
starcop.model_setup.get_model -- this project's own composition entry point
into vendor/starcop's model construction (see train.py's "SETTING UP MODEL"
step). No vendor/starcop file is modified; get_model is imported unmodified
via _vendor_starcop_training.py (composition-only rule, TASK-2.2 decision 0).

Test Size: Small -- real ModelModule/smp.Unet built from vendor's own
config.yaml (unmodified, no data on disk needed) with a synthetic input
tensor; no mocking. encoder_weights is None for this project's real config
(num_channels=4 != 3), so construction needs no network access.
"""

import torch
from _vendor_starcop_training import get_model
from omegaconf import OmegaConf

_VENDOR_CONFIG = "vendor/starcop/scripts/configs/config.yaml"


def _build_model():
    settings = OmegaConf.load(_VENDOR_CONFIG)
    settings.model.test = False  # skip loading weights from disk
    return get_model(settings, experiment_name=None), settings


class TestUNetForwardShape:
    def test_forward_pass_output_matches_input_spatial_dims_and_num_classes(self):
        model, settings = _build_model()
        model.eval()
        num_channels = len(settings.dataset.input_products)
        batch_size, height, width = 2, 128, 128
        x = torch.rand(batch_size, num_channels, height, width)

        with torch.no_grad():
            output = model.forward(x)

        assert output.shape == (batch_size, settings.model.num_classes, height, width)

    def test_forward_pass_output_is_float32_with_no_nans_or_infs(self):
        model, settings = _build_model()
        model.eval()
        num_channels = len(settings.dataset.input_products)
        x = torch.rand(1, num_channels, 128, 128)

        with torch.no_grad():
            output = model.forward(x)

        assert output.dtype == torch.float32
        assert torch.isfinite(output).all()

    def test_forward_pass_tracks_gradients_in_train_mode(self):
        model, settings = _build_model()
        model.train()
        num_channels = len(settings.dataset.input_products)
        x = torch.rand(1, num_channels, 128, 128)

        output = model.forward(x)
        output.sum().backward()

        first_param = next(model.network.parameters())
        assert first_param.grad is not None
