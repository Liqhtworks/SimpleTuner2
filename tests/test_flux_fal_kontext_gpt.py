import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from helpers.training.adapter import determine_adapter_target_modules
from helpers.models.flux.model import Flux


class TestFluxFalKontextGpt(unittest.TestCase):
    def setUp(self):
        self.expected_targets = [
            "to_qkv",
            "add_qkv_proj",
            "to_out.0",
            "to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]

    def test_adapter_target_modules_fal_kontext_gpt(self):
        args = SimpleNamespace(
            model_family="flux",
            flux_lora_target="fal-kontext-gpt",
        )
        # trigger transformer branch by passing a non-None transformer
        out = determine_adapter_target_modules(args, unet=None, transformer=object())
        for key in self.expected_targets:
            self.assertIn(key, out)

    def test_flux_get_lora_target_layers_fal_kontext_gpt(self):
        # Create an instance without invoking __init__ to avoid heavy setup
        flux = Flux.__new__(Flux)
        flux.DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0", "to_qkv"]
        flux.config = SimpleNamespace(
            model_type="lora",
            controlnet=False,
            control=False,
            lora_type="standard",
            flux_lora_target="fal-kontext-gpt",
        )
        layers = Flux.get_lora_target_layers(flux)
        # Ensure all expected keys are present and a couple of unique ones too
        for key in self.expected_targets:
            self.assertIn(key, layers)
        # Ensure single-block MLP and norm modulation entries are present per preset
        for extra in ["proj_mlp", "proj_out", "norm.linear", "norm1.linear", "norm1_context.linear"]:
            self.assertIn(extra, layers)

    def test_check_user_config_auto_enables_fused_qkv(self):
        flux = Flux.__new__(Flux)
        # minimal config to avoid unrelated branches
        flux.config = SimpleNamespace(
            # core flags used by the logic under test
            model_type="lora",
            flux_lora_target="fal-kontext-gpt",
            fuse_qkv_projections=False,
            # flags to bypass other behaviours in check_user_config
            unet_attention_slice=False,
            aspect_bucket_alignment=64,
            prediction_type=None,
            tokenizer_max_length=None,
            model_flavour="krea",
            validation_num_inference_steps=28,
            validation_guidance_real=1.0,
            flux_attention_masked_training=False,
        )
        # The method may reference these in warnings; keep them available
        flux.get_trained_component = MagicMock(return_value=None)
        flux.noise_schedule = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))

        # Should flip fuse_qkv_projections to True for this preset
        Flux.check_user_config(flux)
        self.assertTrue(flux.config.fuse_qkv_projections)


if __name__ == "__main__":
    unittest.main()
