import torch

from scripts.jepa_visual_adapters import (
    apply_visual_shift,
    build_image_adapter,
)


def test_adapters_start_as_identity():
    images = torch.rand(2, 3, 224, 224)
    for name in ("unet", "grid_color"):
        adapter = build_image_adapter(name).eval()
        torch.testing.assert_close(adapter(images), images, rtol=0, atol=3e-5)


def test_rbg_swap_is_exact_and_involutive():
    images = torch.rand(3, 3, 16, 16)
    indices = torch.arange(3)
    shifted = apply_visual_shift(images, "rbg", indices)
    torch.testing.assert_close(shifted, images[:, (0, 2, 1)], rtol=0, atol=0)
    restored = apply_visual_shift(shifted, "rbg", indices)
    torch.testing.assert_close(restored, images, rtol=0, atol=0)


def test_affine_shift_is_deterministic_and_shared_across_target_domain():
    image = torch.rand(1, 3, 32, 32)
    images = image.expand(2, -1, -1, -1).clone()
    indices = torch.tensor([10, 11])
    first = apply_visual_shift(images, "affine", indices)
    second = apply_visual_shift(images, "affine", indices)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not torch.equal(first[0], images[0])
    torch.testing.assert_close(first[0], first[1], rtol=0, atol=0)
