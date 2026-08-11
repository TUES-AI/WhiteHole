import torch
from torch import nn

from scripts.train_vjepa_visual_adapters import adapt_video, shifted_video, stratified_indices


def test_rbg_shift_is_identical_on_every_video_frame():
    clips = torch.rand(2, 3, 4, 8, 8)

    shifted = shifted_video(clips)

    assert torch.equal(shifted[:, 0], clips[:, 0])
    assert torch.equal(shifted[:, 1], clips[:, 2])
    assert torch.equal(shifted[:, 2], clips[:, 1])


def test_image_adapter_is_shared_framewise():
    class AddOne(nn.Module):
        def forward(self, images):
            return images + 1

    clips = torch.rand(2, 3, 4, 8, 8)

    adapted = adapt_video(AddOne(), clips)

    assert torch.equal(adapted, clips + 1)


def test_loss_subset_is_class_balanced():
    rows = [{"label": label} for label in range(5) for _ in range(5)]

    selected = stratified_indices(rows, 5)

    assert selected == [0, 5, 10, 15, 20]
