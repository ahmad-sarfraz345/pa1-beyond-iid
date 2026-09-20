"""Frozen final features and model-specific normalization (no extra crop)."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms as T


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class FrozenBackbone(nn.Module):
    def __init__(self, name: str, device: torch.device):
        super().__init__()
        self.name = name
        if name == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            net.fc = nn.Identity()  # forward now returns global-average-pooled 2048-d features
            self.dim = 2048
            self.norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        elif name == "vit_b_16":
            net = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            net.heads = nn.Identity()  # final encoded class token, 768-d
            self.dim = 768
            self.norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        elif name == "clip_vit_b_32":
            import open_clip

            # OpenAI's ViT-B/32 checkpoint was trained with QuickGELU.
            net, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
            self.dim = net.visual.output_dim
            self.norm = T.Normalize(CLIP_MEAN, CLIP_STD)
        else:
            raise ValueError(name)
        self.net = net.eval().to(device)
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)
        self.device = device
        self.to_tensor = T.ToTensor()

    @torch.inference_mode()
    def encode(self, images):
        x = torch.stack([self.norm(self.to_tensor(im)) for im in images]).to(self.device)
        if self.name == "clip_vit_b_32":
            return F.normalize(self.net.encode_image(x).float(), dim=-1)
        return self.net(x).float()

    @torch.inference_mode()
    def zero_shot(self, classes):
        if self.name != "clip_vit_b_32":
            raise ValueError("Zero-shot evaluation requires CLIP")
        import open_clip

        tokens = open_clip.get_tokenizer("ViT-B-32-quickgelu")(
            [f"a photo of a {name}." for name in classes]
        ).to(self.device)
        texts = F.normalize(self.net.encode_text(tokens).float(), dim=-1)
        return texts, self.net.logit_scale.exp().float().item()
