import torch
import torch.nn as nn
import torch.nn.functional as F


class PIJEPA(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        target_encoder: nn.Module,
        predictors: list,
        embed_dim: int,
        num_patches: int = None,
        patch_size: int = 16,
    ):
        super().__init__()

        self.encoder = encoder
        self.target_encoder = target_encoder
        self.predictors = nn.ModuleList(predictors)

        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.patch_size = patch_size

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def mask_input(self, x, target_indices):
        B, C, H, W = x.shape
        
        # Get patch_size from encoder if available, otherwise use self.patch_size
        if hasattr(self.encoder, 'patch_size'):
            patch_size = self.encoder.patch_size
        else:
            patch_size = self.patch_size
        
        grid_size = H // patch_size
        num_patches = grid_size * grid_size
        
        # Clamp target_indices to valid range to prevent out-of-bounds errors
        # This handles cases where masker grid_size doesn't match encoder grid_size
        target_indices_clamped = target_indices.clamp(0, num_patches - 1)

        mask = torch.ones(B, num_patches, device=x.device)

        mask = mask.scatter(
            1,
            target_indices_clamped,
            torch.zeros_like(target_indices_clamped, dtype=mask.dtype)
        )

        mask = mask.view(B, grid_size, grid_size)
        mask = mask.repeat_interleave(patch_size, dim=1)
        mask = mask.repeat_interleave(patch_size, dim=2)
        mask = mask.unsqueeze(1)

        return x * mask
    
    def get_num_patches(self, image_size: int) -> int:
        """Get the number of patches for a given image size."""
        if hasattr(self.encoder, 'patch_size'):
            patch_size = self.encoder.patch_size
        else:
            patch_size = self.patch_size
        grid_size = image_size // patch_size
        return grid_size * grid_size

    def forward(self, x, context_indices, target_indices):
        B = x.shape[0]

        x_masked = self.mask_input(x, target_indices)

        with torch.no_grad():
            z_target_full = self.target_encoder(x)

        z_full = self.encoder(x_masked)

        z = z_full.clone()

        mask_tokens = self.mask_token.expand(
            B, target_indices.shape[1], self.embed_dim
        )

        z = z.scatter(
            1,
            target_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim),
            mask_tokens
        )

        for predictor in self.predictors:
            z_delta, _ = predictor(z, context_indices, target_indices)

            z_old = torch.gather(
                z,
                1,
                target_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
            )

            z_new = z_old + 0.5 * z_delta

            z = z.scatter(
                1,
                target_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim),
                z_new
            )

        z_pred = torch.gather(
            z,
            1,
            target_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        )

        z_target = torch.gather(
            z_target_full,
            1,
            target_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        )

        z_pred = F.layer_norm(z_pred, (self.embed_dim,))
        z_target = F.layer_norm(z_target, (self.embed_dim,))

        z_pred = F.normalize(z_pred, dim=-1)
        z_target = F.normalize(z_target, dim=-1)

        return z_pred, z_target

    def encode(self, x):
        return self.encoder(x)

    def encode_target(self, x):
        with torch.no_grad():
            return self.target_encoder(x)

    def predict_latent(self, z_full, context_indices, target_indices):
        B = z_full.shape[0]

        z = z_full.clone()

        mask_tokens = self.mask_token.expand(
            B, target_indices.shape[1], self.embed_dim
        )

        z = z.scatter(
            1,
            target_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim),
            mask_tokens
        )

        for predictor in self.predictors:
            z_delta, _ = predictor(z, context_indices, target_indices)

            z_old = torch.gather(
                z,
                1,
                target_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
            )

            z_new = z_old + 0.5 * z_delta

            z = z.scatter(
                1,
                target_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim),
                z_new
            )

        z_out = torch.gather(
            z,
            1,
            target_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        )

        return F.normalize(z_out, dim=-1)

    def rollout(self, x_init, steps, decoder=None, noise_std=0.0):
        x = x_init
        outputs = []

        for _ in range(steps):
            z_full = self.encoder(x)

            B, N, D = z_full.shape
            idx = torch.arange(N, device=x.device).unsqueeze(0).repeat(B, 1)

            z_pred = self.predict_latent(z_full, idx, idx)

            z_recon = z_full.clone()
            z_recon = z_recon.scatter(
                1,
                idx.unsqueeze(-1).expand(-1, -1, D),
                z_pred
            )

            if noise_std > 0.0:
                z_recon = z_recon + noise_std * torch.randn_like(z_recon)

            if decoder is not None:
                x = decoder(z_recon)
            else:
                x = z_recon

            outputs.append(x)

        return torch.stack(outputs, dim=1)
