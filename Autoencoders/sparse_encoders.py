"""
Sparse Autoencoder (SAE) — pretrained, load-and-run inference.
This SAE reconstructs
GPT-2 small's internal residual-stream activations through a sparse,
overcomplete bottleneck. This is the modern "interpretability SAE" --
the technique underlying Anthropic's monosemanticity research.

Pretrained SAE: jbloom/gpt2-small-res-jb (loaded via SAELens)
Base model:     gpt2-small (loaded via TransformerLens)
"""

import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

DEVICE = "cpu"
HOOK_LAYER = 8
HOOK_NAME = "blocks.8.hook_resid_pre"

# ---------------------------------------------------------------------
# 1. Load base model + pretrained SAE
# ---------------------------------------------------------------------
print("[SAE] Loading GPT-2 small...")
model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)

print("[SAE] Loading pretrained SAE (blocks.8.hook_resid_pre)...")
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gpt2-small-res-jb",   # official SAELens release name
    sae_id=HOOK_NAME,
    device=DEVICE,
)

print(f"  SAE input dim  (d_in) : {sae.cfg.d_in}")
print(f"  SAE hidden dim (d_sae): {sae.cfg.d_sae}")
print(f"  Expansion factor      : {sae.cfg.d_sae / sae.cfg.d_in:.1f}x")

# ---------------------------------------------------------------------
# 2. Run the base model, capture activations at the hook point
# ---------------------------------------------------------------------
prompt = "The capital of France is Paris"
tokens = model.to_tokens(prompt)

with torch.no_grad():
    _, cache = model.run_with_cache(tokens)
    activations = cache[HOOK_NAME]          # shape: [batch, seq_len, d_in]

    # ---------------------------------------------------------------
    # 3. SAE inference: encode -> sparse features -> decode -> reconstruction
    # ---------------------------------------------------------------
    features = sae.encode(activations)      # [batch, seq_len, d_sae]  (sparse)
    reconstruction = sae.decode(features)   # [batch, seq_len, d_in]   (back to original space)

# ---------------------------------------------------------------------
# 4. Inspect sparsity + reconstruction quality
# ---------------------------------------------------------------------
l0 = (features > 0).float().sum(-1)                     # active features per token
recon_mse = (reconstruction - activations).pow(2).mean().item()

print(f"\n[SAE — Inference on prompt]: \"{prompt}\"")
print(f"  Activations shape      : {tuple(activations.shape)}")
print(f"  Sparse features shape  : {tuple(features.shape)}")
print(f"  Avg active features/token (L0): {l0.mean().item():.1f} / {sae.cfg.d_sae}")
print(f"  Reconstruction MSE     : {recon_mse:.6f}")

# ---------------------------------------------------------------------
# 5. Per-token top active features (interpretability view)
# ---------------------------------------------------------------------
print("\n[Top-5 active SAE features per token]")
str_tokens = model.to_str_tokens(tokens)
for pos in range(tokens.shape[1]):
    top = features[0, pos].topk(5)
    feat_ids = top.indices.tolist()
    print(f"  '{str_tokens[pos]}': features {feat_ids}")

print("\nTip: look up any feature id on Neuronpedia to see what it represents, e.g.")
print(f"  https://www.neuronpedia.org/gpt2-small/8-res-jb/{features[0, -1].topk(1).indices.item()}")