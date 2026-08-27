import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from util.lazy_load import instantiate
import configs.position_detr.position_detr_resnet50_asr_only as model_config

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

checkpoint_path = os.path.join(current_dir, '../output_hfs_fgfe_token_asr/best_ap.pth')
if not os.path.exists(checkpoint_path):
    checkpoint_path = 'best_ap.pth'

model = instantiate(model_config.model)
checkpoint = torch.load(checkpoint_path, map_location='cpu')

if 'model' in checkpoint:
    model.load_state_dict(checkpoint['model'], strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)

model.to(device)
model.eval()

# Hook for Transformer / Neck features
activation = {}
def get_activation(name):
    def hook(m, inp, out):
        activation[name] = out
    return hook

# Hook directly into Backbone Stage 4 or Neck
if hasattr(model, 'neck') and model.neck is not None:
    model.neck.register_forward_hook(get_activation('feat'))
else:
    model.backbone.register_forward_hook(get_activation('feat'))

image_path = 'test.jpg'
if not os.path.exists(image_path):
    image_path = os.path.join(current_dir, '../test.jpg')

orig_img = Image.open(image_path).convert('RGB')
w, h = orig_img.size

# Transform
img_tensor = TF.to_tensor(orig_img.resize((1024, 1024)))
img_tensor = TF.normalize(img_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
img_tensor = img_tensor.unsqueeze(0).to(device)

with torch.no_grad():
    _ = model(img_tensor)

raw_feat = activation['feat']
if isinstance(raw_feat, (list, tuple)):
    # Finest scale feature (High spatial resolution for small objects)
    f = raw_feat[0]
elif isinstance(raw_feat, dict) and 'features' in raw_feat:
    f = raw_feat['features'][0]
else:
    f = raw_feat

# Compute L2 Norm across channels to capture true object energy/gradient
heatmap = torch.norm(f[0], dim=0, keepdim=True).unsqueeze(0)  # (1, 1, H, W)

# Normalize strictly between 0 and 1
heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

# Upsample to full image resolution
heatmap_up = F.interpolate(heatmap, size=(h, w), mode='bilinear', align_corners=False)
heatmap_grid = heatmap_up.squeeze().cpu()

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

axs[0].imshow(orig_img)
axs[0].set_title("Original UAV Image", fontsize=13)
axs[0].axis('off')

im1 = axs[1].imshow(heatmap_grid, cmap='jet')
axs[1].set_title("Feature Saliency Response Map", fontsize=13)
axs[1].axis('off')
plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

axs[2].imshow(orig_img)
axs[2].imshow(heatmap_grid, cmap='jet', alpha=0.5)
axs[2].set_title("Overlay (Target Focus vs Background)", fontsize=13)
axs[2].axis('off')

plt.tight_layout()
output_path = 'attention_heatmap_result.jpg'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Refined Heatmap saved to {output_path}")
