import os
import sys
import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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

print(f"Loading checkpoint from: {checkpoint_path}")
model = instantiate(model_config.model)
checkpoint = torch.load(checkpoint_path, map_location='cpu')

if 'model' in checkpoint:
    model.load_state_dict(checkpoint['model'], strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)

model.to(device)
model.eval()
print("Model and weights loaded successfully!")

CLASSES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van',
    'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor'
]

transform = T.Compose([
    T.Resize((1024, 1024)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image_path = 'test.jpg'
if not os.path.exists(image_path):
    image_path = os.path.join(current_dir, '../test.jpg')

orig_image = Image.open(image_path).convert('RGB')
w, h = orig_image.size
img_tensor = transform(orig_image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(img_tensor)

# Postprocessed Format သို့မဟုတ် Raw Format ကို ခွဲခြားဖတ်ယူခြင်း
if isinstance(outputs, list) and isinstance(outputs[0], dict) and 'scores' in outputs[0]:
    # Postprocessed format: [{'scores': ..., 'labels': ..., 'boxes': ...}]
    pred = outputs[0]
    scores = pred['scores'].cpu()
    labels = pred['labels'].cpu()
    boxes = pred['boxes'].cpu()
    is_absolute_boxes = True
else:
    # Raw format: {'pred_logits': ..., 'pred_boxes': ...}
    output_dict = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    logits = output_dict['pred_logits'].sigmoid()[0].cpu()
    boxes = output_dict['pred_boxes'][0].cpu()
    scores, labels = logits.max(-1)
    is_absolute_boxes = False

# Confidence filter (0.30 အထက်)
keep = scores > 0.30
scores = scores[keep]
labels = labels[keep]
boxes = boxes[keep]

print(f"Detected objects: {len(scores)}")

fig, ax = plt.subplots(1, figsize=(14, 9))
ax.imshow(orig_image)

for score, label, box in zip(scores, labels, boxes):
    if is_absolute_boxes:
        # xyxy (1024x1024 scale) မှ Original Image scale သို့ ပြောင်းခြင်း
        xmin, ymin, xmax, ymax = box.unbind()
        xmin = (xmin.item() / 1024.0) * w
        ymin = (ymin.item() / 1024.0) * h
        box_w = ((xmax.item() - box[0].item()) / 1024.0) * w
        box_h = ((ymax.item() - box[1].item()) / 1024.0) * h
    else:
        # Normalized cxcywh format
        cx, cy, bw, bh = box.unbind()
        xmin = (cx - 0.5 * bw).item() * w
        ymin = (cy - 0.5 * bh).item() * h
        box_w = bw.item() * w
        box_h = bh.item() * h

    lbl_idx = label.item()
    class_name = CLASSES[lbl_idx] if lbl_idx < len(CLASSES) else f"cls_{lbl_idx}"
    
    rect = patches.Rectangle((xmin, ymin), box_w, box_h, linewidth=2, edgecolor='#00FFCC', facecolor='none')
    ax.add_patch(rect)
    ax.text(
        xmin, ymin - 4,
        f'{class_name}: {score:.2f}',
        color='yellow',
        fontsize=8,
        fontweight='bold',
        bbox=dict(facecolor='black', alpha=0.6, pad=1, edgecolor='none')
    )

plt.axis('off')
output_path = 'output_result.jpg'
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Detection result saved successfully to {output_path}")
