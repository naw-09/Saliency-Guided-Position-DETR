import os
import sys
import glob
import random

import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ============================================================
# 1. PROJECT PATH
# ============================================================

current_dir = os.path.dirname(os.path.abspath(__file__))

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


from util.lazy_load import instantiate
import configs.position_detr.position_detr_resnet50_asr_only as model_config


# ============================================================
# 2. SETTINGS
# ============================================================

IMAGE_DIR = (
    "/Users/nawnaw/Documents/Saliency-Guided-Position-DETR/"
    "datasets/VisDrone2019-DET-val/"
    "VisDrone2019-DET-val/images"
)

NUM_RANDOM_IMAGES = 5

CONF_THRESHOLD = 0.30

INPUT_SIZE = 1024

# Reproducible random selection
random.seed(42)


# ============================================================
# 3. OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = os.path.join(
    current_dir,
    "detection_results"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print("Output directory:")
print(OUTPUT_DIR)


# ============================================================
# 4. DEVICE
# ============================================================

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"\nUsing device: {device}")


# ============================================================
# 5. CHECKPOINT
# ============================================================

checkpoint_path = os.path.join(
    current_dir,
    "output_hfs_fgfe_token_asr",
    "best_ap.pth"
)

if not os.path.exists(checkpoint_path):

    checkpoint_path = os.path.join(
        current_dir,
        "../output_hfs_fgfe_token_asr",
        "best_ap.pth"
    )

if not os.path.exists(checkpoint_path):

    checkpoint_path = "best_ap.pth"


if not os.path.exists(checkpoint_path):

    raise FileNotFoundError(
        f"Checkpoint not found:\n{checkpoint_path}"
    )


print("\nLoading checkpoint from:")
print(checkpoint_path)


# ============================================================
# 6. LOAD MODEL
# ============================================================

model = instantiate(
    model_config.model
)

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu"
)


if "model" in checkpoint:

    model.load_state_dict(
        checkpoint["model"],
        strict=False
    )

else:

    model.load_state_dict(
        checkpoint,
        strict=False
    )


model.to(device)

model.eval()

print("\nModel and weights loaded successfully!")


# ============================================================
# 7. CLASS NAMES
# ============================================================

CLASSES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor"
]


# ============================================================
# 8. TRANSFORM
# ============================================================

transform = T.Compose([
    T.Resize(
        (
            INPUT_SIZE,
            INPUT_SIZE
        )
    ),
    T.ToTensor(),
    T.Normalize(
        [
            0.485,
            0.456,
            0.406
        ],
        [
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# 9. GET IMAGES
# ============================================================

image_files = sorted(
    glob.glob(
        os.path.join(
            IMAGE_DIR,
            "*.jpg"
        )
    )
)


if len(image_files) == 0:

    raise RuntimeError(
        f"No JPG images found in:\n{IMAGE_DIR}"
    )


print(
    "\nTotal validation images found:",
    len(image_files)
)


# ============================================================
# 10. RANDOMLY SELECT 5
# ============================================================

num_to_select = min(
    NUM_RANDOM_IMAGES,
    len(image_files)
)


selected_images = random.sample(
    image_files,
    num_to_select
)


print(
    f"\nSelected {num_to_select} random images:"
)


for i, path in enumerate(
    selected_images,
    start=1
):

    print(
        f"{i}. {os.path.basename(path)}"
    )


# ============================================================
# 11. PROCESS EACH IMAGE
# ============================================================

for index, image_path in enumerate(
    selected_images,
    start=1
):

    print("\n" + "=" * 70)

    print(
        f"Processing image "
        f"{index}/{num_to_select}"
    )

    print(
        os.path.basename(image_path)
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Load original image
    # --------------------------------------------------------

    orig_image = Image.open(
        image_path
    ).convert("RGB")

    w, h = orig_image.size


    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    img_tensor = (
        transform(orig_image)
        .unsqueeze(0)
        .to(device)
    )


    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            img_tensor
        )


    # ========================================================
    # FORMAT A:
    # Postprocessed
    # [{'scores': ..., 'labels': ..., 'boxes': ...}]
    # ========================================================

    if (
        isinstance(outputs, list)
        and len(outputs) > 0
        and isinstance(outputs[0], dict)
        and "scores" in outputs[0]
    ):

        pred = outputs[0]

        scores = (
            pred["scores"]
            .detach()
            .cpu()
        )

        labels = (
            pred["labels"]
            .detach()
            .cpu()
        )

        boxes = (
            pred["boxes"]
            .detach()
            .cpu()
        )

        is_absolute_boxes = True


    # ========================================================
    # FORMAT B:
    # Raw DETR
    # {'pred_logits': ..., 'pred_boxes': ...}
    # ========================================================

    else:

        output_dict = (
            outputs[0]
            if isinstance(
                outputs,
                (list, tuple)
            )
            else outputs
        )


        logits = (
            output_dict["pred_logits"]
            .sigmoid()[0]
            .detach()
            .cpu()
        )


        boxes = (
            output_dict["pred_boxes"][0]
            .detach()
            .cpu()
        )


        scores, labels = logits.max(
            dim=-1
        )


        is_absolute_boxes = False


    # ========================================================
    # CONFIDENCE FILTER
    # ========================================================

    keep = (
        scores >= CONF_THRESHOLD
    )


    scores = scores[keep]

    labels = labels[keep]

    boxes = boxes[keep]


    print(
        f"Detected objects: "
        f"{len(scores)}"
    )


    # ========================================================
    # DRAW DETECTIONS
    # ========================================================

    fig, ax = plt.subplots(
        1,
        figsize=(16, 10)
    )


    ax.imshow(
        orig_image
    )


    for score, label, box in zip(
        scores,
        labels,
        boxes
    ):

        # ----------------------------------------------------
        # POSTPROCESSED xyxy boxes
        # ----------------------------------------------------

        if is_absolute_boxes:

            xmin, ymin, xmax, ymax = (
                box.unbind()
            )


            xmin = (
                xmin.item()
                / INPUT_SIZE
                * w
            )


            ymin = (
                ymin.item()
                / INPUT_SIZE
                * h
            )


            xmax = (
                xmax.item()
                / INPUT_SIZE
                * w
            )


            ymax = (
                ymax.item()
                / INPUT_SIZE
                * h
            )


            box_w = xmax - xmin

            box_h = ymax - ymin


        # ----------------------------------------------------
        # RAW normalized cxcywh boxes
        # ----------------------------------------------------

        else:

            cx, cy, bw, bh = (
                box.unbind()
            )


            xmin = (
                cx
                - 0.5 * bw
            ).item() * w


            ymin = (
                cy
                - 0.5 * bh
            ).item() * h


            box_w = (
                bw.item()
                * w
            )


            box_h = (
                bh.item()
                * h
            )


        # ----------------------------------------------------
        # Class name
        # ----------------------------------------------------

        lbl_idx = int(
            label.item()
        )


        if 0 <= lbl_idx < len(CLASSES):

            class_name = CLASSES[
                lbl_idx
            ]

        else:

            class_name = (
                f"cls_{lbl_idx}"
            )


        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        rect = patches.Rectangle(
            (
                xmin,
                ymin
            ),
            box_w,
            box_h,
            linewidth=1.5,
            edgecolor="lime",
            facecolor="none"
        )


        ax.add_patch(
            rect
        )


        # ----------------------------------------------------
        # Label + confidence
        # ----------------------------------------------------

        ax.text(
            xmin,
            max(
                ymin - 4,
                0
            ),
            f"{class_name}, {score:.3f}",
            fontsize=7,
            color="yellow",
            bbox=dict(
                facecolor="black",
                alpha=0.65,
                pad=1,
                edgecolor="none"
            )
        )


    ax.axis(
        "off"
    )


    ax.set_title(
        f"Proposed Model Detection Result\n"
        f"{os.path.basename(image_path)}",
        fontsize=14
    )


    plt.tight_layout()


    # ========================================================
    # SAVE RESULT
    # ========================================================

    image_stem = os.path.splitext(
        os.path.basename(image_path)
    )[0]


    output_path = os.path.join(
        OUTPUT_DIR,
        f"{index:02d}_detection_{image_stem}.jpg"
    )


    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close(
        fig
    )


    print(
        "Saved:"
    )

    print(
        output_path
    )


# ============================================================
# 12. FINISH
# ============================================================

print("\n" + "=" * 70)

print(
    f"Finished. "
    f"{num_to_select} detection results generated."
)

print(
    "Results saved in:"
)

print(
    OUTPUT_DIR
)

print("=" * 70)