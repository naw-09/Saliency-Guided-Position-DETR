import os
import sys
import glob
import csv

import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.ops import box_iou


# ============================================================
# 1. PROJECT PATH
# ============================================================

ROOT = os.path.dirname(os.path.abspath(__file__))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from util.lazy_load import instantiate

import configs.position_detr.position_detr_resnet50_asr_only as model_config


# ============================================================
# 2. DATASET PATHS
# ============================================================

VAL_ROOT = os.path.join(
    ROOT,
    "datasets",
    "VisDrone2019-DET-val",
    "VisDrone2019-DET-val"
)

VAL_IMAGE_DIR = os.path.join(
    VAL_ROOT,
    "images"
)

VAL_ANN_DIR = os.path.join(
    VAL_ROOT,
    "annotations"
)


# ============================================================
# 3. CHECKPOINT
# ============================================================

CHECKPOINT = os.path.join(
    ROOT,
    "output_hfs_fgfe_token_asr",
    "best_ap.pth"
)


# ============================================================
# 4. SETTINGS
# ============================================================

INPUT_SIZE = 1024

CONF_THRESHOLD = 0.50

IOU_THRESHOLD = 0.50


# ============================================================
# 5. VISDRONE CLASSES
#
# Native VisDrone:
# 1 pedestrian
# 2 people
# 3 bicycle
# 4 car
# 5 van
# 6 truck
# 7 tricycle
# 8 awning-tricycle
# 9 bus
# 10 motor
#
# Model indices:
# 0 - 9
# ============================================================

CLASS_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

NUM_CLASSES = len(CLASS_NAMES)

BACKGROUND = NUM_CLASSES


# ============================================================
# 6. DEVICE
# ============================================================

if torch.backends.mps.is_available():

    device = torch.device("mps")

else:

    device = torch.device("cpu")


print("Device:", device)


# ============================================================
# 7. CHECK PATHS
# ============================================================

if not os.path.exists(VAL_IMAGE_DIR):

    raise FileNotFoundError(
        f"Image directory not found:\n{VAL_IMAGE_DIR}"
    )


if not os.path.exists(VAL_ANN_DIR):

    raise FileNotFoundError(
        f"Annotation directory not found:\n{VAL_ANN_DIR}"
    )


if not os.path.exists(CHECKPOINT):

    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT}"
    )


print("\nValidation images:")
print(VAL_IMAGE_DIR)

print("\nValidation annotations:")
print(VAL_ANN_DIR)

print("\nCheckpoint:")
print(CHECKPOINT)


# ============================================================
# 8. LOAD MODEL
# ============================================================

print("\nLoading model...")

model = instantiate(
    model_config.model
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu"
)


if "model" in checkpoint:

    missing, unexpected = model.load_state_dict(
        checkpoint["model"],
        strict=False
    )

else:

    missing, unexpected = model.load_state_dict(
        checkpoint,
        strict=False
    )


print(
    "Missing keys:",
    len(missing)
)

print(
    "Unexpected keys:",
    len(unexpected)
)


model.to(device)

model.eval()

print("Model loaded successfully.")


# ============================================================
# 9. CONFUSION MATRIX
#
# Rows    = Ground Truth
# Columns = Prediction
#
# Last row/column = Background
# ============================================================

cm = np.zeros(
    (
        NUM_CLASSES + 1,
        NUM_CLASSES + 1
    ),
    dtype=np.int64
)


# ============================================================
# 10. READ VISDRONE ANNOTATION
# ============================================================

def read_visdrone_annotation(annotation_path):

    boxes = []
    labels = []

    if not os.path.exists(annotation_path):

        return (
            torch.empty(
                (0, 4),
                dtype=torch.float32
            ),
            torch.empty(
                (0,),
                dtype=torch.long
            )
        )


    with open(
        annotation_path,
        "r"
    ) as f:

        lines = f.readlines()


    for line in lines:

        line = line.strip()

        if not line:
            continue


        parts = line.split(",")

        if len(parts) < 6:
            continue


        try:

            x = float(parts[0])
            y = float(parts[1])
            w = float(parts[2])
            h = float(parts[3])

            category_id = int(parts[5])

        except ValueError:

            continue


        # ----------------------------------------------------
        # Ignore invalid boxes
        # ----------------------------------------------------

        if w <= 0 or h <= 0:
            continue


        # ----------------------------------------------------
        # VisDrone categories:
        #
        # 0 = ignored region
        # 1-10 = target classes
        # 11 = others
        #
        # Only evaluate 1-10.
        # ----------------------------------------------------

        if category_id < 1 or category_id > 10:
            continue


        # ----------------------------------------------------
        # Convert:
        #
        # VisDrone 1-10
        # ->
        # Model 0-9
        # ----------------------------------------------------

        class_index = category_id - 1


        # xywh -> xyxy
        x1 = x
        y1 = y

        x2 = x + w
        y2 = y + h


        boxes.append(
            [
                x1,
                y1,
                x2,
                y2
            ]
        )


        labels.append(
            class_index
        )


    if len(boxes) == 0:

        return (
            torch.empty(
                (0, 4),
                dtype=torch.float32
            ),
            torch.empty(
                (0,),
                dtype=torch.long
            )
        )


    return (
        torch.tensor(
            boxes,
            dtype=torch.float32
        ),
        torch.tensor(
            labels,
            dtype=torch.long
        )
    )


# ============================================================
# 11. CONVERT CXCYWH -> XYXY
# ============================================================

def cxcywh_to_xyxy(boxes):

    cx = boxes[:, 0]
    cy = boxes[:, 1]

    w = boxes[:, 2]
    h = boxes[:, 3]


    x1 = cx - w / 2
    y1 = cy - h / 2

    x2 = cx + w / 2
    y2 = cy + h / 2


    return torch.stack(
        [
            x1,
            y1,
            x2,
            y2
        ],
        dim=1
    )


# ============================================================
# 12. MODEL PREDICTION
# ============================================================

def predict(image):

    original_width, original_height = image.size


    # --------------------------------------------------------
    # Same preprocessing used in your visualization
    # --------------------------------------------------------

    resized = image.resize(
        (
            INPUT_SIZE,
            INPUT_SIZE
        )
    )


    tensor = TF.to_tensor(
        resized
    )


    tensor = TF.normalize(
        tensor,
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


    tensor = (
        tensor
        .unsqueeze(0)
        .to(device)
    )


    with torch.no_grad():

        outputs = model(
            tensor
        )


    # --------------------------------------------------------
    # Handle list / tuple wrapper
    # --------------------------------------------------------

    if isinstance(
        outputs,
        (list, tuple)
    ):

        output = outputs[0]

    else:

        output = outputs


    # ========================================================
    # CASE A:
    # Post-processed predictions
    #
    # boxes / scores / labels
    # ========================================================

    if (
        isinstance(output, dict)
        and "scores" in output
        and "labels" in output
        and "boxes" in output
    ):

        scores = (
            output["scores"]
            .detach()
            .cpu()
        )

        labels = (
            output["labels"]
            .detach()
            .cpu()
            .long()
        )

        boxes = (
            output["boxes"]
            .detach()
            .cpu()
            .float()
        )


        keep = (
            scores >= CONF_THRESHOLD
        )


        scores = scores[keep]

        labels = labels[keep]

        boxes = boxes[keep]


        return (
            boxes,
            labels,
            scores
        )


    # ========================================================
    # CASE B:
    # Raw DETR output
    #
    # pred_logits
    # pred_boxes
    # ========================================================

    if not isinstance(output, dict):

        raise RuntimeError(
            f"Unexpected model output type: "
            f"{type(output)}"
        )


    if "pred_logits" not in output:

        raise RuntimeError(
            "Model output does not contain "
            "'pred_logits'.\n"
            f"Available keys: {output.keys()}"
        )


    if "pred_boxes" not in output:

        raise RuntimeError(
            "Model output does not contain "
            "'pred_boxes'."
        )


    logits = (
        output["pred_logits"][0]
        .detach()
        .cpu()
    )


    boxes = (
        output["pred_boxes"][0]
        .detach()
        .cpu()
        .float()
    )


    # --------------------------------------------------------
    # Position-DETR / DINO style uses sigmoid class scores
    # --------------------------------------------------------

    probabilities = logits.sigmoid()


    scores, labels = probabilities.max(
        dim=-1
    )


    keep = (
        scores >= CONF_THRESHOLD
    )


    scores = scores[keep]

    labels = labels[keep]

    boxes = boxes[keep]


    # --------------------------------------------------------
    # Normalized cxcywh -> xyxy
    # --------------------------------------------------------

    if len(boxes) > 0:

        boxes = cxcywh_to_xyxy(
            boxes
        )


        boxes[:, [0, 2]] *= (
            original_width
        )


        boxes[:, [1, 3]] *= (
            original_height
        )


    return (
        boxes,
        labels,
        scores
    )


# ============================================================
# 13. MATCH PREDICTIONS TO GROUND TRUTH
# ============================================================

def update_confusion_matrix(
    gt_boxes,
    gt_labels,
    pred_boxes,
    pred_labels,
    pred_scores
):

    num_gt = len(gt_boxes)

    num_pred = len(pred_boxes)


    # --------------------------------------------------------
    # No GT objects
    # --------------------------------------------------------

    if num_gt == 0:

        for pred_label in pred_labels:

            pred_class = int(
                pred_label
            )


            if 0 <= pred_class < NUM_CLASSES:

                cm[
                    BACKGROUND,
                    pred_class
                ] += 1

        return


    # --------------------------------------------------------
    # No predictions
    # --------------------------------------------------------

    if num_pred == 0:

        for gt_label in gt_labels:

            gt_class = int(
                gt_label
            )


            cm[
                gt_class,
                BACKGROUND
            ] += 1

        return


    # --------------------------------------------------------
    # IoU matrix
    # --------------------------------------------------------

    ious = box_iou(
        gt_boxes,
        pred_boxes
    )


    # --------------------------------------------------------
    # Candidate matches
    # --------------------------------------------------------

    candidates = []


    for gt_index in range(num_gt):

        for pred_index in range(num_pred):

            iou = float(
                ious[
                    gt_index,
                    pred_index
                ]
            )


            if iou >= IOU_THRESHOLD:

                candidates.append(
                    (
                        iou,
                        gt_index,
                        pred_index
                    )
                )


    # --------------------------------------------------------
    # Highest IoU first
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )


    matched_gt = set()

    matched_pred = set()


    # --------------------------------------------------------
    # One-to-one matching
    # --------------------------------------------------------

    for (
        iou,
        gt_index,
        pred_index
    ) in candidates:


        if gt_index in matched_gt:
            continue


        if pred_index in matched_pred:
            continue


        gt_class = int(
            gt_labels[
                gt_index
            ]
        )


        pred_class = int(
            pred_labels[
                pred_index
            ]
        )


        if (
            0 <= gt_class < NUM_CLASSES
            and
            0 <= pred_class < NUM_CLASSES
        ):

            cm[
                gt_class,
                pred_class
            ] += 1


        matched_gt.add(
            gt_index
        )

        matched_pred.add(
            pred_index
        )


    # --------------------------------------------------------
    # Unmatched GT = False Negative
    # --------------------------------------------------------

    for gt_index in range(
        num_gt
    ):

        if gt_index not in matched_gt:

            gt_class = int(
                gt_labels[
                    gt_index
                ]
            )


            if 0 <= gt_class < NUM_CLASSES:

                cm[
                    gt_class,
                    BACKGROUND
                ] += 1


    # --------------------------------------------------------
    # Unmatched prediction = False Positive
    # --------------------------------------------------------

    for pred_index in range(
        num_pred
    ):

        if pred_index not in matched_pred:

            pred_class = int(
                pred_labels[
                    pred_index
                ]
            )


            if 0 <= pred_class < NUM_CLASSES:

                cm[
                    BACKGROUND,
                    pred_class
                ] += 1


# ============================================================
# 14. GET VALIDATION IMAGES
# ============================================================

image_files = sorted(
    glob.glob(
        os.path.join(
            VAL_IMAGE_DIR,
            "*.jpg"
        )
    )
)


if len(image_files) == 0:

    raise RuntimeError(
        "No validation JPG images found."
    )


print(
    "\nNumber of validation images:",
    len(image_files)
)


# ============================================================
# 15. EVALUATION
# ============================================================

print("\nStarting evaluation...")
print(
    f"Confidence threshold = "
    f"{CONF_THRESHOLD}"
)

print(
    f"IoU threshold = "
    f"{IOU_THRESHOLD}"
)


processed = 0


for index, image_path in enumerate(
    image_files,
    start=1
):

    image_name = os.path.basename(
        image_path
    )


    image_stem = os.path.splitext(
        image_name
    )[0]


    annotation_path = os.path.join(
        VAL_ANN_DIR,
        image_stem + ".txt"
    )


    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    gt_boxes, gt_labels = (
        read_visdrone_annotation(
            annotation_path
        )
    )


    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:

        (
            pred_boxes,
            pred_labels,
            pred_scores
        ) = predict(
            image
        )

    except Exception as error:

        print(
            f"\nERROR processing "
            f"{image_name}:"
        )

        print(error)

        raise


    # --------------------------------------------------------
    # Update matrix
    # --------------------------------------------------------

    update_confusion_matrix(
        gt_boxes,
        gt_labels,
        pred_boxes,
        pred_labels,
        pred_scores
    )


    processed += 1


    if (
        index == 1
        or index % 50 == 0
        or index == len(image_files)
    ):

        print(
            f"Processed "
            f"{index}/"
            f"{len(image_files)}"
        )


print(
    "\nEvaluation complete."
)

print(
    "Processed images:",
    processed
)


# ============================================================
# 16. PER-CLASS METRICS
# ============================================================

precisions = []

recalls = []

f1_scores = []


total_tp = 0

total_fp = 0

total_fn = 0


print("\n")
print("=" * 85)

print(
    f"{'Class':20s}"
    f"{'Precision':>12s}"
    f"{'Recall':>12s}"
    f"{'F1':>12s}"
    f"{'TP':>10s}"
    f"{'FP':>10s}"
    f"{'FN':>10s}"
)

print("=" * 85)


for class_index, class_name in enumerate(
    CLASS_NAMES
):

    TP = int(
        cm[
            class_index,
            class_index
        ]
    )


    FP = int(
        cm[
            :,
            class_index
        ].sum()
        - TP
    )


    FN = int(
        cm[
            class_index,
            :
        ].sum()
        - TP
    )


    precision = (
        TP / (TP + FP)
        if (TP + FP) > 0
        else 0.0
    )


    recall = (
        TP / (TP + FN)
        if (TP + FN) > 0
        else 0.0
    )


    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            precision
            + recall
        ) > 0
        else 0.0
    )


    precisions.append(
        precision
    )

    recalls.append(
        recall
    )

    f1_scores.append(
        f1
    )


    total_tp += TP

    total_fp += FP

    total_fn += FN


    print(
        f"{class_name:20s}"
        f"{precision:12.4f}"
        f"{recall:12.4f}"
        f"{f1:12.4f}"
        f"{TP:10d}"
        f"{FP:10d}"
        f"{FN:10d}"
    )


print("=" * 85)


# ============================================================
# 17. MACRO METRICS
# ============================================================

macro_precision = float(
    np.mean(
        precisions
    )
)

macro_recall = float(
    np.mean(
        recalls
    )
)

macro_f1 = float(
    np.mean(
        f1_scores
    )
)


# ============================================================
# 18. MICRO METRICS
# ============================================================

micro_precision = (
    total_tp
    / (
        total_tp
        + total_fp
    )
    if (
        total_tp
        + total_fp
    ) > 0
    else 0.0
)


micro_recall = (
    total_tp
    / (
        total_tp
        + total_fn
    )
    if (
        total_tp
        + total_fn
    ) > 0
    else 0.0
)


micro_f1 = (
    2
    * micro_precision
    * micro_recall
    / (
        micro_precision
        + micro_recall
    )
    if (
        micro_precision
        + micro_recall
    ) > 0
    else 0.0
)


print("\nMACRO METRICS")

print(
    f"Macro Precision : "
    f"{macro_precision:.4f}"
)

print(
    f"Macro Recall    : "
    f"{macro_recall:.4f}"
)

print(
    f"Macro F1        : "
    f"{macro_f1:.4f}"
)


print("\nMICRO METRICS")

print(
    f"Micro Precision : "
    f"{micro_precision:.4f}"
)

print(
    f"Micro Recall    : "
    f"{micro_recall:.4f}"
)

print(
    f"Micro F1        : "
    f"{micro_f1:.4f}"
)


print("\nTOTAL")

print(
    "TP:",
    total_tp
)

print(
    "FP:",
    total_fp
)

print(
    "FN:",
    total_fn
)


# ============================================================
# 19. SAVE CSV
# ============================================================

csv_path = os.path.join(
    ROOT,
    "precision_recall_f1.csv"
)


with open(
    csv_path,
    "w",
    newline=""
) as csv_file:

    writer = csv.writer(
        csv_file
    )


    writer.writerow(
        [
            "Class",
            "Precision",
            "Recall",
            "F1",
            "TP",
            "FP",
            "FN"
        ]
    )


    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        TP = int(
            cm[i, i]
        )


        FP = int(
            cm[:, i].sum()
            - TP
        )


        FN = int(
            cm[i, :].sum()
            - TP
        )


        writer.writerow(
            [
                class_name,
                f"{precisions[i]:.6f}",
                f"{recalls[i]:.6f}",
                f"{f1_scores[i]:.6f}",
                TP,
                FP,
                FN
            ]
        )


    writer.writerow([])


    writer.writerow(
        [
            "Macro Average",
            f"{macro_precision:.6f}",
            f"{macro_recall:.6f}",
            f"{macro_f1:.6f}",
            "",
            "",
            ""
        ]
    )


    writer.writerow(
        [
            "Micro Average",
            f"{micro_precision:.6f}",
            f"{micro_recall:.6f}",
            f"{micro_f1:.6f}",
            total_tp,
            total_fp,
            total_fn
        ]
    )


print(
    "\nSaved CSV:"
)

print(
    csv_path
)


# ============================================================
# 20. SAVE RAW CONFUSION MATRIX
# ============================================================

DISPLAY_NAMES = (
    CLASS_NAMES
    + ["Background"]
)


fig, ax = plt.subplots(
    figsize=(14, 12)
)


im = ax.imshow(
    cm
)


fig.colorbar(
    im,
    ax=ax,
    fraction=0.046,
    pad=0.04
)


ax.set_xticks(
    np.arange(
        NUM_CLASSES + 1
    )
)

ax.set_yticks(
    np.arange(
        NUM_CLASSES + 1
    )
)


ax.set_xticklabels(
    DISPLAY_NAMES,
    rotation=45,
    ha="right"
)

ax.set_yticklabels(
    DISPLAY_NAMES
)


ax.set_xlabel(
    "Predicted Class"
)

ax.set_ylabel(
    "Ground Truth Class"
)


ax.set_title(
    "Confusion Matrix\n"
    f"Confidence >= {CONF_THRESHOLD}, "
    f"IoU >= {IOU_THRESHOLD}"
)


# ------------------------------------------------------------
# Add values to cells
# ------------------------------------------------------------

for i in range(
    NUM_CLASSES + 1
):

    for j in range(
        NUM_CLASSES + 1
    ):

        value = cm[i, j]


        if value > 0:

            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                fontsize=7
            )


plt.tight_layout()


raw_cm_path = os.path.join(
    ROOT,
    "confusion_matrix_raw.png"
)


plt.savefig(
    raw_cm_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close(
    fig
)


print(
    "\nSaved raw confusion matrix:"
)

print(
    raw_cm_path
)


# ============================================================
# 21. NORMALIZED CONFUSION MATRIX
# ============================================================

row_sums = cm.sum(
    axis=1,
    keepdims=True
)


normalized_cm = np.divide(
    cm,
    row_sums,
    out=np.zeros_like(
        cm,
        dtype=float
    ),
    where=row_sums != 0
)


fig, ax = plt.subplots(
    figsize=(14, 12)
)


im = ax.imshow(
    normalized_cm,
    vmin=0,
    vmax=1
)


fig.colorbar(
    im,
    ax=ax,
    fraction=0.046,
    pad=0.04
)


ax.set_xticks(
    np.arange(
        NUM_CLASSES + 1
    )
)

ax.set_yticks(
    np.arange(
        NUM_CLASSES + 1
    )
)


ax.set_xticklabels(
    DISPLAY_NAMES,
    rotation=45,
    ha="right"
)

ax.set_yticklabels(
    DISPLAY_NAMES
)


ax.set_xlabel(
    "Predicted Class"
)

ax.set_ylabel(
    "Ground Truth Class"
)


ax.set_title(
    "Normalized Confusion Matrix\n"
    f"Confidence >= {CONF_THRESHOLD}, "
    f"IoU >= {IOU_THRESHOLD}"
)


for i in range(
    NUM_CLASSES + 1
):

    for j in range(
        NUM_CLASSES + 1
    ):

        value = normalized_cm[
            i,
            j
        ]


        if value >= 0.01:

            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7
            )


plt.tight_layout()


normalized_cm_path = os.path.join(
    ROOT,
    "confusion_matrix_normalized.png"
)


plt.savefig(
    normalized_cm_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close(
    fig
)


print(
    "\nSaved normalized confusion matrix:"
)

print(
    normalized_cm_path
)


# ============================================================
# 22. FINISHED
# ============================================================

print("\n")
print("=" * 70)

print("FINISHED")

print(
    "Confidence threshold:",
    CONF_THRESHOLD
)

print(
    "IoU threshold:",
    IOU_THRESHOLD
)

print(
    "Images evaluated:",
    processed
)

print("=" * 70)