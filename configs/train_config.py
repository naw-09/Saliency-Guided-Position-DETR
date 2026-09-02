


from torch import optim
from functools import partial


from datasets.coco import CocoDetection
from transforms import presets
from optimizer import param_dict


num_epochs = 12
batch_size = 1
num_workers = 4
pin_memory = True
print_freq = 50

starting_epoch = 0
max_norm = 0.1

# output_dir = "/workspace/ASR-Position-DETR/output_asr_position_detr"
# output_dir = "/workspace/ASR-Position-DETR/output_asr_position_detr_saliency_v2"
# output_dir = (
#     "/workspace/ASR-Position-DETR/"
#     "output_position_detr_saliency_v1_original"
# )
# output_dir = (
#     "/workspace/ASR-Position-DETR/"
#     "output_position_detr_saliency_p2p3_original"
# )
# output_dir = (
#     "/workspace/ASR-Position-DETR/"
#     "output_asr_only_no_hfs"
# )
output_dir = (
    "/workspace/ASR-Position-DETR/"
    "output_hfs_fgfe_token_asr"
)
# output_dir = (
#     "/workspace/ASR-Position-DETR/"
#     "output_position_detr_saliency_all_k3_concat1x1"
# )
# output_dir = (
#     "/workspace/ASR-Position-DETR/"
#     "output_position_detr_saliency_hungarian_matcher"
# )


# resume_from_checkpoint = (
#     "/workspace/ASR-Position-DETR/"
#     "output_position_detr_saliency_all_k3_original"
# )


resume_from_checkpoint = (
    "/workspace/ASR-Position-DETR/position_detr_coco.pth"
)

# resume_from_checkpoint = ( "/workspace/ASR-Position-DETR/" 
#                            "output_asr_only_no_hfs" )

find_unused_parameters = False


# coco_path = "/workspace/ASR-Position-DETR/data/coco_visdrone/"
coco_path = (
    "/workspace/ASR-Position-DETR/"
    "data/coco_visdrone_original/"
)


train_dataset = CocoDetection(
    img_folder=coco_path + "train",
    ann_file=coco_path + "annotations/instances_train.json",
    transforms=presets.detr,
    train=True,
)


test_dataset = CocoDetection(
    img_folder=coco_path + "val",
    ann_file=coco_path + "annotations/instances_val.json",
    transforms=None,
)


# model_path = "configs/position_detr/position_detr_resnet50.py"
model_path = "configs/position_detr/position_detr_resnet50_asr_only.py"


learning_rate = 1e-4

optimizer = optim.AdamW(
    lr=learning_rate,
    weight_decay=1e-4,
    betas=(0.9, 0.999),
)

lr_scheduler = optim.lr_scheduler.MultiStepLR(
    milestones=[10],
    gamma=0.1,
)

def param_dicts(model):
    return param_dict.finetune_backbone_and_linear_projection(
        model,
        lr=learning_rate,
    )