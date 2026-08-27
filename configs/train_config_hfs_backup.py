# from torch import optim

# from datasets.coco import CocoDetection
# from transforms import presets
# from optimizer import param_dict

# # Commonly changed training configurations
# num_epochs = 12   # train epochs
# batch_size = 1    # total_batch_size = #GPU x batch_size
# num_workers = 4   # workers for pytorch DataLoader
# pin_memory = True # whether pin_memory for pytorch DataLoader
# print_freq = 50   # frequency to print logs

# # Code မှ Directory ကို ဖတ်ပြီး Auto-Resume လုပ်မည်ဖြစ်၍ 0 သာ ထားပါ
# starting_epoch = 0
# max_norm = 0.1    # clip gradient norm

# # 💡 1. Output & Resume Directory (main.py အတွက် resume_from_checkpoint ဟု သုံးရပါမည်)
# output_dir = "/workspace/ASR-Position-DETR/output" 
# resume_from_checkpoint = "/workspace/ASR-Position-DETR/output"

# find_unused_parameters = False  # useful for debugging distributed training

# # 💡 2. COCO VisDrone Path
# coco_path = "/workspace/ASR-Position-DETR/data/coco_visdrone/"

# train_transform = presets.detr  # see transforms/presets to choose a transform

# # 💡 3. VisDrone Folders & JSON Files ချိတ်ဆက်ခြင်း
# train_dataset = CocoDetection(
#     img_folder=coco_path + 'train',
#     ann_file=coco_path + 'annotations/instances_train.json',
#     transforms=train_transform,
#     train=True,
# )
# test_dataset = CocoDetection(
#     img_folder=coco_path + 'val',
#     ann_file=coco_path + 'annotations/instances_val.json',
#     transforms=None,  # the eval_transform is integrated in the model
# )

# # model config to train
# model_path = "configs/position_detr/position_detr_resnet50.py"

# learning_rate = 1e-4  # initial learning rate
# optimizer = optim.AdamW(lr=learning_rate, weight_decay=1e-4, betas=(0.9, 0.999))
# lr_scheduler = optim.lr_scheduler.MultiStepLR(milestones=[10], gamma=0.1)

# # This define parameter groups with different learning rate
# param_dicts = param_dict.finetune_backbone_and_linear_projection(lr=learning_rate)


from torch import optim
from functools import partial


from datasets.coco import CocoDetection
from transforms import presets
from optimizer import param_dict


num_epochs = 25
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
output_dir = (
    "/workspace/ASR-Position-DETR/"
    "output_position_detr_saliency_all_k3_original"
)
# output_dir = (
#     "/workspace/ASR-Position-DETR/"
#     "output_position_detr_saliency_all_k3_concat1x1"
# )
# output_dir = (
#     "/workspace/ASR-Position-DETR/"
#     "output_position_detr_saliency_hungarian_matcher"
# )


resume_from_checkpoint = (
    "/workspace/ASR-Position-DETR/"
    "output_position_detr_saliency_all_k3_original"
)


# resume_from_checkpoint = (
#     "/workspace/ASR-Position-DETR/position_detr_coco.pth"
# )

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


model_path = "configs/position_detr/position_detr_resnet50.py"


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