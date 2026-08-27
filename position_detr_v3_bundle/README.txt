Position-DETR v3 controlled ablation bundle
================================================

Copy files
----------

1. Copy these two files to:
   /workspace/ASR-Position-DETR/src/Position-DETR/models/bricks/

   - background_suppression.py
   - adaptive_multiscale_fusion.py

2. Back up the current detector:

   cp models/detectors/position_detr.py \
      models/detectors/position_detr_v2_backup.py

3. Copy the provided position_detr.py to:

   /workspace/ASR-Position-DETR/src/Position-DETR/models/detectors/position_detr.py


Experiment variants
-------------------

v3_variant="none"
    Adaptive Saliency v2 on P2/P3 only.
    No background suppression.
    No multi-scale fusion.

v3_variant="a"
    Adaptive Saliency v2 on P2/P3.
    Background suppression on P2/P3.
    No multi-scale fusion.

v3_variant="b"
    Adaptive Saliency v2 on P2/P3.
    No background suppression.
    P2 + P3 + P4 -> P3 fusion.

v3_variant="c"
    Adaptive Saliency v2 on P2/P3.
    Background suppression on P2/P3.
    P2 + P3 + P4 -> P3 fusion.


Configuration
-------------

The model constructor must receive one of:

    v3_variant: "a"
    v3_variant: "b"
    v3_variant: "c"

The exact YAML key depends on the repository's model-building/configuration system.

Do not initialize v3-B from v3-A or v3-C from v3-B.
All three experiments must start from the same pretrained checkpoint and use
the same seed, dataset, 12 epochs, optimizer, learning-rate schedule, batch size,
and evaluation protocol.


Quick syntax check
------------------

python -m py_compile \
  models/bricks/background_suppression.py \
  models/bricks/adaptive_multiscale_fusion.py \
  models/detectors/position_detr.py
