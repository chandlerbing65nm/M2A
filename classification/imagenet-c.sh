
cd classification
python test_time.py \
      --cfg cfgs/imagenet_c/m2a.yaml \
      DATA_DIR "/path/to/datasets/imagenet" \
      TEST.BATCH_SIZE 64 \
      OPTIM.STEPS 1 \
      CORRUPTION.DATASET imagenet_c \
      CORRUPTION.NUM_EX 5000 \
      RNG_SEED 1 \
      M2A.SEED 1 \
      M2A.M 0.1 \
      M2A.N 3 \
      M2A.NUM_SQUARES 1 \
      M2A.LAMBDA_EML 1.0 \
      M2A.RANDOM_MASKING spatial \
      M2A.SPATIAL_TYPE patch \
      M2A.SPECTRAL_TYPE low \
      M2A.DISABLE_MCL False \
      M2A.DISABLE_EML False \