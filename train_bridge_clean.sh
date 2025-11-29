export MODEL_DIR="./PhotoDoodle_Pretrain" # you may need to modity this in order to train your own model
export OUTPUT_DIR="outputs/bridge_clean"
export CONFIG="./default_config.yaml"
export TRAIN_DATA="/root/PhotoDoodle/data/bridge_train/augmented_index.jsonl"
export LOG_PATH="$OUTPUT_DIR/log"

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file $CONFIG train_lora_flux_pe.py \
    --pretrained_model_name_or_path $MODEL_DIR \
    --width 960 \
    --height 720 \
    --source_column="source" \
    --target_column="target" \
    --caption_column="text" \
    --output_dir=$OUTPUT_DIR \
    --logging_dir=$LOG_PATH \
    --mixed_precision="bf16" \
    --train_data_dir=$TRAIN_DATA \
    --rank=128 \
    --learning_rate=1e-4 \
    --train_batch_size=1 \
    --num_validation_images=2 \
    --validation_image "/root/PhotoDoodle/assets/id00001_corner.png" \
    --validation_prompt "This collective 3x3 framework of sequential frames illustrates the phases of the robot's move the spatula to the far left corner behind the pot." \
    --num_train_epochs=10 \
    --validation_steps=2000 \
    --checkpointing_steps=2000 \
    --resume_from_checkpoint "outputs/bridge_clean/checkpoint-4000" \
