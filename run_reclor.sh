export RECLOR_DIR=data_reclor
export TASK_NAME=reclor
export MODEL_NAME=roberta-large

#CUDA_VISIBLE_DEVICES=0,1
export OUTPUT_NAME=reclor-large-len256-bs8-acc3-lr8e6
python train.py \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --evaluate_during_training \
    --do_test \
    --do_lower_case \
    --data_dir $RECLOR_DIR \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size 8   \
    --per_gpu_train_batch_size 8   \
    --gradient_accumulation_steps 3 \
    --learning_rate 8e-06 \
    --num_train_epochs 15.0 \
    --output_dir Checkpoints/$TASK_NAME/${MODEL_NAME} \
    --logging_steps 200 \
    --save_steps 800 \
    --adam_betas "(0.9, 0.98)" \
    --adam_epsilon 1e-6 \
    --no_clip_grad_norm \
    --warmup_proportion 0.1 \
    --weight_decay 0.01