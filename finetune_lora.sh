# transformers 41a2f3529c6b56866c317031375ffd3e7b8bea01
#OMP_NUM_THREADS=8 WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=2 --master_port=9967  \
    python3 finetune.py \
    --base_model '../llama-13b/' \
    --data_path 'alpaca_data.json' \
    --output_dir './lora-alpaca_2' \
    --num_epochs 1 \
    --lora_target_modules ['q_proj','k_proj','v_proj','o_proj']
    