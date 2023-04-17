# transformers 41a2f3529c6b56866c317031375ffd3e7b8bea01
# FastChat 4732169b937acc1b0ecb3f88656f9097f33ae841
# baes model https://huggingface.co/huggyllama/llama-13b
#OMP_NUM_THREADS=8 WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=2 --master_port=9967  \
    python3 finetune_fastchat.py \
    --base_model '../llama-13b/' \
    --data_path '../dataset/cleaned_processed.json' \
    --output_dir './lora-fastchat-alpaca' \
    --num_epochs 3
    