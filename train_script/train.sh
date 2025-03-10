export CUDA_VISIBLE_DEVICES=0

cd /D/predrnn-pytorch
source venv/bin/activate


python -u run.py \
    --is_training 1 \
    --device cuda \
    --dataset_name mnist \
    --train_data_paths /D/predrnn-pytorch/data/1_train.npz \
    --valid_data_paths  /D/predrnn-pytorch/data/1_valid.npz\
    --save_dir /D/predrnn-pytorch/output/checkpoint/tes \
    --gen_frm_dir /D/predrnn-pytorch/output/images \
    --model_name predrnn \
    --reverse_input 1 \
    --img_height 850 \
    --img_width 2350 \
    --img_channel 1 \
    --input_length 10 \
    --total_length 20 \
    --num_hidden 128,128,128,128 \
    --filter_size 3 \
    --stride 1 \
    --patch_size 5 \
    --layer_norm 0 \
    --scheduled_sampling 1 \
    --sampling_stop_iter 500 \
    --sampling_start_value 1.0 \
    --sampling_changing_rate 0.00002 \
    --lr 0.0003 \
    --batch_size 1 \
    --max_iterations 80000 \
    --display_interval 100 \
    --test_interval 5000 \
    --snapshot_interval 5000
