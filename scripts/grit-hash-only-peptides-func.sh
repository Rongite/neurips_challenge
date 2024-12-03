config_dir=configs/GRIT-hash-only
model_name=peptides-func-GRIT-RRWP

# for batch_size in 16 32 64; do
    for layer in 4 6 8; do
        # for n_head in 8 16 32 64; do
        for dim_hidden in 64 96; do
            for lr in 0.0003 0.001 0.003; do
                # Perform modulo operation and check condition
                # if (( dim_hidden % n_head == 0 )); then
                batch_size=16
                n_head=8
                file_name="${config_dir}/${model_name}-batch${batch_size}-layers${layer}-head${n_head}-dim_hidden${dim_hidden}-lr${lr}.yaml"
                echo "$file_name"
                python main.py --cfg "$file_name" wandb.use False accelerator "cuda:0" seed 41 dataset.dir '/mnt/vstor/CSE_CSDS_VXC204/dhl64/datalake'
                # fi
            done
        done
    done
    # done
# done
