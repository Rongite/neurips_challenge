import yaml
import os
from copy import deepcopy

file_name = 'peptides-func-GRIT-RRWP.yaml'
config_dir = f'configs/GRIT/{file_name}'
config = None

with open(config_dir) as config_file:
    try:
        config = yaml.safe_load(config_file)
        print(config)
    except yaml.YAMLError as exc:
        print(exc)

# batch_sizes = [16, 32, 64]
layers = [4, 6, 8]
# n_heads = [8, 16, 32, 64]
dim_hiddens = [64, 96]
# dim_hiddens = [128, 256]
lrs = [0.0003, 0.001, 0.003]

# for batch_size in batch_sizes:
for layer in layers:
        # for n_head in n_heads:
    for dim_hidden in dim_hiddens:
        for lr in lrs:
            # if dim_hidden % n_head == 0:
            new_config = deepcopy(config)
            # new_config['train']['batch_size'] = batch_size
            batch_size = new_config['train']['batch_size']
            new_config['gt']['layers'] = layer
            # new_config['gt']['n_heads'] = n_head
            n_head = new_config['gt']['n_heads']
            new_config['gt']['dim_hidden'] = dim_hidden
            new_config['gnn']['dim_inner'] = dim_hidden
            new_config['optim']['base_lr'] = lr

            new_file_name = file_name.split('.')[0] + f'-batch{batch_size}-layers{layer}-head{n_head}-dim_hidden{dim_hidden}-lr{lr}.yaml'
            new_config_dir = f'configs/GRIT-hash-only/'
            if not os.path.exists(new_config_dir):
                os.makedirs(new_config_dir)
            new_config_dir = f'configs/GRIT-hash-only/{new_file_name}'

            with open(new_config_dir, 'w') as config_file:
                try:
                    yaml.dump(new_config, config_file, sort_keys=False)
                except yaml.YAMLError as exc:
                    print(exc)

