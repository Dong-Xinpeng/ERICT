cd ..


CUDA_VISIBLE_DEVICES=0 python main.py --dataset celebA --backbone ViTL14 --model zsclip_twice  --seed 1 --batch_size 64  \
                            --help_prompt '_class'  --mask_mode 'weight_last_two' --score_mode 'sparse' --tau 1.5

CUDA_VISIBLE_DEVICES=0 python main.py --dataset celebA --backbone ViTB16 --model zsclip_twice  --seed 1 --batch_size 64  \
                            --help_prompt '_class'  --mask_mode 'weight_last_two' --score_mode 'sparse' --tau 1.5


CUDA_VISIBLE_DEVICES=0 python main.py --dataset waterbirds --backbone ViTL14 --model zsclip_twice  --seed 1 --batch_size 64  \
                            --help_prompt '_class'  --mask_mode 'weight_last_two' --score_mode 'sparse' --tau 0.2

CUDA_VISIBLE_DEVICES=0 python main.py --dataset waterbirds --backbone ViTB16 --model zsclip_twice  --seed 1 --batch_size 64  \
                            --help_prompt '_class'  --mask_mode 'weight_last_two' --score_mode 'sparse' --tau 0.2






CUDA_VISIBLE_DEVICES=0 python main.py --dataset celebA --backbone ViTL14 --model zsclip_twice  --seed 1 --batch_size 64  \
                            --help_prompt 'hair in photo'  --mask_mode 'weight_last_two' --score_mode 'sparse' --tau 1.5

CUDA_VISIBLE_DEVICES=0 python main.py --dataset celebA --backbone ViTB16 --model zsclip_twice  --seed 1 --batch_size 64  \
                            --help_prompt 'hair in photo'  --mask_mode 'weight_last_two' --score_mode 'sparse' --tau 0.75


CUDA_VISIBLE_DEVICES=0 python main.py --dataset waterbirds --backbone ViTL14 --model zsclip_twice  --seed 1 --batch_size 64  \
                            --help_prompt 'bird in photo'  --mask_mode 'weight_last_two' --score_mode 'sparse' --tau 0.5

CUDA_VISIBLE_DEVICES=0 python main.py --dataset waterbirds --backbone ViTB16 --model zsclip_twice  --seed 1 --batch_size 64  \
                            --help_prompt 'bird in photo'  --mask_mode 'weight_last_two' --score_mode 'sparse' --tau 0.2