# CHECK: c1 = 5, c2 = 1
CUDA_VISIBLE_DEVICES=1 python main_yf --seed=1; CUDA_VISIBLE_DEVICES=1 python main_yf --seed=2; CUDA_VISIBLE_DEVICES=1 python main_yf --seed=3;

{
    CUDA_VISIBLE_DEVICES=1 python main_sgdm_fixed_alpha.py --alpha=0.4 --seed=1;
    CUDA_VISIBLE_DEVICES=1 python main_sgdm_fixed_alpha.py --alpha=0.4 --seed=2;
    CUDA_VISIBLE_DEVICES=1 python main_sgdm_fixed_alpha.py --alpha=0.4 --seed=3;
} &\
{
    CUDA_VISIBLE_DEVICES=2 python main_sgdm_fixed_alpha.py --alpha=0.05714285714285714 --seed=1;
    CUDA_VISIBLE_DEVICES=2 python main_sgdm_fixed_alpha.py --alpha=0.05714285714285714 --seed=2;
    CUDA_VISIBLE_DEVICES=2 python main_sgdm_fixed_alpha.py --alpha=0.05714285714285714 --seed=3;
}


# baseline
{
    CUDA_VISIBLE_DEVICES=1 python main_baseline.py --beta_val=0.9 --seed=1; 
    CUDA_VISIBLE_DEVICES=1 python main_baseline.py --beta_val=0.9 --seed=2; 
    CUDA_VISIBLE_DEVICES=1 python main_baseline.py --beta_val=0.9 --seed=3;
    echo "done Baseline"
} & \
{
    CUDA_VISIBLE_DEVICES=2 python main_sgdm.py --seed=1;
    CUDA_VISIBLE_DEVICES=2 python main_sgdm.py --seed=2;
    CUDA_VISIBLE_DEVICES=2 python main_sgdm.py --seed=3;
    echo "done MS-SGDM"
}

CUDA_VISIBLE_DEVICES=1 python main_baseline.py --seed=1 & CUDA_VISIBLE_DEVICES=2 python main_baseline.py --seed=2 & CUDA_VISIBLE_DEVICES=3 python main_baseline.py --seed=3

# baseline beta = 0 and beta = 0.9, repeat over 3 seeds
CUDA_VISIBLE_DEVICES=2 python main_baseline.py --beta_val=0 --seed=1; CUDA_VISIBLE_DEVICES=2 python main_baseline.py --beta_val=0 --seed=2; CUDA_VISIBLE_DEVICES=2 python main_baseline.py --beta_val=0 --seed=3; CUDA_VISIBLE_DEVICES=2 python main_baseline.py --beta_val=0.9 --seed=1; CUDA_VISIBLE_DEVICES=2 python main_baseline.py --beta_val=0.9 --seed=2; CUDA_VISIBLE_DEVICES=2 python main_baseline.py --beta_val=0.9 --seed=3

# baseline with beta = 0.9 and 3 seeds
CUDA_VISIBLE_DEVICES=1 python main_baseline.py --beta_val=0.9 --seed=1;
CUDA_VISIBLE_DEVICES=1 python main_baseline.py --beta_val=0.9 --seed=2;
CUDA_VISIBLE_DEVICES=1 python main_baseline.py --beta_val=0.9 --seed=3;
echo "done"

# sgmd alpha, beta both change, repeat over 3 seeds
CUDA_VISIBLE_DEVICES=0 python main_sgdm.py --seed=1; CUDA_VISIBLE_DEVICES=0 python main_sgdm.py --seed=2; CUDA_VISIBLE_DEVICES=0 python main_sgdm.py --seed=3