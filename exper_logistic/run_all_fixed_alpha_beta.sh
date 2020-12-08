# beta 越小 alpha 的安全范围越大

{
    CUDA_VISIBLE_DEVICES=1 python logistic_mnist_fixed_alpha_beta.py --alpha=0.1 --beta=0.99 --seed=1 & 
    CUDA_VISIBLE_DEVICES=1 python logistic_mnist_fixed_alpha_beta.py --alpha=0.1 --beta=0.99 --seed=2 &
    CUDA_VISIBLE_DEVICES=1 python logistic_mnist_fixed_alpha_beta.py --alpha=0.1 --beta=0.99 --seed=3 &
    CUDA_VISIBLE_DEVICES=1 python logistic_mnist_fixed_alpha_beta.py --alpha=0.2 --beta=0.99 --seed=1 & 
    CUDA_VISIBLE_DEVICES=1 python logistic_mnist_fixed_alpha_beta.py --alpha=0.2 --beta=0.99 --seed=2 &
    CUDA_VISIBLE_DEVICES=1 python logistic_mnist_fixed_alpha_beta.py --alpha=0.2 --beta=0.99 --seed=3 &
    CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=0.5 --beta=0.99 --seed=1 & 
    CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=0.5 --beta=0.99 --seed=2 &
    CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=0.5 --beta=0.99 --seed=3 & 
    CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=0.8 --beta=0.99 --seed=1 & 
    CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=0.8 --beta=0.99 --seed=2 &
    CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=0.8 --beta=0.99 --seed=3 &
    CUDA_VISIBLE_DEVICES=3 python logistic_mnist_fixed_alpha_beta.py --alpha=1.0 --beta=0.99 --seed=1 & 
    CUDA_VISIBLE_DEVICES=3 python logistic_mnist_fixed_alpha_beta.py --alpha=1.0 --beta=0.99 --seed=2 &
    CUDA_VISIBLE_DEVICES=3 python logistic_mnist_fixed_alpha_beta.py --alpha=1.0 --beta=0.99 --seed=3 &
    CUDA_VISIBLE_DEVICES=3 python logistic_mnist_fixed_alpha_beta.py --alpha=5.0 --beta=0.99 --seed=1 & 
    CUDA_VISIBLE_DEVICES=3 python logistic_mnist_fixed_alpha_beta.py --alpha=5.0 --beta=0.99 --seed=2 &
    CUDA_VISIBLE_DEVICES=3 python logistic_mnist_fixed_alpha_beta.py --alpha=5.0 --beta=0.99 --seed=3
}
echo "all done"


{
    CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=0.1 --beta=0.2 --seed=1 & 
    CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=0.1 --beta=0.2 --seed=2 &
    CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=0.1 --beta=0.2 --seed=3 &
    CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=0.5 --beta=0.2 --seed=1 & 
    CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=0.5 --beta=0.2 --seed=2 &
    CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=0.5 --beta=0.2 --seed=3; 
} &\
{
    CUDA_VISIBLE_DEVICES=1 python logistic_mnist_fixed_alpha_beta.py --alpha=1.0 --beta=0.2 --seed=1 & 
    CUDA_VISIBLE_DEVICES=1 python logistic_mnist_fixed_alpha_beta.py --alpha=1.0 --beta=0.2 --seed=2 &
    CUDA_VISIBLE_DEVICES=1 python logistic_mnist_fixed_alpha_beta.py --alpha=1.0 --beta=0.2 --seed=3 &
    CUDA_VISIBLE_DEVICES=1 python logistic_mnist_fixed_alpha_beta.py --alpha=2.0 --beta=0.2 --seed=1 & 
    CUDA_VISIBLE_DEVICES=1 python logistic_mnist_fixed_alpha_beta.py --alpha=2.0 --beta=0.2 --seed=2 &
    CUDA_VISIBLE_DEVICES=1 python logistic_mnist_fixed_alpha_beta.py --alpha=2.0 --beta=0.2 --seed=3; 

    CUDA_VISIBLE_DEVICES=1 python logistic_mnist_fixed_alpha_beta.py --alpha=2.5 --beta=0.2 --seed=1 & 
    CUDA_VISIBLE_DEVICES=1 python logistic_mnist_fixed_alpha_beta.py --alpha=2.5 --beta=0.2 --seed=2 &
    CUDA_VISIBLE_DEVICES=1 python logistic_mnist_fixed_alpha_beta.py --alpha=2.5 --beta=0.2 --seed=3;

} &\

{
    CUDA_VISIBLE_DEVICES=3 python logistic_mnist_fixed_alpha_beta.py --alpha=5.0 --beta=0.2 --seed=1 & 
    CUDA_VISIBLE_DEVICES=3 python logistic_mnist_fixed_alpha_beta.py --alpha=5.0 --beta=0.2 --seed=2 &
    CUDA_VISIBLE_DEVICES=3 python logistic_mnist_fixed_alpha_beta.py --alpha=5.0 --beta=0.2 --seed=3 &
    CUDA_VISIBLE_DEVICES=3 python logistic_mnist_fixed_alpha_beta.py --alpha=0.1 --beta=0.1 --seed=1 & 
    CUDA_VISIBLE_DEVICES=3 python logistic_mnist_fixed_alpha_beta.py --alpha=0.1 --beta=0.1 --seed=2 &
    CUDA_VISIBLE_DEVICES=3 python logistic_mnist_fixed_alpha_beta.py --alpha=0.1 --beta=0.1 --seed=3; 
} &\
{
    CUDA_VISIBLE_DEVICES=0 python logistic_mnist_fixed_alpha_beta.py --alpha=0.5 --beta=0.1 --seed=1 & 
    CUDA_VISIBLE_DEVICES=0 python logistic_mnist_fixed_alpha_beta.py --alpha=0.5 --beta=0.1 --seed=2 &
    CUDA_VISIBLE_DEVICES=0 python logistic_mnist_fixed_alpha_beta.py --alpha=0.5 --beta=0.1 --seed=3 &
    CUDA_VISIBLE_DEVICES=0 python logistic_mnist_fixed_alpha_beta.py --alpha=1.0 --beta=0.1 --seed=1 & 
    CUDA_VISIBLE_DEVICES=0 python logistic_mnist_fixed_alpha_beta.py --alpha=1.0 --beta=0.1 --seed=2 &
    CUDA_VISIBLE_DEVICES=0 python logistic_mnist_fixed_alpha_beta.py --alpha=1.0 --beta=0.1 --seed=3; 
}

CUDA_VISIBLE_DEVICES=0 python logistic_mnist_fixed_alpha_beta.py --alpha=2.5 --beta=0.1 --seed=1 & 
CUDA_VISIBLE_DEVICES=0 python logistic_mnist_fixed_alpha_beta.py --alpha=2.5 --beta=0.1 --seed=2 &
CUDA_VISIBLE_DEVICES=0 python logistic_mnist_fixed_alpha_beta.py --alpha=2.5 --beta=0.1 --seed=3; 

CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=2.0 --beta=0.1 --seed=1 & 
CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=2.0 --beta=0.1 --seed=2 &
CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=2.0 --beta=0.1 --seed=3 & 
CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=5.0 --beta=0.1 --seed=1 & 
CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=5.0 --beta=0.1 --seed=2 &
CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=5.0 --beta=0.1 --seed=3;

##########################################
##########################################
##########################################
# old ones
CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=0.5 --beta=0.5 --seed=1 &
CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=0.5 --beta=0.5 --seed=2 &
CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=0.5 --beta=0.5 --seed=3;

CUDA_VISIBLE_DEVICES=1 python logistic_mnist_fixed_alpha_beta.py --alpha=1.0 --beta=0.5 --seed=1 & 
CUDA_VISIBLE_DEVICES=1 python logistic_mnist_fixed_alpha_beta.py --alpha=1.0 --beta=0.5 --seed=2 &
CUDA_VISIBLE_DEVICES=1 python logistic_mnist_fixed_alpha_beta.py --alpha=1.0 --beta=0.5 --seed=3;

CUDA_VISIBLE_DEVICES=1 python logistic_mnist_fixed_alpha_beta.py --alpha=2.0 --beta=0.5 --seed=1 & 
CUDA_VISIBLE_DEVICES=1 python logistic_mnist_fixed_alpha_beta.py --alpha=2.0 --beta=0.5 --seed=2 &
CUDA_VISIBLE_DEVICES=1 python logistic_mnist_fixed_alpha_beta.py --alpha=2.0 --beta=0.5 --seed=3;

CUDA_VISIBLE_DEVICES=1 python logistic_mnist_fixed_alpha_beta.py --alpha=2.5 --beta=0.5 --seed=1 & 
CUDA_VISIBLE_DEVICES=1 python logistic_mnist_fixed_alpha_beta.py --alpha=2.5 --beta=0.5 --seed=2 &
CUDA_VISIBLE_DEVICES=1 python logistic_mnist_fixed_alpha_beta.py --alpha=2.5 --beta=0.5 --seed=3;

CUDA_VISIBLE_DEVICES=1 python logistic_mnist_fixed_alpha_beta.py --alpha=5.0 --beta=0.5 --seed=1 & 
CUDA_VISIBLE_DEVICES=1 python logistic_mnist_fixed_alpha_beta.py --alpha=5.0 --beta=0.5 --seed=2 &
CUDA_VISIBLE_DEVICES=1 python logistic_mnist_fixed_alpha_beta.py --alpha=5.0 --beta=0.5 --seed=3;
echo "beta = 0.5 okay"

CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=0.1 --beta=0.9 --seed=1 & 
CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=0.1 --beta=0.9 --seed=2 &
CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=0.1 --beta=0.9 --seed=3; 

CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=0.5 --beta=0.9 --seed=1 &
CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=0.5 --beta=0.9 --seed=2 &
CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=0.5 --beta=0.9 --seed=3;

CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=1.0 --beta=0.9 --seed=1 & 
CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=1.0 --beta=0.9 --seed=2 &
CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=1.0 --beta=0.9 --seed=3;

CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=2.0 --beta=0.9 --seed=1 & 
CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=2.0 --beta=0.9 --seed=2 &
CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=2.0 --beta=0.9 --seed=3;

CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=2.5 --beta=0.9 --seed=1 & 
CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=2.5 --beta=0.9 --seed=2 &
CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=2.5 --beta=0.9 --seed=3;

CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=5.0 --beta=0.9 --seed=1 & 
CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=5.0 --beta=0.9 --seed=2 &
CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=5.0 --beta=0.9 --seed=3;
echo "beta = 0.9 okay"

# CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=0.6666666666666666 --beta=0.9 --seed=1 & 
# CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=0.6666666666666666 --beta=0.9 --seed=2 & 
# CUDA_VISIBLE_DEVICES=2 python logistic_mnist_fixed_alpha_beta.py --alpha=0.6666666666666666 --beta=0.9 --seed=3 & 
# CUDA_VISIBLE_DEVICES=1 python logistic_mnist_fixed_alpha_beta.py --alpha=0.09523809523809523 --beta=0.9 --seed=1 & 
# CUDA_VISIBLE_DEVICES=1 python logistic_mnist_fixed_alpha_beta.py --alpha=0.09523809523809523 --beta=0.9 --seed=2 & 
# CUDA_VISIBLE_DEVICES=1 python logistic_mnist_fixed_alpha_beta.py --alpha=0.09523809523809523 --beta=0.9 --seed=3


