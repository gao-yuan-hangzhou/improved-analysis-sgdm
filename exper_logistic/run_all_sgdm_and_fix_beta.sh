# fixed beta change alpha

{
    CUDA_VISIBLE_DEVICES=1 python logistic_mnist_multistage_sgdm.py --seed=1;
    CUDA_VISIBLE_DEVICES=1 python logistic_mnist_multistage_sgdm.py --seed=2;
    CUDA_VISIBLE_DEVICES=1 python logistic_mnist_multistage_sgdm.py --seed=3;
} &\
{
    CUDA_VISIBLE_DEVICES=1 python logistic_mnist_changing_alpha_fix_beta.py --momentum_val=0.0 --seed=1;
    CUDA_VISIBLE_DEVICES=1 python logistic_mnist_changing_alpha_fix_beta.py --momentum_val=0.0 --seed=2;
    CUDA_VISIBLE_DEVICES=1 python logistic_mnist_changing_alpha_fix_beta.py --momentum_val=0.0 --seed=3;
} &\
{
    CUDA_VISIBLE_DEVICES=1 python logistic_mnist_changing_alpha_fix_beta.py --momentum_val=0.6 --seed=1;
    CUDA_VISIBLE_DEVICES=1 python logistic_mnist_changing_alpha_fix_beta.py --momentum_val=0.6 --seed=2;
    CUDA_VISIBLE_DEVICES=1 python logistic_mnist_changing_alpha_fix_beta.py --momentum_val=0.6 --seed=3;
} &\
{
    CUDA_VISIBLE_DEVICES=1 python logistic_mnist_changing_alpha_fix_beta.py --momentum_val=0.9 --seed=1;
    CUDA_VISIBLE_DEVICES=1 python logistic_mnist_changing_alpha_fix_beta.py --momentum_val=0.9 --seed=2;
    CUDA_VISIBLE_DEVICES=1 python logistic_mnist_changing_alpha_fix_beta.py --momentum_val=0.9 --seed=3;
}