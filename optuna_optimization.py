import optuna
from optuna.samplers import TPESampler
import torch
import numpy as np
from TCN_SA_Guss_opti import TemporalConvNet, train, tst, valid, create_dataloader
import torch
import numpy as np
import random
import pandas as pd


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def optimize_hyperparameters(args, device):
    def objective(trial):
        trial_seed = 42 + trial.number
        set_seed(trial_seed)


        train_loader, test_loader, valid_loader, scaler = create_dataloader(args, device)

        args.input_size = next(iter(train_loader))[0].shape[-1]
        args.kernel_sizes = trial.suggest_int('kernel_sizes', 3, 9)
        args.drop_out = trial.suggest_float('drop_out', 0.01, 0.5)


        legal_nheads = [h for h in [1, 2, 4, 8] if args.output_size % h == 0]
        args.nheads = trial.suggest_categorical('nheads', legal_nheads)


        legal_dims = [i for i in range(64, 257) if i % args.nheads == 0]
        dim2 = trial.suggest_categorical('model_dim_2', legal_dims)
        dim3 = trial.suggest_categorical('model_dim_3', legal_dims)
        args.model_dim = [args.input_size, dim2, dim3]

        args.lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        args.batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        args.epochs = trial.suggest_int('epochs', 20, 50)

        args.trial_num = trial.number

        model = TemporalConvNet(
            args.input_size, args.output_size, args.pre_len,
            args.model_dim, args.kernel_sizes, args.drop_out, args.nheads
        ).to(device)

        train(model, args, scaler, device, train_loader)

        rmse_val, nse_val = valid(model, args, valid_loader, scaler, trial_num=trial.number)

        rmse_test, nse_test = tst(model, args, test_loader, scaler, trial_num=trial.number)

        trial.set_user_attr("rmse", rmse_val)
        trial.set_user_attr("test_nse", nse_test)
        trial.set_user_attr("test_rmse", rmse_test)

        return nse_val

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=70)   #需要手动调整参数寻优训练的次数

    print("Best trial:")
    trial = study.best_trial
    print(f"  NSE: {trial.value:.4f} (目标)")
    print(f"  RMSE: {trial.user_attrs['rmse']:.4f} (参考)")
    print(f"  Test NSE: {trial.user_attrs['test_nse']:.4f}")
    print(f"  Test RMSE: {trial.user_attrs['test_rmse']:.4f}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    best_result = {
        'Val_NSE': trial.value,
        'Val_RMSE': trial.user_attrs['rmse'],
        'Test_NSE': trial.user_attrs['test_nse'],
        'Test_RMSE': trial.user_attrs['test_rmse'],
        **trial.params
    }
    pd.DataFrame([best_result]).to_csv("Best_trial.csv", index=False)

    return trial.params
