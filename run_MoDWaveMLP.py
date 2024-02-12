
import os
 



from datetime import datetime
import tensorflow as tf
import numpy as np
import argparse
print("Tensorflow version:", tf.__version__)


from Datasets.dataset_wt_preprocessing import Dataset
from Model.Trainer import Trainer
from Parameters import Parameters
from Parameters import hyperparams_defaults as hyperparams_dict

def getEmptyGPU(e_gpu):
    empty_gpu = e_gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = empty_gpu
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
            exit(-1)


def insert_dict(d, k, v):
    previous = d.get(k, [])
    d[k] = previous + [v]
    return d

def subtime(date1, date2):
    date1 = datetime.strptime(date1, "%Y-%m-%d %H:%M:%S")
    date2 = datetime.strptime(date2, "%Y-%m-%d %H:%M:%S")
    return date2 - date1


if __name__ == '__main__':
    getEmptyGPU('1')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        default='metr-la', help='the name of dataset')
    parser.add_argument('--horizon', type=int,
                        default=12, help='the historical time step')
    parser.add_argument('--history_length', type=int,
                        default=12, help='the length of time step to be predicted')   
    parser.add_argument('--datapath', type=str,
                        default="data", help='the folder where the dataset is located')                                        
    args = parser.parse_args()

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("------------------Start training------------------", start_time)
    hyperparams_dict["dataset"] = args.dataset
    hyperparams_dict["horizon"] = args.horizon
    hyperparams_dict["history_length"] = args.history_length
  

    dataset = Dataset(name=hyperparams_dict["dataset"],
                    horizon=hyperparams_dict["horizon"],
                    history_length=hyperparams_dict["history_length"],
                    path=args.datapath)
    
    hyperparams_dict["num_nodes"] = dataset.num_nodes
    print("ModWaveMLP parameters:")
    print(hyperparams_dict)
    hyperparams = Parameters(**hyperparams_dict)
    


    trainer = Trainer(hyperparams=hyperparams, logdir="logdir")
    trainer.fit(dataset=dataset)


    

    early_stop_mae_h_repeats = dict()
    early_stop_mape_h_repeats = dict()
    early_stop_rmse_h_repeats = dict()
    early_stop_mae_h_ave = dict()
    early_stop_mape_h_ave = dict()
    early_stop_rmse_h_ave = dict()
    for i, h in enumerate(trainer.history):
        early_stop_idx = np.argmin(h['mae_val'])
        early_stop_mae = np.round(h['mae_test'][early_stop_idx], decimals=3)
        print(f"Early stop test error model {trainer.folder_names[i]}:", "Avg MAE", early_stop_mae)
        for horizon in range(1, hyperparams.horizon + 1,1):
            early_stop_mae_h_repeats = insert_dict(early_stop_mae_h_repeats, k=horizon,
                                                v=h[f'mae_test_h{horizon}'][early_stop_idx])
            early_stop_mape_h_repeats = insert_dict(early_stop_mape_h_repeats, k=horizon,
                                                    v=h[f'mape_test_h{horizon}'][early_stop_idx])
            early_stop_rmse_h_repeats = insert_dict(early_stop_rmse_h_repeats, k=horizon,
                                                    v=h[f'rmse_test_h{horizon}'][early_stop_idx])

            print(f"Horizon {horizon} MAE:", np.round(early_stop_mae_h_repeats[horizon][-1], decimals=2),
                f"Horizon {horizon} MAPE:", np.round(early_stop_mape_h_repeats[horizon][-1], decimals=2),
                f"Horizon {horizon} RMSE:", np.round(early_stop_rmse_h_repeats[horizon][-1], decimals=2))

        for horizon in range(3, hyperparams.horizon + 1,1):
            early_stop_mae_h_ave[horizon] = np.round(np.mean(early_stop_mae_h_repeats[horizon]), decimals=2)
            early_stop_mape_h_ave[horizon] = np.round(np.mean(early_stop_mape_h_repeats[horizon]), decimals=2)
            early_stop_rmse_h_ave[horizon] = np.round(np.mean(early_stop_rmse_h_repeats[horizon]), decimals=2)

    print()
    print("Average MAE:", early_stop_mae_h_ave)
    print("Average MAPE:", early_stop_mape_h_ave)
    print("Average RMSE:", early_stop_rmse_h_ave)

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("------------------Completed!------------------", end_time)
    print("total time spent{}".format(subtime(start_time, end_time)))


