from run import CDMOE_run

CDMOE_run(model_name='cdmoe', dataset_name='mosi', is_tune=True, seeds=[1111,1112], model_save_dir="./pt",
         res_save_dir="./result", log_dir="./log", mode='train')