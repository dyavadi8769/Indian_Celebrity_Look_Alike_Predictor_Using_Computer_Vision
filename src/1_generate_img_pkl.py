from src.utils.all_utils import read_yaml, create_directory
import argparse
import os
import pickle
import logging

logging_str="[%(asctime)s:%(levelname)s: %(module)s: %(message)s]"
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir,'running_log.log'),
level=logging.INFO, format=logging_str,filemode='a')

def generate_data_pickle_file(config_path,params_path):
    config=read_yaml(config_path)
    params=read_yaml(params_path)
    artifacts=config['artifacts']
    artifacts_dir=artifacts['artifacts_dir']
    pickle_format_data_dir =artifacts['pickle_format_data_dir']
    img_pickle_file_name=artifacts['img_pickle_file_name']
    raw_local_dir_path=os.path.join(artifacts_dir,pickle_format_data_dir )
    create_directory(dirs=[raw_local_dir_path])

if __name__=="__main__":
    args=argparse.ArgumentParser()
    args.add_argument('--config',"-c", default='config/config.yaml')
    args.add_argument('--params',"-p", default='params.yaml')
    parsed_args=args.parse_args()

    try:
        logging.info(">>>>> stage_01 is started")
        generate_data_pickle_file(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info("stage_01 is completed")
    except Exception as e:
        logging.exception(e)
        raise e

