import os

def execute_system():
    bash1= 'python src/1_generate_img_pkl.py'
    bash2= 'python src/2_feature_extractor.py'
    os.system(bash1)
    os.system(bash2)
    print('Executed Succesfully')


if __name__ == '__main__':
    execute_system()