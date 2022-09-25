import glob
import os
import tqdm
import shutil


class_list = ['00_best', '01_best_pet', '10_normal_all', '11_normal_pack', '20_thumnail', '21_thumnail_back', '22_box', '23_text', '24_chapture']

if __name__=='__main__':
    DEFAULT_PATH = '../data/target_220607'
    SAVE_ROOT = '../data/220607/rename'

    for idx, class_name in enumerate(os.listdir(DEFAULT_PATH)):
        print(idx, class_name)
        print(class_list[idx])
        os.makedirs(os.path.join(SAVE_ROOT, class_list[idx]), exist_ok=True)
        for file_idx, file in enumerate(tqdm.tqdm(os.listdir(os.path.join(DEFAULT_PATH, class_name)))):
            shutil.copy(os.path.join(DEFAULT_PATH, class_name, file), f'{SAVE_ROOT}/{class_list[idx]}/{class_list[idx]}_{file_idx:06}{os.path.splitext(file)[-1]}')