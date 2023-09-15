from easydict import EasyDict as edict

__C                                             = edict()

cfg                                             = __C
__C.DATASET                                     = edict()
__C.DATASET.USED_DATA_FOLDER_1                  = [r"C:\Users\...\FL\rob1_big_out_r_1"]
__C.DATASET.USED_DATA_FOLDER_2                  = [r"C:\Users\...\FL\rob1_forbidden_1"]
__C.DATASET.USED_DATA_FOLDER_3                  = [r"C:\Users\...\FL\rob1_forbidden_2"]
__C.DATASET.USED_DATA_FOLDER_4                  = [r"C:\Users\...\FL\rob1_small_in_1"]
__C.DATASET.USED_DATA_FOLDER_5                  = [r"C:\Users\...\FL\rob3_big_in_1"]
__C.DATASET.USED_DATA_FOLDER_6                  = [r"C:\Users\...\FL\rob3_big_out_1"]
__C.DATASET.USED_DATA_FOLDER_7                  = [r"C:\Users\...\FL\rob3_big_out_stop"]
__C.DATASET.USED_DATA_FOLDER_8                  = [r"C:\Users\...\FL\rob3_small_in_1"]
__C.DATASET.USED_DATA_FOLDER_9                  = [r"C:\Users\...\FL\rob3_small_out_1"]
__C.DATASET.USED_DATA_FOLDER_10                 = [r"C:\Users\...\FL\rob3_small_out_stop"]



# TRAIN options
__C.TRAIN                                       = edict()
__C.TRAIN.LEARNING_RATE                         = 0.001
__C.TRAIN.NBR_EPOCH                             = 50
__C.TRAIN.CHECKPOINT_SAVE_PATH                  = r"C:\Users\...\results\checkpoint"
__C.TRAIN.VALIDATION_RATIO                      = 2
__C.TRAIN.GRADIANT_ACCUMULATION                 = 1
__C.TRAIN.IMAGE_SHAPE                           = (3,120,424)