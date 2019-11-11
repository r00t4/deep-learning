import cv2
import glob
import os
import pandas as pd
import numpy as np

ls = glob.glob('./test/*.jpg')

# data = pd.read_csv("sample_submission.csv") 

# data_list = list(data.columns.values)


# str = '"'

# for ls in data_list:
#     str += ls + '","'

# print(str)


arr = ["affenpinscher","afghan_hound","african_hunting_dog","airedale","american_staffordshire_terrier","appenzeller","australian_terrier","basenji","basset","beagle","bedlington_terrier","bernese_mountain_dog","black-and-tan_coonhound","blenheim_spaniel","bloodhound","bluetick","border_collie","border_terrier","borzoi","boston_bull","bouvier_des_flandres","boxer","brabancon_griffon","briard","brittany_spaniel","bull_mastiff","cairn","cardigan","chesapeake_bay_retriever","chihuahua","chow","clumber","cocker_spaniel","collie","curly-coated_retriever","dandie_dinmont","dhole","dingo","doberman","english_foxhound","english_setter","english_springer","entlebucher","eskimo_dog","flat-coated_retriever","french_bulldog","german_shepherd","german_short-haired_pointer","giant_schnauzer","golden_retriever","gordon_setter","great_dane","great_pyrenees","greater_swiss_mountain_dog","groenendael","ibizan_hound","irish_setter","irish_terrier","irish_water_spaniel","irish_wolfhound","italian_greyhound","japanese_spaniel","keeshond","kelpie","kerry_blue_terrier","komondor","kuvasz","labrador_retriever","lakeland_terrier","leonberg","lhasa","malamute","malinois","maltese_dog","mexican_hairless","miniature_pinscher","miniature_poodle","miniature_schnauzer","newfoundland","norfolk_terrier","norwegian_elkhound","norwich_terrier","old_english_sheepdog","otterhound","papillon","pekinese","pembroke","pomeranian","pug","redbone","rhodesian_ridgeback","rottweiler","saint_bernard","saluki","samoyed","schipperke","scotch_terrier","scottish_deerhound","sealyham_terrier","shetland_sheepdog","shih-tzu","siberian_husky","silky_terrier","soft-coated_wheaten_terrier","staffordshire_bullterrier","standard_poodle","standard_schnauzer","sussex_spaniel","tibetan_mastiff","tibetan_terrier","toy_poodle","toy_terrier","vizsla","walker_hound","weimaraner","welsh_springer_spaniel","west_highland_white_terrier","whippet","wire-haired_fox_terrier","yorkshire_terrier"]

# y_arr = np.zeros(120)

# y = "affenpinscher"

# print(len(arr))
print(arr.index("afghan_hound"))
# print(y_arr)


# print(len(ls))
# for path in ls:
#     img = cv2.imread(path)
#     width = img.shape[1]
#     height = img.shape[0]
#     dim = (100,100)
#     resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
#     print(path)
#     cv2.imwrite('./test_resized/' + os.path.basename(path),resized)
    # cv2.imshow("img", resized)

    