import cv2
import glob
import torch
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score,accuracy_score

from model import NeuralNetwork

net = NeuralNetwork()
net.load_state_dict(torch.load('./model.pth'))

ls = glob.glob('./test_resized/*.jpg')

arr = ["affenpinscher","afghan_hound","african_hunting_dog","airedale","american_staffordshire_terrier","appenzeller","australian_terrier","basenji","basset","beagle","bedlington_terrier","bernese_mountain_dog","black-and-tan_coonhound","blenheim_spaniel","bloodhound","bluetick","border_collie","border_terrier","borzoi","boston_bull","bouvier_des_flandres","boxer","brabancon_griffon","briard","brittany_spaniel","bull_mastiff","cairn","cardigan","chesapeake_bay_retriever","chihuahua","chow","clumber","cocker_spaniel","collie","curly-coated_retriever","dandie_dinmont","dhole","dingo","doberman","english_foxhound","english_setter","english_springer","entlebucher","eskimo_dog","flat-coated_retriever","french_bulldog","german_shepherd","german_short-haired_pointer","giant_schnauzer","golden_retriever","gordon_setter","great_dane","great_pyrenees","greater_swiss_mountain_dog","groenendael","ibizan_hound","irish_setter","irish_terrier","irish_water_spaniel","irish_wolfhound","italian_greyhound","japanese_spaniel","keeshond","kelpie","kerry_blue_terrier","komondor","kuvasz","labrador_retriever","lakeland_terrier","leonberg","lhasa","malamute","malinois","maltese_dog","mexican_hairless","miniature_pinscher","miniature_poodle","miniature_schnauzer","newfoundland","norfolk_terrier","norwegian_elkhound","norwich_terrier","old_english_sheepdog","otterhound","papillon","pekinese","pembroke","pomeranian","pug","redbone","rhodesian_ridgeback","rottweiler","saint_bernard","saluki","samoyed","schipperke","scotch_terrier","scottish_deerhound","sealyham_terrier","shetland_sheepdog","shih-tzu","siberian_husky","silky_terrier","soft-coated_wheaten_terrier","staffordshire_bullterrier","standard_poodle","standard_schnauzer","sussex_spaniel","tibetan_mastiff","tibetan_terrier","toy_poodle","toy_terrier","vizsla","walker_hound","weimaraner","welsh_springer_spaniel","west_highland_white_terrier","whippet","wire-haired_fox_terrier","yorkshire_terrier"]
for path in ls:
    img = cv2.imread(path)
    x = img.reshape(1, 3, 100, 100)
    pred = net(torch.from_numpy(x).float())
    pred = pred.data.numpy()
    cv2.imshow("resized", img)
    print(path, arr[pred[0].argmax()])
    cv2.waitKey(0)
