import cv2
import numpy as np
from math import hypot, pi
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pandas as pd
from pathlib import Path

SAVE_PATH = 'data/fixed_data/output_test'
all_files = list()

if os.path.exists('data/fixed_data/tmp_csv_test'):
    pass
else:
    os.mkdir('data/fixed_data/tmp_csv_test')

for root, dirs, files in os.walk('data/fixed_data/test/'):
    for file in files:
        if file.endswith('.jpeg'):
            all_files.append(os.path.join(root, file))

for file in tqdm(all_files):
    if os.path.exists(SAVE_PATH):
        pass
    else:
        os.mkdir(SAVE_PATH)
    if os.path.exists(os.path.join(SAVE_PATH, file.split(os.sep)[-2])):
        pass
    else:
        os.mkdir(os.path.join(SAVE_PATH, file.split(os.sep)[-2]))

    new_save_path = os.path.join(SAVE_PATH, file.split(os.sep)[-2])
    folder_name = file.split(os.sep)[-2]
    file_name = file.split(os.sep)[-1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.imread(file)
    cv2.imwrite(os.path.join(new_save_path, file.split(os.sep)[-1]), img)
    height = img.shape[0]
    width = img.shape[1]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    ret, thresh = cv2.threshold(gray, 40, 255, 0, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(
        image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    a = 1
    b = 0

    for i, contour in enumerate(contours):
        if a < len(contour) and [0, 0] not in contour and [0, height] not in contour and [width, height] not in contour and [width, 0] not in contour:
            a = len(contour)
            b = i
            bigger_contour = contour
        else:
            pass

    try:
        M = cv2.moments(bigger_contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centre_coordinstes = (cX, cY)
        cv2.putText(img, ".", (cX, cY), font, 0.5, (255, 255, 255), 2)
    except Exception as e:
        pass

    distance = []

    for point in bigger_contour:
        new_distance = hypot(point[0][0] - cX, point[0][1] - cY)
        distance.append(new_distance)
    distance = np.array(distance)
    dist_res = distance
    distance = min(distance)
    radius = distance
    color = (255, 0, 0)
    new_img = cv2.circle(img, centre_coordinstes,
                         int(distance), color, thickness=4)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 9 * cv2.arcLength(cnt, True), True)
        cv2.drawContours(image=img, contours=bigger_contour, contourIdx=-1, color=(0, 255, 0), thickness=4,
                         lineType=cv2.LINE_AA)
    cv2.imwrite(os.path.join(new_save_path, file.split(
        os.sep)[-1][:-5]+'_contours.jpeg'), img)

    collections = []
    for point in bigger_contour:
        new_distance2 = int(hypot(point[0][0] - cX, point[0][1] - cY)-distance)
        collections.append(new_distance2)

    MinDistance = min(collections)
    MaxDistance = max(collections)
    MeanDistance = np.mean(collections)
    StandartDeviation = np.std(collections)
    scale_bar = 500
    coeff = (scale_bar * 4) / width
    area = sum(dist_res) + pi * radius ** 2
    res_area = coeff * area
    PhotoName = file.split('/')[4].split('.')[0]
    FileName = file.split('/')[3].split('_')[0]

    df = pd.DataFrame(
        {
            "Photo Name": [PhotoName],
            "Concentration": [FileName],
            "Area": [res_area],
            "Min distance": [MinDistance],
            "Mean distance": [MeanDistance],
            "Max distance": [MaxDistance],
            "Standart deviation": [StandartDeviation]
        }
    )

    df.to_csv(os.path.join(new_save_path, file.split(
        os.sep)[-1][:-5]+"_description.csv"))


water_0_dir = Path("data/fixed_data/output_test/0_water")
alcohol_5_dir = Path("data/fixed_data/output_test/5_alcohol")
alcohol_12_5_dir = Path("data/fixed_data/output_test/12.5_alcohol")
alcohol_25_dir = Path("data/fixed_data/output_test/25_alcohol")
alcohol_50_dir = Path("data/fixed_data/output_test/50_alcohol")
alcohol_75_dir = Path("data/fixed_data/output_test/75_alcohol")
alcohol_96_dir = Path("data/fixed_data/output_test/96_alcohol")
ResultTest_dir = Path("data/fixed_data/tmp_csv_test")

df_0_water = pd.concat([pd.read_csv(f)
                       for f in water_0_dir.glob("*.csv")], ignore_index=True)
df_0_water.drop(columns=df_0_water.columns[0], axis=1, inplace=True)
df_0_water.to_csv("data/fixed_data/tmp_csv_test/0_water.csv", index=False)

df_5_alcohol = pd.concat([pd.read_csv(f)
                         for f in alcohol_5_dir.glob("*.csv")], ignore_index=True)
df_5_alcohol.drop(columns=df_5_alcohol.columns[0], axis=1, inplace=True)
df_5_alcohol.to_csv("data/fixed_data/tmp_csv_test/96_alcohol.csv", index=False)

df_12_5_alcohol = pd.concat(
    [pd.read_csv(f) for f in alcohol_12_5_dir.glob("*.csv")], ignore_index=True)
df_12_5_alcohol.drop(columns=df_12_5_alcohol.columns[0], axis=1, inplace=True)
df_12_5_alcohol.to_csv(
    "data/fixed_data/tmp_csv_test/12.5_alcohol.csv", index=False)

df_25_alcohol = pd.concat(
    [pd.read_csv(f) for f in alcohol_25_dir.glob("*.csv")], ignore_index=True)
df_25_alcohol.drop(columns=df_25_alcohol.columns[0], axis=1, inplace=True)
df_25_alcohol.to_csv(
    "data/fixed_data/tmp_csv_test/25_alcohol.csv", index=False)

df_50_alcohol = pd.concat(
    [pd.read_csv(f) for f in alcohol_50_dir.glob("*.csv")], ignore_index=True)
df_50_alcohol.drop(columns=df_50_alcohol.columns[0], axis=1, inplace=True)
df_50_alcohol.to_csv(
    "data/fixed_data/tmp_csv_test/50_alcohol.csv", index=False)

df_75_alcohol = pd.concat(
    [pd.read_csv(f) for f in alcohol_75_dir.glob("*.csv")], ignore_index=True)
df_75_alcohol.drop(columns=df_75_alcohol.columns[0], axis=1, inplace=True)
df_75_alcohol.to_csv("data/fixed_data/tmp_csv_test/75_alchol.csv", index=False)

df_96_alcohol = pd.concat(
    [pd.read_csv(f) for f in alcohol_96_dir.glob("*.csv")], ignore_index=True)
df_96_alcohol.drop(columns=df_96_alcohol.columns[0], axis=1, inplace=True)
df_96_alcohol.to_csv(
    "data/fixed_data/tmp_csv_test/96_alcohol.csv", index=False)

df_ResultTest = pd.concat(
    [pd.read_csv(f) for f in ResultTest_dir.glob("*.csv")], ignore_index=True)
df_ResultTest.to_csv("data/fixed_data/ResultTest_FixedData.csv", index=False)
