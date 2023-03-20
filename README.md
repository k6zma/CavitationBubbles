# Ultrasound research cavitation bubbles for determination of alcohol concentration in water-ethanol solutions with using methods of machine learning and neural networks.

## Problem:

Today there is a serious problem with determination of the concentration of alcohol in water-ethanol solutions, at the moment the most popular device for measuring the concentration of alcohol is a hydrometer (alcohol meter). A hydrometer is a fairly versatile thing, it is certainly a good device for measuring the density of liquids and solids, but unfortunately it copes rather poorly with determining the concentration of alcohol. The hydrometer problem is a high error, inconvenience of use due to the small number of divisions, and you also need to have a large volume of solution. That is why there is a need to create a new method, as well as a device for determining the concentration of alcohol.

## Targets

The aim of this project is to train machine learning models and neural networks using computer vision and multimodal data to quickly learn the concentration of alcohol in various solutions.

## Tasks

- To study the phenomenon of cavitation;
- Explore and filter the database;
- To study and select the necessary methods of image augmentation;
- Training various machine learning models;
- Creation and training of a neural network.

## Formulas for cavitation reserve

$$
F_g = {m_{gas}g}
$$

$$
F_b = {V(P_{liquid} - P_{bubble})g}
$$

$$
P = {P_{in} - p_{out} = {{2σ} \over R}}
$$

$$
P = {P_σ - P_0}
$$

## Preprocessing

For preprocessing we used computer vision methods
To work with machine learning models, you need to create csv tables with information about bubbles.
To collect information about the bubbles was used:

- Contour selection method for each cavitation bubble

The following data were selected as the data entered in the table:

- Cavitation bubble area in µm
- Concentration of water-ethanol solution
- Minimal distance from the center to the limits of the contour of the cavitation bubble
- Average distance from the center to the limits of the contour of the cavitation bubble
- Maximum distance from the center to the limits of the contour of the cavitation bubble
- Standard deviation

Example of collected dataframe:

| №   | Photo Name | Concentration | Area, µm      | Min distance | Mean distance | Max distance | Standart Deviation |
| --- | ---------- | ------------- | ------------- | ------------ | ------------- | ------------ | ------------------ |
| 0   | 41063      | 50%           | 1591,55973    | 0            | 1,2600        | 3            | 0,844038           |
| 1   | 30333      | 50%           | 347418,947967 | 0            | 42,416164     | 77           | 21,222681          |
| 2   | 40165      | 50%           | 295141,965856 | 0            | 12,321006     | 21           | 4,952039           |
| 3   | 40638      | 50%           | 57641,281449  | 0            | 1,298611      | 3            | 1,034676           |
| 4   | 40168      | 50%           | 30310,149762  | 0            | 4,853881      | 9            | 2,306754           |

## Maсhine Learning

KNeighborsClassifier - average model accuracy on test and validation data ~ 99%

Best Hyperparameters:

- metric: manhattan
- n_neighbors: 97
- weights: distance

GradientBoostingClassifier - average model accuracy on test and validation data ~ 99%

Best Hyperparameters:

- max_depth: 5
- max_features: sqrt
- min_samples_leaf: 0.1
- min_samples_split: 0.1
- n_estimators: 10
- subsample: 1.0

RandomForestClassifier - average model accuracy on test and validation data ~ 99%

Best Hyperparameters:

- bootstrap: True
- max_depth: 90
- min_samples_leaf: 1
- min_samples_split: 5
- n_estimators: 600

## Neural Network

ResNet50V2 with batch normalization was used to create a neural network. ResNet50V2 is a convolutional neural network that has 50 depth layers. These architectures have been trained on more than 1 million images from the ImageNet database and are able to classify more than 1000 types of objects in images (computer accessories, animals, plants, chemical elements, vehicles, air transport, bacteria, etc.). For this study, a neural network was created with two outputs: convolutional for images and linear for tables, and then these two outputs were combined into one.

## Сonclusions

- The most accurate machine learning model for determining concentration was KNeighborsClassifier;
- The accuracy of the neural network on test and validation data was 99.8%;
- To solve the regression problem, there is a need to increase the database;
- A single model was created, consisting of the KNeighborsClassifier model and a neural network;
- Such results were obtained due to the fact that we used an image classification method, and not regression. Now we can only predict 7 concentrations, but if we increase the database and add more concentrations, then we can solve the regression problem to determine a larger range of concentration values, and it will be possible to predict the concentration of the solution with an accuracy of tenths, of course, the accuracy of the model will decrease, but it decrease by ~2% - 3%. I also wanted to say that based on this study, we can understand that neural networks and machine learning models can be used to study other various solutions, by studying the cavitation bubbles of this solution.

## Future Plans

- Expanding the database to determine the alcohol concentration of a larger range;
- Creation of an expansion board and development of software for a high-speed camera to determine in real time the concentration of alcohol, the radius and area of the bubble;
- Improvement of the neural network to work with solutions of other substances, to reduce the number of accidents associated with cavitation bubbles.
