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
