# Rice-image-classification

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tensorflow)
![GitHub last commit](https://img.shields.io/github/last-commit/Haliacal/Rice-Image-Classification)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/Haliacal/Rice-Image-Classification)
[![](https://tokei.rs/b1/github/Haliacal/Rice-Image-Classification?category=lines)](https://github.com/Haliacal/Rice-Image-Classification) 
![GitHub Repo stars](https://img.shields.io/github/stars/Haliacal/Rice-Image-Classification?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/Haliacal/Rice-Image-Classification?style=social)

## Summary

Predicting type of rice using the keras and tensorflow. Achieving a test accuracy: 0.9451 and a validation accuracy: 0.9299. Dataset had over 75000 images.

``` test.ipynb ```
* Importing data
* testing functions
* Creating different combinations of the model

``` main.py ```
* Finalising Model
* Options to save and load trained data

## Background

This is a rice image classifier using the [data](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset) set from Kaggle. I wanted to test and apply concepts within AI to create a convolution network. In the end I was able to apply data augmentation and dropout to prevent overfitting. Resizing the data also helped to fit the data and cut down on computation expenses. I went with a resizing of 28x28.
Since te data is quite simple, only 2 epochs was needed to achieve a good fit which can be seen in the results.

## Results
![Results](assets/output.png)

