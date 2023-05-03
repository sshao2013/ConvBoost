# ConvBoost

Official pytorch implementation of "ConvBoost: Boosting ConvNets for Sensor-based Activity Recognition". (accepted at Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 2023)

This work proposes ConvBoost – a novel, three layer, structured model architecture and boosting framework for convolutional network based HAR. Our framework generates additional training data from three different perspectives for improved HAR, aiming to alleviate the shortness of labeled training data in the field

This code provides an implementation for ConvBoost. This repository is implemented using PyTorch and it includes code for running experiments on OPP, Pamap2, GOTOV datasets.

## Dataset Download
* [OPP Dataset](https://drive.google.com/drive/folders/1X0PHkBXWP3Td08kxwAxNBOlFjeBZkjlv?usp=sharing)
* [PAMAP2 Dataset](https://drive.google.com/drive/folders/1X0PHkBXWP3Td08kxwAxNBOlFjeBZkjlv?usp=sharing)
* [GOTOV Dataset](https://data.4tu.nl/articles/dataset/GOTOV_Human_Physical_Activity_and_Energy_Expenditure_Dataset_on_Older_Individuals/12716081)

## Set up environment
To ensure the experiments run smoothly, please create a **python 3.8** environment.

## Running Experiments
you could run a baseline model by:

    python main.py --dataset=1 --model='CNN-3L'

To run the CovBoost model by:

    python main.py --dataset=1 --model='CNN-3L'  --epoch_wise=True

More configuration settings can be found in the file 'cfg_data.py'

## Citation
If you found this paper is helpful and like it. Please don't mind citing it and thank you.
```
todo

```
