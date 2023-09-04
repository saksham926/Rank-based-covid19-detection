A Multi-stage Framework for COVID-19 Detection and Severity Assessment from Chest Radiography Images using Advanced Fuzzy Ensemble Technique

We have added all the lung segmentation, disease classification and infection segmentation models folder wise in the google drive. Please find the link: https://drive.google.com/drive/folders/1Vo-J9wIWX7cPcZJjyl4__jO609HDlvBe?usp=sharing. We have added a small test dataset also to test the models. 

First run the best performing segmentation models () using this file on the test dataset and save the output then load these three classification models and run this script. For infection segmentation use these models 

The classification models folders contain three different experiment folders that we performed during the result analysis. First folder contains the models with full-fine tuning, the second folder contains partial fine-tuning models, and the third folder contains models with custom layer fine-tuning only. 

To run the models on the given sample dataset use this code

