A Multi-stage Framework for COVID-19 Detection and Severity Assessment from Chest Radiography Images using Advanced Fuzzy Ensemble Technique

We have added all the lung segmentation, disease classification and infection segmentation models folder wise in the google drive. Please find the link: https://drive.google.com/drive/folders/1Vo-J9wIWX7cPcZJjyl4__jO609HDlvBe?usp=sharing. We have added a small test dataset also to test the models. 

First run the best performing segmentation models UNet, VGG16 UNet, and DenseNet121 UNet using segmentation_code_lung_segmentation_data.ipynb file on the test dataset and save the output, and then load these three classification models which include SE Inception V3, DenseNet201, and SE SqueezeNet and run this script Classification on segmented lung portions.py. For infection segmentation use these models UNet, VGG16 UNet, and DenseNet121 UNetagain using this code file segmentation_code_on_infection_segmentation_data.ipynb.

The classification models folders contain nine different models that we performed during the result analysis. Each model is executed with full-fine tuning, partial fine-tuning, and custom layer fine-tuning setting only. 

To run the models on the given sample dataset use this code Testing_code_classification_models.ipynb. 

