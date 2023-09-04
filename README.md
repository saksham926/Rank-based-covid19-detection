Paper: "Multi-stage Framework for COVID-19 Detection and Severity Assessment from Chest Radiography Images using Advanced Fuzzy Ensemble Technique"
Author List: Pranab Sahoo, Sriparna Saha, Saksham Kumar Sharma, Samrat Mondal, and Suraj Gowda

We have added all the lung segmentation, disease classification, and infection segmentation models folder-wise to Google Drive. Please find the link: https://drive.google.com/drive/folders/1Vo-J9wIWX7cPcZJjyl4__jO609HDlvBe?usp=sharing. We have added a small test dataset to test the classification models. The classification model folders contain 9 different models that we performed during the result analysis. Each model is trained with full-fine tuning, partial fine-tuning, and custom layer fine-tuning settings. To test the results of the pre-trained classification models, use the “Testing_code_classification_models.ipynb” notebook file to load the models and use the “Test_data.zip” dataset.

To validate the models on the private dataset, first, use the “Testing_code_classification_models.ipynb” notebook, load the three classification models, and give the private dataset path. 

The "Lung Segmentation Models" folder contains 9 pre-trained segmentation models developed using "segmentation_code_lung_segmentation_data.ipynb" notebook. Next, select the three best-performing segmentation models (ex., in our experiment, UNet, VGG16UNet, and DenseNet121UNet) and perform lung segmentation ensemble using "Ensembling_results.ipynb" on the test dataset and save the outputs. Load three classification models (SE Inception V3, DenseNet201, and SE SqueezeNet) and execute "Classification on segmented lung portions.py" to get the individual classification accuracy. To ensemble the model predictions using Rank-based ensemble technique, first, save the model outputs and execute the code "Ensembling_results.ipynb". For infection segmentation use the three best-performing infection segmentation models (UNet, ResNet50UNet, and DenseNet121UNet) using the code "segmentation_code_on_infection_segmentation_data.ipynb". 

To develop the proposed architecture, first use notebook "segmentation_code_lung_segmentation_data.ipynb" for lung segmentation, followed by lung segmentation region ensemble using the "Segmentation_ensembling.ipynb" notebook. Use those segmented lung regions as input to the three classification models using the "Classification on segmented lung portions.py" code and save the predictions. Use these predicted values for the rank-based ensemble prediction using "Ensembling_results.ipynb". Once the output is identified as COVID-19, use "Segmentation_ensembling.ipynb" for the infection segmentation and severity assessment, followed by an ensemble of infected regions. 

**Prerequisites:** \
numpy                        1.21.0
tensorflow                   2.9.1
keras                        2.9.0
opencv-python                54



