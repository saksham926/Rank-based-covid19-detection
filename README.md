Paper: "Multi-stage Framework for COVID-19 Detection and Severity Assessment from Chest Radiography Images using Advanced Fuzzy Ensemble Technique"

We have added all the lung segmentation, disease classification and infection segmentation models folder wise in the google drive. Please find the link: https://drive.google.com/drive/folders/1Vo-J9wIWX7cPcZJjyl4__jO609HDlvBe?usp=sharing. We have added a small test dataset also to test the models. 

The "Lung Segmentation Models" folder contain 9 pre-trained segmentation models. First,  use the three best performing segmentation models (ex., in our experiment UNet, VGG16UNet, and DenseNet121UNet) using "segmentation_code_lung_segmentation_data.ipynb" file on the test dataset and save the outputs, and then load the three classification models (SE Inception V3, DenseNet201, and SE SqueezeNet) and execute "Classification on segmented lung portions.py" to get the individual classification accuracy. To ensemble the models, first save the models and then run the code "Ensembling_results.ipynb" to ensemble the results using Rank-based ensemble technique. For infection segmentation use (UNet, ResNet50UNet, and DenseNet121UNet) using the code "segmentation_code_on_infection_segmentation_data.ipynb". 

The classification models folders contain nine different models that we performed during the result analysis. Each model is executed with full-fine tuning, partial fine-tuning, and custom layer fine-tuning setting. To run the models on the given sample test dataset use the code "Testing_code_classification_models.ipynb". 

