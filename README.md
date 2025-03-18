# Multitask Prostate Segmentation and Cancer Classification Model Using Deep Learning

## Project Overview
This project aims to develop a dual-purpose model capable of both segmenting the prostate region from medical images and classifying the presence of prostate cancer. The multitask approach leverages deep learning techniques to streamline the workflow and improve disease detection capabilities.

## Dataset
- **Dataset:** PI-CAI
- **Number of Patients:** 673 anonymized patients
- **Patient Breakdown:** 352 patients diagnosed with prostate cancer
- **Data Modalities:**
  - ADC
  - HBV
  - T2W (middle slice ±1)

## Technologies Used
- **Python:** Main programming language used in the project.
- **Deep Learning Frameworks:** Likely using frameworks such as TensorFlow or PyTorch.
- **UNet Architecture:** Core deep learning model for segmentation and classification tasks.
- **Other Libraries and Tools:**
  - Numpy and Pandas for data manipulation.
  - OpenCV for image processing.
  - Scikit-learn for additional machine learning utilities and evaluations.

## Results (Overview)
- **Average Accuracy:** 73.8%
- **AUROC:** 83.5%
- **Recall:** 85.0%
- **Specificity:** 62.2%
*Note: These results were obtained without the segmentation feature being active during classification.*

## Getting Started
To get started with the project:
1. Clone the repository.
2. Set up your Python environment and install the required packages listed in `requirements.txt`.
3. Review the project structure to understand how data preprocessing, model training, and evaluation scripts are organized.
4. Run the experiments using the provided scripts and configurations.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request to discuss improvements or new features.

## License
Include the license information here.
