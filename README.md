# Gender and Speaker Classification from Audio Files

## Overview

This project aims to perform gender and speaker classification from audio files using various machine learning techniques. The tasks involve feature extraction from audio signals, data analysis, and application of different classifiers to achieve the desired results. The project is divided into two main parts: Gender Classification and Speaker Classification.

## Repository Structure

- `solution.ipynb`: The main Jupyter Notebook containing all the code and analysis for the project.
- `data/`: Directory containing the audio data used in the project (not included in the repository; instructions for downloading are provided).
- `README.md`: Project documentation.

## Part 1: Gender Classification

1. **Data Acquisition**:
   - Downloaded the `dev-clean` corpus from the LibriSpeech dataset.
   - Selected a subset of the dataset containing audio files from at least 18 speakers (9 male and 9 female).

2. **Feature Extraction**:
   - Extracted Mel-Frequency Cepstral Coefficients (MFCC) features from the audio files using the `librosa` library.

3. **Data Analysis**:
   - Performed exploratory data analysis (EDA) to understand feature distributions and detect outliers.
   - Applied normalization techniques to prepare data for classification.

4. **Model Training**:
   - Split the dataset into training and testing sets.
   - Implemented and evaluated four classifiers:
     - Gradient Boosting algorithm 
     - Neural Network
     - Convolutional Neural Network (CNN, Deep Learning Model)
     - Hugging Face model
   - Measured and reported the accuracy, precision and recall of each model and compared the respective outcomes.

## Part 2: Speaker Classification

1. **Wav2Vec2.0 Feature Extraction**:
   - Downloaded and used the Wav2Vec2.0 base model to extract features from the audio signals.
   - Processed audio files in 2-second chunks to increase the number of samples.

2. **Quantitative Analysis**:
   - Implemented prediction models using the extracted features such as:
       - Gaussian Mixture Model
       - Neural Network
   - Evaluated the models using accuracy metrics.

3. **Qualitative Analysis**:
   - Applied dimensionality reduction and clustering algorithms:
        - Performed PCA on the features provided
        - Used K-means clustering
   - Visualized the clustering results to analyze the performance qualitatively.

## Results and Discussion

- Provided detailed analysis and interpretation of the results for each classifier.
- Explained the performance differences among the algorithms.
- Highlighted the strengths and weaknesses of each approach.

## How to Run

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/cat3699/Data-Science-Cbranco.git
   cd Data-Science-Cbranco
