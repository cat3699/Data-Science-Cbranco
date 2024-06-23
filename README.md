# Gender and Speaker Classification from Audio Files

# Introduction

This project aims to perform gender and speaker classification from audio files using various machine learning techniques. The tasks involve feature extraction from audio signals, data analysis, and application of different classifiers to achieve the desired results. The project is divided into two main parts: Gender Classification and Speaker Classification. 

The data chosen to do this project is part of the LibriSpeech that can be donwload [here](https://www.openslr.org/12/). More specificaly we used the dev-clean corpus, which is relatively small in size. This database contains directories with ids. Each id represents one speaker and contains audio files associated with that speaker. You will also find a file that indicates
the gender of each speaker.


# Tools Used

- **Programming Language**: Python
- **Libraries**:
  - `librosa` for audio processing and feature extraction
  - `numpy`, `pandas`, `matplotlib`, and `seaborn` for data analysis and visualization
  - `scikit-learn` for traditional machine learning models
  - `tensorflow` and `keras` for deep learning models
  - `fairseq` for Wav2Vec2.0 model
- **Jupyter Notebook** for coding and presenting the analysis

- **Git & GitHub**: Essential for version control and sharing my Jupyter Notebook with all the analysis and implementation of my project.


# Analysis

## Part 1: Gender Classification

1. **Data Acquisition**:
   - Downloaded the `dev-clean` corpus from the LibriSpeech dataset.
   - Selected a subset of the dataset containing audio files from at least 18 speakers (9 male and 9 female).

2. **Feature Extraction**:
   - Extracted Mel-Frequency Cepstral Coefficients (MFCC) features from the audio files using the `librosa` library.

````python
def extract_feat(filename):
    
   # Load audio using librosa_load
    audio, sample_rate = librosa_load(filename)  

    # Extract MFCC features
    mfcc_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)

    return mfcc_features
````

3. **Data Analysis**:
   - Performed exploratory data analysis (EDA) to understand feature distributions and detect outliers using z-score and the Interquile Range (IQR) method
   ### Z-Score Mehtod 
````python
    def calculate_zcore(df):

    # Extract MFCC column names
    mfcc_columns = [f'MFCC_{i}' for i in range(13)]

    # Extract MFCC data from the DataFrame
    mfcc_data = df[mfcc_columns]

    # Calculate z-scores for each MFCC dimension
    z_scores = (mfcc_data - mfcc_data.mean()) / mfcc_data.std()

    # Define a threshold for outlier detection (e.g., z-score > 3)
    threshold = 3

    # Mark outliers (True if the z-score exceeds the threshold, indicating an outlier)
    outliers = np.abs(z_scores) > threshold

    # Print the count of outliers for each MFCC dimension
    print("Count of outliers for each MFCC:")
    print(outliers.sum())

    # Mark rows with any outlier (True if any MFCC dimension is an outlier)
    df['Is_Outlier'] = outliers.any(axis=1)

    # Filter the DataFrame to exclude rows with outliers
    df_filtered = df[~df['Is_Outlier']]
````
### Interquartile Range (IQR) method
````python
    def calculate_outliers_iqr(df):

    # Select only the MFCC columns
    mfcc_columns = [f'MFCC_{i}' for i in range(13)]
    mfcc_data = df[mfcc_columns]

    # Calculate the first quartile (Q1)
    Q1 = mfcc_data.quantile(0.25)

    # Calculate the third quartile (Q3)
    Q3 = mfcc_data.quantile(0.75)

    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outlier detection
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Detect outliers
    outliers = (mfcc_data < lower_bound) | (mfcc_data > upper_bound)

    # Print the count of outliers for each MFCC dimension
    print("Count of outliers for each MFCC:")
    print(outliers.sum())

    # Add a new column to the DataFrame to indicate if a row is an outlier
    df['Is_Outlier'] = outliers.any(axis=1)

    # Filter the DataFrame to exclude rows with outliers
    df_filtered = df[~df['Is_Outlier']]

````


   - Applied normalization techniques to prepare data for classification using Global cepstral mean and variance normalization(CMVN).
````python
   def perform_cmvn(df, mfcc_columns):

    df_normalized = df.copy()  # Create a copy to avoid modifying the original DataFrame

    for col in mfcc_columns:
        # Calculate mean and variance for the current column
        mean = df[col].mean()
        variance = df[col].var()

        # Perform mean normalization
        df_normalized[col] = df[col] - mean

        # Perform variance normalization
        epsilon = 1e-10  # Small constant to prevent division by zero
        df_normalized[col] = df_normalized[col] / (np.sqrt(variance) + epsilon)

    return df_normalized
````
4. **Model Training**:
- Split the dataset into training and testing sets. The data was divided in 80% train and 20% test ensuring that we have an equal representation between the two in both sets, if not we should look for another set.

- Implemented and evaluated four classifiers:
### Gradient Boosting algorithm

````python
# Import necessary library for Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

# Train a Gradient Boosting model
gbrt = GradientBoostingClassifier(random_state=0).fit(X_train, y_train_gb)
```` 
### Neural Network

````python
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.callbacks import EarlyStopping
from keras import backend as K

class NeuralNetwork:
    """
    Adaptation of a Neural Network found on GitHub for gender speech recognition

    Attributes:
        model (Sequential): Keras Sequential model for the baseline.
        early_stop (EarlyStopping): Early stopping to prevent overfitting.

    Methods:
        compile_model: Compiles the baseline model.
    """

    def __init__(self):
        """
        Initializes the baseline model.
        """
        self.model = Sequential()
        self.model.add(Dense(13, input_shape=(13,), activation='relu'))
        self.model.add(BatchNormalization())  # Batch normalization after the first dense layer
        self.model.add(Dense(128, activation='relu'))
        self.model.add(BatchNormalization())  # Batch normalization after the second dense layer
        self.model.add(Dense(64, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(2, activation='softmax'))

        self.early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=200, verbose=1, mode='auto')

    def compile_model(self):
        """
        Compiles the baseline model.

        Returns:
            None
        """
        self.model.compile(loss='categorical_crossentropy', metrics=['accuracy', recall_m, precision_m], optimizer='rmsprop')
```` 

## Convolutional Neural Network (CNN, Deep Learning Model)
````python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class GenderClassificationCNN:
    """
    Convolutional Neural Network (CNN) model for gender classification.

    Attributes:
        model (Sequential): Keras Sequential model for CNN-based gender classification.

    Methods:
        compile_model: Compiles the CNN model.
    """

    def __init__(self):
        """
        Initializes the CNN model for gender classification.
        """
        self.model = Sequential()
        self.model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(13, 1)))
        self.model.add(Conv1D(filters=48, kernel_size=3, activation='relu'))
        self.model.add(Conv1D(filters=120, kernel_size=3, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(2, activation='softmax'))

    def compile_model(self):
        """
        Compiles the CNN model.

        Returns:
            None
        """
        optimizer = Adam()  # Learning rate: 0.001
        self.model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy',recall_m,precision_m])
````
## Hugging Face model

````python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Wav2Vec2FeatureExtractor
from src.models import Wav2Vec2ForSpeechClassification, HubertForSpeechClassification

import librosa
import IPython.display as ipd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name_or_path = "m3hrdadfi/hubert-base-persian-speech-gender-recognition"
config = AutoConfig.from_pretrained(model_name_or_path)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
sampling_rate = feature_extractor.sampling_rate
model = HubertForSpeechClassification.from_pretrained(model_name_or_path).to(device)
````

- Measured and reported the accuracy, precision and recall of each model and compared the respective outcomes.

## Part 2: Speaker Classification

1. **Wav2Vec2.0 Feature Extraction**:

- Downloaded and used the Wav2Vec2.0 base model to extract features from the audio signals.

````python
import torch
import fairseq
import torchaudio

wav2vec2_checkpoint_path = '/content/drive/MyDrive/wav2vec_small.pt'
checkpoint = torch.load(wav2vec2_checkpoint_path)
cfg = fairseq.dataclass.utils.convert_namespace_to_omegaconf(checkpoint['args'])
wav2vec2_encoder = fairseq.models.wav2vec.Wav2Vec2Model.build_model(cfg.model)
wav2vec2_encoder.load_state_dict(checkpoint['model'])
````
- Processed audio files in 2-second chunks to increase the number of samples.

````python
def get_feat_labels_wav2vec(wav2vec2_encoder, file_path):
  
  dir = '/content/drive/MyDrive/dev-clean'
  speakers = '/content/drive/MyDrive/SPEAKERS.TXT'
  id_speaker,gender_speaker=get_speaker_id_gender(speakers)

  # List of column names for the DataFrame
  column_names = ['Features', 'Speaker_ID', 'Gender','Audio']

  # Number of samples in a 2 seconds clip (2*sample_rate(16Khz))
  num_samples_per_chunk = 32000

  # Initialize an empty DataFrame with the specified column names
  df_final = pd.DataFrame(columns=column_names)

  for index, id in enumerate(id_speaker):

    dir_audios = dir + '/' + id + '/*/*.flac'
    file_list = glob.glob(dir_audios)
    audio_files = []

    for audio in file_list:

      #load audio
      waveform, sample_rate = torchaudio.load(audio)

      #Check if audio is shorter than 2 seconds
      if waveform.shape[1] < 32000:
        chunks = waveform
      else:
        # Unfold the waveform into subsequent 2-second chunks
        chunks = waveform.unfold(1, num_samples_per_chunk, num_samples_per_chunk)
        chunks = chunks.squeeze(0)

      # Iterate over the chunks and apply the model to each chunk
      for i in range(chunks.size(0)):
          chunk = chunks[i]

          # Ensure the chunk has the right size (num_samples_per_chunk samples)
          if chunk.size(0) < num_samples_per_chunk:
              padding = torch.zeros(num_samples_per_chunk - chunk.size(0))
              chunk = torch.cat((chunk, padding), dim=0)

          # Extract features for the chunk
          features_wav2vec =  wav2vec2_encoder(chunk.unsqueeze(0), features_only=True, mask=False)['x']
          features_np = features_wav2vec.detach().numpy()

          df_features = pd.DataFrame({'Features': [features_np]})

          # Add speaker ID, gender, and audio to df_features
          df_features['Speaker_ID'] = id
          df_features['Gender'] = gender_speaker[index]
          df_features['Audio'] = audio

        # Concatenate the DataFrames
          df_final = pd.concat([df_features, df_final])
````

2. **Quantitative Analysis**:
- Implemented prediction models using the extracted features such as:
## Gaussian Mixture Model
````python
from sklearn.mixture import GaussianMixture

def train_gmm(gmm_models_file_path, unique_speakers, X_train_wav2vec, y_train_wav2vec):
    """
    Trains Gaussian Mixture Models (GMM) for each unique speaker in the dataset and saves the models to a file.

    Parameters:
    gmm_models_file_path (str): File path to save the GMM models.
    unique_speakers (list): List of unique speaker identifiers.
    X_train_wav2vec (numpy.ndarray): Feature matrix containing training data (wave2vec features).
    y_train_wav2vec (numpy.ndarray): Labels corresponding to the training data.

    Returns:
    None
    """
    gmm_models = {}

    for speaker_id in unique_speakers:
        print("Training GMM for speaker:", speaker_id)
        # Select features for the current speaker
        speaker_features = np.vstack(X_train_wav2vec[y_train_wav2vec == speaker_id])

        # Initialize and fit Gaussian Mixture Model
        gmm = GaussianMixture(n_components=16, covariance_type='diag', n_init=3)
        gmm.fit(speaker_features)
        gmm_models[speaker_id] = gmm

    # Save the trained GMM models to a file
    with open(gmm_models_file_path, 'wb') as gmm_models_file:
        pickle.dump(gmm_models, gmm_models_file)

    print('GMM models saved to', gmm_models_file_path)
````
## Neural Network
````python
from keras.optimizers import SGD

class NeuralNetworkClassifier:
    
    def __init__(self, input_dim, output_dim):
      """
      Initializes the classifier model.

      Parameters:
      input_dim (int): Dimension of the input features.
      output_dim (int): Dimension of the output labels.
      """
      self.model = Sequential()
      self.model.add(Dense(64, input_dim=input_dim, kernel_initializer='glorot_normal'))
      self.model.add(BatchNormalization())
      self.model.add(Activation('tanh'))
      self.model.add(Dense(250, kernel_initializer='glorot_normal'))
      self.model.add(BatchNormalization())
      self.model.add(Dropout(0.5))
      self.model.add(Dense(output_dim, kernel_initializer='glorot_normal'))
      self.model.add(Activation('softmax'))


    def compile_model(self):
      """
      Compiles the classifier model.

      Returns:
          None
      """
      sgd = SGD(learning_rate=0.1, momentum=0.9, nesterov=True)
      self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
````

- Evaluated the models using accuracy metrics.

3. **Qualitative Analysis**:
- Applied dimensionality reduction and clustering algorithms:
## Performed PCA 

````python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def pca_for_audio_features(X_wav2vec):
    
    # Initialize a list to store transformed features for each audio file
    transformed_features_list = []
    num_comp = []

    for features_audio_file in X_wav2vec:
        # Step 1: Standardize the features for the current audio file
        scaler = StandardScaler()
        features_standardized = scaler.fit_transform(features_audio_file)

        # Step 2-4: Perform PCA for the current audio file
        pca = PCA()
        features_pca = pca.fit_transform(features_standardized)

        # Calculate explained variance ratio for the current audio file
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

        # Find the number of components needed for 85% variance
        num_components_85_variance = np.argmax(cumulative_variance_ratio >= 0.85) + 1
        num_comp.append(num_components_85_variance)

        # Step 5: Select desired number of components for the current audio file
        k_components = num_components_85_variance
        features_pca_selected = features_pca[:, :num_components_85_variance]

        # Append the transformed features for the current audio file to the list
        transformed_features_list.append(features_pca_selected)

    return transformed_features_list, num_comp

````

## K-means clustering
````python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Reduce dimensionality using PCA
pca = PCA(n_components=47)  # Initialize PCA with 47 components
xpca_path = '/content/drive/MyDrive/x_pca.pickle'  # Path to save/load PCA-transformed data
num_clusters = 40  # Choose the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)  # Initialize KMeans clustering

# Check if PCA-transformed data exists at the specified path
if os.path.exists(xpca_path):
  # Load PCA-transformed data from the saved pickle file
  with open(xpca_path, 'rb') as f:
    x_pca = pickle.load(f)
else:
  # If the PCA-transformed data doesn't exist, compute it
  X = np.concatenate(X_wav2vec).reshape(-1, 768)  # Concatenate and reshape input features
  x_scale = StandardScaler().fit_transform(X)  # Standardize the features
  x_pca = pca.fit_transform(x_scale)  # Perform PCA and obtain the transformed data

````
- Visualized the clustering results to analyze the performance qualitatively.

# Results 
## Task 1: Count of outliers
| Method                                     | MFCC_0 | MFCC_1 | MFCC_2 | MFCC_3 | MFCC_4 | MFCC_5 | MFCC_6 | MFCC_7 | MFCC_8 | MFCC_9 | MFCC_10 | MFCC_11 | MFCC_12 |
|---------------------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|---------|---------|
| Z-score                 |   80   |  2332  |  1857  |  1943  |  3455  |  2672  |  1558  |  2619  |  1640  |  2359  |   3514  |   2916  |   3279  |
| Interquartile Range (IQR)|  4607  |  7940  |  6680  |  8135  |  7602  |  7054  |  3061  |  5477  |  3817  |  5391  |   8252  |   7826  |   9336  |

## Task 1: Classifier Performance Results

| Classifier                      | Evaluation Type | Accuracy      | Precision     | Recall        |
|---------------------------------|-----------------|---------------|---------------|---------------|
| **Gradient Boosting**           | Per audio       | 0.901845      | 0.909161      | 0.888638      |
|                                 | Per frame       | 0.752792      | 0.758723      | 0.736892      |
| **Neural Network**              | Per audio       | 0.998155      | 0.997744      | 0.998495      |
|                                 | Per frame       | 0.896041      | 0.884290      | 0.909866      |
| **Convolutional Neural Network**| Per audio       | 0.998893      | 0.999247      | 0.998495      |
|                                 | Per frame       | 0.907791      | 0.905460      | 0.909404      |
| **Hugging Face Model**          | Per audio       | 0.918081      | 0.902545      | 0.933785      |

## Task 2: Speaker Prediction task Performance Results

| Classifier                      | Evaluation Type             | Accuracy       |
|---------------------------------|-----------------------------|----------------|
| **Gaussian Mixture Model**      | Speaker prediction          | 0.803544       |
| **Neural Network**              | Test data (unseen before)   | 0.747152       |

## Task 2: PCA 

![Histogram of number of components obtained with PCA](images\pca_results.PNG)


## Task 2: K-means clustering

![K-means clustering with the first two PCA features](images\kmeans.PNG)


# Conclusions

## Task 1: Outlier methods

The counts of outliers vary notably between the z-score and IQR methods for each MFCC, indicating differing sensitivity levels to outliers. Nevertheless, certain coefficients consistently exhibit a high number of outliers across both methods. This suggests that these coefficients may have a lower impact when training a model.

## Task 1: Prefered classifier for classifyng gender based on audio files



**Introduction**

The four chosen algorithms for performing gender classification were: the Gradient Boosting algorithm available in the scikit-learn library; a Neural Network adapted from a repository found on GitHub that performed the same task; a CNN inspired by the following paper; and finally, we used a hubert-base-persian-speech-gender-recognition model available on Hugging Face.

For the first three models, we trained the model on frames and evaluated both on a per-frame basis and per audio (determining the gender of the person in the audio by considering the majority prediction of the frames). The last model was evaluated only at the audio level.


- Gradient Boosting algorithm

This model was selected due to its ease of implementation and minimal computational requirements. The primary aim was to assess accuracy in our dataset using a straightforward approach. In a frame-based evaluation, the model achieved an accuracy of 0.75, a precision of 0.76, and a recall of 0.74. From an audio-based perspective, the model achieved an accuracy of 0.90, a precision of 0.91, and a recall of 0.89.

It is evident from these scores that there was a significant enhancement in model performance when evaluating on both a per-frame and per-audio basis. This stresses the importance of the evaluation strategy in understanding the model's behavior. In both evaluation scenarios, the model exhibited substantial progress, indicating its potential for accurate gender classification. Required around 8 min to compile.


* Neural Network

We adapted a Neural Network sourced from GitHub, initially designed with three layers and only one hidden layer. To enhance its complexity, we added an additional hidden layer and incorporated batch normalization for improved performance assessment. However, due to its increased complexity, this model required a longer compilation time, approximately 30 minutes.

From a frame evaluation perspective, the model achieved an accuracy of 0.896, precision of 0.88, and recall of 0.91. On an audio-level evaluation, the model excelled with an accuracy, precision, and recall of 0.998, showcasing near-perfect performance.

Comparatively, the performance of this model was notably superior, particularly from an audio perspective. The significant improvement in performance is evident when comparing the results at the frame and audio levels. Moreover, this model demonstrated exceptional audio-level performance, emphasizing its potential for accurate gender classification. Notably, the computational demands for this model remained manageable despite its enhanced performance.


*   CNN Model

We implemented a CNN based on the model described in 'DGR: Gender Recognition of Human Speech Using One-Dimensional Conventional Neural Network' by Rami S. Alkhawaldeh. The author reported a remarkable recall of 99.7% in this paper, making it a promising candidate for testing on our dataset. In comparison to the previous two models, this CNN exhibited significantly higher complexity. Training the model for 200 epochs took approximately 3 to 4 hours. It's worth noting that the author trained for 1000 epochs, but due to computational limitations, we opted to reduce the number of epochs.

In terms of frame evaluation, the model achieved an accuracy of 0.908, a precision of 0.905, and a recall of 0.91. When evaluated at the audio level, the model displayed exceptional performance with an accuracy of 0.998, a precision of 0.999, and a recall of 0.998, approaching near-perfection.

Comparatively, significant improvements were observed in both frame-based and audio-level evaluations. However, when comparing with the previously discussed Neural Network, there was a slight improvement from a frame perspective, and the audio-level performance was comparable. Considering the computational power required for this model, which was around 10 times more in terms of time than the Neural Network, it may not be the most efficient choice for this particular task.



*   Hugging Face model

This model is fundamentally different from the previous ones. Unlike the previous models, the features fed into this model were not MFCC features, and it was not trained and tested at a frame level. We chose this model because of its simplicity and its availability on Hugging Face. The model is notably user-friendly in terms of implementation. While the model architecture itself is not inherently simple, its ease of use for predicting gender based on audio makes it accessible and straightforward. Moreover, we saw an opportunity to assess the model on an English dataset, allowing us to investigate whether gender recognition is language-dependent or not, since the model is pre-trained in persian.

It took approximately 2 hours to predict the gender for all the audio samples. The model achieved an accuracy of 0.92, a recall of 0.9, and a precision of 0.93. We observed a decrease in performance compared to the Neural Network and the CNN, but a slight increase compared to the gradient boosting algorithm. Nevertheless, the computational weight required is comparable to the CNN. Consequently, its implementation does not justify its usage. However, it did demonstrate that gender recognition is language-dependent in this case. Additionally, its implementation is far less complex than the other models.




**Conclusion**

The best-performing model in terms of both results and training time was the Neural Network. Despite being less complex than the CNN and the Hugging Face model, its architecture combined with the extracted MFCCs proved effective in capturing crucial patterns and features from the dataset for gender recognition.


## Task 2: Comments on the results obtained in the prediction task.

- **GMM model**

The gaussian mixture models are commonly used in the task of speaker classification. In this case this implementation utilizes GMM, a probabilistic model, for speaker identification, where for each speaker in the dataset a GMM is trained, modeling their speech features. Each GMM is initialized with 16 components and a diagonal covariance matrix for each component. This resulted in a accuracy of 0.803,suggesting that the GMM approach is promising for speaker identification. Fine-tuning the GMM parameters, such as the number of components or the covariance type, may lead to potential improvements in accuracy.

-**Neural Network**

In the second model, we initiated from a neural network architecture obtained from GitHub and extended it by adding an additional hidden layer, incorporating batch normalization, and training with a batch size of 32. Surprisingly, this augmented model achieved an accuracy of 0.747, which is actually lower than the accuracy attained by the GMM approach. One would anticipate that a more complex model would yield better accuracy, but it's highly probable that the chosen parameters were not the most suitable for the given task. In this case, the GMM, being simpler in design, performed better due to its appropriateness for the task.

For this case further exploration of diverse architectures for this task is essential. Additionally, employing a pre-trained ResNet model, fine-tuned to the dataset with mel-spectrogram features, could potentially yield improved performance. Research suggests that such an approach has exhibited notable success in various audio classification tasks.


## Task 2: PCA

Due to the substantial volume of data, Principal Component Analysis (PCA) was performed on each audio file to determine the number of components necessary to account for 85% of the variance in the data.

The resulting values were aggregated and averaged, revealing that approximately 47 components are required to explain the variance in the dataset. Moreover, a graphical representation of these components for each audio clip demonstrated that the majority of audio clips necessitate between 40 and 55 components. This highlights the diverse characteristics and complexities associated with each speaker, reinforcing the challenge of speaker classification.

Interestingly, a subset of audio clips required only around 20 components. This suggests that these particular clips may represent unique vocal patterns, distinct accents, or specific speech characteristics associated with particular speakers.

Overall, the finding that approximately 47 components are sufficient to explain 85% of the variance in the data is a noteworthy advancement in computational efficiency compared to the initial 768 features extracted using the wav2vec encoder.

## Task 2: K-means clustering 

In this scenario, we utilized the 47 PCA components to reduce the dimensionality of the features, thus making the computational demands more manageable. Subsequently, I employed 40 clusters, each corresponding to a specific speaker, and applied the k-means algorithm to visualize these clusters based on the first two PCA features.

From the visual representation shown in the image above, it becomes evident that relying solely on these two features is insufficient to accurately identify the speaker. However, some clusters appear distinguishable, suggesting that there are speakers with distinct voices for which two features might suffice. This further aligns with our findings from the PCA analysis, which indicated the need for a minimum of 47 features to accurately represent the data.

Regrettably, visualizing the clusters in all 47 dimensions is impractical. Nevertheless, our plot provides a valuable visualization, offering insights into the nature of the data under analysis.

# What I Learned

- **Feature Extraction**: Understanding the importance of MFCC features in audio classification tasks.
- **Model Comparison**: Gaining insights into how different models perform on the same task and the trade-offs between traditional machine learning models and deep learning models.
- **Advanced Models**: Learning to use advanced models like Wav2Vec2.0 for feature extraction and understanding their impact on classification tasks.
- **Data Analysis**: Improving skills in data analysis and visualization to interpret and present results effectively.

