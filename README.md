# Fake News Detection Kaggle Competition

This project focuses on building a **Fake News Detection System** as part of the Kaggle competition. It employs advanced **NLP techniques** and **deep learning models** to identify and classify fake news articles effectively. The solution achieved a **third-place ranking on the leaderboard (late submission)** with an impressive score of  **0.97527** .

## Key Features of the Solution

### 1. **Dataset Handling**

* The competition dataset was downloaded using Kaggle's API.
* Preprocessing steps included:
  * Removing rows with missing values.
  * Text cleaning to remove special characters and convert text to lowercase.
  * Removing stop words to enhance the signal in the text data.

### 2. **Custom Dataset Class**

A `FakeNewsDataset` class was created to manage the dataset efficiently:

* Converts the text into numerical indices based on a vocabulary.
* Handles tokenization and padding for input to the model.

### 3. **Vocabulary Creation**

* A vocabulary was generated using all words from the training and testing datasets.
* `<unk>` and `<pad>` tokens were added to handle out-of-vocabulary words and padding, respectively.

### 4. **Deep Learning Model**

* The **SentimentLSTM** model was built using PyTorch:
  * **Embedding Layer** : Encodes text into dense vector representations.
  * **Bidirectional LSTM** : Captures long-term dependencies in the text in both forward and backward directions.
  * **Layer Normalization** : Stabilizes training and speeds up convergence.
  * **Dropout Layer** : Prevents overfitting.
  * **Fully Connected Layer** : Maps the LSTM output to the final classification.

### 5. **Training and Evaluation**

* The dataset was split into training and validation subsets (70%-30% split).
* Used **Binary Cross Entropy with Logits Loss** as the loss function.
* **Adam Optimizer** was employed for efficient parameter updates with weight decay for regularization.
* A **learning rate scheduler** adjusted the learning rate dynamically based on validation loss.
* Metrics like  **accuracy** ,  **F1 score** , and **validation loss** were monitored during training.

### 6. **Inference**

* Predictions on the test dataset were generated using the best model (saved during training).
* Results were saved in the required Kaggle submission format.

### 7. **Late Submission and Leaderboard Placement**

Despite a late submission, the solution achieved a score of  **0.97527** , earning  **third place on the leaderboard** .

## Results

* **Validation Accuracy** : ~98%
* **Leaderboard Score** : 0.97527
* **Rank** : Third Place (Late Submission)

## How to Reproduce

1. Clone this repository and set up the environment by installing the dependencies (e.g., `torch`, `pandas`, `nltk`).
2. install requirements using: **pip install -r requirements.txt**
3. Place your Kaggle API token (`kaggle.json`) in the project directory.
4. Download the dataset using the Kaggle API:
   ```bash
   kaggle competitions download -c fake-news
   ```
5. Train the model by running the notebook.
6. Submit the predictions in the format required by Kaggle.

## Technical Highlights

* **NLP Preprocessing** : Custom text cleaning and stopword removal.
* **Deep Learning Architecture** : Use of bidirectional LSTM with dropout and layer normalization.
* **Robust Optimization** : Integrated learning rate scheduling and gradient clipping.
* **Model Generalization** : Efficient handling of overfitting through dropout and regularization.

---

This repository demonstrates advanced NLP techniques and deep learning expertise, resulting in a high-performing solution to the Fake News Detection problem.
