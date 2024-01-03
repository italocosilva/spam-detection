# SMS Spam Detection with RNNs

📱 Welcome to the SMS Spam Detection repository! This project leverages Natural Language Processing (NLP) and deep learning techniques, specifically Recurrent Neural Networks (RNNs), to identify and classify SMS messages as spam or not.

## Repository Structure

### 1. `src` Directory
- **config.yaml**: Configuration file for model training and other settings.
- **train.py**: Python script for training the SMS spam detection model.

### 2. `notebooks` Directory
- **spam-detection.ipynb**: Jupyter notebook showcasing the end-to-end process of SMS spam detection using RNNs.
- **nlp-basics.ipynb**: Jupyter notebook covering fundamental concepts of Natural Language Processing.

### 3. `artifacts` Directory
- **model.h5**: Trained RNN model for SMS spam detection.
- **training_history.png**: Visualization of the training history.

### 4. `data` Directory
- **spam.csv**: Dataset containing SMS messages labeled as spam or non-spam.

### 5. `test` Directory
- **test_train.py**: Unit tests for the training script.

### 6. Other Files
- **LICENSE**: The license governing the use and distribution of this project.
- **.gitignore**: Specifies intentionally untracked files that Git should ignore.
- **pyproject.toml**: Configuration file for Python project metadata.

## Getting Started
To get started with SMS spam detection, follow these steps:

1. Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

2. Train the model using the provided script:

```bash
python src/train.py
```

3. Explore the provided Jupyter notebooks in the `notebooks` directory for a deeper understanding of the NLP techniques used.

## Model Artifacts
- The trained model is saved as `artifacts/model.h5`.
- View the training history in `artifacts/training_history.png`.

## Dataset
The SMS dataset is located in the `data` directory. The file `spam.csv` contains labeled SMS messages used for training and evaluation.


## License
This project is licensed under the [MIT License](LICENSE).

Feel free to reach out if you have any questions or suggestions! 🚀
