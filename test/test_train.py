import os
import sys

dirname = os.path.dirname(__file__)
sys.path.append(os.path.join(dirname, "..", "src"))

from train import train
from hydra.experimental import compose, initialize


def test_train():
    # Define paths
    data_path = os.path.join(dirname, "..", "data", "test_spam.csv")
    train_plot_path = os.path.join(dirname, "..", "artifacts", "test_training_history.png")
    model_path = os.path.join(dirname, "..", "artifacts", "test_model.h5")

    # Assert files not exist
    assert not os.path.exists(train_plot_path)
    assert not os.path.exists(model_path)

    # Train model
    train(data_path, train_plot_path, model_path)

    # Assert files exist
    assert os.path.exists(train_plot_path)
    assert os.path.exists(model_path)

    # Remove files
    os.remove(train_plot_path)
    os.remove(model_path)
