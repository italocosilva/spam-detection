import os
import sys

dirname = os.path.dirname(__file__)
sys.path.append(os.path.join(dirname, "..", "src"))

from train import train


def test_train():
    train()
