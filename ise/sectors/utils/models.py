"""Utility functions for working with pretrained models."""

import torch


def load_model(
    model_path, model_class, architecture, mc_dropout=False, dropout_prob=0.1
):
    """Loads PyTorch model from saved state_dict.

    Args:
        model_path (str): Filepath to model state_dict.
        model_class (Model): Model class.
        architecture (dict): Defined architecture of pretrained model.
        mc_dropout (bool): Flag denoting wether the model was trained using MC Dropout.
        dropout_prob (float): Value between 0 and 1 denoting the dropout probability.

    Returns:
        model (Model): Pretrained model.
    """
    model = model_class(architecture, mc_dropout=mc_dropout, dropout_prob=dropout_prob)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device)
