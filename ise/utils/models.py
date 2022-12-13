import torch

def load_model(model_path, model_class, architecture):
    """Loads PyTorch model from saved state_dict.

    Args:
        model_path (str): Filepath to model state_dict.
        model_class (Model): Model class.
        architecture (_type_): Defined architecture of pretrained model.

    Returns:
        model (Model): Pretrained model.
    """
    model = model_class(architecture)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model