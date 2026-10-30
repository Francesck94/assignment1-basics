import torch

def linear_weight_init(out_features: int, in_features: int):
    """
    Initialize the weights for a linear layer using truncated normal distribution.

    Args:
        out_features (int): The number of output features
        in_features (int): The number of input features

    Returns:
        torch.Tensor: Initialized weight tensor of shape (out_features, in_features)
    """
    std = torch.sqrt(torch.tensor(2)/(in_features + out_features))
    #return torch.clip(torch.normal(mean=0, std=std, size=(out_features, in_features)), -3*std, 3*std)
    return torch.nn.init.trunc_normal_(torch.empty(out_features, in_features), mean=0, std=std, a=-3*std, b=3*std)

def embedding_init(num_embeddings: int, embedding_dim: int):
    """
    Initialize the weights for an embedding layer using truncated normal distribution.

    Args:
        num_embeddings (int): The number of embeddings (vocabulary size)
        embedding_dim (int): The dimension of each embedding vector

    Returns:
        torch.Tensor: Initialized embedding weight tensor of shape (num_embeddings, embedding_dim)
    """
    _std = 1
    _mean = 0
    return torch.nn.init.trunc_normal_(torch.empty(num_embeddings, embedding_dim), mean=_mean, std=_std, a=-3, b=3)
