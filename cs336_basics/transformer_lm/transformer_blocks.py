import torch
from einops import einsum, rearrange, reduce
from cs336_basics.transformer_lm.helpers.weight_init import linear_weight_init, embedding_init




class LinearLayer(torch.nn.Module):
    """
    Linear layer block that performs linear transformation y = Wx.  
    Args:
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
    """
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        #self.weight = torch.normal(mean=0, std=torch.sqrt(2/(in_features + out_features)))
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(linear_weight_init(out_features, in_features))
        if device is not None:
            self.weight = self.weight.to(device)
        if dtype is not None:
            self.weight = self.weight.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #return x @ self.weight.T # [batch, out_features] 
        y = torch.einsum("o i, b ... i -> b ... o", self.weight, x)
        return y


class EmbeddingLayer(torch.nn.Module):
    """
    Embedding layer block that converts token indices into dense vectors.  
    Args:
        num_embeddings: int size of the vocabulary
        embedding_dim: int dimension of the embedding vectors, usually denoted as d_model
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
    """
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.Parameter(embedding_init(num_embeddings, embedding_dim)) # (num_embeddings, embedding_dim)
        if device is not None:
            self.weight = self.weight.to(device)
        if dtype is not None:
            self.weight = self.weight.to(dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSLayerNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) block.
    Args:
        d_model: hidden dimension of the model
        eps: float = 1e-5 
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__() 
        self.d_model = d_model
        self.eps = eps
        self.g = torch.nn.Parameter(torch.ones(d_model))
        if device is not None:
            self.g = self.g.to(device)
        if dtype is not None:
            self.g = self.g.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms_a = reduce(x**2, "... d_model -> ... 1", reduction="mean") + self.eps
        rms_a = torch.sqrt(rms_a)

        x_norm = x / rms_a
        

        #result = x_norm * self.g
        result = x_norm * self.g
        return result.to(in_dtype)



if __name__ == "__main__":
    # Example usage
    in_features = 4
    out_features = 3
    batch_size = 2
    linear_layer = LinearLayer(in_features, out_features)
    x = torch.randn(batch_size, in_features)  # Example input
    y = linear_layer(x)
    print("linear layer input shape:", x.shape)
    print("linear layer output shape:", y.shape)
    print("linear layer weight shape:", linear_layer.weight.shape)

    print("\n","-"*100)
    # Example usage for EmbeddingLayer
    vocab_size = 10
    embedding_dim = 4
    batch_size = 2
    seq_length = 3
    embedding_layer = EmbeddingLayer(vocab_size, embedding_dim)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    embeddings = embedding_layer(token_ids)
    print("vocab size:", vocab_size)
    print("embedding dim:", embedding_dim)
    print("embedding weight shape:", embedding_layer.weight.shape)
    print("token ids shape:", token_ids.shape)
    print("embeddings shape:", embeddings.shape)