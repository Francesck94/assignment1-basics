import torch
from einops import einsum


def _xavier_init(in_features, out_features):
    std = torch.sqrt(torch.tensor(2)/(in_features + out_features))
    #return torch.clip(torch.normal(mean=0, std=std, size=(out_features, in_features)), -3*std, 3*std)
    return torch.nn.init.trunc_normal_(torch.empty(out_features, in_features), mean=0, std=std, a=-3*std, b=3*std)

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
        self.weight = torch.nn.Parameter(_xavier_init(in_features, out_features))
        if device is not None:
            self.weight = self.weight.to(device)
        if dtype is not None:
            self.weight = self.weight.to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #return x @ self.weight.T
        y = torch.einsum("o i, b ... i -> b ... o", self.weight, x)
        return y


def _initialize_weight(in_features: int, out_features: int) -> torch.Tensor:
    return torch.normal(mean=0, std=torch.sqrt(torch.tensor(2)/(in_features + out_features)), size=(out_features, in_features))


if __name__ == "__main__":
    # Example usage
    in_features = 4
    out_features = 3
    batch_size = 2
    linear_layer = LinearLayer(in_features, out_features)
    x = torch.randn(batch_size, in_features)  # Example input
    y = linear_layer(x)
    print(y.shape)