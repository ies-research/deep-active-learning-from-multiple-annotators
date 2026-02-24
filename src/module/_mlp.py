import torch
from torch import nn


class MultiLayerPerceptron(nn.Module):
    """
    Multi-layer perceptron (MLP) that returns both logits and the learned
    embedding (last hidden representation).

    The embedding is the output of the final hidden block
    (after activation/dropout). If you configure zero hidden layers, the
    embedding is just the (optionally projected) input to the head (here:
    identical to input features unless you change the design).

    Parameters
    ----------
    in_features : int
        Input feature dimension.
    out_features : int
        Output feature dimension.
    hidden_units : int | Sequence[int]
        - If int: hidden width used for each hidden layer (requires
          ``n_hidden_layers``).
        - If sequence: explicit hidden sizes for each hidden layer.
    n_hidden_layers : int, optional
        - Number of hidden layers when ``hidden_units`` is an int.
        - Ignored if ``hidden_units`` is a sequence.
    dropout : float, default=0.0
        Dropout probability applied after activation (and after
        BatchNorm if enabled).
    use_batchnorm : bool, default=True
        If True, use ``BatchNorm1d`` after each hidden linear layer.
    activation : str, default="gelu"
        Activation name: ``{"gelu", "relu", "silu", "tanh"}```.
    bias : bool, default=True
        Whether Linear layers use a bias term.
    """

    def __init__(
        self,
        in_features,
        out_features,
        hidden_units,
        *,
        n_hidden_layers=None,
        dropout=0.0,
        use_batchnorm=True,
        activation="gelu",
        bias=True,
    ) -> None:
        super().__init__()

        if isinstance(hidden_units, int):
            if n_hidden_layers is None or n_hidden_layers < 0:
                raise ValueError(
                    "If `hidden_units` is an int, n_hidden_layers must be a "
                    "non-negative int."
                )
            hidden_sizes = [hidden_units] * n_hidden_layers
        else:
            hidden_sizes = list(hidden_units)

        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1).")

        act = self._make_activation(activation)

        trunk: list[nn.Module] = []
        prev = in_features

        for h in hidden_sizes:
            trunk.append(nn.Linear(prev, h, bias=bias))
            if use_batchnorm:
                trunk.append(nn.BatchNorm1d(h))
            trunk.append(act)
            if dropout > 0.0:
                trunk.append(nn.Dropout(p=dropout))
            prev = h

        self.trunk = nn.Sequential(*trunk) if trunk else nn.Identity()
        self.embedding_dim = (
            prev  # last hidden size (or in_features if no hidden layers)
        )
        self.head = nn.Linear(self.embedding_dim, out_features, bias=bias)

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "gelu":
            return nn.GELU()
        if name == "relu":
            return nn.ReLU(inplace=True)
        if name in {"silu", "swish"}:
            return nn.SiLU(inplace=True)
        if name == "tanh":
            return nn.Tanh()
        raise ValueError(f"Unknown activation: {name}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits : torch.Tensor
            Shape (batch_size, out_features).
        embeddings : torch.Tensor
            Shape (batch_size, embedding_dim).
        """
        embeddings = self.trunk(x)
        logits = self.head(embeddings)
        return logits, embeddings
