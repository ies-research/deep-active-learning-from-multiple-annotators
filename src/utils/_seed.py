def seed_everything(seed: int = 0, deterministic: bool = True) -> None:
    import os
    import random

    import numpy as np
    import torch

    # --- Python ---
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # --- NumPy ---
    np.random.seed(seed)

    # --- PyTorch ---
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU

    if deterministic:
        # Forces deterministic algorithms when available (may throw if not possible)
        torch.use_deterministic_algorithms(True)

        # cuDNN settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # cuBLAS determinism (needed for some matmul/conv paths)
        # Must be set *before* CUDA context is initialized to be fully reliable.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # Alternative: ":16:8" (less memory)

    else:
        # Faster, but can be non-deterministic
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
