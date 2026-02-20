import random

import torch
import numpy as np


def set_seed(seed: int = 43) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
