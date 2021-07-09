import json
import torch
import time

DATA_FILE_NAME = "MOCK_DATA.json"

if __name__ == "__main__":
    T = torch.randint(0, 10, [10, 10])
    T, _ = T.sort(dim=-1)
    print(T.diag(0))
    print(T)