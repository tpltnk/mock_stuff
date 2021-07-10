import json
import torch

DATA_FILE_NAME = "MOCK_DATA.json"
GENDER_INFO = {
    "Male": 0,
    "Female": 1
}

def read_mock_data() -> torch.Tensor:
    with open(DATA_FILE_NAME) as json_file:
        mock_data = [[GENDER_INFO[entry["gender"]], entry["likes_sandwich"]] for entry in json.load(json_file)]
    return torch.tensor(mock_data)

def get_correlations(tensor: torch.Tensor) -> dict:
    cor = {}
    for entry in tensor:
        gender = int(entry[0])
        likes = int(entry[1])
        if not cor.get(gender):
            cor[gender] = 0
        cor[gender] += likes
    v1 = (cor[0] * 100) / float((cor[0] + cor[1]))
    v2 = (cor[1] * 100) / float((cor[0] + cor[1]))
    cor[0] = round(v1, 3)
    cor[1] = round(v2, 3)
    assert cor[0] + cor[1] == 100.0
    return cor


class CorrelatorModel(torch.nn.Module):
    def __init__(self, insize, outsize):
        super().__init__()
        self.linear = torch.nn.Linear(insize, outsize)

    def forward(self, tensor):
        out = self.linear(tensor)
        return out

class CorrelatorConvolutionModel(torch.nn.Module):
    def __init__(self, insize, outsize):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(insize, outsize)
        self.conv2 = torch.nn.Conv2d(insize, outsize)
    
    def forward(self, tensor):
        relu = torch.nn.ReLU

if __name__ == "__main__":
    T = read_mock_data()
    print(T.dim())
    print(T)
    print(get_correlations(T))