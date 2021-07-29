import torch
import torch.nn as nn

class actorModel(nn.Module) :
    def __init__(self) -> None:
        super(actorModel, self).__init__()
        self.linear1 = nn.Linear()

    
    
