from credential_classifier.bit_pytorch.models import FCMaxPool
from credential_classifier.bit_pytorch.grid_divider import read_img_reverse
import torch
import torch.nn.functional as F

def credential_config(checkpoint):
    '''
    Load credential classifier configurations
    :param checkpoint: classifier weights
    :return model: classifier
    '''
    # load weights
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FCMaxPool()
    checkpoint = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model

def credential_classifier(img:str, coords, types, model):
    '''
    Run credential classifier
    :param img: path to image
    :param coords: torch.Tensor/np.ndarray Nx4 bbox coords
    :param types: torch.Tensor/np.ndarray Nx4 bbox types
    :param model: classifier 
    :return pred: predicted class 'credential': 0, 'noncredential': 1
    :return conf: prediction confidence
    '''
    # process it into grid_array
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    grid_arr = read_img_reverse(img, coords, types)
    assert grid_arr.shape == (9, 10, 10) # ensure correct shape
    
    # inference
    with torch.no_grad():
        pred_orig = model(grid_arr.type(torch.float).to(device))
        pred = F.softmax(pred_orig, dim=-1).argmax(dim=-1).item() # 'credential': 0, 'noncredential': 1
        conf, _ = torch.max(F.softmax(pred_orig, dim=-1), dim=-1)
        conf = conf.item()
        
    return pred, conf