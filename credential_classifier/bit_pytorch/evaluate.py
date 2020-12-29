
import torch
import numpy as np
import bit_pytorch.models as models
from bit_pytorch.dataloader import GetLoader

def evaluate(model, train_loader):

    model.eval()
    # num_ones = 0
    # num_zeros = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for b, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True, dtype=torch.float)
            y = y.to(device, non_blocking=True, dtype=torch.long)

            # Compute output, measure accuracy
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            correct += preds.eq(y).sum().item()
            total += len(logits)
            # num_ones += np.sum(preds == 1)
            # num_zeros += np.sum(preds == 0)
            print("GT:", y)
            print("Pred:", preds)
            print(correct, total)

    return float(correct/total)

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = models.KNOWN_MODELS['FCMax'](head_size=2, grid_num=10)
    checkpoint = torch.load('./output/website/FCMax_0.01.pth.tar', map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    val_set = GetLoader(img_folder='./data/val_imgs',
                        annot_path='./data/val_coords.txt')
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=512, drop_last=False, shuffle=False)

    acc = evaluate(model, val_loader)
    print(acc)
