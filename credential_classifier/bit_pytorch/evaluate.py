
import torch
import numpy as np
import bit_pytorch.models as models
from bit_pytorch.dataloader import GetLoader, ImageLoader
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def vis_helper(x):
    fig = plt.figure(figsize=(20, 20))
    ax = plt.subplot("331")
    ax.set_title("X", fontsize=15)

    ax.imshow(x[0].numpy() if not isinstance(x[0], np.ndarray) else x[0], cmap='gray')
    ax = plt.subplot("332")
    ax.set_title("Y", fontsize=15)

    ax.imshow(x[1].numpy() if not isinstance(x[1], np.ndarray) else x[1], cmap='gray')
    ax = plt.subplot("333")
    ax.set_title("W", fontsize=15)

    ax.imshow(x[2].numpy() if not isinstance(x[2], np.ndarray) else x[2], cmap='gray')
    ax = plt.subplot("334")
    ax.set_title("H", fontsize=15)

    ax.imshow(x[3].numpy() if not isinstance(x[3], np.ndarray) else x[3], cmap='gray')
    ax = plt.subplot("335")
    ax.set_title("C1(logo)", fontsize=15)

    ax.imshow(x[4].numpy() if not isinstance(x[4], np.ndarray) else x[4], cmap='gray')
    ax = plt.subplot("336")
    ax.set_title("C2(input)", fontsize=15)

    ax.imshow(x[5].numpy() if not isinstance(x[5], np.ndarray) else x[5], cmap='gray')
    ax = plt.subplot("337")
    ax.set_title("C3(button)", fontsize=15)
    ax.imshow(x[6].numpy() if not isinstance(x[6], np.ndarray) else x[6], cmap='gray')

    ax = plt.subplot("338")
    ax.set_title("C4(label)", fontsize=15)
    ax.imshow(x[7].numpy() if not isinstance(x[7], np.ndarray) else x[7], cmap='gray')

    ax = plt.subplot("339")
    ax.set_title("C5(block)", fontsize=15)
    ax.imshow(x[8].numpy() if not isinstance(x[8], np.ndarray) else x[8], cmap='gray')
    
    plt.show()
    
def evaluate(model, train_loader):
    '''
    :param model: model to be evaluated
    :param train_loader: dataloader to be evaluated
    :return: classification acc
    '''

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
    model = models.KNOWN_MODELS['BiT-M-R50x1'](head_size=2)
    checkpoint = torch.load('./output/screenshot/screenshot/BiT-M-R50x1_0.001.pth.tar', map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    val_set = ImageLoader(img_folder='../datasets/val_imgs',
                          annot_path='../datasets/val_coords.txt')
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=8, drop_last=False, shuffle=False)

    acc = evaluate(model, val_loader)
    print(acc)
