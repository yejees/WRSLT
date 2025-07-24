# 1. Imports and Setup
import os
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.signal import savgol_filter
import clip
from Models.models import CNN2DModel, CNN3DModel
from Utils import *
from scipy.interpolate import CubicSpline
import natsort

torch.autograd.set_detect_anomaly(True)

# 2. Argument Parser and Seeding
def create_argparser():
    parser = argparse.ArgumentParser(description='Train CLIP-ASL model')
    parser.add_argument('--fold', type=int, default=1, help='Cross validation fold (1-4)')
    parser.add_argument('--save_root', type=str, default='./checkpoints', help='Checkpoint save path')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to data directory')
    return parser

def seed_everything(seed=1):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PL_SEED_WORKERS"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

# 3. Label Definitions
label_sets_dict = {
    'ASL_ISL': [...],  # 생략: 기존 리스트 복사
    'Only_ASL': [...],
}
label_names = [label for group in label_sets_dict.values() for label in group]

# 4. Training Function
def Train(args, logname='3Dconv', epochs=1000, batch_size=65, early_stop=100):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and process datasets
    def load_and_preprocess(file_prefix):
        data = np.load(f'{args.data_path}/{file_prefix}_data.npy', allow_pickle=True)
        labels = np.load(f'{args.data_path}/{file_prefix}_label.npy').astype(np.int64)

        n, c, f_n, f_s = data.shape
        label_num = len(label_names)

        data = data.reshape(label_num, n // label_num, c, f_n, f_s)
        data = np.transpose(data, (1, 0, 2, 3, 4)).reshape(n, c, f_n, f_s)

        labels = labels.reshape(label_num, n // label_num)
        labels = np.transpose(labels, (1, 0)).reshape(n)

        return data, labels

    data_a, labels_a = load_and_preprocess("case1")
    data_b, labels_b = load_and_preprocess("case2")

    data = np.concatenate([data_a, data_b], axis=0)
    labels = np.concatenate([labels_a, labels_b], axis=0)

    label_num = len(label_names)
    total_dataset_num = data.shape[0] // label_num
    print('Total dataset count:', total_dataset_num)

    # Savitzky–Golay filtering
    for a in range(total_dataset_num * label_num):
        for b in range(data.shape[1]):
            for v in range(data.shape[2]):
                data[a, b, v, :] = savgol_filter(data[a, b, v, :], 11, 3)

    # Cross-validation folds
    indices = list(range(total_dataset_num))
    random.shuffle(indices)
    fold_size = total_dataset_num // 4
    folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(3)]
    folds.append(indices[3 * fold_size:])

    test_indices = folds[args.fold]
    train_indices = [i for i in indices if i not in test_indices]

    def split_dataset(indices):
        X = np.empty((0, 3, 10, 200))
        Y = np.empty((0,))
        for i in indices:
            X = np.append(X, data[i * label_num:(i + 1) * label_num], axis=0)
            Y = np.append(Y, labels[i * label_num:(i + 1) * label_num], axis=0)
        return X, Y

    train_X, train_Y = split_dataset(train_indices)
    test_X, test_Y = split_dataset(test_indices)

    # Preprocessing r0 and normalization
    def categorize_r0(x):
        return np.where(x >= 105, 0,
               np.where(x >= 61, 1,
               np.where(x >= -60, 2,
               np.where(x >= -104, 3, 4))))

    train_r0 = categorize_r0(np.concatenate((train_X[:, 2:3, 1:2, 0:1], train_X[:, 2:3, 6:7, 0:1]), axis=2))
    test_r0 = categorize_r0(np.concatenate((test_X[:, 2:3, 1:2, 0:1], test_X[:, 2:3, 6:7, 0:1]), axis=2))

    # DC offset removal
    train_X -= ((train_X[:, :, :, 1:2] + train_X[:, :, :, -2:-1]) / 2)
    test_X -= ((test_X[:, :, :, 1:2] + test_X[:, :, :, -2:-1]) / 2)

    # Frame selection
    def select_frames(X):
        return np.concatenate([X[:, :, i:i+1] for i in [1, 3, 4, 5, 6, 7, 8]], axis=2)
    
    train_X = select_frames(train_X)
    test_X = select_frames(test_X)

    # Load CLIP
    clip_encoder, _ = clip.load("ViT-B/32")
    clip_encoder = clip_encoder.to(device).eval()

    # Tokenize text prompts
    def generate_text_prompts(labels):
        return [
            clip.tokenize([
                f"a sign language representation of {label_names[int(l)]}",
                f"the sign '{label_names[int(l)]}' in International Sign Language",
                f"the meaning of this sign is '{label_names[int(l)]}'",
                f"a gesture that represents the concept of '{label_names[int(l)]}'"
            ]) for l in labels
        ]

    tokenized_texts = generate_text_prompts(train_Y)
    tokenized_texts = [torch.stack([t.cuda() for t in prompts]) for prompts in tokenized_texts]

    # Model & training setup
    model = CNN2DModel(in_ch=3, class_num=label_num).to(device)
    temperature = nn.Parameter(torch.tensor(1.).cuda())

    hard_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {'params': model.parameters()},
        {'params': temperature}
    ], lr=1e-5)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

    save_path = f"{args.save_root}_fold{args.fold}"
    os.makedirs(save_path, exist_ok=True)

    # Training loop
    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []
    best_val_acc = 0
    early_stop_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for i in tqdm(range(len(train_X) // batch_size)):
            idx = slice(i * batch_size, (i + 1) * batch_size)
            x = torch.tensor(train_X[idx], dtype=torch.float32, device=device)
            y = torch.tensor(train_Y[idx], dtype=torch.long, device=device)
            r0 = torch.tensor(train_r0[idx], dtype=torch.long, device=device).view(len(x), -1)
            text_feats = torch.stack([clip_encoder.encode_text(t[i]) for t in tokenized_texts], dim=0).mean(1)

            optimizer.zero_grad()
            out, sign_feats, _ = model(x, r0, train=True)
            noise = torch.randn_like(text_feats) * 0.2
            clip_loss = (1 - F.cosine_similarity(sign_feats, text_feats + noise, dim=1)).mean()
            hard_loss = hard_criterion(out, y)
            loss = hard_loss + clip_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total += y.size(0)
            correct += (out.argmax(1) == y).sum().item()

        acc = correct / total
        print(f"[Epoch {epoch}] Train Loss: {train_loss/total:.4f} | Train Acc: {acc:.4f}")

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for i in range(len(test_X)):
                x = torch.tensor(test_X[i:i+1], dtype=torch.float32, device=device)
                y = torch.tensor(test_Y[i:i+1], dtype=torch.long, device=device)
                r0 = torch.tensor(test_r0[i:i+1], dtype=torch.long, device=device).view(1, -1)

                out, _, _ = model(x, r0)
                loss = hard_criterion(out, y)
                val_loss += loss.item()
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += 1

        val_acc = val_correct / val_total
        print(f"[Epoch {epoch}] Val Loss: {val_loss/val_total:.4f} | Val Acc: {val_acc:.4f}")
        print('-'*80)

        train_loss_list.append(train_loss / total)
        val_loss_list.append(val_loss / val_total)
        train_acc_list.append(acc)
        val_acc_list.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(save_path, f'best_model_epoch{epoch}_acc{int(val_acc*100)}.pt'))
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop:
                print("Early stopping triggered.")
                break

    return 

# 5. Main Block
if __name__ == '__main__':
    args = create_argparser().parse_args()
    Train(args, batch_size=len(label_names))
