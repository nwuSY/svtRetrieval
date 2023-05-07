import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from thop import profile, clever_format

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SV1KDataset, train_transform, test_valid_transform
from model import Model
import numpy as np


def train(net, data_loader, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)

    for name, pos_1, pos_2, text in train_bar:
        if torch.cuda.is_available():
            pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
            text = text.cuda(non_blocking=True)

        feature_1, out_1 = net(pos_1, text)

        # flops, params = profile(model, inputs=(pos_1, text))
        # flops, params = clever_format([flops, params])  # 模型参数计算
        # print('# Model Params: {} FLOPs: {}'.format(params, flops))

        feature_2, out_2 = net(pos_2, text)

        # [2 * B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2 * B, 2 * B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2 * B, 2 * B - 1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2 * B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


def test(epoch, net, test_data_loader, valid_data_loader):
    net.eval()
    name_bank, feature_bank = [], []
    valid_name_bank, valid_feature_bank = [], []
    with torch.no_grad():
        for name, pos, _, text in tqdm(test_data_loader, desc="Feature Extracting"):
            name_bank.append(name[0])
            if torch.cuda.is_available():
                pos = pos.cuda(non_blocking=True)
                text = text.cuda(non_blocking=True)
            feature, out = net(pos, text)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).contiguous()

        for name, pos, _, text in tqdm(valid_data_loader, desc="Feature Extracting"):
            valid_name_bank.append(name[0])
            if torch.cuda.is_available():
                pos = pos.cuda(non_blocking=True)
                text = text.cuda(non_blocking=True)
            feature, out = net(pos, text)
            valid_feature_bank.append(feature)
        # [D, N]
        valid_feature_bank = torch.cat(valid_feature_bank, dim=0).contiguous()

    result = torch.mm(feature_bank, valid_feature_bank.t())
    if torch.cuda.is_available():
        idx = torch.argsort(-result).detach().cpu().numpy()
    else:
        idx = torch.argsort(-result).numpy()

    acc = np.sum([np.array(name_bank) == np.array(valid_name_bank)[idx[:, 0]]]) / len(name_bank)
    print("Test Epoch: {} Acc:{:.2f}%".format(epoch, acc * 100))

    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Sim On SV1K Dataset')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.1, type=float, help='Temperature used in softmax')
    parser.add_argument('--batch_size', default=5, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=1000, type=int, help='Number of sweeps over the dataset to train')

    args = parser.parse_args()
    feature_dim, temperature = args.feature_dim, args.temperature
    batch_size, epochs = args.batch_size, args.epochs

    train_data = SV1KDataset(transform=train_transform, gw_root_dir="./data", train="train")
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                              drop_last=True)
    test_data = SV1KDataset(transform=test_valid_transform, gw_root_dir="./data", train="test")
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=0)
    valid_data = SV1KDataset(transform=test_valid_transform, gw_root_dir="./data", train="valid")
    valid_loader = DataLoader(valid_data, batch_size=1, shuffle=True, num_workers=0)

    f_data_1 = SV1KDataset(transform=test_valid_transform, gw_root_dir="./data_", train="test")
    f_data_1_loader = DataLoader(f_data_1, batch_size=1, shuffle=True, num_workers=0)
    f_data_2 = SV1KDataset(transform=test_valid_transform, gw_root_dir="./data_", train="valid")
    f_data_2_loader = DataLoader(f_data_2, batch_size=1, shuffle=True, num_workers=0)

    model = Model()
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    total_params = sum(p.numel() for p in model.parameters())
    total_params += sum(p.numel() for p in model.buffers())
    print(f'{total_params:,} total parameters.')
    print(f'{total_params / (1024 * 1024):.2f}M total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    print(f'{total_trainable_params / (1024 * 1024):.2f}M training parameters.')

    # training loop
    results = {'train_loss': [], 'train_acc@1': [], 'test_acc@1': []}
    save_name_pre = '{}_{}_{}_{}'.format(feature_dim, temperature, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer)
        results["train_loss"].append(train_loss)

        train_acc = test(epoch, model, f_data_1_loader, f_data_2_loader)
        results['train_acc@1'].append(train_acc)

        test_acc = test(epoch, model, test_loader, valid_loader)
        results['test_acc@1'].append(test_acc)

        # print("Epoch:{}, Acc:{}".format(epoch + 1, test_acc))

        with open("./results/record.txt", "a+") as f:
            f.write(str(epoch) + "\t" + str(train_loss) + "\t" + str(train_acc) + "\t" + str(test_acc) + "\n")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "./results/_%.4f_model.pth".format(test_acc))
