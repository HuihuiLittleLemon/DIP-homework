# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from cgan_network import Generator, Discriminator
from facades_dataset import FacadesDataset
from config import Config as conf
from torchvision.utils import save_image
import os
import time


def save_images(inputs, targets, outputs, folder_name, epoch, batch_idx=0):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        batch_idx (int): Index of the batch in the current epoch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    # 将图像的像素值从 [-1, 1] 还原到 [0, 1]
    inputs = (inputs + 1) / 2.0
    targets = (targets + 1) / 2.0
    outputs = (outputs + 1) / 2.0

    # 保存前5张图像
    num_images = min(inputs.size(0), 5)
    for i in range(num_images):
        input_img = inputs[i]
        target_img = targets[i]
        output_img = outputs[i]

        # 拼接输入、目标和输出图像
        comparison = torch.cat(
            (input_img, target_img, output_img), dim=2)  # 在宽度维度拼接

        save_image(
            comparison, f'{folder_name}/epoch_{epoch}/batch_{batch_idx}_sample_{i+1}.png')


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化数据集和数据加载器
    train_dataset = FacadesDataset(conf.data_path, phase='train')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    val_dataset = FacadesDataset(conf.data_path, phase='val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # 初始化模型、损失函数和优化器
    generator = Generator(conv_channel_base=conf.conv_channel_base,
                          img_channel=conf.img_channel).to(device)
    discriminator = Discriminator(
        conv_channel_base=conf.conv_channel_base, img_channel=conf.img_channel).to(device)

    criterion_GAN = torch.nn.BCEWithLogitsLoss()
    criterion_L1 = torch.nn.L1Loss()

    optimizer_G = optim.Adam(generator.parameters(
    ), lr=conf.learning_rate, betas=(conf.beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(
    ), lr=conf.learning_rate, betas=(conf.beta1, 0.999))

    # 学习率调度器（可选）
    scheduler_G = optim.lr_scheduler.StepLR(
        optimizer_G, step_size=100, gamma=0.5)
    scheduler_D = optim.lr_scheduler.StepLR(
        optimizer_D, step_size=100, gamma=0.5)

    if not os.path.exists(conf.output_path):
        os.makedirs(conf.output_path)

    start_time = time.time()
    for epoch in range(conf.max_epoch):
        generator.train()
        discriminator.train()
        for i, batch in enumerate(train_loader):
            img_A = batch['A'].to(device)
            img_B = batch['B'].to(device)

            # 生成标签
            pred_shape = discriminator(img_B, img_A).shape
            valid = torch.ones(pred_shape, device=device)
            fake = torch.zeros(pred_shape, device=device)

            # ---------------------
            #  训练判别器
            # ---------------------
            optimizer_D.zero_grad()

            # 真实图像
            pred_real = discriminator(img_B, img_A)
            loss_D_real = criterion_GAN(pred_real, valid)

            # 生成图像
            fake_B = generator(img_A)
            pred_fake = discriminator(fake_B.detach(), img_A)
            loss_D_fake = criterion_GAN(pred_fake, fake)

            # 判别器总损失
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # -----------------
            #  训练生成器
            # -----------------
            optimizer_G.zero_grad()

            # 对抗性损失
            pred_fake = discriminator(fake_B, img_A)
            loss_G_GAN = criterion_GAN(pred_fake, valid)

            # L1 损失
            loss_G_L1 = criterion_L1(fake_B, img_B) * conf.L1_lambda

            # 生成器总损失
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizer_G.step()

            # 输出训练信息
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{conf.max_epoch}], Batch [{i+1}/{len(train_loader)}], "
                      f"Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}, Time: {time.time() - start_time:.4f}")

            # 保存训练结果的图像，每 5 个 epoch 保存一次
            if (epoch + 1) % 5 == 0 and i == 0:
                generator.eval()
                with torch.no_grad():
                    fake_B = generator(img_A)
                save_images(img_A, img_B, fake_B, 'train_results',
                            epoch + 1, batch_idx=i)
                generator.train()

        # 在验证集上进行评估，每 5 个 epoch 保存一次验证集结果
        if (epoch + 1) % 5 == 0:
            generator.eval()
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    img_A = batch['A'].to(device)
                    img_B = batch['B'].to(device)

                    # 生成图像
                    fake_B = generator(img_A)
                    save_images(img_A, img_B, fake_B,
                                'val_results', epoch + 1, batch_idx=i)
            generator.train()

        # 更新学习率
        scheduler_G.step()
        scheduler_D.step()

        # 保存模型
        if (epoch + 1) % conf.save_per_epoch == 0:
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
            }, os.path.join(conf.output_path, f"model_{epoch+1}.pth"))
            print(f"Model saved at epoch {epoch+1}")

    # 训练结束后保存最终模型
    torch.save({
        'epoch': conf.max_epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
    }, os.path.join(conf.output_path, "model_final.pth"))
    print("Training complete. Final model saved.")


if __name__ == "__main__":
    train()
