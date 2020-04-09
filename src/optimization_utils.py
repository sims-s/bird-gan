import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from modeling_utils import *
import datetime
import tqdm

# Mostly inspired by: https://github.com/akanimax/pro_gan_pytorch/blob/master/pro_gan_pytorch/CustomLayers.py
# function to calculate the Exponential moving averages for the Generator weights
# This function updates the exponential average weights based on the current training
def update_average(model_tgt, model_src, beta):
    """
    update the model_target using exponential moving averages
    :param model_tgt: target model
    :param model_src: source model
    :param beta: value of decay beta
    :return: None (updates the target model)
    """

    # utility function for toggling the gradient requirements of the models
    def toggle_grad(model, requires_grad):
        for p in model.parameters():
            p.requires_grad_(requires_grad)

    # turn off gradient calculation
    toggle_grad(model_tgt, False)
    toggle_grad(model_src, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)

    # turn back on the gradient calculation
    toggle_grad(model_tgt, True)
    toggle_grad(model_src, True)



def discrim_loss_wasserstein_gp(discrim, real_images, fake_images, depth, alpha, grad_pen_weight, device, drift=0.001):
    real_pred = discrim(real_images, depth, alpha)
    fake_pred = discrim(fake_images, depth, alpha)
    
    convex_combo_lambda = torch.rand((real_images.size()[0])).to(device)
    convex_combo_lambda = convex_combo_lambda.view((-1,1,1,1))

    
    convex_combo = convex_combo_lambda * real_images + (1-convex_combo_lambda)*fake_images
    convex_combo.requires_grad = True
    
    discrim_convex_combo = discrim(convex_combo, depth, alpha)
    
    
    grad = torch.autograd.grad(outputs=discrim_convex_combo, inputs=convex_combo, 
                               grad_outputs = torch.ones_like(discrim_convex_combo), create_graph=True,
                              retain_graph = True, only_inputs=True)[0]

    grad = grad.view((grad.size()[0], -1))
    gradient_penalty = grad_pen_weight*((grad.norm(p=2, dim=1)-1)**2).mean()
    return fake_pred.mean() - real_pred.mean() + gradient_penalty + drift * (real_pred**2).mean()
    


def gen_loss_wasserstein(discrim, fake_samples, depth, alpha):
    return -discrim(fake_samples, depth, alpha).mean()


def discrim_step_wasserstein_gp(discrim, discrim_opt, gen, real_images, batch_noise, depth, alpha, grad_pen_weight, device, n_iters=2):
    total_loss = 0
    fake_images = gen(batch_noise, depth, alpha).detach()
    for _ in range(n_iters):
        loss = discrim_loss_wasserstein_gp(discrim, real_images, fake_images, depth, alpha, grad_pen_weight, device)
        discrim_opt.zero_grad()
        loss.backward()
        discrim_opt.step()
        total_loss += loss.item()
        
    return total_loss / n_iters
        
def gen_step_wasserstein(gen, gen_opt, gen_ema, discrim, batch_noise, depth, alpha):
    fake_images = gen(batch_noise, depth, alpha)
    loss = gen_loss_wasserstein(discrim, fake_images, depth, alpha)
    gen_opt.zero_grad()
    loss.backward()
    gen_opt.step()
    update_average(gen_ema, gen, .999)
    return loss.item()

def train_on_depth_from_batch_wasserstein_gp(gen, gen_opt, gen_ema, discrim, discrim_opt, depth, fade_in, x_batch, batch_noise, grad_pen_weight, device):
    d_loss = discrim_step_wasserstein_gp(discrim, discrim_opt, gen, x_batch, batch_noise, depth, fade_in, grad_pen_weight, device, 1)
    g_loss = gen_step_wasserstein(gen, gen_opt, gen_ema, discrim, batch_noise, depth, fade_in)
    return d_loss, g_loss


def checkpoint(gen, gen_ema, discrim, d_loss_hist, g_loss_hist, depth, fade_in, counter, fixed_noise, noise_size, device, save_path, save_samples):
    if save_samples:
        print('Mean d loss: ', np.mean(d_loss_hist))
        print('Mean g loss: ', np.mean(g_loss_hist))
    plot_gen_images(gen_ema, depth, fade_in, noise_size, device, save_path, save_samples)
    torch.save(discrim.state_dict(), save_path + 'discrim_%d.pt'%(depth))
    torch.save(gen.state_dict(), save_path + 'gen_%d.pt'%(depth))
    torch.save(gen_ema.state_dict(), save_path + 'gen_ema_%d.pt'%(depth))
    
    save_gen_fixed_noise(gen_ema, depth, fade_in, counter, fixed_noise, save_path)


def train_on_depth_wasserstein_gp(gen, gen_opt, gen_ema, discrim, discrim_opt, depth, img_size, nb_epochs, fade_in_pct, 
                                noise_size, grad_pen_weight, data_loader, device, fixed_noise, save_path, save_samples, sample_interval=1000):

    gen.train()
    discrim.train()
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    d_loss_hist = []
    g_loss_hist = []
    counter = 0

    '''
    downsampler = nn.AvgPool2d(int(img_size / (4*2**depth)))
    if depth > 0:
        prev_depth_downsampler = nn.AvgPool2d(int(img_size / (4*2**(depth-1))))
    '''
    epoch_pbar = tqdm.tqdm(total = nb_epochs)

    for epoch in range(nb_epochs):
        pbar = tqdm.tqdm(total = len(data_loader), leave=False)
        for x_batch in data_loader:
            fade_in = min(1, counter/(fade_in_pct * len(data_loader) * nb_epochs))
            
            if depth==0 or fade_in >= 1:
                pass
                '''
                x_batch = downsampler(x_batch)
                '''
            else:
                '''
                x_batch_norm = downsampler(x_batch)
                x_batch_down_up = F.interpolate(prev_depth_downsampler(x_batch), scale_factor=2)
                x_batch = fade_in*x_batch_norm + ((1-fade_in)*x_batch_down_up)
                x_batch = x_batch
                '''
                x_batch_down_up = F.interpolate(nn.AvgPool2d(2)(x_batch), scale_factor=2)
                x_batch = fade_in * x_batch + ((1-fade_in) * x_batch_down_up)
            x_batch = x_batch.to(device)
            
            batch_noise = generate_noise(len(x_batch), noise_size, device)
            
            d_loss, g_loss = train_on_depth_from_batch_wasserstein_gp(gen, gen_opt, gen_ema, discrim, discrim_opt, depth, fade_in, x_batch, batch_noise, grad_pen_weight, device)
            d_loss_hist.append(d_loss)
            g_loss_hist.append(g_loss)

            counter+=1

            if not counter % (sample_interval):
                checkpoint(gen, gen_ema, discrim, d_loss_hist, g_loss_hist, depth, fade_in, counter, fixed_noise, noise_size, device, save_path, save_samples)
                d_loss_hist = []
                g_loss_hist = []
                return
                
            pbar.update(1)
            
        epoch_pbar.update(1)
        pbar.close()
        
    checkpoint(gen, gen_ema, discrim, d_loss_hist, g_loss_hist, depth, 1, counter, fixed_noise, noise_size, device, save_path, save_samples)