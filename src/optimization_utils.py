import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np
import os
import modeling_utils
import datetime
from tqdm.auto import tqdm
import warnings


# This function updates the exponential average weights based on the current training
def update_average(model_tgt, model_src, beta):
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


def discrim_step_wasserstein_gp(discrim, discrim_opt, gen, real_images, batch_noise, depth, alpha, grad_pen_weight, device):
    fake_images = gen(batch_noise, depth, alpha).detach()
    loss = discrim_loss_wasserstein_gp(discrim, real_images, fake_images, depth, alpha, grad_pen_weight, device)
    return loss
    '''
    discrim_opt.zero_grad()
    loss.backward()
    discrim_opt.step()
    return loss
    '''
        
def gen_step_wasserstein(gen, gen_opt, gen_ema, discrim, batch_noise, depth, alpha):
    fake_images = gen(batch_noise, depth, alpha)
    loss = gen_loss_wasserstein(discrim, fake_images, depth, alpha)
    return loss
    '''
    gen_opt.zero_grad()
    loss.backward()
    gen_opt.step()
    update_average(gen_ema, gen, .999)
    return loss.item()
    '''


def loss_stop(d_loss, g_loss, g_stop_g_min=-128, g_stop_d_max=1.5, d_stop_g_max=128, d_stop_d_min=.2):
    zero_g = False
    zero_d = False
    if d_loss < d_stop_d_min or g_loss > d_stop_g_max:
        zero_d = True
    if g_loss < g_stop_g_min or d_loss > g_stop_d_max:
        zero_g = True

    if zero_d:
        d_loss = 0*d_loss
    if zero_g:
        g_loss = 0*g_loss

    return d_loss, g_loss

def train_on_depth_from_batch_wasserstein_gp(gen, gen_opt, gen_ema, discrim, discrim_opt, depth, fade_in, x_batch, batch_noise, grad_pen_weight, device):
    d_loss = discrim_step_wasserstein_gp(discrim, discrim_opt, gen, x_batch, batch_noise, depth, fade_in, grad_pen_weight, device)
    g_loss = gen_step_wasserstein(gen, gen_opt, gen_ema, discrim, batch_noise, depth, fade_in)

    # Stop loss block
    d_loss, g_loss = loss_stop(d_loss, g_loss)
    
    discrim_opt.zero_grad()
    d_loss.backward()
    discrim_opt.step()

    gen_opt.zero_grad()
    g_loss.backward()
    gen_opt.step()
    update_average(gen_ema, gen, .999)


    return d_loss.item(), g_loss.item()


# def checkpoint(gen, gen_ema, discrim, d_loss_hist, g_loss_hist, depth, fade_in, counter, fixed_noise, noise_size, device, save_path, save_samples):
#     if save_samples:
#         print('Mean d loss: ', np.mean(d_loss_hist))
#         print('Mean g loss: ', np.mean(g_loss_hist))
#     plot_gen_images(gen_ema, depth, fade_in, noise_size, device, save_path, save_samples)
#     torch.save(discrim.state_dict(), save_path + 'discrim_%d.pt'%(depth))
#     torch.save(gen.state_dict(), save_path + 'gen_%d.pt'%(depth))
#     torch.save(gen_ema.state_dict(), save_path + 'gen_ema_%d.pt'%(depth))
    
#     save_gen_fixed_noise(gen_ema, depth, fade_in, counter, fixed_noise, save_path)


def train_on_depth_wasserstein_gp(gen, gen_opt, gen_ema, discrim, discrim_opt, depth, img_size, nb_epochs, fade_in_pct, 
                                noise_size, grad_pen_weight, data_loader, device, fixed_noise, save_path, save_samples=False, sample_interval=1000,
                                notebook=True):
    warnings.warn("stop loss currently enabled!")

    gen.train()
    discrim.train()
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    d_loss_hist = []
    g_loss_hist = []
    counter = 0

    epoch_pbar = tqdm.tqdm(total = nb_epochs) if not notebook else tqdm.notebook.tqdm(total = nb_epochs)

    for epoch in range(nb_epochs):
        if not notebook:
            print('START OF EPOCH %d'%epoch)
        pbar = tqdm.tqdm(total = len(data_loader), leave=False) if not notebook else tqdm.notebook.tqdm(total = len(data_loader), leave=False)
        for x_batch in data_loader:
            fade_in = min(1, counter/(fade_in_pct * len(data_loader) * nb_epochs))
            
            if depth==0 or fade_in >= 1:
                pass
            else:
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
            pbar.update(1)
            
        epoch_pbar.update(1)
        pbar.close()
        
    checkpoint(gen, gen_ema, discrim, d_loss_hist, g_loss_hist, depth, 1, counter, fixed_noise, noise_size, device, save_path, save_samples)

"""General functions for saving metrics"""
def setup_model_save_directory(save_dir, save_gen_samples, save_gen_fixed, tensorboard):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    sample_dir = os.path.join(save_dir, 'random_samples/')
    fixed_dir = os.path.join(save_dir, 'fixed_samples/')
    model_dir = os.path.join(save_dir, 'models/')
    tensorboard_dir = os.path.join(save_dir, 'tensorboard/')

    if save_gen_samples and not os.path.exists(sample_dir):
        os.mkdir(sample_dir)
    if save_gen_fixed and not os.path.exists(fixed_dir):
        os.mkdir(fixed_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if tensorboard and not os.path.exists(tensorboard_dir):
        os.mkdir(tensorboard_dir)

    return {
        'sample' : sample_dir,
        'fixed' : fixed_dir,
        'model' : model_dir,
        'tensorboard' : tensorboard_dir
    }

def setup_tensorboard(tensorboard, save_path):
    if not tensorboard:
        return
    writer = SummaryWriter(save_path)
    return writer

def checkpoint(gen, gen_ema, discrim, device, noise_size, all_save_dirs, print_metrics, fixed_noise, plot_gen_samples, 
                        save_gen_samples, save_gen_fixed, fid, tensorboard, writer, metric_dict, **kwargs):
    save_idx = max([-1] + [int(f[f.rfind('_')+1:-3]) for f in os.listdir(all_save_dirs['model'])])+1
    torch.save(gen.state_dict(), all_save_dirs['model'] + 'gen_%d.pt'%save_idx)
    torch.save(discrim.state_dict(), all_save_dirs['model'] + 'discrim_%d.pt'%save_idx)
    if gen_ema is not None:
        torch.save(gen_ema.state_dict(), all_save_dirs['model'] + 'gen_ema_%d.pt'%save_idx)
    
    if save_gen_fixed:
        modeling_utils.save_gen_fixed_noise(gen_ema if gen_ema is not None else gen, 
                fixed_noise, all_save_dirs['fixed'], save_idx)
    
    if plot_gen_samples or save_gen_samples:
        samples = modeling_utils.sample_gen_images(gen_ema if gen_ema is not None else gen, noise_size,
                                                    device, **kwargs)
        if plot_gen_samples:
            modeling_utils.plot_imgs(samples)
        if save_gen_samples:
            modeling_utils.save_imgs(samples, all_save_dirs['sample'] + '%d.png'%save_idx)
    else:
        samples = None
    if print_metrics:
        print(metric_dict)
    if tensorboard:
        for metric, value in metric_dict.items():
            writer.add_scalar(metric, value, save_idx)
        writer.flush()
        if samples is None:
            samples = modeling_utils.sample_gen_images(gen_ema if gen_ema is not None else gen,
                                                        noise_size, device, **kwargs)
        samples = modeling_utils.swap_channels_batch(samples)
        samples = torch.tensor(samples)
        grid = torchvision.utils.make_grid(samples).unsqueeze(0)
        writer.add_images('random_samples', grid, save_idx)

        if fixed_noise is not None:
            fixed_samples = gen_ema(fixed_noise, **kwargs) if gen_ema is not None else gen(fixed_noise, **kwargs)
            grid = torchvision.utils.make_grid(fixed_samples).unsqueeze(0)
            writer.add_images('fixed_samples', grid, save_idx)
    

"""Optimization steps for traiditinoal gan"""
def train_for_epoch_traditional(gen, gen_opt, discrim, discrim_opt, loader, device, noise_size, discrim_noise_level, 
    checkpoint_interval = -1, save_dir=None, print_metrics=False, fixed_noise=None, plot_gen_samples=True, 
    save_gen_samples=True, save_gen_fixed=True, tensorboard=True, fid=False, **kwargs):

    all_save_dirs = setup_model_save_directory(save_dir, save_gen_samples, save_gen_fixed, tensorboard)
    writer = setup_tensorboard(tensorboard, all_save_dirs['tensorboard'])

    d_real_loss_hist = []
    d_fake_loss_hist = []
    d_real_pred_hist = []
    d_fake_pred_hist = []
    gen_loss_hist = []
    pbar = tqdm(total = len(loader), leave=False)
    if not isinstance(checkpoint_interval, int):
        checkpoint_interval = int(checkpoint_interval * len(loader))
    for i, batch in enumerate(loader):
        d_real_loss, d_real_pred, d_fake_loss, d_fake_pred, gen_loss = train_on_batch_traditional(gen, gen_opt, discrim, 
                                    discrim_opt, batch, device, noise_size, discrim_noise_level)
        d_real_loss_hist.append(d_real_loss)
        d_fake_loss_hist.append(d_fake_loss)
        gen_loss_hist.append(gen_loss)
        d_real_pred_hist.append(d_real_pred)
        d_fake_pred_hist.append(d_fake_pred)
        if checkpoint_interval > 0 and i > 0 and not i % checkpoint_interval:
            metric_dict = {'d_real_loss' : np.mean(d_real_loss_hist), 'd_real_pred' : np.mean(d_real_pred_hist),
                            'd_fake_loss' : np.mean(d_fake_loss_hist), 'd_fake_pred' : np.mean(d_fake_pred_hist),
                            'g_loss' : np.mean(gen_loss_hist)}
            d_real_loss_hist = []
            d_fake_loss_hist = []
            gen_loss_hist = []
            d_real_pred_hist = []
            d_fake_pred_hist = []
            checkpoint(gen, None, discrim, device, noise_size, all_save_dirs, print_metrics, fixed_noise, plot_gen_samples, 
                        save_gen_samples, save_gen_fixed, fid, tensorboard, writer, metric_dict, **kwargs)
        pbar.update(1)
    pbar.close()
    metric_dict = {'d_real_loss' : np.mean(d_real_loss_hist), 'd_real_pred' : np.mean(d_real_pred_hist),
                            'd_fake_loss' : np.mean(d_fake_loss_hist), 'd_fake_pred' : np.mean(d_fake_pred_hist),
                            'g_loss' : np.mean(gen_loss_hist)}
    checkpoint(gen, None, discrim, device, noise_size, all_save_dirs, print_metrics, fixed_noise, plot_gen_samples, 
                        save_gen_samples, save_gen_fixed, fid, tensorboard, writer, metric_dict, **kwargs)

def train_on_batch_traditional(gen, gen_opt, discrim, discrim_opt, batch, device, noise_size, discrim_noise_level):
    bce_loss = nn.BCEWithLogitsLoss()
    batch_real = batch.to(device)
    discrim.zero_grad()
    # Train discrim
    batch_size = batch_real.size(0)
    label = torch.full((batch_size, ), 1, device=device)
    if discrim_noise_level:
        label[torch.rand(label.size()) < discrim_noise_level] = 0
    # Train on real
    d_real_pred = discrim(batch_real)
    d_real_loss = bce_loss(d_real_pred.squeeze(), label)
    d_real_loss.backward()
    d_real_pred_prob = torch.sigmoid(d_real_pred).mean()
    # Train on fake
    noise = modeling_utils.generate_noise(batch_size, noise_size, device)
    batch_fake = gen(noise)
    d_fake_pred = discrim(batch_fake.detach())
    
    label = torch.full((batch_size, ), 0, device=device)
    if discrim_noise_level:
        label[torch.rand(label.size()) < discrim_noise_level] = 1

    d_fake_loss = bce_loss(d_fake_pred.squeeze(), label)
    d_fake_loss.backward()
    d_fake_pred_prob = torch.sigmoid(d_fake_pred).mean()
    discrim_opt.step()

    # Train the generator
    gen.zero_grad()
    label = torch.full((batch_size, ), 1, device=device)
    d_fake_pred_for_gen = discrim(batch_fake)
    gen_loss = bce_loss(d_fake_pred_for_gen.squeeze(), label)
    gen_loss.backward()
    gen_opt.step()

    return d_real_loss.item(), d_real_pred_prob.item(), d_fake_loss.item(), d_fake_pred_prob.item(), gen_loss.item()