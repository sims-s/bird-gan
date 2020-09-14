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
from collections import defaultdict
import fid_utils


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

"""General checkpointing functions and such"""
def checkpoint(gen, gen_ema, discrim, device, noise_size, all_save_dirs, print_metrics, fixed_noise, plot_gen_samples, 
                    save_gen_samples, save_gen_fixed, fid, n_fid_samples, tensorboard, writer, metric_dict, model_desc="", 
                    **kwargs):
    gen.eval()
    if gen_ema is not None:
        gen_ema.eval() 
    save_idx = max([-1] + [int(f[f.rfind('_')+1:-3]) for f in os.listdir(all_save_dirs['model'])])+1
    if len(model_desc) > 0:
        torch.save(gen.state_dict(), all_save_dirs['model'] + 'gen_%s_step_%d.pt'%(model_desc, save_idx))
        torch.save(discrim.state_dict(), all_save_dirs['model'] + 'discrim_%s_step_%d.pt'%(model_desc, save_idx))
        if gen_ema is not None:
            torch.save(gen_ema.state_dict(), all_save_dirs['model'] + 'gen_ema_%s_step_%d.pt'%(model_desc, save_idx))
    else:
        torch.save(gen.state_dict(), all_save_dirs['model'] + 'gen_step_%d.pt'%save_idx)
        torch.save(discrim.state_dict(), all_save_dirs['model'] + 'discrim_step_%d.pt'%save_idx)
        if gen_ema is not None:
            torch.save(gen_ema.state_dict(), all_save_dirs['model'] + 'gen_ema_step_%d.pt'%save_idx)
    
    if save_gen_fixed:
        modeling_utils.save_gen_fixed_noise(gen_ema if gen_ema is not None else gen, 
                fixed_noise, all_save_dirs['fixed'], save_idx, **kwargs)
    
    if plot_gen_samples or save_gen_samples:
        samples = modeling_utils.sample_gen_images(gen_ema if gen_ema is not None else gen, noise_size,
                                                    device, **kwargs)
        if plot_gen_samples:
            modeling_utils.plot_imgs(samples)
        if save_gen_samples:
            modeling_utils.save_imgs(samples, all_save_dirs['sample'] + 'step_%d.png'%save_idx)
    else:
        samples = None
    if fid: #TODO
        print('making images...')
        modeling_utils.save_gen_fid_images(gen_ema if gen_ema is not None else gen, noise_size, all_save_dirs['fid'], n_fid_samples, device, **kwargs)
        print('calcuatling stats')
        fid_utils.calculate_statistics_for_dataset(all_save_dirs['fid'] + 'tmp_gen_images/', all_save_dirs['fid'] + 'fake.npy', 8, device)
        print('calcuatling fid')
        metric_dict['fid'] = fid_utils.calculate_fid(all_save_dirs['fid'] + 'real.npy', all_save_dirs['fid'] + 'fake.npy')
        print('cleaning up')
        fid_utils.cleanup_fid(all_save_dirs['fid'])
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



"""General functions for saving metrics"""
def setup_model_save_directory(save_dir, save_gen_samples, save_gen_fixed, tensorboard, 
                    fid, n_fid_samples, fid_real_path, device):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    sample_dir = os.path.join(save_dir, 'random_samples/')
    fixed_dir = os.path.join(save_dir, 'fixed_samples/')
    model_dir = os.path.join(save_dir, 'models/')
    tensorboard_dir = os.path.join(save_dir, 'tensorboard/')
    fid_dir = os.path.join(save_dir, 'fid/')

    if save_gen_samples and not os.path.exists(sample_dir):
        os.mkdir(sample_dir)
    if save_gen_fixed and not os.path.exists(fixed_dir):
        os.mkdir(fixed_dir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if tensorboard and not os.path.exists(tensorboard_dir):
        os.mkdir(tensorboard_dir)
    if fid and not os.path.exists(fid_dir):
        os.mkdir(fid_dir)
        os.mkdir(fid_dir + 'tmp_gen_images/')
        fid_utils.calculate_statistics_for_dataset(fid_real_path, fid_dir + 'real.npy', 
                        8, device, n_fid_samples//8)


    return {
        'sample' : sample_dir,
        'fixed' : fixed_dir,
        'model' : model_dir,
        'tensorboard' : tensorboard_dir,
        'fid' : fid_dir,
    }

def setup_tensorboard(tensorboard, save_path):
    if not tensorboard:
        return
    writer = SummaryWriter(save_path)
    return writer


def get_checkpoint_interval(checkpoint_interval, loader):
    final_checkpoint = True
    # If checkpoint interval is a float, then treat it as a % - e.g. .25 -> evaulte every 25% of training
    # final checkpoint: checkpoint when we're done; We don't want to if we've checkpointed very recently
    # e.g. len(loader)==401, checkpoint_interval=.25; final checkpoint will be for one batch -> noisy
    if not isinstance(checkpoint_interval, int):
        checkpoint_interval = int(checkpoint_interval * len(loader))
        if not checkpoint_interval % len(loader):
            final_checkpoint = False
        else:
            final_checkpoint = True
    return final_checkpoint, checkpoint_interval

"""Optimization steps for wasserstein gan + gradient penalty"""
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
    return fake_pred.mean(), real_pred.mean(), gradient_penalty, drift * (real_pred**2).mean()
    


def gen_loss_wasserstein(discrim, fake_samples, depth, alpha):
    return -discrim(fake_samples, depth, alpha).mean()


def discrim_step_wasserstein_gp(discrim, discrim_opt, gen, real_images, batch_noise, depth, 
                                    alpha, grad_pen_weight, device):
    fake_images = gen(batch_noise, depth, alpha).detach()
    fake_pred, real_pred, gp, drift = discrim_loss_wasserstein_gp(discrim, real_images, fake_images, depth, alpha, grad_pen_weight, device)
    loss = fake_pred - real_pred + gp + drift
    discrim_opt.zero_grad()
    loss.backward()
    discrim_opt.step()
    return fake_pred.item(), real_pred.item(), gp.item(), drift.item()
        
def gen_step_wasserstein(gen, gen_opt, gen_ema, discrim, batch_noise, depth, alpha):
    fake_images = gen(batch_noise, depth, alpha)
    loss = gen_loss_wasserstein(discrim, fake_images, depth, alpha)
    gen_opt.zero_grad()
    loss.backward()
    gen_opt.step()
    return loss.item()


def wasserstein_gp_batch(gen, gen_opt, gen_ema, discrim, discrim_opt, depth, fade_in, 
                                x_batch, batch_noise, device, grad_pen_weight, **kwargs):

    fake_pred, real_pred, gp, drift = discrim_step_wasserstein_gp(discrim, discrim_opt, gen, x_batch, \
                        batch_noise, depth, fade_in, grad_pen_weight, device)
    g_loss = gen_step_wasserstein(gen, gen_opt, gen_ema, discrim, batch_noise, depth, fade_in)

    return {"d_fake_pred" : fake_pred, "d_real_pred" : real_pred, "grad_pen" : gp, "g_loss" : g_loss, "drift" : drift, }



def discrim_loss_hinge(discrim, batch_real, batch_fake, depth, fade_in):
    real_preds = discrim(batch_real, depth, fade_in)
    fake_preds = discrim(batch_fake, depth, fade_in)

    real_loss = torch.mean(F.relu(1-real_preds))
    fake_loss = torch.mean(F.relu(1+fake_preds))
    return real_loss, fake_loss

def gen_loss_hinge(discrim, fake_imgs, depth, fade_in):
    return -torch.mean(discrim(fake_imgs, depth, fade_in))

def discrim_step_hinge(discrim, discrim_opt, gen, batch_real, batch_noise, depth, fade_in):
    fake_images = gen(batch_noise, depth, fade_in).detach()
    real_loss, fake_loss = discrim_loss_hinge(discrim, batch_real, fake_images, depth, fade_in)
    loss = real_loss + fake_loss
    discrim_opt.zero_grad()
    loss.backward()
    discrim_opt.step()
    return real_loss.item(), fake_loss.item()

def gen_step_hinge(gen, gen_opt, gen_ema, discrim, batch_noise, depth, fade_in):
    fake_images = gen(batch_noise, depth, fade_in)
    gen_loss = gen_loss_hinge(discrim, fake_images, depth, fade_in)
    gen_opt.zero_grad()
    gen_loss.backward()
    gen_opt.step()
    update_average(gen_ema, gen, .999)
    return gen_loss.item()


def hinge_batch(gen, gen_opt, gen_ema, discrim, discrim_opt, depth, fade_in, x_batch, batch_noise, device, **kwargs):
    d_real_loss, d_fake_loss = discrim_step_hinge(discrim, discrim_opt, gen, x_batch, batch_noise, depth, fade_in)
    g_loss = gen_step_hinge(gen, gen_opt, gen_ema, discrim, batch_noise, depth, fade_in)
    return {'d_loss_real' : d_real_loss, 'd_fake_loss' : d_fake_loss, 'g_loss' : g_loss}
    

def dict_mean(dictionary):
    return {k : np.mean(v) for k, v in dictionary.items()}
def list_dict_update(old, new):
    for k, v in new.items():
        old[k].append(v)
    return old
# getattr that function
# pass aguments for taking a step into that function

def train_on_depth_progan(gen, gen_opt, gen_ema, discrim, discrim_opt, depth, nb_epochs, fade_in_pct,
    loader, device, noise_size, loss, grad_step_every = 1, checkpoint_interval = -1, save_dir=None, print_metrics=False,
    fixed_noise = None, plot_gen_samples=True, save_gen_samples=True, save_gen_fixed=True, tensorboard=True, 
    fid=False, n_fid_samples=10000, fid_real_path=None, **kwargs):
    
    all_save_dirs = setup_model_save_directory(save_dir, save_gen_samples, save_gen_fixed, tensorboard, 
                        fid, n_fid_samples, fid_real_path, device)
    writer = setup_tensorboard(tensorboard, all_save_dirs['tensorboard'])
    loss_function = globals()[loss]

    final_checkpoint, checkpoint_interval = get_checkpoint_interval(checkpoint_interval, loader)
    
    counter = 0
    epoch_pbar = tqdm(total = nb_epochs, leave=False)
    for epoch in range(nb_epochs):
        batch_pbar = tqdm(total = len(loader), leave=False)
        metric_dict = defaultdict(list)
        for i, x_batch in enumerate(loader):
            gen.train()
            gen_ema.train()
            discrim.train()
            if fade_in_pct > 0:
                fade_in = min(1, counter/(fade_in_pct * len(loader) * nb_epochs))
            else:
                fade_in = 1
            if depth==0 or fade_in >= 1:
                pass
            else:
                x_batch_down_up = F.interpolate(nn.AvgPool2d(2)(x_batch), scale_factor=2)
                x_batch = fade_in * x_batch + ((1-fade_in) * x_batch_down_up)
            x_batch = x_batch.to(device)
            
            batch_noise = modeling_utils.generate_noise(len(x_batch), noise_size, device)
            
            this_batch_metrics = loss_function(gen, gen_opt, gen_ema, discrim, discrim_opt, depth, fade_in, \
                                                x_batch, batch_noise, device, **kwargs)
            metric_dict = list_dict_update(metric_dict, this_batch_metrics)

            counter+=1
            if checkpoint_interval > 0 and i > 0 and not i % checkpoint_interval:
                metric_dict = dict_mean(metric_dict)
                checkpoint(gen, gen_ema, discrim, device, noise_size, all_save_dirs, print_metrics, fixed_noise, plot_gen_samples,
                            save_gen_samples, save_gen_fixed, fid and i > int(.98*len(loader))  , n_fid_samples, 
                            tensorboard, writer, metric_dict, 
                            depth=depth, alpha=fade_in, model_desc="depth_%d_fade_%d"%(depth, int(100*min(fade_in, 1))))
                metric_dict = defaultdict(list)

            batch_pbar.update(1)
            
        epoch_pbar.update(1)
        batch_pbar.close()
        if final_checkpoint:
            metric_dict = dict_mean(metric_dict)
            checkpoint(gen, gen_ema, discrim, device, noise_size, all_save_dirs, print_metrics, fixed_noise, plot_gen_samples,
                    save_gen_samples, save_gen_fixed, fid, n_fid_samples, tensorboard, writer, metric_dict, depth=depth, alpha=1,
                    model_desc="depth_%d_fade_%d"%(depth, int(100*min(fade_in, 1))))
    epoch_pbar.close()
    


"""Optimization steps for traditional gan"""
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
    final_checkpoint, checkpoint_interval = get_checkpoint_interval(checkpoint_interval, loader)
    
    for i, batch in enumerate(loader):
        gen.train()
        discrim.train()
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
    if final_checkpoint:
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