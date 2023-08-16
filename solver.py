from model import Generator
from model import Discriminator
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import pandas as pd


class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, train_loader,test_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
#         self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
#         self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.torch.cuda.FloatTensor
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        self.G = Generator(self.g_conv_dim, self.c_dim, self.g_repeat_num)
        self.D = Discriminator(self.image_size, self.d_conv_dim, self.c_dim, self.d_repeat_num) 


        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), self.d_lr, [self.beta1, self.beta2])
        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
            
        self.G.to(self.device)
        self.D.to(self.device)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(resume_iters))
        self.G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
        self.D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))


    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import Logger
        self.logger = Logger(self.log_dir)

    def update_lr(self, g_lr, d_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.g_optimizer.param_groups:
            param_group['lr'] = g_lr
        for param_group in self.d_optimizer.param_groups:
            param_group['lr'] = d_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)
    
    def mean_absolute_error(self,image_true, image_generated):
        return torch.abs(image_true - image_generated).mean()


    def peak_signal_to_noise_ratio(self,image_true, image_generated):
        mse = ((image_true - image_generated) ** 2).mean().cpu()
        return -10 * np.log10(mse)


    def structural_similarity_index(self,image_true, image_generated, C1=0.01, C2=0.03):

        mean_true = image_true.mean()
        mean_generated = image_generated.mean()
        std_true = image_true.std()
        std_generated = image_generated.std()
        covariance = ((image_true - mean_true) * (image_generated - mean_generated)).mean()
    
        numerator = (2 * mean_true * mean_generated + C1) * (2 * covariance + C2)
        denominator = ((mean_true ** 2 + mean_generated ** 2 + C1) *(std_true ** 2 + std_generated ** 2 + C2))
        return numerator / denominator


#     def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
#         """Generate target domain labels for debugging and testing."""
#         # Get hair color indices.
#         if dataset == 'CelebA':
#             hair_color_indices = []
#             for i, attr_name in enumerate(selected_attrs):
#                 if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
#                     hair_color_indices.append(i)

#         c_trg_list = []
#         for i in range(c_dim):
#             c_trg = c_org.clone()
#             if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
#                 c_trg[:, i] = 1
#                 for j in hair_color_indices:
#                     if j != i:
#                         c_trg[:, j] = 0
#             else:
#                 c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.

#             c_trg_list.append(c_trg.to(self.device))
#         return c_trg_list

    
    def train(self):
        """Train StarGAN within a single dataset."""


        # Fetch fixed inputs for debugging.
        data_iter = iter(self.train_loader)
        x_fixed_whole = next(data_iter)
        x_fixed = x_fixed_whole[0].to(self.device, dtype=torch.float)
        x_fixed=x_fixed.reshape(1,1,240,240)
        c_fixed_list =torch.zeros((3,3)).to(self.device)
        c_fixed_list[0][0]=1
        c_fixed_list[1][1]=1
        c_fixed_list[2][2]=1

        # Learning rate cache for decaying.
        g_lr = self.g_lr
        d_lr = self.d_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(start_iters, self.num_iters):
            try:
                x_fixed_whole = next(data_iter)
            except:
                data_iter = iter(self.train_loader)
                x_fixed_whole = next(data_iter)
                

            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

                # Inputs T1-w and T2-w
            real_t1 = x_fixed_whole[0].to(self.device, dtype=torch.float)
            c_t1 = torch.zeros((1,3)).to(self.device)
            c_t1[0][0]=1

            real_ir = x_fixed_whole[1].to(self.device, dtype=torch.float)
            c_ir = torch.zeros((1,3)).to(self.device)
            c_ir[0][1]=1

            real_flr = x_fixed_whole[2].to(self.device, dtype=torch.float)
            c_flr = torch.zeros((1,3)).to(self.device)
            c_flr[0][2]=1

            real_t1=real_t1.reshape(1,1,240,240)
            real_ir=real_ir.reshape(1,1,240,240)
            real_flr=real_flr.reshape(1,1,240,240)  

#                 real_t1=real_t1.to(self.device)
#                 real_ir=real_ir.to(self.device)
#                 real_flr=real_flr.to(self.device)

            # =================================================================================== #
            #                             2. Train the discriminator                              #
            # =================================================================================== #

            # Compute loss with real images.
            out_src, out_cls = self.D(real_flr)
            d_loss_real = - torch.mean(out_src)
            d_loss_cls = F.binary_cross_entropy_with_logits(out_cls, c_flr, size_average=False) / out_cls.size(0)

            # Compute loss with fake images.
            fake_flr = self.G(real_t1, c_flr)
            out_src, out_cls = self.D(fake_flr.detach())
            d_loss_fake = torch.mean(out_src)

            # Compute loss for gradient penalty.
            alpha = torch.rand(real_flr.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * real_flr.data + (1 - alpha) * fake_flr.data).requires_grad_(True)
            out_src, _ = self.D(x_hat)
            d_loss_gp = self.gradient_penalty(out_src, x_hat)

            # Backward and optimize.
            d_loss = d_loss_real + d_loss_fake + self.lambda_cls * d_loss_cls + self.lambda_gp * d_loss_gp
            self.reset_grad()
            d_loss.backward()
            self.d_optimizer.step()

            # Logging.
            loss = {}
            loss['D/loss_real'] = d_loss_real.item()
            loss['D/loss_fake'] = d_loss_fake.item()
            loss['D/loss_cls'] = d_loss_cls.item()    

            # =================================================================================== #
            #                               3. Train the generator                                #
            # =================================================================================== #

            if (i+1) % self.n_critic == 0:
                # Original-to-target domain.
                fake_flr = self.G(real_t1, c_flr)
                out_src, out_cls = self.D(fake_flr)
                g_loss_fake = -torch.mean(out_src)
                g_loss_cls =  F.binary_cross_entropy_with_logits(out_cls, c_flr, size_average=False) / out_cls.size(0)


                # Target-to-original domain.
                x_reconst = self.G(fake_flr, c_t1)
                g_loss_rec = torch.mean(torch.abs(real_t1 - x_reconst))

                # Backward and optimize.
                g_loss = g_loss_fake + self.lambda_rec * g_loss_rec + self.lambda_cls * g_loss_cls
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # Logging.
                loss['G/loss_fake'] = g_loss_fake.item()
                loss['G/loss_rec'] = g_loss_rec.item()
                loss['G/loss_cls'] = g_loss_cls.item()
                loss['D/loss_gp'] = d_loss_gp.item()

            # =================================================================================== #
            #                                 4. Miscellaneous                                    #
            # =================================================================================== #

            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i+1)

            # Translate fixed images for debugging.
            if (i+1) % self.sample_step == 0:
                with torch.no_grad():
                    x_fake_list = [x_fixed]
                    for c_fixed in c_fixed_list:
                        c_fixed=c_fixed.reshape(1,3)
                        x_fake_list.append(self.G(x_fixed, c_fixed))
                    x_concat = torch.cat(x_fake_list, dim=3)
                    sample_path = os.path.join(self.sample_dir, '{}-images.jpg'.format(i+1))
#                     save_image(self.denorm(x_concat.data.cpu()), sample_path, nrow=1, padding=0)
                    save_image(x_concat.data.cpu(), sample_path, nrow=1, padding=0)
                    print('Saved real and fake images into {}...'.format(sample_path))

            # Save model checkpoints.
            if (i+1) % self.model_save_step == 0:
                G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                D_path = os.path.join(self.model_save_dir, '{}-D.ckpt'.format(i+1))
                torch.save(self.G.state_dict(), G_path)
                torch.save(self.D.state_dict(), D_path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

            # Decay learning rates.
            if (i+1) % self.lr_update_step == 0 and (i+1) > (self.num_iters - self.num_iters_decay):
                g_lr -= (self.g_lr / float(self.num_iters_decay))
                d_lr -= (self.d_lr / float(self.num_iters_decay))
                self.update_lr(g_lr, d_lr)
                print ('Decayed learning rates, g_lr: {}, d_lr: {}.'.format(g_lr, d_lr))


    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)       

        data_loader = self.test_loader
        
        with torch.no_grad():
            for i, batch in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_real = batch[0].to(self.device, dtype=torch.float)
                x_real=x_real.reshape(1,1,240,240)
                x_ir = batch[1].to(self.device, dtype=torch.float)
                x_ir = x_ir.reshape(1,1,240,240)
                x_flr = batch[2].to(self.device, dtype=torch.float)
                x_flr = x_flr.reshape(1,1,240,240)
#                 c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                c_trg_list = c_fixed_list=torch.zeros((3,3)).to(self.device)
                c_trg_list[0][0]=1
                c_trg_list[1][1]=1
                c_trg_list[2][2]=1

                # Translate images.
                
                x_fake_list = [x_real]
                x_fake_list.append(x_ir)
                x_fake_list.append(x_flr)
                for c_trg in c_trg_list:
                    c_trg=c_trg.reshape(1,3)
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.

                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
#                 save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                save_image(x_concat.data.cpu(), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))
        
        #################################################################################################################
                                                   #Evaluate Generator#
            #############################################################################################################
        
    def evaluate(self):

        res_train, res_test = [], []

        with torch.no_grad():

            for i, batch in enumerate(self.train_loader):

                # Inputs T1-w and T2-w
                x_t1 = batch[0].to(self.device, dtype=torch.float)
                x_t1 = x_t1.reshape(1,1,240,240)
                x_ir=batch[1].to(self.device, dtype=torch.float)
                x_ir=x_ir.reshape(1,1,240,240)
                x_flr=batch[2].to(self.device, dtype=torch.float)
                x_flr=x_flr.reshape(1,1,240,240)

                c_trg_list = c_fixed_list=torch.zeros((3,3)).to(self.device)
                c_trg_list[0][0]=1
                c_trg_list[1][1]=1
                c_trg_list[2][2]=1


    #             fake_t2 = generator(x_t1)
                x_fake_list = []
                for c_trg in c_trg_list:
                    c_trg=c_trg.reshape(1,3)
                    x_fake_list.append(self.G(x_t1, c_trg))

                mae_t1 = self.mean_absolute_error(x_t1, x_fake_list[0]).item()
                psnr_t1 = self.peak_signal_to_noise_ratio(x_t1, x_fake_list[0]).item()
                ssim_t1 = self.structural_similarity_index(x_t1, x_fake_list[0]).item()

                mae_ir = self.mean_absolute_error(x_ir, x_fake_list[1]).item()
                psnr_ir = self.peak_signal_to_noise_ratio(x_ir, x_fake_list[1]).item()
                ssim_ir = self.structural_similarity_index(x_ir, x_fake_list[1]).item()

                mae_flr = self.mean_absolute_error(x_flr,x_fake_list[2] ).item()
                psnr_flr = self.peak_signal_to_noise_ratio(x_flr, x_fake_list[2]).item()
                ssim_flr = self.structural_similarity_index(x_flr, x_fake_list[2]).item()

                res_train.append([mae_t1, psnr_t1, ssim_t1,mae_ir, psnr_ir, ssim_ir,mae_flr, psnr_flr, ssim_flr])



            for i, batch in enumerate(self.test_loader):

                # Inputs T1-w and T2-w
                x_t1 = batch[0].to(self.device, dtype=torch.float)
                x_t1 = x_t1.reshape(1,1,240,240)
                x_ir=batch[1].to(self.device, dtype=torch.float)
                x_ir=x_ir.reshape(1,1,240,240)
                x_flr=batch[2].to(self.device, dtype=torch.float)
                x_flr=x_flr.reshape(1,1,240,240)

                c_trg_list = c_fixed_list=torch.zeros((3,3)).to(self.device)
                c_trg_list[0][0]=1
                c_trg_list[1][1]=1
                c_trg_list[2][2]=1


    #             fake_t2 = generator(x_t1)
                x_fake_list = []
                for c_trg in c_trg_list:
                    c_trg=c_trg.reshape(1,3)
                    x_fake_list.append(self.G(x_t1, c_trg))

                mae_t1 = self.mean_absolute_error(x_t1, x_fake_list[0]).item()
                psnr_t1 = self.peak_signal_to_noise_ratio(x_t1, x_fake_list[0]).item()
                ssim_t1 = self.structural_similarity_index(x_t1, x_fake_list[0]).item()

                mae_ir = self.mean_absolute_error(x_ir, x_fake_list[1]).item()
                psnr_ir = self.peak_signal_to_noise_ratio(x_ir, x_fake_list[1]).item()
                ssim_ir = self.structural_similarity_index(x_ir, x_fake_list[1]).item()

                mae_flr = self.mean_absolute_error(x_flr,x_fake_list[2] ).item()
                psnr_flr = self.peak_signal_to_noise_ratio(x_flr, x_fake_list[2]).item()
                ssim_flr = self.structural_similarity_index(x_flr, x_fake_list[2]).item()

                res_test.append([mae_t1, psnr_t1, ssim_t1,mae_ir, psnr_ir, ssim_ir,mae_flr, psnr_flr, ssim_flr])

        df = pd.DataFrame([
            pd.DataFrame(res_train, columns=['MAE_T1', 'PSNR_T1', 'SSIM_T1','MAE_IR', 'PSNR_IR', 'SSIM_IR',
                                             'MAE_FLR', 'PSNR_FLR', 'SSIM_FLR']).mean().squeeze(),
            pd.DataFrame(res_test, columns=['MAE_T1', 'PSNR_T1', 'SSIM_T1','MAE_IR', 'PSNR_IR', 'SSIM_IR',
                                             'MAE_FLR', 'PSNR_FLR', 'SSIM_FLR']).mean().squeeze()
        ], index=['Training set', 'Test set']).T
        return df

