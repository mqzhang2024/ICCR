import torch
import torch.nn as nn
from .Discriminator import MultiscaleDiscriminator,GANLoss
from cldm.model import create_model, load_state_dict
import functools
import pytorch_lightning as pl
import os
import json
import einops
from .PiPNet import PipDetection
import torch.nn.functional as F

class CR(pl.LightningModule):
    def __init__(self,bs,use_FM,use_PiP,sd_file,pip_face_file,pip_weight_file,gpu_id=0,show_params=False,num_D=2,n_layers_D=3,every_n_D=40,use_Grad=False):
        super().__init__()
        self.automatic_optimization = False
        # Configs
        self.bs=bs
        self.gpu_id=gpu_id
        self.num_D=num_D
        self.n_layers_D=n_layers_D
        self.every_n_D=every_n_D
        self.use_FM=use_FM
        self.use_PiP=use_PiP

        self.use_Grad=use_Grad
        
        # Loss of GAN
        self.criterionGAN=GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor)
        
        # Loss of Feature matching 
        if self.use_FM:
            self.criterionFeat = torch.nn.L1Loss() 
        
        # Loss of PiPNet 
        if self.use_PiP:
            self.detector_land = PipDetection(device='cuda:'+str(self.gpu_id),face_file=pip_face_file,weight_file=pip_weight_file,use_grad=False).to('cuda:'+str(self.gpu_id))
            self.criterionPiP=nn.MSELoss()
        

        self.fake_pool = ImagePool(0)

        # loading ControlNet
        self.generator = create_model('./models/cldm_v15.yaml').cpu()   
        self.generator.load_state_dict(load_state_dict(sd_file, location='cpu'))
        self.generator.sd_locked = True
        self.generator.only_mid_control = False

        # loading MultiscaleDiscriminator
        self.discriminator = self.define_D(6, 64, self.n_layers_D,'instance', 'false',self.num_D, self.use_FM)


        if show_params:
            g_params=self.get_parameter_number(self.generator)
            print('parameters of Controlnet',g_params)
            d_params=self.get_parameter_number(self.discriminator)
            print('parameters of Dicriminator',d_params)
            if self.use_PiP:
                pip_params=self.get_parameter_number(self.detector_land)
                print('parameters of PiPNet',pip_params)
            

        

    def get_parameter_number(self,net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total parameters': total_num, 'Trainable parameters': trainable_num}


    def configure_optimizers(self):
        params1 = list(self.discriminator.parameters())    
        opt_d = torch.optim.AdamW(params1, lr=2e-7)

        params2 = list(self.generator.control_model.parameters())
        opt_g = torch.optim.AdamW(params2, lr=1e-5)

        return opt_g, opt_d
    
    
    def training_step(self, batch, batch_idx):   
        self.discriminator.train()
        self.generator.train()

        g_opt,d_opt=self.optimizers()

        real_images=einops.rearrange( batch['jpg'], 'b h w c -> b c h w')  
        cond_images=einops.rearrange( batch['hint'], 'b h w c -> b c h w')
        
        # Loss of ContorlNet
        x, c = self.generator.get_input(batch,'jpg')   
        loss_Noise,fake_images = self.generator(x,c)


        # ================= Optimise the Discriminator ==================
        # Fake Detection and Loss  
        pred_fake_pool = self.discriminate(cond_images, fake_images) 
        loss_fake = self.criterionGAN(pred_fake_pool, False)     

        # Real Detection and Loss  
        pred_real = self.discriminate(cond_images, real_images) 
        loss_real = self.criterionGAN(pred_real, True)

        D_loss = (loss_real + loss_fake) * 0.5
        self.log('D_loss', D_loss, prog_bar=True)


        # ================= Optimise the ContorlNet =================
        loss_Noise=loss_Noise*10
                    
        # Loss of GAN    
        pred_fake=self.discriminator.forward(torch.cat((cond_images,fake_images),dim=1))        
        loss_GAN = self.criterionGAN(pred_fake, True) 

        # Loss of Feature matching 
        loss_FM = 0
        if self.use_FM: 
            feat_weights = 4.0 / (self.n_layers_D + 1)   
            D_weights = 1.0 / self.num_D
            for i in range(self.num_D):  
                for j in range(len(pred_fake[i])-1):  
                    cos=(F.cosine_similarity(pred_real[i][j].detach(), pred_fake[i][j], dim=0)).mean()
                    loss_FM +=feat_weights* D_weights * (1 - cos) * 7.0
        
        # Loss of PiPNet
        loss_PiP=0
        if self.use_PiP:
            x = (real_images * 127.5 + 127.5).clip(0, 255).div(255.0)
            x_landmarks =self.detector_land(x).reshape(x.shape[0] * 1, -1, 2)
            y = (fake_images * 127.5 + 127.5).clip(0, 255).div(255.0)
            y_landmarks =self.detector_land(y).reshape(y.shape[0] * 1, -1, 2)
            loss_PiP =self.criterionPiP(torch.Tensor(x_landmarks), torch.Tensor(y_landmarks))*150

        # total loss
        G_loss=loss_GAN + loss_Noise + loss_FM  + loss_PiP 
        self.log('G_loss', G_loss, prog_bar=True)

        g_opt.zero_grad()
        self.manual_backward(G_loss)
        g_opt.step()

        if self.global_step%self.every_n_D==0:
            d_opt.zero_grad()
            self.manual_backward(D_loss)
            d_opt.step()
        

        # ======save train loss=====
        self.save_json('train',float(loss_GAN),float(loss_Noise),float(G_loss),float(loss_fake),float(loss_real),float(D_loss),float(loss_FM),float(loss_PiP))

 
    @torch.no_grad()
    def validation_step(self, batch, batch_idx): 
        self.discriminator.eval()
        self.generator.eval()

        real_images=einops.rearrange( batch['jpg'], 'b h w c -> b c h w')   
        cond_images=einops.rearrange( batch['hint'], 'b h w c -> b c h w')
        
        # Loss of ContorlNet
        x, c = self.generator.get_input(batch,'jpg')   
        loss_Noise,fake_images = self.generator(x,c)


        # ================= Optimise the Discriminator ==================
        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(cond_images, fake_images, use_pool=True) 
        loss_fake = self.criterionGAN(pred_fake_pool, False)     

        # Real Detection and Loss  
        pred_real = self.discriminate(cond_images, real_images)  
        loss_real = self.criterionGAN(pred_real, True)

        D_loss = (loss_real + loss_fake) * 0.5
        self.log('D_loss', D_loss, prog_bar=True)


        # ================= Optimise the ContorlNet =================
        loss_Noise=loss_Noise*10
                    
        # Loss of GAN        
        pred_fake=self.discriminator.forward(torch.cat((cond_images,fake_images),dim=1))        
        loss_GAN = self.criterionGAN(pred_fake, True) 

        # Loss of Feature matching 
        loss_FM = 0
        if self.use_FM: 
            feat_weights = 4.0 / (self.n_layers_D + 1) 
            D_weights = 1.0 / self.num_D
            for i in range(self.num_D):   
                for j in range(len(pred_fake[i])-1):  
                    cos=(F.cosine_similarity(pred_real[i][j].detach(), pred_fake[i][j], dim=0)).mean()
                    loss_FM +=feat_weights* D_weights * (1 - cos) * 7.0
        
        # Loss of PiPNet
        loss_PiP=0
        if self.use_PiP:
            x = (real_images * 127.5 + 127.5).clip(0, 255).div(255.0)
            x_landmarks =self.detector_land(x).reshape(x.shape[0] * 1, -1, 2)
            y = (fake_images * 127.5 + 127.5).clip(0, 255).div(255.0)
            y_landmarks =self.detector_land(y).reshape(y.shape[0] * 1, -1, 2)
            loss_PiP =self.criterionPiP(torch.Tensor(x_landmarks), torch.Tensor(y_landmarks))*150

        # total loss
        G_loss=loss_GAN + loss_Noise + loss_FM + loss_PiP
        self.log('G_loss', G_loss, prog_bar=True)


        # ======save valid loss=====
        self.save_json('valid',float(loss_GAN),float(loss_Noise),float(G_loss),float(loss_fake),float(loss_real),float(D_loss),float(loss_FM),float(loss_PiP))
        
         
    def update_learning_rate(self):  
        lrd = 1e-5 / 10
        lrg = 1e-5 / 10
        lrd1 = self.old_lrd - lrd
        lrg1 = self.old_lrg - lrg    
        opt_g,opt_d=self.optimizers()    
        for param_group in opt_d.param_groups:
            param_group['lr'] = lrd1
        for param_group in opt_g.param_groups:
            param_group['lr'] = lrg1
        print('update discriminator learning rate: '+str(self.old_lrd)+' -> '+str(lrd1))
        print('update generator learning rate: '+str(self.old_lrg)+' -> '+str(lrg1))
        self.old_lrd = lrd1
        self.old_lrg = lrg1

   
    # save loss to ./lightning_logs/image_log/
    def save_json(self,perfix,loss_GAN,loss_Noise,G_loss,loss_fake,loss_real,D_loss,loss_FM,loss_PiP):
        os.makedirs('./lightning_logs/image_log',exist_ok=True)
        with open('./lightning_logs/image_log/'+perfix+'_loss.json', 'a') as f1:
            data={
                  'epoch':self.current_epoch,
                  'global_step':self.global_step,
                  'loss_GAN':loss_GAN,
                  'loss_Noise':loss_Noise,
                  'loss_FM':loss_FM,
                  'loss_PiP':loss_PiP,
                  'G_loss':G_loss,

                  'loss_fake':loss_fake,
                  'loss_real':loss_real,
                  'D_loss':D_loss,
                  'prefix':perfix
                  }
            json_str = json.dumps(data)
            f1.write(json_str)
            f1.write('\n') 

   

    # ==============Network of Discriminator==================
    def weights_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def get_norm_layer(self,norm_type='instance'):
        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
        return norm_layer

    def define_D(self,input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False):        
        norm_layer = self.get_norm_layer(norm_type=norm)   
        netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)   
        # print(netD)
        netD.to('cuda:'+str(self.gpu_id))
        netD.apply(self.weights_init)
        return netD

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.discriminator.forward(fake_query)
        else:
            return self.discriminator.forward(input_concat)
        



class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = Variable(torch.cat(return_images, 0))
        return return_images
