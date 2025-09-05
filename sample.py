import torch
from muti_control.ICCR import ICCR
from cldm.model import load_state_dict
import os
from cldm.ddim_hacked import DDIMSampler
from pytorch_lightning import seed_everything
import cv2
from annotator.util import resize_image, HWC3
import numpy as np
import einops
from PIL import Image


def get_prompt(sex,age,body_mass):
    if age<14:
        age_level=1
    elif age<30:
        age_level=2
    elif age<50:
        age_level=3
    elif age<65:
        age_level=4
    else:
        age_level=5
       
    prompt="3D face scan,Asian "+sex+",age level "+str(age_level)+","

    if body_mass=='1':
        prompt+= 'very thin face.'
    elif body_mass=='2':
        prompt+= 'thin face.' 
    elif body_mass=='3':
        prompt+= 'slightly thin face.' 
    elif body_mass=='4':
        prompt+= 'normal face.'
    elif body_mass=='5':
        prompt+= 'slightly fat face.' 

    return prompt


def process_img(file_path,num_samples=1,size=256):
    image = cv2.imread(file_path)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize_image(HWC3(image), size)
    image = (image.astype(np.float32) / 127.5) - 1.0 
    image = torch.tensor(image).to(device)
    image=torch.stack([image for _ in range(num_samples)], dim=0)
    image= einops.rearrange(image, 'b h w c -> b c h w').clone()
    return image


def iccr_sample(prompt,skull_image): 
    control = skull_image.to(memory_format=torch.contiguous_format).float()

    # ddim sample config
    ddim_steps=50
    scale=9.0    
    eta=1.0 
    seed=17027
    seed_everything(seed)  

    # path of results
    os.makedirs('./results',exist_ok=True)
    
    # iccr sample config
    iccr_model.use_Grad=True
    cld_model=iccr_model.generator
    ddim_sampler = DDIMSampler(iccr_model)

    # sampling
    with torch.no_grad():
        cond = {"c_concat": [control], "c_crossattn": [cld_model.get_learned_conditioning([prompt])]}
        un_cond = {"c_concat": [control], "c_crossattn": [cld_model.get_learned_conditioning([''] * len([prompt]))]}
        shape = (4, 256 // 8, 256 // 8)     

        samples, _ = ddim_sampler.sample(ddim_steps, 1, shape, cond, verbose=False, eta=eta,unconditional_guidance_scale=scale,unconditional_conditioning=un_cond)
        x_samples = cld_model.decode_first_stage(samples)            
        x_samples=torch.clamp(x_samples, -1., 1.)   
        x_samples=x_samples[0].detach().cpu().permute(1,2,0).numpy()   
        x_samples = (((x_samples+ 1.0) / 2.0)* 255).astype(np.uint8)
            
        Image.fromarray(x_samples).save('./results/1.png')


               

if __name__ == '__main__':
    # Configs
    device='cuda:0'
    gpu_id=int(device.split(':')[-1])
    use_PiP=True
    use_FM=True
    pip_face_file='meanface.txt'    # meanface file from PiPNet
    pip_weight_file='pipnet.pth'    # weight file from PiPNet
    sd_file='./models/control_sd15_ini.ckpt'     
    iccr_weight=''                  # ICCR weight

    # load model
    iccr_model = ICCR(1,use_FM,use_PiP,sd_file,pip_face_file,pip_weight_file)  
    iccr_model.load_state_dict(load_state_dict(iccr_weight,location=device))
    iccr_model = iccr_model.to(device) 
    iccr_model.eval()   

    # biometric information
    age=30          #  0-120
    sex='male'      #  male;female
    body_mass='1'   #  1,2,3,4,5
    prompt=get_prompt(sex,age,body_mass)

    # skull image
    skull_path=''   # path of skull image
    skull_image=process_img(skull_path)
    
    # sample
    iccr_sample(prompt,skull_image)
            
        
