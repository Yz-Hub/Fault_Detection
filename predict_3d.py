import os
import torch
from torch.amp import autocast
import numpy as np
from models.build import build_model
from data.build_data import get_test_loader
from utils.config import get_config
from monai.inferers import SlidingWindowInferer
from collections import OrderedDict

def predict_3d(config):

    input_folder = os.path.abspath("datasets/test/seismic")
    output_folder = os.path.abspath('datasets/test/fault')
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"[Input Directory] {input_folder}")
    print(f"[Output Directory] {output_folder}\n")
    
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder {input_folder} does not exist")
    

    window_infer = SlidingWindowInferer(
        roi_size=[config.data.img_size, config.data.img_size, config.data.img_size],
        sw_batch_size=2,
        overlap=0.5,
        mode='gaussian',
        progress=True,
        device=torch.device('cpu'),
    )
    

    model = build_model(config)
    model_path = os.path.join(config.train.output, config.model.name, 'checkpoints')
    model_file = os.path.join(model_path, config.model.name + '.pt')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory {model_path} does not exist")
    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"Model file {model_file} not found")
    

    newdict = OrderedDict()
    model_dict = torch.load(model_file, map_location='cpu', weights_only=False)
    for k, v in model_dict.items():
        newdict[k.replace('module.', '')] = v  
    model.load_state_dict(newdict)
    
    
    model.cuda()
    model.eval()
    
    
    with autocast(device_type='cuda', enabled=config.train.amp):
        with torch.no_grad():
            data_loader = get_test_loader(input_folder)
            print(f"Data loader length: {len(data_loader)}")
            
            for idx, data in enumerate(data_loader):
                test_input = data['image'].cuda()
                
                
                raw_path = data['image_meta_dict']['filename_or_obj'][0]
                filename = os.path.basename(raw_path)  
                print(f"Filename: {filename}")
                
                print(f"• Processing input file: {filename}")
                
                
                test_output = window_infer(inputs=test_input, network=model)
                print(f"Model output shape: {test_output.shape}")
                
                
                save_path = os.path.join(output_folder, filename)
                print(f"Save path: {save_path}")
                
                
                output_array = np.squeeze(torch.sigmoid(test_output[0]).cpu().numpy())
                np.save(save_path, output_array)
                
                print(f"✓ Output file saved to: {save_path}\n")
                print(f'Finished processing {filename}')

    print(f"Prediction completed! Processed {len(data_loader)} files in total")

if __name__ == '__main__':
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    
    config_path = 'configs/config.yaml'
    config = get_config(config_path)
    
    
    predict_3d(config)
