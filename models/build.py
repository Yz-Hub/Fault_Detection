from .LightEGEUnet import LightEGEUnet3D as LightEGEUnet
def build_model(config):
    model_type = config.model.name.upper()

		
    #LightGEUnet
    if model_type == "LIGHTEGEUNET":
        return LightEGEUnet(  
            in_chans=config.model.in_chans,
            num_classes=config.model.num_classes,
            c_list=config.model.c_list,
            bridge=config.model.bridge,
            drop_rate=config.model.drop
        )		

    
    else:
        raise NotImplementedError(f"Unsupported model: {model_type}")
