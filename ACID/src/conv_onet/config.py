import os
from src.encoder import encoder_dict
from src.conv_onet import models, training
from src.conv_onet import generation
from src import data

def get_model(cfg,device=None, dataset=None, **kwargs):
    if cfg['model']['type'] == 'geom':
        return get_geom_model(cfg,device,dataset)
    elif cfg['model']['type'] == 'combined':
        return get_combined_model(cfg,device,dataset)


def get_combined_model(cfg, device=None, dataset=None, **kwargs):
    ''' Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    dim = cfg['data']['dim']
    act_dim = cfg['data']['act_dim']
    obj_c_dim = cfg['model']['obj_c_dim']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    obj_encoder_kwargs = cfg['model']['obj_encoder_kwargs']
    padding = cfg['data']['padding']
    decoder = 'combined_decoder'
    encoder = 'geom_encoder'
    
    if 'env_c_dim' in cfg['model'] and 'env_c_dim' != 0:
        env_c_dim = cfg['model']['env_c_dim']
        env_encoder_kwargs = cfg['model']['env_encoder_kwargs']
        env_encoder = encoder_dict[encoder](
            dim=dim, c_dim=env_c_dim, padding=padding,
            **env_encoder_kwargs
        )
    else:
        env_c_dim = 0
        env_encoder=None
    
    decoder = models.decoder_dict[decoder](
        dim=dim, 
        c_per_dim=obj_c_dim+env_c_dim, 
        c_act_dim=obj_c_dim+env_c_dim, 
        padding=padding,
        **decoder_kwargs
    )

    obj_per_encoder = encoder_dict[encoder](
        dim=dim, c_dim=obj_c_dim, padding=padding,
        **obj_encoder_kwargs
    )
    obj_act_encoder = encoder_dict[encoder](
        dim=act_dim, c_dim=obj_c_dim, padding=padding,
        **obj_encoder_kwargs
    )

    model = models.ConvImpDyn(
        obj_per_encoder, obj_act_encoder, env_encoder, decoder, device=device
    )

    return model

def get_geom_model(cfg, device=None, dataset=None, **kwargs):
    ''' Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    dim = cfg['data']['dim']
    obj_c_dim = cfg['model']['obj_c_dim']
    decoder_kwargs = cfg['model']['decoder_kwargs']
    obj_encoder_kwargs = cfg['model']['obj_encoder_kwargs']
    padding = cfg['data']['padding']
    decoder = 'geom_decoder'
    encoder = 'geom_encoder'
    
    if 'env_c_dim' in cfg['model'] and 'env_c_dim' != 0:
        env_c_dim = cfg['model']['env_c_dim']
        env_encoder_kwargs = cfg['model']['env_encoder_kwargs']
        env_encoder = encoder_dict[encoder](
            dim=dim, c_dim=env_c_dim, padding=padding,
            **env_encoder_kwargs
        )
    else:
        env_c_dim = 0
        env_encoder=None
    
    decoder = models.decoder_dict[decoder](
        dim=dim, c_dim=obj_c_dim+env_c_dim, padding=padding,
        **decoder_kwargs
    )

    obj_encoder = encoder_dict[encoder](
        dim=dim, c_dim=obj_c_dim, padding=padding,
        **obj_encoder_kwargs
    )

    model = models.ConvOccGeom(
        obj_encoder, env_encoder, decoder, device=device
    )

    return model

def get_trainer(model, optimizer, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')

    trainer = training.PlushTrainer(
        model, optimizer, cfg,
        device=device, 
        vis_dir=vis_dir )

    return trainer


def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    generator = generation.Generator3D(
        model,
        device=device,
        threshold=cfg['test']['threshold'],
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        sample=cfg['generation']['use_sampling'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        padding=cfg['data']['padding'],
        vol_info = None,
        vol_bound = None,
    )
    return generator
