import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import matplotlib; matplotlib.use('Agg')
import numpy as np
import os
import argparse
import time, datetime
from src import config, data
from src.checkpoints import CheckpointIO
from collections import defaultdict
import shutil
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas 
from src.utils import common_util
import matplotlib.pyplot as plt
from PIL import Image

# Arguments
parser = argparse.ArgumentParser(
    description='Train a Plush Env dynamics model.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')
parser.add_argument('--exit-after', type=int, default=-1,
                    help='Checkpoint and exit after specified number of seconds'
                         'with exit code 2.')
parser.add_argument('--debug', action='store_true', help='debugging')
parser.add_argument('--eval_only', action='store_true', help='run eval only')

args = parser.parse_args()
cfg = config.load_config(args.config, 'configs/default.yaml')
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")
# Set t0
t0 = time.time()

# Shorthands
out_dir = cfg['training']['out_dir']
if args.debug:  
    cfg['training']['batch_size'] = 2
    cfg['training']['vis_n_outputs'] = 1
    cfg['training']['print_every'] = 1
    cfg['training']['backup_every'] = 1
    cfg['training']['validate_every'] = 1
    cfg['training']['visualize_every'] = 1
    cfg['training']['checkpoint_every'] = 1
    cfg['training']['visualize_total'] = 1

batch_size = cfg['training']['batch_size']
backup_every = cfg['training']['backup_every']
vis_n_outputs = cfg['generation']['vis_n_outputs']
exit_after = args.exit_after

model_selection_metric = cfg['training']['model_selection_metric']
if cfg['training']['model_selection_mode'] == 'maximize':
    model_selection_sign = 1
elif cfg['training']['model_selection_mode'] == 'minimize':
    model_selection_sign = -1
else:
    raise ValueError('model_selection_mode must be '
                     'either maximize or minimize.')

# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

shutil.copyfile(args.config, os.path.join(out_dir, 'config.yaml'))

# Dataset
train_loader = data.core.get_plush_loader(cfg, cfg['model']['type'], split='train')
val_loader = data.core.get_plush_loader(cfg,  cfg['model']['type'], split='test')

# Model
model = config.get_model(cfg, device=device)

# Generator
generator = config.get_generator(model, cfg, device=device)

# Intialize training
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
trainer = config.get_trainer(model, optimizer, cfg, device=device)

checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)
try:
    load_dict = checkpoint_io.load('model_best.pt')
except FileExistsError:
    load_dict = dict()
epoch_it = load_dict.get('epoch_it', 0)
it = load_dict.get('it', 0)
metric_val_best = load_dict.get(
    'loss_val_best', -model_selection_sign * np.inf)

if metric_val_best == np.inf or metric_val_best == -np.inf:
    metric_val_best = -model_selection_sign * np.inf
print('Current best validation metric (%s): %.8f'
      % (model_selection_metric, metric_val_best))
logger = SummaryWriter(os.path.join(out_dir, 'logs'))

# Shorthands
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
validate_every = cfg['training']['validate_every']
visualize_every = cfg['training']['visualize_every']

# Print model
nparameters = sum(p.numel() for p in model.parameters())
print('Total number of parameters: %d' % nparameters)

print('output path: ', cfg['training']['out_dir'])

# For visualizations
data_vis_list = []
if cfg['model']['type'] == 'geom':
    vis_dataset = data.core.get_geom_dataset(cfg, split='vis')
elif cfg['model']['type'] == 'combined':
    vis_dataset = data.core.get_combined_dataset(cfg, split='vis')
# Build a data dictionary for visualization
np.random.seed(0)
data_idxes = np.random.randint(len(vis_dataset), size=cfg['training']['visualize_total'])
for i, id in enumerate(data_idxes):
    data_vis = data.core.collate_pair_fn([vis_dataset[id]]) 
    data_vis_list.append({'it': i, 'data': data_vis})


if args.eval_only:
    eval_dict, figs  = trainer.evaluate(val_loader)
    metric_val = eval_dict[model_selection_metric]
    for k, v in eval_dict.items():
        print(f"metric {k}: {v}")
    print('Validation metric (%s): %.4f'
            % (model_selection_metric, metric_val))
    for k,v in figs.items():
        fig_path = os.path.join(out_dir, 'vis', f"{k}_eval_best.png")
        v.savefig(fig_path)
    for data_vis in data_vis_list:
        out = generator.generate_mesh(data_vis['data'])
        # Get statistics
        try:
            mesh, stats_dict = out
        except TypeError:
            mesh, stats_dict = out, {}
        mesh.export(os.path.join(out_dir, 'vis', f"best_{data_vis['it']}.off"))
        out2 = generator.generate_pointcloud(data_vis['data'])
        for i,pcloud in enumerate(out2):
            ipath = os.path.join(out_dir, 'vis', f"best_{data_vis['it']}_{i}.obj")
            common_util.write_pointcoud_as_obj(ipath,  pcloud)
        pcloud_dict = [{"title":'source'if i == 0 else 'target',
                        "pts": p[:,:3],
                        "col": None if p.shape[1] == 3 else p[:,3:]
                        } for i,p in enumerate(out2)] 
        fig = common_util.side_by_side_point_clouds(pcloud_dict)
        width, height = fig.get_size_inches() * fig.get_dpi()
        canvas = FigureCanvas(fig)
        canvas.draw() 
        img_path = os.path.join(out_dir, 'vis', f"best_{data_vis['it']}.png")
        Image.fromarray(
            np.frombuffer(
                canvas.tostring_rgb(), 
                dtype='uint8').reshape(int(height), int(width), 3)).save(                                   
                    img_path
                )
        plt.close(fig)
    quit()


while True:
    epoch_it += 1

    for batch in train_loader:
        it += 1
        losses = trainer.train_step(batch, it)
        for k,v in losses.items():
            logger.add_scalar(f'train/{k}_loss', v, it)

        # Print output
        if (it % print_every) == 0:
            t = datetime.datetime.now()
            print_str = f"[Epoch {epoch_it:04d}] it={it:04d}, time: {time.time()-t0:.3f}, "
            print_str += f"{t.hour:02d}:{t.minute:02d}, "
            for k,v in losses.items():
                print_str += f"{k}:{v:.4f}, "
            print(print_str)

        # Save checkpoint
        if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
            print('Saving checkpoint')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)

        # Backup if necessary
        if (backup_every > 0 and (it % backup_every) == 0):
            print('Backup checkpoint')
            checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
        # Run validation
        if validate_every > 0 and (it % validate_every) == 0:
            print('Running Validation')
            eval_dict, figs  = trainer.evaluate(val_loader)
            for k,v in figs.items():
                fig_path = os.path.join(out_dir, 'vis', f"{k}_{it}.png")
                v.savefig(fig_path)
                logger.add_figure(k, v, it)
            metric_val = eval_dict[model_selection_metric]
            print('Validation metric (%s): %.4f'
                  % (model_selection_metric, metric_val))

            for k, v in eval_dict.items():
                print(f"metric {k}: {v}")
                logger.add_scalar('val/%s' % k, v, it)

            if model_selection_sign * (metric_val - metric_val_best) > 0:
                metric_val_best = metric_val
                print('New best model (loss %.4f)' % metric_val_best)
                checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)

        # Visualize output
        if visualize_every > 0 and (it % visualize_every) == 0:
            print('Visualizing')
            renders = []
            for data_vis in data_vis_list:
                out = generator.generate_mesh(data_vis['data'])
                # Get statistics
                try:
                    mesh, stats_dict = out
                except TypeError:
                    mesh, stats_dict = out, {}
                mesh.export(os.path.join(out_dir, 'vis', '{}_{}.off'.format(it, data_vis['it'])))
                out2 = generator.generate_pointcloud(data_vis['data'])
                for i,pcloud in enumerate(out2):
                    ipath = os.path.join(out_dir, 'vis', f"{it}_{data_vis['it']}_{i}.obj")
                    common_util.write_pointcoud_as_obj(ipath,  pcloud)
                name_dict = ['source', 'target', 'source_rollout', 'target_rollout']
                pcloud_dict = [{"title":name_dict[i],
                                "pts": p[:,:3],
                                "col": None if p.shape[1] == 3 else p[:,3:]
                                } for i,p in enumerate(out2)] 
                fig = common_util.side_by_side_point_clouds(pcloud_dict)
                width, height = fig.get_size_inches() * fig.get_dpi()
                canvas = FigureCanvas(fig)
                canvas.draw() 
                img_path = os.path.join(out_dir, 'vis', f"{it}_{data_vis['it']}.png")
                Image.fromarray(
                    np.frombuffer(
                        canvas.tostring_rgb(), 
                        dtype='uint8').reshape(int(height), int(width), 3)).save(                                   
                            img_path
                        )
                plt.close(fig)

        # Exit if necessary
        if exit_after > 0 and (time.time() - t0) >= exit_after:
            print('Time limit reached. Exiting.')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
            exit(3)
