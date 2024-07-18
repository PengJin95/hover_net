import argparse
import collections
import importlib
import logging
import os

import joblib
import yaml

from dataloader.train_loader import FileLoader
from misc.utils import mkdir, recur_find_ext, rm_n_mkdir, rmdir


####
def load_yaml(path):
    with open(path) as fptr:
        info = yaml.full_load(fptr)
    return info


def update_nested_dict(orig_dict, new_dict):
    for key, val in new_dict.items():
        if isinstance(val, collections.Mapping):
            tmp = update_nested_dict(orig_dict.get(key, { }), val)
            orig_dict[key] = tmp
        # elif isinstance(val, list):
        #     orig_dict[key] = (orig_dict.get(key, []) + val)
        else:
            orig_dict[key] = new_dict[key]
    return orig_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('-e', '--epoch', type=int, default=60, help='diffusion fine-tuned epoch, for locating checkpoint')
    parser.add_argument('-s', '--step', type=int, default=10000, help='diffusion fine-tuned step, for locating checkpoint')
    parser.add_argument('--syn_suffix', nargs='+', default=None, help='suffix of fake images (.npy) ')
    parser.add_argument('--syn_suffix_anno', nargs='+', default=None, help='suffix of gt masks of fake images (.npy)')
    parser.add_argument('-sas', '--syn_anno_same', action='store_true', help='whether use the same mask (e.g. train on fold1, fake fold1)')
    parser.add_argument('--name', type=str, default='qkv')
    parser.add_argument('--epoch_factor', type=float, default=1., help='to keep the same number of trainig steps (e.g. "2" for 100 percent augmentation)')
    parser.add_argument('--template', type=str, default='param/template.yaml')

    args = parser.parse_args()
    # print(args)

    logging.basicConfig(level=logging.INFO,)

    seed = 5
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    FOLD_IDX = 0
    WORKSPACE_DIR = '/data/peng/hover_net/exp'
    SAVE_ROOT = f'{WORKSPACE_DIR}/pannuke/'

    # splits = joblib.load('splits.dat')

    def run_one_split_with_param_set(save_path, split_info, param_kwargs):
        mkdir(save_path)

        template_paramset = load_yaml(args.template)
        template_paramset['epoch_factor'] = args.epoch_factor
        # repopulate loader arg according to available subset info
        template_loader_kwargs = template_paramset['loader_kwargs']
        loader_kwargs = {
            k: template_loader_kwargs['train'] if 'train' in k else
            template_loader_kwargs['infer'] for k in split_info.keys()}
        template_paramset['loader_kwargs'] = loader_kwargs

        # * reset logger handler
        log_formatter = logging.Formatter(
            '|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d|%H:%M:%S'
        )
        log = logging.getLogger()  # root logger
        for hdlr in log.handlers[:]:  # remove all old handlers
            log.removeHandler(hdlr)
        new_hdlr_list = [
            logging.FileHandler(f"{save_path}/debug.log"),
            logging.StreamHandler()
        ]
        for hdlr in new_hdlr_list:
            hdlr.setFormatter(log_formatter)
            log.addHandler(hdlr)
        #

        train_loader_list = [
            v for v in split_info.keys() if 'train' in v]
        infer_loader_list = [
            v for v in split_info.keys() if not ('train' in v)]

        cfg_module = importlib.import_module('models.hovernet.opt')
        cfg_getter = getattr(cfg_module, 'get_config')

        with open(f'{save_path}/settings.yml', 'w') as fptr:
            yaml.dump(template_paramset, fptr, default_flow_style=False)

        model_config = cfg_getter(
                            train_loader_list,
                            infer_loader_list,
                            **template_paramset)

        def create_dataset(
                run_mode=None, subset_name=None, setup_augmentor=None):
            target_gen_func = getattr(
                importlib.import_module('models.hovernet.targets'),
                'gen_targets'
            )
            print(subset_name)
            # img_path = f'/data/peng/datasets/Lizard/images_{subset_name}.npy'
            # ann_path = f'/data/peng/datasets/Lizard/labels_{subset_name}.npy'
            img_path = f'/data/peng/datasets/PanNuke/images_{subset_name}.npy'
            ann_path = f'/data/peng/datasets/PanNuke/masks_{subset_name}.npy'
            if 'train' in subset_name and args.syn_suffix is not None:
                img_syn_path = [f'/data/peng/HistoDiffAug/fake_PanNuke/images_qkv_{args.epoch}_{args.step}_fold{suffix}.npy'
                                for suffix in args.syn_suffix]
                if args.syn_suffix_anno is not None:
                    ann_syn_path = [
                        f'/data/peng/datasets/PanNuke/masks{suffix}.npy'
                        for suffix in args.syn_suffix_anno]
                else:
                    assert args.syn_anno_same is True
                    ann_syn_path = None
                # img_syn_path = [f'/data/peng/HistoDiffAug/fake_Lizard_400/images_{suffix}.npy'
                #                 for suffix in args.syn_suffix]
                # img_syn_path = f'/data/peng/HistoDiffAug/fake_Lizard/images_{args.syn_suffix}.npy'

            else:
                img_syn_path = None
                ann_syn_path = None
            # indices = split_info[subset_name]
            print('img_syn_path:', img_syn_path)
            return FileLoader(
                        img_path,
                        ann_path,
                        None, #indices
                        img_syn_path=img_syn_path,
                        ann_syn_path=ann_syn_path,
                        with_type=True,
                        input_shape=[256, 256],
                        mask_shape=[256, 256],
                        run_mode=run_mode,
                        target_gen_func=[target_gen_func, {}]
                    )

        run_kwargs = {
            'seed': seed,
            'debug': False,
            'logging': True,
            'log_dir': save_path + '/model/',
            'create_dataset': create_dataset,
            'model_config': model_config,
        }

        from run_train import RunManager
        trainer = RunManager(**run_kwargs)
        trainer.run()

    save_path_ = f'{SAVE_ROOT}/{args.name}/'
    # split_info = splits[FOLD_IDX]
    split_info = {'train': [], 'val': []}
    run_one_split_with_param_set(save_path_, split_info, {})