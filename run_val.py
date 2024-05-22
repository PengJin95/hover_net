import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse
from run_utils.utils import convert_pytorch_checkpoint
from models.hovernet.net_desc import create_model
from dataloader.train_loader import FileLoader
from models.hovernet.targets import gen_targets
from tqdm import tqdm
from collections import OrderedDict
import torch.nn.functional as F
from models.hovernet.post_proc import post_process_batch_hovernet
from metrics.stats_utils import (
    get_dice_1,
    get_fast_aji,
    get_fast_aji_plus,
    get_fast_dice_2,
    get_fast_pq,
    remap_label,
    pair_coordinates
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--name', type=str, default='qkv_100_25k_run2')
    parser.add_argument('--phase', type=str, default='01')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=24)
    
    parser.add_argument('--template', type=str, default='param/template.yaml')
    args = parser.parse_args()

    device = torch.device('cuda:%d' % args.gpu)
    
    WORKSPACE_DIR = '/fs/ess/PCON0521/pjin/hovernet_exp'
    save_path_ = f'{WORKSPACE_DIR}/lizard/{args.name}/'
    model = create_model(num_types=7, pretrained_backbone='/users/PAS2606/pqj5125/hover_net/resnet50-0676ba61.pth').to(device)

    state_dict = torch.load(f'{save_path_}/model/{args.phase}/net_epoch={args.epoch}.tar', map_location='cpu')["desc"]
    state_dict = convert_pytorch_checkpoint(state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    dataset = FileLoader(
        img_path='/fs/ess/PCON0521/pjin/Lizard/images_val.npy',
        ann_path='/fs/ess/PCON0521/pjin/Lizard/labels_val.npy',
        with_type=True,
        input_shape=[256, 256],
        mask_shape=[256, 256],
        run_mode='val',
        target_gen_func=[gen_targets, {}]
    )

    dataloader = DataLoader(dataset, num_workers=4, batch_size=args.batch_size, shuffle=False, drop_last=False)

    with torch.no_grad():
        metrics = [[], [], [], [], [], []]
        metrics2 = [[], [], [], [], [], []]
        for i, batch_data in tqdm(enumerate(dataloader)):
            imgs = batch_data["img"].permute(0, 3, 1, 2).contiguous()
            imgs = imgs.to(device).type(torch.float32)
            pred_dict = model(imgs) # tp, hv, np
            outputs = [pred_dict['np'], pred_dict['hv'], pred_dict['tp']]

            preds_detection, preds_classification = post_process_batch_hovernet(outputs, 7)
            
            true_inst = batch_data['inst_map'] # NHW
            true_segm = batch_data['tp_map'] # NHW
            for j in range(imgs.shape[0]):
                if len(list(np.unique(true_inst[j]))) == 1:
                    continue
                for k in range(1, 7):
                    pred = preds_detection[j] * (preds_classification[j, k] > 0)
                    pred = remap_label(pred, by_size=False)
                    true = true_inst[j] * (true_segm[j] == k)
                    true = remap_label(true, by_size=False)
                    if len(list(np.unique(true))) == 1:
                        continue
                    # pq_info = get_fast_pq(true, pred, match_iou=0.5)[0]
                    # metrics[k-1].append(pq_info[2])
                    metrics[k-1].append(get_dice_1(true, pred))
                    # metrics2[k-1].append(get_fast_dice_2(true, pred))
                # pred = remap_label(preds_detection[j], by_size=False)
                # true = remap_label(true_inst[j], by_size=False)

                # pq_info = get_fast_pq(true, pred, match_iou=0.5)[0]
                # metrics[0].append(get_dice_1(true, pred))
                # metrics[1].append(get_fast_aji(true, pred))
                # metrics[2].append(pq_info[0])  # dq
                # metrics[3].append(pq_info[1])  # sq
                # metrics[4].append(pq_info[2])  # pq
                # metrics[5].append(get_fast_aji_plus(true, pred))
        
        print('%.5f' % np.mean([np.mean(m) for m in metrics]))
        print('%.5f' % np.mean([m for metric in metrics for m in metric]))
        # print('%.5f' % np.mean([np.mean(m) for m in metrics2]))
        # print('%.5f' % np.mean([m for metric in metrics2 for m in metric]))        
        # metrics = np.array(metrics)
        # metrics_avg = np.mean(metrics, axis=-1)
        # np.set_printoptions(formatter={"float": "{: 0.5f}".format})
        # print(metrics_avg)                

            # pred_dict = OrderedDict(
            #     [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
            # )
            # pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)
            # type_map = F.softmax(pred_dict["tp"], dim=-1)
            # type_map = torch.argmax(type_map, dim=-1, keepdim=True)
            # type_map = type_map.type(torch.float32)
            # pred_dict["tp"] = type_map
            # pred_output = torch.cat(list(pred_dict.values()), -1) # tp1, hv2, np1
            # # need to change order to tp np hv

            # true_np = batch_data["np_map"]
            # true_hv = batch_data["hv_map"]
            # true_np = torch.squeeze(true_np).to("cuda").type(torch.int64)
            # true_hv = torch.squeeze(true_hv).to("cuda").type(torch.float32)
            # true_tp = batch_data["tp_map"]
            # true_tp = torch.squeeze(true_tp).to("cuda").type(torch.int64)

