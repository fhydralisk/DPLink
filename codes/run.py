from __future__ import print_function, division

import os
import sys
import time
import torch
import mlflow
import argparse
import setproctitle
import numpy as np
import shutil
import pathlib
import json
import uuid

from urllib.parse import unquote, urlparse
from match import run_experiments


class CustomSettings(object):
    def __init__(self, seed=None, data='weibo', epoch=20, threshold=2000, lr_pretrain=0.001, lr_match=0.0005,
                 layers=1, rnn_mod="GRU", attn_mod='dot', lr_step=5, lr_decay=0.1, batch_size=32, l2=0.0,
                 poi=None, loss_mode="BCELoss", neg=32, intersect=1, topk=5, noise=0, pretrain=1, dropout_p=0.5,
                 hidden_size=200, loc_emb_size=200, tim_emb_size=10, poi_type=0, poi_size=21, poi_emb_size=10,
                 data_path="./data", save_path="./output", use_aug=False, aug_name="1000_1000"):

        if seed is None:
            np.random.seed(1)
            torch.manual_seed(1)
        else:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.data_path = data_path
        self.use_aug=use_aug
        self.save_path = save_path
        self.data_name = data
        self.epoch = epoch
        self.threshold = threshold
        self.poi = poi
        self.lr_pretrain = lr_pretrain
        self.intersect = intersect
        self.lr_step = lr_step
        self.lr_decay = lr_decay
        self.topk = topk
        self.aug_name = aug_name

        self.rnn_mod = rnn_mod
        self.attn_mod = attn_mod
        self.layers = layers

        self.neg = neg
        self.loss_mode = loss_mode
        self.noise = noise
        self.pretrain = pretrain

        self.lr_match = lr_match
        self.batch_size = batch_size
        self.l2 = l2
        self.dropout_p = dropout_p

        self.hidden_size = hidden_size
        self.loc_emb_size = loc_emb_size
        self.tim_emb_size = tim_emb_size

        self.poi_type = poi_type
        self.poi_size = poi_size
        self.poi_emb_size = poi_emb_size


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="weibo")
    parser.add_argument("--city", type=str, default="NewYork")
    parser.add_argument("--gpu", type=str, default="0", choices=["0", "1", "2", "3", "4", "5", "6", "7"])
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--pretrain", type=int, default=1)
    parser.add_argument("--model", type=str, default="ERC", choices=["ERC", "ERPC"])
    parser.add_argument("--rnn", type=str, default="GRU")
    parser.add_argument("--noise_level", type=int, default=0)
    parser.add_argument("--poi_type", type=int, default=0)
    parser.add_argument("--use_poi", type=int, default=0)
    parser.add_argument("--threshold", type=int, default=3200)
    parser.add_argument("--epoch", type=int, default=2)
    parser.add_argument("--pretrain_unit", type=str, default="ERCF", choices=["N", "E", "R", "C", "F", "ERCF"])
    parser.add_argument("--data_path", type=str, default="/data3/duyuwei/AugMove/dataset/input_timeloc")
    parser.add_argument("--save_path", type=str, default="./output")
    parser.add_argument("--use_aug", action="store_true")
    parser.add_argument("--result_path", type=str, default="/data3/duyuwei/AugMove/dataset/model_output")
    parser.add_argument("--aug_name", type=str, default="1000_1000")
    parser.add_argument("--task", type=str, default="Trajectory_User_Linkage")
    parser.add_argument("--config_path", type=str, default="/data3/duyuwei/DPLink/codes/DPLink.json")
    parser.add_argument('--param_op', action='store_true')
    parser.add_argument('--optim_path', type=str)
    parser.add_argument('--max_step', type=int)
    args = parser.parse_args()
    with open(args.config_path,"r") as f:
        settings = json.load(f)



    USE_POI = (args.use_poi == 1)
    device = torch.device("cuda:" + args.gpu)

    data_path = os.path.abspath(args.data_path)
    save_path = os.path.abspath(args.save_path)
    save_path_uri = pathlib.Path(save_path).as_uri()

    mlflow.set_tracking_uri(save_path_uri)
    experiment_name = "Default"
    experiment_ID = 0
    try:
        experiment_ID = mlflow.create_experiment(name=experiment_name)
        print("Initial Create!")
    except:
        service = mlflow.tracking.MlflowClient()
        experiment = service.get_experiment_by_name(experiment_name)
        if experiment is not None:
            experiment_ID = experiment.experiment_id
            print("Experiment Exists!")

        # for exp in experiments:
        #     if exp.name == experiment_name:
        #         experiment_ID = exp.experiment_id
        #         print("Experiment Exists!")
        #         break

    setproctitle.setproctitle('DPLink')

    thre = args.threshold
    rnn_unit = 'GRU'
    attn_unit = 'dot'
    test_pretrain = False  # test the effect of different pretrain degree, working with run_pretrain
    pre_path, rank_pre2, hit_pre2 = None, None, None
    all_rank, all_hit, all_acc = 0,0, 0
    for run_id in range(args.repeat):
        with mlflow.start_run(experiment_id=experiment_ID):
            archive_path = mlflow.get_artifact_uri()
            archive_path = unquote(urlparse(archive_path).path)
            if sys.platform == 'win32':
                archive_path = archive_path[1:]

            if run_id == 0:
                pre_path = archive_path
            else:
                if test_pretrain:
                    shutil.copy2(pre_path + "/SN-pre-" + str(run_id) + ".m", archive_path)
                    # os.system("cp " + pre_path + "/SN-pre-" + str(run_id) + ".m " + archive_path + "/")
                else:
                    shutil.copy2(pre_path + "/SN-pre.m", archive_path)
                    # os.system("cp " + pre_path + "/SN-pre.m " + archive_path + "/")
            hidden_size = settings["hidden_size"]
            loc_emb_size = settings["loc_emb_size"]
            tim_emb_size = settings["tim_emb_size"]
            dropout_p = settings["dropout_p"]
            l2 = settings["l2"]
            lr_match = settings["lr_match"]
            if run_id == 0:
                loss_mode = "BCELoss"
            else:
                loss_mode = settings["loss_mode"]
            mlflow.log_param("loss_mode", loss_mode)
            mlflow.log_param("data_name", args.dataset)
            mlflow.log_param("rnn", rnn_unit)
            mlflow.log_param("attn", attn_unit)
            mlflow.log_param("lr_match", lr_match)
            mlflow.log_param("hidden_loc_size", hidden_size)
            mlflow.log_param("tim_size", tim_emb_size)
            mlflow.log_param("dropout", dropout_p)
            mlflow.log_param("pretrain", args.pretrain)
            mlflow.log_param("noise_level", 1000 if USE_POI else args.noise_level)
            mlflow.log_param("poi_type", args.poi_type if USE_POI else 1000)
            mlflow.log_param("threshold", args.threshold)
            mlflow.log_param("model", args.model)
            mlflow.log_param("step", run_id)
            mlflow.log_param("pretrain_unit", args.pretrain_unit)
            run_settings = CustomSettings(
                data=args.dataset, neg=32, seed=int(time.time()), pretrain=args.pretrain,
                loss_mode=loss_mode, lr_match=lr_match, l2=l2, dropout_p=dropout_p,
                tim_emb_size=tim_emb_size, loc_emb_size=loc_emb_size,
                hidden_size=hidden_size, epoch=args.epoch, threshold=args.threshold,
                rnn_mod=rnn_unit, attn_mod=attn_unit,
                data_path=data_path, save_path=archive_path,
                noise=0 if USE_POI else args.noise_level, poi_type=args.poi_type,
                use_aug=args.use_aug, aug_name=args.aug_name
            )
            #SN_aug, rank_neg, hit_neg, acc_32,rank_pre, hit_pre
            # (args, run_id=0, device=None, USE_POI=False, model_type='S', unit=None)
            model, rank, hit, acc, rank_pre, hit_pre = run_experiments(run_settings, model_type=args.model,
                                                                  run_id=run_id,
                                                                  device=device, USE_POI=USE_POI,
                                                                  unit=args.pretrain_unit)
            if run_id == 0:
                rank_pre2, hit_pre2 = rank_pre, hit_pre
            mlflow.log_metric("rank_32", rank)
            mlflow.log_metric("hit_32", hit)
            all_rank += rank
            all_hit += hit
            all_acc += acc
            if rank_pre2 and hit_pre2:
                mlflow.log_metric("rank_pre", rank_pre2)
                mlflow.log_metric("hit_pre", hit_pre2)
    ave_rank = all_rank/args.repeat
    ave_hit = all_hit/args.repeat
    ave_acc = all_acc/args.repeat
    results = {
    "Rank_32": ave_rank,
    "Hit_32": ave_hit,
    "Acc@5": ave_acc
    }
    
#
#         if args.param_op:
#             with open(args.config_path, 'r') as f:
#                 model_config = json.load(f)
#             # 保存至结果文件，供Agent读取
#             uuid_path = os.path.dirname(args.config_path)
#             results['config'] = model_config
#             flag = ''.join(str(uuid.uuid4()).split('-'))
#             file_name = "MainTUL_{}_{}_{}_epoch_{}_step_{}_{}.json".format(args.dataset, args.city, args.aug_name, settings["epochs"], args.max_step, flag)
#                     # 保存至结果文件，供Agent读取
#             with open(os.path.join(args.optim_path,file_name), 'w') as f:
#                 json.dump(results, f)
#             with open(os.path.join(uuid_path, 'uuid.json'), 'w') as f:
#                 json.dump(flag, f)
#         else:
#             file_name = "MainTUL_{}_{}_{}_epoch_{}_step_{}.json".format(args.dataset, args.city, args.aug_name, args.epochs, args.max_step)
#             print(args.result_path,file_name)
#             with open(os.path.join(args.result_path,file_name),'w') as f:
#                 json.dump(results, f)
# #
    if args.param_op:
        with open(args.config_path, 'r') as f:
            model_config = json.load(f)
        uuid_path = os.path.dirname(args.config_path)
        # 保存至结果文件，供Agent读取
        results['config'] = model_config
        results['Hit_32'] = ave_hit
        results['Acc@5'] = ave_acc
        flag = ''.join(str(uuid.uuid4()).split('-'))
        file_name = "DPLink_{}_{}_{}_epoch_{}_step_{}_{}.json".format(args.dataset, args.city, args.aug_name, args.epoch, args.max_step, flag)
                # 保存至结果文件，供Agent读取
        with open(os.path.join(args.optim_path, file_name), 'w') as f:
            json.dump(results, f)
        print("Result file done!")
        with open(os.path.join(uuid_path, 'uuid.json'), 'w') as f:
            json.dump(flag, f)
        print("UUID file done!")
    else:
        file_name = "DPLink_{}_{}_{}_epoch_{}_step_{}.json".format(args.dataset, args.city, args.aug_name, args.epoch, args.max_step)
        with open(os.path.join(args.result_path,file_name),'w') as f:
            json.dump(results, f)
