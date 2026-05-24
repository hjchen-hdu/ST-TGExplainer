import logging
import time
import sys
import os
import numpy as np
import warnings
import json
import torch
import torch.nn as nn


from models.ST_TGExplainer import ST_TGExplainer
from models.modules import MergeLayer
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler, compute_src_dst_max_time_shifts
from evaluate_models_utils import evaluate_model_link_prediction
from evaluate_models_utils import evaluate_model_link_prediction_multi_negs
from utils.metrics import get_link_prediction_metrics
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args
from explain import explain
from tqdm import tqdm


def train_epoch(model, args, logger, epoch, train_idx_data_loader, train_neighbor_sampler, train_neg_edge_sampler, train_data, optimizer, loss_func, full_neighbor_sampler, val_data, val_idx_data_loader, val_neg_edge_sampler, full_data, optimizer_disc=None):
        model.train()
        model[0].set_neighbor_sampler(train_neighbor_sampler)
        
        # store train losses and metrics
        train_losses, train_metrics = [], []
        train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
        
        epoch_start_time = time.time()
        for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
            train_data_indices = train_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]
            if args.collision_check: 
                batch_neg_dst_node_ids = train_neg_edge_sampler.sample_with_time_collision_check(num_negs=1, batch_src_node_ids=batch_src_node_ids, batch_node_interact_times=batch_node_interact_times, neighbor_sampler=train_neighbor_sampler).flatten()
                batch_neg_src_node_ids = batch_src_node_ids
            else:
                batch_neg_src_node_ids, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
            
            # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
            # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
            
            # get temporal embedding of source and destination nodes
            # two Tensors, with shape (batch_size, node_feat_dim)
            pos_loss, _, pos_node_embedding, pos_sub_embedding = \
                model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                    dst_node_ids=batch_dst_node_ids,
                                                                    node_interact_times=batch_node_interact_times,
                                                                    num_neighbors=args.num_neighbors,
                                                                    epoch=epoch+1)

            # get temporal embedding of negative source and negative destination nodes
            # two Tensors, with shape (batch_size, node_feat_dim)
            neg_loss, _, neg_node_embedding, neg_sub_embedding = \
                model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                    dst_node_ids=batch_neg_dst_node_ids,
                                                                    node_interact_times=batch_node_interact_times,
                                                                    num_neighbors=args.num_neighbors,
                                                                    epoch=epoch+1)
            
            positive_probabilities = model[1](
                input_1=pos_node_embedding, input_2=pos_sub_embedding).squeeze(dim=-1).sigmoid()
            negative_probabilities = model[1](
                input_1=neg_node_embedding, input_2=neg_sub_embedding).squeeze(dim=-1).sigmoid()
            
            predicts = torch.cat(
                [positive_probabilities, negative_probabilities], dim=0)
            labels = torch.cat([torch.ones_like(
                positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)
            train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

            ce_loss = loss_func(input=predicts, target=labels)
            info_loss, dis_loss = torch.tensor(0), torch.tensor(0)
            info_loss = (pos_loss['info_loss']+neg_loss['info_loss'])
            if optimizer_disc is not None:
                pos_h_S, pos_h_T = pos_loss['h_S'], pos_loss['h_T']
                neg_h_S, neg_h_T = neg_loss['h_S'], neg_loss['h_T']
                h_S_all = torch.cat([pos_h_S, neg_h_S], dim=0)
                h_T_all = torch.cat([pos_h_T, neg_h_T], dim=0)
                cur_bs = pos_h_S.shape[0]
                Y_disc = torch.cat([
                    torch.ones(cur_bs, 1, device=h_S_all.device),
                    torch.zeros(cur_bs, 1, device=h_S_all.device)
                ], dim=0)

                perm_pos = torch.randperm(cur_bs, device=h_S_all.device)
                perm_neg = torch.randperm(cur_bs, device=h_S_all.device)
                h_T_tilde = torch.cat([pos_h_T[perm_pos], neg_h_T[perm_neg]], dim=0)

                real_input_d = torch.cat([h_S_all.detach(), h_T_all.detach(), Y_disc], dim=-1)
                fake_input_d = torch.cat([h_S_all.detach(), h_T_tilde.detach(), Y_disc], dim=-1)
                real_pred_d = model[0].discriminator(real_input_d).sigmoid()
                fake_pred_d = model[0].discriminator(fake_input_d).sigmoid()
                disc_loss = -(torch.log(real_pred_d + 1e-6).mean() + torch.log(1 - fake_pred_d + 1e-6).mean())
                optimizer_disc.zero_grad()
                disc_loss.backward()
                optimizer_disc.step()

                real_input_g = torch.cat([h_S_all, h_T_all, Y_disc], dim=-1)
                fake_input_g = torch.cat([h_S_all, h_T_tilde.detach(), Y_disc], dim=-1)
                real_pred_g = model[0].discriminator(real_input_g).sigmoid()
                fake_pred_g = model[0].discriminator(fake_input_g).sigmoid()
                dis_loss = torch.log(real_pred_g + 1e-6).mean() + torch.log(1 - fake_pred_g + 1e-6).mean()

            loss = ce_loss + args.info_weight*info_loss + args.dis_weight*dis_loss
            
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.8f}')
        if len(train_metrics)>0:
            for metric_name in train_metrics[0].keys():
                logger.info(f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.8f}')
        

        val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                model=model,
                                                                neighbor_sampler=full_neighbor_sampler,
                                                                evaluate_idx_data_loader=val_idx_data_loader,
                                                                evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                evaluate_data=val_data,
                                                                loss_func=loss_func,
                                                                device=args.device,
                                                                num_neighbors=args.num_neighbors,
                                                                time_gap=args.time_gap,mode='val', loss_type = args.loss, full_data=full_data, collision_check=args.collision_check, dataset_name=args.dataset_name)
        
        for metric_name in val_metrics.keys():
            logger.info(f'validate {metric_name}, {val_metrics[metric_name]:.8f}')

        for handler in logger.handlers:
            handler.flush()

        return val_metrics

def get_model(args, train_data, node_raw_features, edge_raw_features, train_neighbor_sampler, full_data,logger):

    max_time_shift = compute_src_dst_max_time_shifts(full_data.src_node_ids, full_data.dst_node_ids,
                                                             full_data.node_interact_times)
    dynamic_backbone = ST_TGExplainer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                        time_feat_dim=args.time_feat_dim, num_neighbors=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=args.device, max_time_shift=max_time_shift)
    
    link_predictor = MergeLayer(input_dim1=args.output_dim, input_dim2=args.output_dim, hidden_dim=args.output_dim, output_dim=1)
    model = nn.Sequential(dynamic_backbone, link_predictor)
    
    logger.info(f'model -> {model}')
    logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')
    return model

def get_loss_fn(args):
    loss_func = nn.BCELoss()
    return loss_func


if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    # get arguments
    args = get_link_prediction_args(is_evaluation=False)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    postfix = ''
    if args.use_edge_feat:
        postfix += '_e'
    if args.use_node_feat:
        postfix+='_n'
    if args.version is not None:
        args.save_model_name = f'{args.model_name}_seed{args.seed}_batchsize{args.batch_size}_num_neighbors{args.num_neighbors}_dropout{args.dropout}_sample_neighbor_strategy{args.sample_neighbor_strategy}_numlayers{args.num_layers}{postfix}_v{args.version}'
    else:
        args.save_model_name = f'{args.model_name}_seed{args.seed}_batchsize{args.batch_size}_num_neighbors{args.num_neighbors}_dropout{args.dropout}_sample_neighbor_strategy{args.sample_neighbor_strategy}_numlayers{args.num_layers}{postfix}'
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    log_dir = f"{current_dir}/logs/{args.dataset_name}/{args.model_name}/{args.version}_{args.save_model_name}"
    os.makedirs(log_dir, exist_ok=True)
    log_file = f"{log_dir}/{str(time.time())}.log"
    print("log in: ", log_file)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    fh.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)
    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, val_test_data = \
        get_link_prediction_data(
            dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio, dataset_path=args.dataset_path, use_edge_feat=args.use_edge_feat, use_node_feat=args.use_node_feat, logger=logger)

    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,time_scaling_factor=args.time_scaling_factor, seed=0)

    # initialize validation and test neighbor sample to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,time_scaling_factor=args.time_scaling_factor, seed=1)
        
    if (args.is_bipartite or args.dataset_name in ['GoogleLocal', 'ML-20M', 'Taobao', 'Yelp', 'mooc', 'lastfm', 'reddit', 'wikipedia']):
        args.user_size = full_data.src_node_ids.max()-full_data.src_node_ids.min()+1
        args.item_size = full_data.dst_node_ids.max()-full_data.dst_node_ids.min()+1
        args.node_size = args.user_size + args.item_size
        args.dst_min_idx = full_data.dst_node_ids.min()
        args.src_min_idx = full_data.src_node_ids.min()
    else:
        args.user_size = full_data.max_node_id
        args.item_size = full_data.max_node_id
        args.node_size = args.user_size
        args.dst_min_idx = 1
        args.src_min_idx = 1
    
    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    train_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids, seed=0)
    val_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=1)
    test_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2)
    if args.dataset_name in ['wikipedia', 'reddit', 'mooc', 'lastfm', 'uci', 'Flights' ]: # dataset with a small number of nodes
        args.collision_check = True

    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=args.shuffle)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    # new_node_val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(
        range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    multi_negs_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(
        len(test_data.src_node_ids))), batch_size=args.multi_negs_batch_size, shuffle=False)
    # new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    val_metric_all_runs, test_metric_all_runs = [], []
    for run in range(args.num_runs):
        set_random_seed(seed=args.seed+run)
        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")
        logger.info(f'configuration is {args}')
        logger.info(f'{sys.argv}')
        model = get_model(args, train_data, node_raw_features, edge_raw_features, train_neighbor_sampler, full_data, logger)
        loss_func = get_loss_fn(args)
        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer,
                                        learning_rate=args.learning_rate, weight_decay=args.weight_decay)
        optimizer_disc = None
        disc_params = list(model[0].discriminator.parameters())
        disc_param_ids = set(id(p) for p in disc_params)
        model_params = [p for p in model.parameters() if id(p) not in disc_param_ids]
        optimizer = torch.optim.Adam(model_params, lr=args.learning_rate, weight_decay=args.weight_decay)
        optimizer_disc = torch.optim.Adam(disc_params, lr=args.disc_lr)
        args.save_model_name = f'{args.model_name}_seed{args.seed+run}_batchsize{args.batch_size}_num_neighbors{args.num_neighbors}_dropout{args.dropout}_numlayers{args.num_layers}{postfix}'
        save_model_folder = f"{args.save_model_path}/{args.dataset_name}/{args.model_name}/{args.save_model_name}/"
        if not os.path.exists(save_model_folder):
            os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)
        if args.load_pretrained:
            early_stopping.load_checkpoint(model, map_location='cpu')
        model = convert_to_gpu(model, device=args.device)

        for epoch in range(args.num_epochs):
            val_metrics = train_epoch(model, args, logger, epoch, train_idx_data_loader, train_neighbor_sampler, train_neg_edge_sampler, train_data, optimizer, loss_func, full_neighbor_sampler, val_data, val_idx_data_loader, val_neg_edge_sampler, full_data, optimizer_disc=optimizer_disc)   
            if 'mrr' in val_metrics:
                val_metric_indicator = [('mrr', val_metrics['mrr'], True)]
            elif 'average_precision' in val_metrics:
                val_metric_indicator = [('average_precision', val_metrics['average_precision'], True)]
            else:
                raise ValueError(f"No valid metric found in val_metrics: {val_metrics}")
            early_stop = early_stopping.step(val_metric_indicator, model)
            if early_stop:
                break
            
        # load the best model
        early_stopping.load_checkpoint(model, map_location='cpu')
        model = convert_to_gpu(model, device=args.device)
        # evaluate the best model
        logger.info(f'get final performance on dataset {args.dataset_name}...')
        test_metrics={}
        # For memory based models, we need to deal with their val set first in the evaluate_model_link_prediction function.
        test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                   model=model,
                                                                   neighbor_sampler=full_neighbor_sampler,
                                                                   evaluate_idx_data_loader=test_idx_data_loader,
                                                                   evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                   evaluate_data=test_data,
                                                                   loss_func=loss_func,
                                                                   device=args.device,
                                                                   num_neighbors=args.num_neighbors,
                                                                   time_gap=args.time_gap,mode='test', loss_type = args.loss, full_data=full_data, collision_check=args.collision_check, dataset_name=args.dataset_name)
        
        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')
        # reload the model, so that the memory bank is reloaded
        early_stopping.load_checkpoint(model, map_location='cpu')
        model = convert_to_gpu(model, device=args.device)
        for metric_name in test_metrics.keys():
            logger.info(f'test {metric_name}, {test_metrics[metric_name]:.8f}')

        test_metric_all_runs.append(test_metrics)

        result_json = {
            "test metrics": {metric_name: f'{test_metrics[metric_name]:.8f}' for metric_name in test_metrics},
        }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.dataset_name}/{args.model_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(
            save_result_folder, f"{args.save_model_name}.json")

        with open(save_result_path, 'w') as file:
            file.write(result_json)
        
    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.8f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.8f}')
    print(log_file)
    sys.exit()