import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
import numpy as np
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.metrics import Metrics
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_,  test_dataloader) = builder.completion_dataset_builder(args, config.dataset.train), \
                                                            builder.completion_dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()
    
    # parameter setting
    start_epoch = 0
    best_metrics = None
    metrics = None

    # resume ckpts
    if args.resume:
        start_epoch, best_metrics = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Metrics(config.consider_metric, best_metrics)
    else:
        if args.ckpts is not None:
            base_model.base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()


    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['SparseLoss', 'DenseLoss'])

        num_iter = 0

        # validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger=logger)
        features = []
        categories = []
        taxonomy_list = ['02691156', '02933112', '02958343', '03001627', '03636649', '04256520', '04379243', '04530566']

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train._base_.N_POINTS
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()
                if config.dataset.train._base_.CARS:
                    if idx == 0:
                        print_log('padding while KITTI training', logger=logger)
                    partial = misc.random_dropping(partial, epoch) # specially for KITTI finetune
            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                partial = partial.cuda()
            elif dataset_name == 'MVP_16384':
                partial = data[0].cuda()
                gt = data[1].cuda()  
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            num_iter += 1
           
            ret = base_model(partial)

            # ''' 获取global feature '''
            # for taxonomy_id in taxonomy_ids:
            #     taxonomy_id = taxonomy_id if isinstance(taxonomy_id, str) else taxonomy_id.item()
            #     categories.append(taxonomy_list.index(taxonomy_id))
            # features.append(global_feature)
            
            sparse_loss, dense_loss = base_model.module.get_loss(ret, gt)
         
            _loss = sparse_loss + dense_loss 
            _loss.backward()
            if config.clip_gradients:
                norm = builder.clip_gradients(base_model, config.clip_grad)

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                sparse_loss = dist_utils.reduce_tensor(sparse_loss, args)
                dense_loss = dist_utils.reduce_tensor(dense_loss, args)
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])
            else:
                losses.update([sparse_loss.item() * 1000, dense_loss.item() * 1000])


            if args.distributed:
                torch.cuda.synchronize()

            n_itr = epoch * n_batches + idx
            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Sparse', sparse_loss.item() * 1000, n_itr)
                train_writer.add_scalar('Loss/Batch/Dense', dense_loss.item() * 1000, n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 100 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)

        # features = torch.cat(features, dim=0)
        # target_path = os.path.join(args.experiment_path, 'tsne_vis_result_train_set')
        # if not os.path.exists(target_path):
        #     os.makedirs(target_path)
        # np.save(os.path.join(target_path, 'features.npy'), features.detach().cpu().numpy())
        # np.save(os.path.join(target_path, 'categories.npy'), np.array(categories))
        # print_log('Saving success...')

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Sparse', losses.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Dense', losses.avg(1), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()]), logger = logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger=logger)

            # Save ckeckpoints
            if  metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
        # if (config.max_epoch - epoch) < 10 or epoch%10 == 0:
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)     
    train_writer.close()
    val_writer.close()

def validate(base_model, test_dataloader, epoch, ChamferDisL1, ChamferDisL2, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1
    # print(f'n_samples: {n_samples}')

    features = []
    categories = []
    taxonomy_list = ['02691156', '02933112', '02958343', '03001627', '03636649', '04256520', '04379243', '04530566']
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.val._base_.N_POINTS
            dataset_name = config.dataset.val._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()
            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                partial, _ = misc.seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                partial = partial.cuda()
            elif dataset_name == 'MVP_16384':
                partial = data[0].cuda()
                gt = data[1].cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            ret = base_model(partial)
            coarse_points = ret[0]
            dense_points = ret[1]

            # features.append(global_feature)
            # categories.append(taxonomy_list.index(taxonomy_id))

            sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
            sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
            dense_loss_l1 =  ChamferDisL1(dense_points, gt)
            dense_loss_l2 =  ChamferDisL2(dense_points, gt)

            if args.distributed:
                sparse_loss_l1 = dist_utils.reduce_tensor(sparse_loss_l1, args)
                sparse_loss_l2 = dist_utils.reduce_tensor(sparse_loss_l2, args)
                dense_loss_l1 = dist_utils.reduce_tensor(dense_loss_l1, args)
                dense_loss_l2 = dist_utils.reduce_tensor(dense_loss_l2, args)

            test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

            _metrics = Metrics.get(dense_points, gt)

            if taxonomy_id not in category_metrics:
                category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
            category_metrics[taxonomy_id].update(_metrics)

            if val_writer is not None and idx % 200 == 0:
                input_pc = partial.squeeze().detach().cpu().numpy()
                input_pc = misc.get_ptcloud_img(input_pc)
                val_writer.add_image('Model%02d/Input'% idx , input_pc, epoch, dataformats='HWC')

                sparse = coarse_points.squeeze().cpu().numpy()
                sparse_img = misc.get_ptcloud_img(sparse)
                val_writer.add_image('Model%02d/Sparse' % idx, sparse_img, epoch, dataformats='HWC')

                dense = dense_points.squeeze().cpu().numpy()
                dense_img = misc.get_ptcloud_img(dense)
                val_writer.add_image('Model%02d/Dense' % idx, dense_img, epoch, dataformats='HWC')
                
                gt_ptcloud = gt.squeeze().cpu().numpy()
                gt_ptcloud_img = misc.get_ptcloud_img(gt_ptcloud)
                val_writer.add_image('Model%02d/DenseGT' % idx, gt_ptcloud_img, epoch, dataformats='HWC')
        
            if (idx+1) % 20 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)

        # features = torch.cat(features, dim=0)
        # target_path = os.path.join(args.experiment_path, 'tsne_vis_result_test_set')
        # if not os.path.exists(target_path):
        #     os.makedirs(target_path)
        # np.save(os.path.join(target_path, 'features.npy'), features.detach().cpu().numpy())
        # np.save(os.path.join(target_path, 'categories.npy'), np.array(categories))
        # print_log('Saving success...')
        
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[Validation] EPOCH: %d  Metrics = %s' % (epoch, ['%.4f' % m for m in test_metrics.avg()]), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()
     
    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)

    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall\t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Loss/Epoch/Sparse', test_losses.avg(0), epoch)
        val_writer.add_scalar('Loss/Epoch/Dense', test_losses.avg(2), epoch)
        for i, metric in enumerate(test_metrics.items):
            val_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch)

    return Metrics(config.consider_metric, test_metrics.avg())


crop_ratio = {
    'easy': 1/4,
    'median' :1/2,
    'hard':3/4
}

def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.completion_dataset_builder(args, config.dataset.test)
 
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger)
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()

    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger=logger)

def test(base_model, test_dataloader, ChamferDisL1, ChamferDisL2, args, config, logger = None):

    base_model.eval()  # set model to eval mode

    test_losses = AverageMeter(['SparseLossL1', 'SparseLossL2', 'DenseLossL1', 'DenseLossL2'])
    test_metrics = AverageMeter(Metrics.names())
    category_metrics = dict()
    n_samples = len(test_dataloader) # bs is 1
    print(n_samples)

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            taxonomy_id = taxonomy_ids[0] if isinstance(taxonomy_ids[0], str) else taxonomy_ids[0].item()
            model_id = model_ids[0]

            npoints = config.dataset.test._base_.N_POINTS
            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'PCN':
                partial = data[0].cuda()
                gt = data[1].cuda()

                ret = base_model(partial)
                coarse_points = ret[0]
                dense_points = ret[1]

                target_path = os.path.join("/data/feiben1/PCPT-vis/PCN/", 'PCPT_vis_result', taxonomy_id)
                if not os.path.exists(target_path):
                    os.makedirs(target_path)
                misc.visualize_KITTI(
                    os.path.join(target_path, f'{idx:03d}_{model_id}'),
                    [partial[0].cpu(), dense_points[0].cpu()]
                )

                sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                dense_loss_l2 =  ChamferDisL2(dense_points, gt)

                test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

                _metrics = Metrics.get(dense_points ,gt)
                test_metrics.update(_metrics)

                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)

            elif dataset_name == 'ShapeNet':
                gt = data.cuda()
                choice = [torch.Tensor([1,1,1]),torch.Tensor([1,1,-1]),torch.Tensor([1,-1,1]),torch.Tensor([-1,1,1]),
                            torch.Tensor([-1,-1,1]),torch.Tensor([-1,1,-1]), torch.Tensor([1,-1,-1]),torch.Tensor([-1,-1,-1])]
                num_crop = int(npoints * crop_ratio[args.mode])
                for item in choice:           
                    partial, _ = misc.seprate_point_cloud(gt, npoints, num_crop, fixed_points = item)
                    # NOTE: subsample the input
                    partial = misc.fps(partial, 2048)
                    ret = base_model(partial)
                    coarse_points = ret[0]
                    dense_points = ret[1]
                    
                    # target_path = os.path.join("/data/feiben1/PCPT-vis/ShapeNetUnseen21/", 'PCPT_vis_result', taxonomy_id)
                    # if not os.path.exists(target_path):
                    #     os.makedirs(target_path)
                    # misc.visualize_KITTI(
                    #     os.path.join(target_path, f'{idx:03d}_{model_id}'),
                    #     [partial[0].cpu(), dense_points[0].cpu()]
                    # )
                    # idxs = [34, 88, 221, 394, 486, 2099, 3888, 3899, 4136, 5518, 5885, 6338, 6839, 8035, 8749, 8944, 9367]
                    # idxs = [350, 435, 668, 735, 1010, 1277, 1490, 1506, 1830, 1835, 1842, 1864, 1908, 1923, 1974, 1994, 2062, 2269]
                    # if idx in idxs:     
                    #     target_path = os.path.join(args.experiment_path, 'vis_result', taxonomy_id)
                    #     if not os.path.exists(target_path):
                    #         os.makedirs(target_path)
                    #     misc.visualize_KITTI(
                    #         os.path.join(target_path, f'{idx:03d}_{model_id}'),
                    #         [partial[0].cpu(), dense_points[0].cpu()]
                    #     )

                    sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                    sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                    dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                    dense_loss_l2 =  ChamferDisL2(dense_points, gt)

                    test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

                    _metrics = Metrics.get(dense_points ,gt)

                    # test_metrics.update(_metrics)

                    if taxonomy_id not in category_metrics:
                        category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                    category_metrics[taxonomy_id].update(_metrics)
            elif dataset_name == 'MVP_16384':
                partial = data[0].cuda()
                gt = data[1].cuda()

                ret = base_model(partial)
                coarse_points = ret[0]
                dense_points = ret[1]

                # idxs = [41509, 40337, 40335, 39046, 39054, 39116, 39127, 39132, 40298, 37787, 38985, 38727, 38622, 36431, 36445, 36708, 36789, 35896, 36122, 36193, 33846, 34322, 34567, 33571, 33520, 33383, 33374, 33362, 32992, 32005, 31894, 31816, 31330, 32001]
                # # if idx in idxs:
                #     target_path = os.path.join(args.experiment_path, 'vis_result', taxonomy_id)
                #     if not os.path.exists(target_path):
                #         os.makedirs(target_path)
                #     misc.visualize_KITTI(
                #         os.path.join(target_path, f'{idx:03d}_{model_id}'),
                #         [partial[0].cpu(), dense_points[0].cpu()]
                #     )

                target_path = os.path.join("/data/feiben1/PCPT-vis/MVP/", 'PCPT_vis_result', taxonomy_id)
                if not os.path.exists(target_path):
                    os.makedirs(target_path)
                misc.visualize_KITTI(
                    os.path.join(target_path, f'{idx:03d}_{model_id}'),
                    [partial[0].cpu(), dense_points[0].cpu()]
                )

                sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
                sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
                dense_loss_l1 =  ChamferDisL1(dense_points, gt)
                dense_loss_l2 =  ChamferDisL2(dense_points, gt)

                test_losses.update([sparse_loss_l1.item() * 1000, sparse_loss_l2.item() * 1000, dense_loss_l1.item() * 1000, dense_loss_l2.item() * 1000])

                _metrics = Metrics.get(dense_points ,gt)
                test_metrics.update(_metrics)

                if taxonomy_id not in category_metrics:
                    category_metrics[taxonomy_id] = AverageMeter(Metrics.names())
                category_metrics[taxonomy_id].update(_metrics)
            elif dataset_name == 'KITTI':
                partial = data.cuda()
                ret = base_model(partial)
                dense_points = ret[1]
                target_path = os.path.join(args.experiment_path, 'vis_result')
                if not os.path.exists(target_path):
                    os.mkdir(target_path)
                misc.visualize_KITTI(
                    os.path.join(target_path, f'{model_id}_{idx:03d}'),
                    [partial[0].cpu(), dense_points[0].cpu()]
                )
                continue
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            if (idx+1) % 200 == 0:
                print_log('Test[%d/%d] Taxonomy = %s Sample = %s Losses = %s Metrics = %s' %
                            (idx + 1, n_samples, taxonomy_id, model_id, ['%.4f' % l for l in test_losses.val()], 
                            ['%.4f' % m for m in _metrics]), logger=logger)
        if dataset_name == 'KITTI':
            return
        for _,v in category_metrics.items():
            test_metrics.update(v.avg())
        print_log('[TEST] Metrics = %s' % (['%.4f' % m for m in test_metrics.avg()]), logger=logger)

     

    # Print testing results
    shapenet_dict = json.load(open('./data/shapenet_synset_dict.json', 'r'))
    print_log('============================ TEST RESULTS ============================',logger=logger)
    msg = ''
    msg += 'Taxonomy\t'
    msg += '#Sample\t'
    for metric in test_metrics.items:
        msg += metric + '\t'
    msg += '#ModelName\t'
    print_log(msg, logger=logger)


    for taxonomy_id in category_metrics:
        msg = ''
        msg += (taxonomy_id + '\t')
        msg += (str(category_metrics[taxonomy_id].count(0)) + '\t')
        for value in category_metrics[taxonomy_id].avg():
            msg += '%.3f \t' % value
        msg += shapenet_dict[taxonomy_id] + '\t'
        print_log(msg, logger=logger)

    msg = ''
    msg += 'Overall \t\t'
    for value in test_metrics.avg():
        msg += '%.3f \t' % value
    print_log(msg, logger=logger)
    return 
