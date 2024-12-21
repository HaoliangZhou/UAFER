import torch

import utils.lr_decay as lrd

def build_optimizer(args, model):
    params_name = None
    params, param_group_names = lrd.param_groups_lrd(model, args.fix_layer, args.weight_decay, layer_decay=args.layer_decay)

    params_name = []
    for k, v in param_group_names.items():
        params_name += v["params"]

    optimizer = torch.optim.AdamW(params, lr=args.lr)
    # optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for name, param in model.named_parameters():
        # if name not in params_name or 'resblocks' not in name:
        #     param.requires_grad = False
        if name not in params_name:
            param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'total parameters: {total_params/1000000:,} M.')

    return optimizer