from collections import OrderedDict
from .ctc_model import Ctcnet
import torch
import re


def freeze_bn(net):
    for module in net.modules():
        if isinstance(module, torch.nn.modules.BatchNorm1d):
            for i in module.parameters():
                i.requires_grad = False


def release_bn(net):
    for module in net.modules():
        if isinstance(module, torch.nn.modules.BatchNorm1d):
            for i in module.parameters():
                i.requires_grad = True


def init_model(cfg):
    num_classes = cfg.num_classes
    device = ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    if cfg.extra:
        num_classes += 1
    elif cfg.model == 'ctc':
        model = Ctcnet(num_inputs=cfg.inputs_dim, embed_size=cfg.embed_size, class_out=cfg.class_out).cuda()
        model.to(device)
    elif cfg.model == 'testnet':
        model = Testnet(num_inputs=cfg.inputs_dim, embed_size=cfg.embed_size, class_out=cfg.class_out).cuda()
        model.to(device)
    elif cfg.model == 'ctcbn':
        model = Ctcnet_BN(num_inputs=cfg.inputs_dim, embed_size=cfg.embed_size, class_out=cfg.class_out).cuda()
        model.to(device)
    elif cfg.model == 'ctcvae':
        model = CTC_VAE(input_dim=cfg.inputs_dim, latent_dim=cfg.embed_size, class_out=cfg.num_classes,
                        recon_loss=cfg.vaeloss,
                        use_bn=cfg.bn).cuda()
        model.to(device)
    elif cfg.model == 'ctccnn':
        model = Ctcnet_cnn(num_inputs=cfg.inputs_dim, embed_size=cfg.embed_size, class_out=cfg.class_out,
                           batch_size=cfg.batch_size).cuda()
        model.to(device)

    if cfg.fix_bn:
        freeze_bn(model)
    else:
        release_bn(model)

    if cfg.init_weight != 'None':
        params = torch.load(cfg.init_weight)
        print('Model restored with weights from : {}'.format(cfg.init_weight))


        try:
            model.load_state_dict(params, strict=True)


        except Exception as e:
            model.load_state_dict(params, strict=False)
        if cfg.neuron_add != 'None':
            model.classifier.add_units(cfg.neuron_add)

    if cfg.multi_gpu:
        model = nn.DataParallel(model)
        model = model.module
    if cfg.train:
        model = model.train().cuda()
        print('Mode --> Train')
    else:
        model = model.eval().cuda()
        print('Mode --> Eval')
    return model
