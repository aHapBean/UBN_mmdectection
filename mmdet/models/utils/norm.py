import torch.nn as nn
# import apex.parallel as ap
from mmdet.models.utils.CBN import *
from mmdet.models.utils.UBN import UnifiedBatchNorm2d

norm_cfg = {
    # format: layer_type: (abbreviation, module)
    'BN': ('bn', nn.BatchNorm2d),
    # 'SyncBN': ('bn', ap.SyncBatchNorm),
    'GN': ('gn', nn.GroupNorm),
    'CBN': ('bn', CBatchNorm2d),
    'UBN': ('bn', UnifiedBatchNorm2d)
    # and potentially 'IN' 'LN' 'BRN'
}
# layer = nn.BatchNorm2d(num_features, **cfg_)

def build_norm_layer(cfg, num_features, postfix=''):
    """ Build normalization layer

    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            frozen (bool): [optional] whether stop gradient updates
                of norm layer, it is helpful to set frozen mode
                in backbone's norms.
        num_features (int): number of channels from input
        postfix (int, str): appended into norm abbreation to
            create named layer.

    Returns:
        name (str): abbreation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    frozen = cfg_.pop('frozen', False)
    # 如果配置中有冻结参数，则将其值赋给 frozen 变量，否则使用默认值 False。在这之后，cfg_ 字典中就不再包含 'frozen' 这个键了。
    cfg_.setdefault('eps', 1e-5)
    
    # print("cfg_: ",cfg_)
    # print("norm_layer: ", norm_layer)
    cfg_.pop('requires_grad')   # Here
    # 注意原代码中的写法
    
    # 在你的代码中，报错是因为 nn.BatchNorm2d 不接受 requires_grad 这个关键字参数。Batch normalization层的权重参数通常是可以学习的，而不需要显式地设置 requires_grad。
    # 如果你希望冻结或禁用 Batch Normalization 层的梯度，你应该在创建层后手动设置相应的参数的 requires_grad。例如：
    
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        # print("layer.weight.requires_grad: ",layer.weight.requires_grad)
        if layer_type == 'SyncBN-pytorch':
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    if frozen:
        for param in layer.parameters():
            param.requires_grad = False

    return name, layer

# def build_norm_layer(cfg: Dict,
#                      num_features: int,
#                      postfix: Union[int, str] = '') -> Tuple[str, nn.Module]:
#     """Build normalization layer.

#     Args:
#         cfg (dict): The norm layer config, which should contain:

#             - type (str): Layer type.
#             - layer args: Args needed to instantiate a norm layer.
#             - requires_grad (bool, optional): Whether stop gradient updates.
#         num_features (int): Number of input channels.
#         postfix (int | str): The postfix to be appended into norm abbreviation
#             to create named layer.

#     Returns:
#         tuple[str, nn.Module]: The first element is the layer name consisting
#         of abbreviation and postfix, e.g., bn1, gn. The second element is the
#         created norm layer.
#     """
#     if not isinstance(cfg, dict):
#         raise TypeError('cfg must be a dict')
#     if 'type' not in cfg:
#         raise KeyError('the cfg dict must contain the key "type"')
#     cfg_ = cfg.copy()

#     layer_type = cfg_.pop('type')

#     if inspect.isclass(layer_type):
#         norm_layer = layer_type
#     else:
#         # Switch registry to the target scope. If `norm_layer` cannot be found
#         # in the registry, fallback to search `norm_layer` in the
#         # mmengine.MODELS.
#         with MODELS.switch_scope_and_registry(None) as registry:
#             norm_layer = registry.get(layer_type)
#         if norm_layer is None:
#             raise KeyError(f'Cannot find {norm_layer} in registry under '
#                            f'scope name {registry.scope}')
#     abbr = infer_abbr(norm_layer)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if norm_layer is not nn.GroupNorm:
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer