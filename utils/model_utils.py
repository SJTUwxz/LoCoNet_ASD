def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()


def set_bn_non_trainable(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.weight.requires_grad = False
        m.bias.requires_grad = False


def freeze_bn_statistics(model):
    """freeze the statistic mean and variance in BN
    Args:
        model (nn.Module): The model to be freezed statistics.
    """
    model.apply(set_bn_eval)


def freeze_bn_parameters(model):
    """

    Args:
        model (nn.Module): The model to be freezed statistics.

    Returns: TODO

    """
    model.apply(set_bn_non_trainable)
