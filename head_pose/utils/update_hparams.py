def update_hparams(
    hparams, dropout, layer_feature_maps, start_fold, batch_size, lr, n_epochs, wavenet_dilation, alpha
):

    if dropout is not None:
        hparams['model']['dropout_rate'] = float(dropout)

    if layer_feature_maps is not None:
        hparams['model']['layer_feature_maps'] = eval(layer_feature_maps)

    if start_fold is not None:
        hparams['start_fold'] = start_fold

    if batch_size is not None:
        hparams['batch_size'] = int(batch_size)

    if lr is not None:
        hparams['optimizer_hparams']['lr'] = float(lr)

    if n_epochs is not None:
        hparams['n_epochs'] = int(n_epochs)

    if wavenet_dilation is not None:
        hparams['wavenet_dilation'] = eval(wavenet_dilation)

    if alpha is not None:
        hparams['model']['alpha'] = float(alpha)

    return hparams
