import click
from utils.update_hparams import update_hparams
from utils.logger import Logger


# import modules
from cv_pipeline import CVPipeline


@click.command()
@click.option('--start_fold', default=None, help='fold to train')
@click.option('--batch_size', default=None, help='batch size')
@click.option('--lr', default=None, help='learning rate')
@click.option('--n_epochs', default=None, help='number of epoches to run')
@click.option('--gpu', default='0', help='list of GPUs will be used for training')
@click.option('--model', default='wavenet_vat', help='Model type, one of following: vgg, resnet,resnet_xt')
@click.option('--layer_feature_maps', default=None, help='')
@click.option('--dropout', default=None, help='')
@click.option('--wavenet_dilation', default=None, help='')
@click.option('--alpha', default=None, help='')
def main(
    start_fold, batch_size, lr, n_epochs, gpu, model, layer_feature_maps, dropout, wavenet_dilation, alpha
):

    # check model type input
    assert (
        model == 'resnet' or model == 'wavenet' or 'wavenet_vat'
    ), 'Following models are supported:  resnet, wavenet,wavenet_vat'

    if model == 'resnet':
        from models.resnet import Model, hparams
        from utils.data_generators.data_generator import Dataset_train
    if model == 'wavenet':
        from models.wavenet import Model, hparams
        from utils.data_generators.data_generator import Dataset_train
    if model == 'wavenet_vat':
        from models.wavenet_vat import Model, hparams
        from utils.data_generators.data_generator import Dataset_train
    if model == 'wavenet_triplet':
        from models.wavenet_triplet import Model, hparams
        from utils.data_generators.data_generator_triplet import Dataset_train

    # process gpu selection string
    if gpu is not None:
        gpu = [int(i) for i in gpu.split(",")]
    if start_fold is not None:
        start_fold = [int(i) for i in start_fold.split(",")]

    hparams = update_hparams(
        hparams=hparams,
        dropout=dropout,
        layer_feature_maps=layer_feature_maps,
        start_fold=start_fold,
        batch_size=batch_size,
        lr=lr,
        n_epochs=n_epochs,
        wavenet_dilation=wavenet_dilation,
        alpha=alpha,
    )

    logger = Logger()

    print(f"Training fold: {hparams['start_fold']}")

    # run cross-val
    print(f'Selected type of the model: {model}')
    cross_val = CVPipeline(hparams=hparams, gpu=gpu, model=Model, Dataset_train=Dataset_train)
    fold_scores_val, fold_scores_test, start_training = cross_val.train()

    # save logs
    logger.kpi_logger.info('=============================================')
    logger.kpi_logger.info(f'Datetime = {start_training}')
    logger.kpi_logger.info(f'Model metric, val = {fold_scores_val}')
    logger.kpi_logger.info(f'Model metric, test = {fold_scores_test}')
    logger.kpi_logger.info(f"Model fold = {hparams['start_fold']}")
    logger.kpi_logger.info(f"Batch size = {hparams['batch_size']}")
    logger.kpi_logger.info(f"Lr = {hparams['optimizer_hparams']['lr']}")
    logger.kpi_logger.info(f"N epochs = {hparams['n_epochs']}")
    logger.kpi_logger.info(f'GPU = {gpu}')
    logger.kpi_logger.info(f"layer_feature_maps = {hparams['model']['layer_feature_maps']}")
    logger.kpi_logger.info(f"Dropout rate = {hparams['model']['dropout_rate']}")
    logger.kpi_logger.info(f"Model name: = {hparams['model_name']}")
    logger.kpi_logger.info('=============================================')


if __name__ == "__main__":
    main()


# TODO: clearn up jupyter notebooks

# TODO: stage models via DVC

# TODO: add calibration check into KPI

# TODO: add time-dependance into KPI

# TODO: clean up all garbage code, check up for hard-coded stuff

# TODO: add CV debug matching with specific model, add selection of the model in

# TODO: deploy the new model
