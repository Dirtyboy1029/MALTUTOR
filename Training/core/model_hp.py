from collections import namedtuple

_TRAIN_HP_TEMPLATE = namedtuple('training',
                                ['random_seed', 'n_epochs', 'batch_size', 'learning_rate', 'clipvalue', 'interval'])
train_hparam = _TRAIN_HP_TEMPLATE(random_seed=23456,
                                  n_epochs=30,
                                  batch_size=8,
                                  learning_rate=0.0001,
                                  clipvalue=100.,
                                  interval=2  # saving model weights
                                  )



# these hyper-paramters of finetuning are used for tuning r2d2 model, which is deprecated owing to the issue of effectiveness
_FINETUNE_TRAIN_HP_TEMPLATE = namedtuple('finetuning',
                                         ['random_seed', 'n_epochs', 'batch_size', 'learning_rate', 'n_epochs_ft',
                                          'learning_rate_ft', 'unfreezed_layers', 'clipvalue', 'interval'])
finetuning_hparam = _FINETUNE_TRAIN_HP_TEMPLATE(random_seed=23456,
                                                n_epochs=10,
                                                batch_size=16,
                                                learning_rate=0.0001,
                                                n_epochs_ft=20,
                                                learning_rate_ft=0.00001,
                                                unfreezed_layers=3,
                                                clipvalue=100.,
                                                interval=2  # saving model weights
                                                )

_MC_DROPOUT_HP_TEMPLATE = namedtuple('mc_dropout', ['dropout_rate', 'n_sampling'])
mc_dropout_hparam = _MC_DROPOUT_HP_TEMPLATE(dropout_rate=0.4,
                                            n_sampling=10
                                            )
_BAYESIAN_HP_TEMPLATE = namedtuple('bayesian', ['n_sampling'])
bayesian_ensemble_hparam = _BAYESIAN_HP_TEMPLATE(n_sampling=10)

_DNN_HP_TEMPLATE = namedtuple('DNN',
                              ['hidden_units', 'dropout_rate', 'activation', 'output_dim'])

dnn_hparam = _DNN_HP_TEMPLATE(hidden_units=[200,200],
                              # DNN has two hidden layers with each having 200 neurons
                              dropout_rate=0.4,
                              activation='relu',
                              output_dim=1  # binary classification#
                              )

_TEXT_CNN_HP_TEMPLATE = namedtuple('textCNN',
                                   ['hidden_units', 'dropout_rate', 'activation', 'output_dim',
                                    'vocab_size', 'n_embedding_dim', 'n_conv_filters', 'kernel_size',
                                    'max_sequence_length',
                                    'use_spatial_dropout', 'use_conv_dropout'])

text_cnn_hparam = _TEXT_CNN_HP_TEMPLATE(hidden_units=[200, 200],
                                        dropout_rate=0.4,
                                        activation='relu',
                                        vocab_size=256,
                                        n_embedding_dim=8,
                                        n_conv_filters=64,
                                        kernel_size=8,
                                        max_sequence_length=500000,  # shall be large for promoting accuracy
                                        use_spatial_dropout=False,
                                        use_conv_dropout=False,
                                        output_dim=1
                                        )

_MULTIMOD_HP_TEMPLATE = namedtuple('multimodalitynn',
                                   ['hidden_units', 'dropout_rate', 'activation', 'output_dim',
                                    'n_modalities', 'initial_hidden_units',
                                    ])

multimodalitynn_hparam = _MULTIMOD_HP_TEMPLATE(
    hidden_units=[200, 200],
    dropout_rate=0.4,
    activation='relu',
    n_modalities=5,
    initial_hidden_units=[500, 500],
    output_dim=1
)

_R2D2_HP_TEMPLATE = namedtuple('r2d2',
                               ['use_small_model', 'trainable', 'unfreezed_layers', 'dropout_rate', 'output_dim'])

r2d2_hparam = _R2D2_HP_TEMPLATE(
    use_small_model=False,  # if True : return MobileNetV2 else InceptionV3
    trainable=False,  # if False : the learned parameters of InceptionV3 or MobileNetV2 will be frozen
    unfreezed_layers=8,
    dropout_rate=0.4,
    output_dim=1
)

_DROIDETEC_HP_TEMPLATE = namedtuple('droidetec',
                                    ['vocab_size', 'n_embedding_dim', 'lstm_units', 'hidden_units',
                                     'dropout_rate', 'max_sequence_length', 'output_dim'])

droidetec_hparam = _DROIDETEC_HP_TEMPLATE(
    vocab_size=100000,  # owing to the GPU memory size, we set 100,000
    n_embedding_dim=8,
    lstm_units=64,
    hidden_units=[200],
    dropout_rate=0.4,
    max_sequence_length=800000,
    output_dim=1
)

