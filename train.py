import argparse
import CNN_type_1
import CNN_type_2
import torch
import trainer_type_1
import trainer_type_2

def train_model_1(vir_train_set,
    bac_train_set,
    train_fragment_length,
    conv_layers_num,
    conv_kernel_size,
    pool_kernel_size,
    learning_rate,
    batch_size,
    n_epochs):

    model_forward = CNN_type_1.CNN(train_fragment_length,
        conv_layers_num,
        conv_kernel_size,
        pool_kernel_size)

    model_reverse = CNN_type_1.CNN(train_fragment_length, 
        conv_layers_num,
        conv_kernel_size,
        pool_kernel_size)

    trainer = trainer_type_1.Trainer(model_forward, model_reverse, learning_rate)

    # train model
    trainer.fit(vir_train_set, bac_train_set, batch_size, n_epochs)


    scores = trainer.get_model_scores(vir_train_set, bac_train_set,4)
    print(scores[0])
    print(scores[1])
    print(scores[2])
    print(scores[3])
    print(scores[4])


def train_model_2(vir_train_set,
    bac_train_set,
    train_fragment_length,
    conv_layers_num,
    conv_kernel_size,
    pool_kernel_size,
    learning_rate,
    batch_size,
    n_epochs):


    model = CNN_type_2.CNN(train_fragment_length,
        conv_layers_num,
        conv_kernel_size,
        pool_kernel_size)

    trainer = trainer_type_2.Trainer(model, learning_rate)

    # train model
    trainer.fit(vir_train_set, bac_train_set, batch_size, n_epochs)


    scores = trainer.get_model_scores(vir_train_set, bac_train_set,4)
    print(scores[0])
    print(scores[1])
    print(scores[2])
    print(scores[3])
    print(scores[4])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vir_train_set',
        help='Virus train set fasta file',
        required = True)
    parser.add_argument('--bac_train_set',
        help='Bacteria train set fasta file',
        required = True)
    parser.add_argument('--vir_test_set',
        help='Virus test set fasta file',
        required = True)
    parser.add_argument('--bac_test_set',
        help='Bacteria test set fasta file',
        required = True)
    parser.add_argument('--batch_size',
        default = 4,
        type=int,
        help='Batch size')

    parser.add_argument('--learning_rate',
        default = 0.3,
        type=float,
        help='Learning rate')

    parser.add_argument('--n_epochs',
        default = 50,
        type=int,
        help='Number of epochs')

    parser.add_argument('--train_fragment_length',
        default = 1000,
        type=int,
        help='Fragment length')

    parser.add_argument('--conv_layers_num',
        default=10,
        type=int,
        help='Number of convolutional modules')

    parser.add_argument('--conv_kernel_size',
        default=5,
        type=int,
        help='Number of convolutional modules')

    parser.add_argument('--pool_kernel_size',
        default=4,
        type=int,
        help='Number of convolutional modules')

    parser.add_argument('--cnn_type',
        default=2,
        type=int,
        help='Type of CNN')

    # dilation

    args = parser.parse_args()

    # create model

    if args.cnn_type == 1:
        train_model_1(args.bac_train_set, 
            args.vir_train_set,
            args.train_fragment_length,
            args.conv_layers_num,
            args.conv_kernel_size,
            args.pool_kernel_size,
            args.learning_rate,
            args.batch_size,
            args.n_epochs)


    elif args.cnn_type == 2:
        train_model_2(args.bac_train_set, 
            args.vir_train_set,
            args.train_fragment_length,
            args.conv_layers_num,
            args.conv_kernel_size,
            args.pool_kernel_size,
            args.learning_rate,
            args.batch_size,
            args.n_epochs)



   
