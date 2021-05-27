import argparse
import logging
import sys
import CNN_type_1
import CNN_type_2
import torch
import trainer_type_1
import trainer_type_2

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
        help='Batch size')
    parser.add_argument('--learning_rate',
        default = 0.3,
        help='Learning rate')
    parser.add_argument('--n_epochs',
        default = 50,
        help='Number of epochs')
    parser.add_argument('--train_fragment_length',
        default = 1000,
        help='Fragment length')
    parser.add_argument('--conv_layers_num',
        default=10,
        help='Number of convolutional modules')
    parser.add_argument('--conv_kernel_size',
        default=5,
        help='Number of convolutional modules')
    parser.add_argument('--pool_kernel_size',
        default=4,
        help='Number of convolutional modules')
    parser.add_argument('--fc_size',
        default=120,
        help='Number of neurons in fully-connected layer')
    parser.add_argument('--cnn_type',
        default=2,
        help='Type of CNN')
    parser.add_argument('--fr_result',
        default="average",
        help='The procedure for obtaining the overall result for forward and reverse strands: max or average, default: average')

    args = parser.parse_args()
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    logging.debug(f'Virus train set: {args.vir_train_set}')
    logging.debug(f'Bacteria train set: {args.bac_train_set}')
    logging.debug(f'Virus test set: {args.vir_test_set}')
    logging.debug(f'Bacteria test set: {args.bac_test_set}')

    train_fragment_length = int(args.train_fragment_length)
    learning_rate = float(args.learning_rate)
    n_epochs = int(args.n_epochs)
    batch_size = int(args.batch_size)
    conv_layers_num = int(args.conv_layers_num)
    conv_kernel_size = int(args.conv_kernel_size)
    pool_kernel_size= int(args.pool_kernel_size)
    fc_size = int(args.fc_size)
    cnn_type = int(args.cnn_type)

    logging.debug(f"""Configuration: train_fragment_length = {train_fragment_length},
        model type = {cnn_type},
        fr_result = {args.fr_result},
        conv_layers_num = {conv_layers_num},
        conv_kernel_size = {conv_kernel_size},
        pool_kernel_size = {pool_kernel_size},
        fc_size = {fc_size}
        """)


    logging.debug(f"""Learning: is_cuda_available = {torch.cuda.is_available()},
        batch_size = {batch_size},
        learning_rate = {learning_rate},
        n_epochs = {n_epochs},
        """)

    # Train model

    if cnn_type == 1:
        model_forward = CNN_type_1.CNN(train_fragment_length,
            conv_layers_num,
            conv_kernel_size,
            pool_kernel_size,
            fc_size)
        model_reverse = CNN_type_1.CNN(train_fragment_length, 
            conv_layers_num,
            conv_kernel_size,
            pool_kernel_size,
            fc_size)

        trainer = trainer_type_1.Trainer(model_forward, model_reverse, args.fr_result, learning_rate)
        trainer.fit(args.vir_train_set, args.bac_train_set, batch_size, n_epochs)

    elif cnn_type == 2:
        model = CNN_type_2.CNN(args.fr_result,
            train_fragment_length,
            conv_layers_num,
            conv_kernel_size,
            pool_kernel_size,
            fc_size)


        trainer = trainer_type_2.Trainer(model, learning_rate)
        trainer.fit(args.vir_train_set, args.bac_train_set, batch_size, n_epochs)
    else:
        logging.debug(f'Unknown CNN type {cnn_type}')
        sys.exit()


    # Test the model

    scores = trainer.get_model_scores(args.vir_test_set, args.bac_test_set, batch_size)
    
    logging.debug(f"""Result metrics: roc_auc_score = {scores[0]},
        precision = {scores[1]},
        recall = {scores[2]},
        precision = {scores[1]},
        f1 = {scores[3]},
        accuracy = {scores[4]}
         """)