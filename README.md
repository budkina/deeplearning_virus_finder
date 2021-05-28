# deeplearning_virus_finder
The idea was taken from https://github.com/jessieren/DeepVirFinder

# Getting Started

Build docker container:

```docker build /path/to/directory/deeplearning_virus_finder -t dvf```

Run the docker container:

```docker run -v /path/to/input:/input dvf python train.py --vir_train_set input/vir_train.fasta \
  --bac_train_set input/bac_train.fasta \
  --vir_test_set input/vir_test.fasta \
  --bac_test_set  input/bac_test.fasta```

# Commands
```
  -h, --help            show this help message and exit
  
  --vir_train_set VIR_TRAIN_SET
                        Virus train set fasta file
                        
  --bac_train_set BAC_TRAIN_SET
                        Bacteria train set fasta file
                        
  --vir_test_set VIR_TEST_SET
                        Virus test set fasta file
                        
  --bac_test_set BAC_TEST_SET
                        Bacteria test set fasta file
                        
  --batch_size BATCH_SIZE
                        Batch size
                        
  --learning_rate LEARNING_RATE
                        Learning rate
                        
  --n_epochs N_EPOCHS   Number of epochs
  
  --train_fragment_length TRAIN_FRAGMENT_LENGTH
                        Fragment length
                        
  --conv_layers_num CONV_LAYERS_NUM
                        Number of convolutional modules
                        
  --conv_kernel_size CONV_KERNEL_SIZE
                        Convolutional filter size
                        
  --pool_kernel_size POOL_KERNEL_SIZE
                        Pool filter size
                        
  --fc_size FC_SIZE     Number of neurons in the fully-connected layer
  
  --cnn_type CNN_TYPE   Type of CNN
  
  --fr_result FR_RESULT
                        The procedure for obtaining the overall result for forward and reverse strands: max or average, default: average
```
