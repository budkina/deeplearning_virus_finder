import torch
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from io import StringIO
from torch.utils.data import Dataset, DataLoader
import re

class OneHotEncDataset(Dataset):
  def __init__(self, bac_fasta, vir_fasta, is_reverse = False):

    self.nuc_order = {y: x for x, y in enumerate(["A", "T", "C", "G"])}

    # covert to features vector
    bac_set = self.get_onehotenc_representation(bac_fasta, is_reverse)
    vir_set = self.get_onehotenc_representation(vir_fasta, is_reverse)

    # generate train data set
    np_set = np.concatenate((bac_set, vir_set)) 
    self.X = torch.LongTensor(np_set)

    # generate labels vector y
    labels_np = np.concatenate((np.zeros(len(bac_set)), np.ones(len(vir_set))))
    self.Y = torch.LongTensor(labels_np)

    #self.ids= np.concatenate((bacteria_ids, phage_ids))

  def __len__(self):
    return len(self.Y)

  def __getitem__(self,index):
    X = self.X[index].float()
    Y = self.Y[index].long()
    return X,Y

  def get_onehotenc_representation(self, set_fasta_filename, is_reverse):
    records = SeqIO.parse(set_fasta_filename, "fasta")
    ids = []
    matrices = []
    for record in records:
      sequence=record.seq.upper()
      if re.match('^[ACGT]+$', str(sequence)) is None:
        continue
      
      if is_reverse:
        sequence = sequence.reverse_complement()
      
      matrix = np.zeros((4, len(sequence)))

      for idx, base in enumerate(sequence):
        matrix[self.nuc_order[base], idx] = 1

      matrices.append(matrix)
      ids.append(record.id)

    return matrices #,ids