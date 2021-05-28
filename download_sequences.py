import argparse
from Bio import Entrez

parser = argparse.ArgumentParser()
parser.add_argument('--ids', help='NCBI sequence ids', required=True)
parser.add_argument('--fasta', help='output fasta filename', required=True)
args = parser.parse_args()
with open(args.ids) as f:
    entries = f.readlines()
entries = [x.strip() for x in entries]
records = []
for entry in entries:
    handle = Entrez.efetch(db="nucleotide", id=entry, rettype="fasta", retmode="text")
    record = handle.read()
    records.append(record.strip())
    handle.close()
    
fasta='\n'.join(records)
fasta_file = open(args.fasta, "w")
fasta_file.write(fasta)
fasta_file.close()
