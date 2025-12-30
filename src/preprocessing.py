import torch
from src.features import BioFeatureEngineer 

class BioPreprocessor:
    """
    Handles the transformation of raw FASTA sequences into 
    latent space embeddings for Neural Inference.
    """
    def __init__(self):
        self.engineer = BioFeatureEngineer()

    def process(self, sequence):
        # Generates the 320-dimensional ESM-2 embedding
        return self.engineer.get_combined_vector(sequence)

    def to_tensor(self, vector, device):
        # Final computational preprocessing 
        tensor = torch.FloatTensor(vector).to(device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return tensor