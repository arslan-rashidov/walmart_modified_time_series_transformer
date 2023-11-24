import torch
from torch import Tensor


def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    Modified from:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Args:

        dim1: int, for both src and tgt masking, this must be target sequence
              length

        dim2: int, for src masking this must be encoder sequence length (i.e.
              the length of the input sequence to the model),
              and for tgt masking, this must be target sequence length


    Return:

        A Tensor of shape [dim1, dim2]
    """
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)
    #return torch.triu(torch.ones(dim1, dim2, device='cuda:0') * float('-inf'), diagonal=1)


def get_src_trg_masks(enc_seq_len, output_sequence_length):
    enc_seq_len = 24

    # Output length
    output_sequence_length = 2

    # Make src mask for decoder with size:
    tgt_mask = generate_square_subsequent_mask(
        dim1=output_sequence_length,
        dim2=output_sequence_length
       )

    src_mask = generate_square_subsequent_mask(
        dim1=output_sequence_length,
        dim2=enc_seq_len
        )

    return src_mask, tgt_mask