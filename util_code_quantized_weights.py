import numpy as np
import torch
import sys
import argparse
from typing import Dict, List
import bitstream


qp = 16
keys_quantize = [
    'reconstructor.up0.kernel',
    'reconstructor.conv0.kernel',
    'reconstructor.up1.kernel',
    'reconstructor.conv1.kernel',
    'reconstructor.up2.kernel',
    'reconstructor.conv2.kernel',
    'reconstructor.conv2_cls.kernel',
    ]

keys_code_as_is = [
    'entropy_coder.sigma',
    'entropy_coder.mu',
    'reconstructor.activation.beta',
    'reconstructor.activation.gamma',
    'reconstructor.activation.pedestal',
    'reconstructor.up0.b',
    'reconstructor.conv0.b',
    'reconstructor.up1.b',
    'reconstructor.conv1.b',
    'reconstructor.up2.b',
    'reconstructor.conv2.b',
    'reconstructor.conv2_cls.b',
    'reconstructor.likelihood_model.sigma',
    'reconstructor.likelihood_model.mu'
]

def read_elements_from_file(fn):
    ws = torch.load(fn, map_location=torch.device('cpu'))
    pool = []
    as_is_pool = []
    for k in keys_quantize:
        pool.append(ws[k].numpy() * qp)
    for k in keys_code_as_is:
        as_is_pool.append(ws[k].numpy())
    eles = np.concatenate([i.reshape(-1) for i in pool])
    try:
        assert np.abs(np.sum(np.round(eles) - eles)) < 1e-3
    except:
        print("Warning: the loaded elements are not discrete!")
        raise ValueError('The loaded elements are not discrete.')
    return eles, pool, as_is_pool

def get_pdf(eles):
    lower_bound = int(np.min(eles))
    upper_bound = int(np.max(eles))
    bins = np.linspace(
        lower_bound, upper_bound + 1, (upper_bound - lower_bound) + 2
        ).astype(int)
    hist = np.histogram(eles, bins)
    freq = hist[0].astype(np.float64)
    pdf = freq / np.sum(freq)
    non_zero_entries = np.where(pdf != 0)
    non_zero_pdf = pdf[non_zero_entries]
    non_zero_p_bins = bins[non_zero_entries]
    return non_zero_pdf, non_zero_p_bins

class HTreeNode:
    def __init__(self, p, children=None, symbol=None) -> None:
        self.p = p
        self.children = children
        self.symbol = symbol

def get_huffman_codebook(pdf, bins):
    initial_nodes = []
    for p, k in zip(pdf, bins):
        initial_nodes.append(HTreeNode(p, None, k))
    nodes_remain = initial_nodes.copy()
    while len(nodes_remain) > 1:
        pool = sorted(nodes_remain, key=lambda x: x.p)
        a, b = pool[0], pool[1]
        merge_node = HTreeNode(a.p + b.p, (a, b), None)
        nodes_remain = pool[2:] + [merge_node,]
    root = nodes_remain[0]
    codebook = {}
    inv_codebook = {}
    def traverse(node: HTreeNode, codeword: List):
        if node.children is None:
            bitword = ''.join([str(c) for c in codeword])
            codebook[node.symbol] = np.array(codeword).astype(bool)
            inv_codebook[bitword] = node.symbol
            return
        traverse(node.children[0], codeword+[0])
        traverse(node.children[1], codeword+[1])
        return
    traverse(root, [])
    return codebook, inv_codebook

def est_rate(pdf, bins, codebook):
    expected_L = 0.
    for i in range(len(pdf)):
        p = pdf[i]
        symbol = bins[i]
        codeword = codebook[symbol]
        l = len(codeword)
        expected_L += l * p
    return expected_L

def entropy_encode(tensor_list, codebook):
    shape_list = []
    codewords = []
    for t in tensor_list:
        shape = t.shape
        shape_list.append(shape)
        flat_t = t.reshape(-1)
        for v in flat_t:
            assert np.abs(int(v) - v) < 1e-3
            codewords.append(codebook[int(v)])
    bit_string = np.concatenate(codewords, 0)
    print("Length of the bit string: ", bit_string.shape)
    bit_stream = bitstream.BitStream()
    bit_stream.write(bit_string, bool)
    if len(bit_stream) % 8 != 0:
        dummy_padding = 8 - len(bit_stream) % 8
        bit_stream.write(np.zeros((dummy_padding)).astype(bool), bool)
    n_words = len(bit_stream) // 8
    result = bit_stream.read(bytes, n_words)
    return result, shape_list

def entropy_decode(byte_str, inv_codebook, nsymbol, shape_list):
    bit_stream = bitstream.BitStream(byte_str)
    bit_str = str(bit_stream)
    head_ptr = 1
    tail_ptr = 0
    symbols = []
    while len(symbols) < nsymbol:
        if bit_str[tail_ptr:head_ptr] in inv_codebook:
            symbols.append(inv_codebook[bit_str[tail_ptr:head_ptr]])
            tail_ptr = head_ptr
            head_ptr = head_ptr + 1
        else:
            head_ptr += 1
    tensors = []
    for s in shape_list:
        n_sym = np.prod(s)
        tensor = np.array(symbols[:n_sym]).astype(np.float32).reshape(s)
        symbols = symbols[n_sym:]
        tensors.append(tensor)
    return tensors

def est_fp_bit_consumption(tensor_list):
    total_eles = 0
    for t in tensor_list:
        n_ele = np.prod(t.shape)
        assert t.dtype == np.float32
        total_eles += n_ele
    return total_eles * 32

def test_huffman_length():
    eles, pool, as_is_pool = read_elements_from_file(args.input)
    pdf, bins = get_pdf(eles)
    codebook, inv_codebook = get_huffman_codebook(pdf, bins)
    eL = est_rate(pdf, bins, codebook)
    print('Estimated E(l): ', eL)
    bit_stream, shape_list = entropy_encode(pool, codebook)
    print('Bit-stream length in bytes: ', len(bit_stream))
    n_bits_as_is = est_fp_bit_consumption(as_is_pool)
    print('Extra bits: ', n_bits_as_is)
    print('Total bits: ', n_bits_as_is + len(bit_stream)*8)

def test_huffman_enc_dec():
    eles, pool, as_is_pool = read_elements_from_file(args.input)
    pdf, bins = get_pdf(eles)
    codebook, inv_codebook = get_huffman_codebook(pdf, bins)
    eL = est_rate(pdf, bins, codebook)
    print('Estimated E(l): ', eL)
    bit_stream, shape_list = entropy_encode(pool, codebook)
    print('Bit-stream length in bytes: ', len(bit_stream))
    n_bits_as_is = est_fp_bit_consumption(as_is_pool)
    print('Extra bits: ', n_bits_as_is)
    print('Total bits: ', n_bits_as_is + len(bit_stream)*8)

    dec_pool = entropy_decode(bit_stream, inv_codebook, len(eles), shape_list)
    for a, b in zip(pool, dec_pool):
        assert np.sum(np.abs(a - b)) < 1e-6

def enc_dec_from_file(filename):
    eles, pool, as_is_pool = read_elements_from_file(filename)
    pdf, bins = get_pdf(eles)
    codebook, inv_codebook = get_huffman_codebook(pdf, bins)
    eL = est_rate(pdf, bins, codebook)
    print('Estimated E(l): ', eL)
    bit_stream, shape_list = entropy_encode(pool, codebook)
    print('Bit-stream length in bytes: ', len(bit_stream))
    n_bits_as_is = est_fp_bit_consumption(as_is_pool)
    print('Extra bits: ', n_bits_as_is)
    print('Total bits: ', n_bits_as_is + len(bit_stream)*8)

    dec_pool = entropy_decode(bit_stream, inv_codebook, len(eles), shape_list)
    for a, b in zip(pool, dec_pool):
        assert np.sum(np.abs(a - b)) < 1e-6
    return {
        'bit_stream': bit_stream,
        'inv_codebook': inv_codebook,
        'element_length': len(eles),
        'shape_list': shape_list,
        'as_is_pool': as_is_pool,
        'keys_quantize': keys_quantize,
        'keys_code_as_is': keys_code_as_is
        }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(    
        "input", nargs="?",
        help="Input filename. For encoding, the input is a checkpoint. For decoding, the input is a bin file.")

    parser.add_argument(    
        "output", nargs="?",
        help="Output filename. For encoding, the output is a bin file. For decoding, the output is a ckpt file.")

    args = parser.parse_args()
    test_huffman_enc_dec()