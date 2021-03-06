import numpy as np
import argparse
import struct

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="The input file (.fvecs)")
    parser.add_argument("--dst", help="The output file (.fvecs)")
    parser.add_argument("--topk", type=int, help="The number of element to pick up")
    return parser.parse_args()


def ivecs_write(fname, m):
    n, d = m.shape
    m1 = np.empty((n, d + 1), dtype='int32')
    m1[:, 0] = d
    m1[:, 1:] = m
    m1.tofile(fname)

def fvecs_write(fname, m):
    m = m.astype('float32')
    ivecs_write(fname, m.view('int32'))

if __name__ == "__main__":
    args = process_args()

    # Read topk vector one by one

    vecs = []    
    with open(args.src, "rb") as f:
        row_bin = f.read(4)
        assert row_bin != b''
        row, = struct.unpack('i', row_bin)

        dim_bin = f.read(4)
        assert dim_bin != b''
        dim, = struct.unpack('i', dim_bin)
        while 1:

            # The next 4 * dim byte is for a vector
            vec = struct.unpack('f' * dim, f.read(4 * dim))
            
            # Store it
            vecs.append(vec)
            if len(vecs) == args.topk:
                break
            
    vecs = np.array(vecs, dtype=np.float32)
    assert vecs.shape[0] == args.topk
    print("vecs.shape:", vecs.shape)

    fvecs_write(args.dst, vecs)
