import numpy as np
import argparse
import struct
from sklearn.cluster import KMeans

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", help="The input file (.fvecs)")
    parser.add_argument("--dst", help="The output file (.fvecs)")
    parser.add_argument("--extra", help="The output extra file (.fvecs)")
    parser.add_argument("--totalForTruth", help="The total for truth generate file (.fvecs)")
    return parser.parse_args()


if __name__ == "__main__":
    args = process_args()

    # Read topk vector one by one
    #change from default to xvec
    vecs = []
    row_bin = "";
    dim_bin = ""; 
    with open(args.src, "rb") as f:

        row_bin = f.read(4)
        assert row_bin != b''
        row, = struct.unpack('i', row_bin)

        dim_bin = f.read(4)
        assert dim_bin != b''
        dim, = struct.unpack('i', dim_bin)

        i = 0
        while 1:

            # The next dim byte is for a vector for spacev
            vec = struct.unpack('b' * dim, f.read(dim))
            
            # Store it
            vecs.append(vec)
            i += 1
            if i == row:
                break
            
    vecs = np.array(vecs, dtype=np.int8)
    assert vecs.shape[0] == row
    print("vecs.shape:", vecs.shape)
    estimator = KMeans(n_clusters=2)
    estimator.fit(vecs)
    label_pred = estimator.labels_ 
    vecs_1 = ""
    vecs_2 = ""
    with open(args.src, "rb") as f:

        row_bin = f.read(4)
        assert row_bin != b''
        row, = struct.unpack('i', row_bin)

        dim_bin = f.read(4)
        assert dim_bin != b''
        dim, = struct.unpack('i', dim_bin)

        i = 0
        num_1 = 0
        num_2 = 0
        while 1:

            # The next dim byte is for a vector for spacev
            vec = f.read(dim)
            
            if label_pred[i] == 0:
                vecs_1 += vec
                num_1 += 1
            else:
                vecs_2 += vec
                num_2 += 1
            i += 1
            if i == row:
                break
        print("cluster1: ", num_1)
        print("cluster2: ", num_2)
    with open(args.dst, "wb") as f:
        f.write(struct.pack('i', num_1))
        f.write(dim_bin)
        f.write(vecs_1)
    with open(args.extra, "wb") as f:
        f.write(struct.pack('i', num_2))
        f.write(dim_bin)
        f.write(vecs_2)
    with open(args.totalForTruth, "wb") as f:
        f.write(struct.pack('i', row))
        f.write(dim_bin)
        f.write(vecs_1)
        f.write(vecs_2)

