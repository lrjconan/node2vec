import os
import scipy
import numpy as np
from scipy import io


def format_mat_data(file_name_in, file_name_out):
  mat_data = scipy.io.loadmat(file_name_in)
  network = mat_data["network"].toarray()
  I, J, _ = scipy.sparse.find(network)

  # pack up data
  with open(file_name_out, "w") as ff:
    for ii in xrange(len(I)):
      ff.write("{} {}\n".format(I[ii] + 1, J[ii] + 1))


def main():
  ppi_file_in = "graph/Homo_sapiens.mat"
  ppi_file_out = "graph/ppi.edgelist"
  wiki_file_in = "graph/POS.mat"
  wiki_file_out = "graph/wiki.edgelist"

  # read data  
  ppi_data = format_mat_data(ppi_file_in, ppi_file_out)
  wiki_data = format_mat_data(wiki_file_in, wiki_file_out)


if __name__ == "__main__":
  main()
