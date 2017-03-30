import os
import scipy
import numpy as np
from scipy import io


def read_csv_file(file_name):
  with open(file_name, "r") as ff:
    count = 0

    for line in ff:
      line_str = line.rstrip().split(",")

      if count == 0:
        num_col = len(line_str)
        results = [[] for _ in xrange(num_col)]

      for ii, xx in enumerate(line_str):
        results[ii] += [int(xx)]

      count += 1

  return results


def read_embedding(file_name):

  with open(file_name) as ff:
    count = 0

    for line in ff:
      line_str = line.rstrip().split()

      if count == 0:
        num_nodes = int(line_str[0])
        dim = int(line_str[1])
        node_feat = np.zeros([num_nodes, dim], dtype=np.float32)
      else:
        for ii, xx in enumerate(line_str):
          if ii == 0:
            node_idx = int(xx) - 1
          else:
            node_feat[node_idx, ii - 1] = float(xx)

      count += 1

  return node_feat


def read_blog_data(folder_name):
  node_file = os.path.join(folder_name, "nodes.csv")
  edge_file = os.path.join(folder_name, "edges.csv")
  label_file = os.path.join(folder_name, "groups.csv")
  annotation_file = os.path.join(folder_name, "group-edges.csv")

  node_id = read_csv_file(node_file)[0]
  edge_id = read_csv_file(edge_file)
  label_id = read_csv_file(label_file)[0]
  node_annotation = read_csv_file(annotation_file)

  # pack up data
  num_node = len(node_id)
  num_class = len(label_id)
  node_id_to_idx = dict(zip(node_id, range(num_node)))
  label_id_to_idx = dict(zip(label_id, range(num_class)))

  node = [node_id_to_idx[xx] for xx in node_id]
  label = dict([(xx, []) for xx in node])

  for ii, xx in enumerate(node_annotation[0]):
    label[node_id_to_idx[xx]] += [label_id_to_idx[node_annotation[1][ii]]]

  graph = dict([(xx, []) for xx in node])
  for ii, xx in enumerate(edge_id[0]):
    graph[node_id_to_idx[xx]] += [node_id_to_idx[edge_id[1][ii]]]
    graph[node_id_to_idx[edge_id[1][ii]]] += [node_id_to_idx[xx]]

  graph_data = {}
  graph_data["node"] = node
  graph_data["label"] = np.zeros([num_node, num_class], dtype=np.int32)

  for xx in node:
    for yy in label[xx]:
      graph_data["label"][xx, yy] = 1

  graph_data["graph"] = graph

  return graph_data


def read_mat_data(filename):
  mat_data = scipy.io.loadmat(filename)
  label = mat_data["group"].toarray()
  network = mat_data["network"].toarray()
  node_id = range(label.shape[0])
  graph = dict([(xx, []) for xx in node_id])
  I, J, _ = scipy.sparse.find(network)

  for ii in xrange(len(I)):
    graph[I[ii]] += [J[ii]]

  # pack up data
  graph_data = {}
  graph_data["node"] = range(label.shape[0])
  graph_data["label"] = label.astype(np.float32)
  graph_data["graph"] = graph

  return graph_data


def main():
  blog_folder = "../../data/node2vec/BlogCatalog"
  ppi_file = "../../data/node2vec/Homo_sapiens.mat"
  wiki_file = "../../data/node2vec/POS.mat"

  # read data
  blog_data = read_blog_data(blog_folder)
  ppi_data = read_mat_data(ppi_file)
  wiki_data = read_mat_data(wiki_file)


if __name__ == "__main__":
  main()
