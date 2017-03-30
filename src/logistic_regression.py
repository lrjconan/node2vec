import os
import numpy as np
from data_reader import read_blog_data, read_mat_data, read_embedding
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression


def main():
  seed = 1234
  dataset = "blog"
  # dataset = "ppi"
  # dataset = "wiki"

  if dataset == "blog":
    embedding_file = "emb/blog.emb"
    data_folder = "graph/BlogCatalog"
    embedding = read_embedding(embedding_file)
    graph_data = read_blog_data(data_folder)
  elif dataset == "ppi":
    embedding_file = "emb/ppi.emb"
    data_folder = "graph"
    embedding = read_embedding(embedding_file)
    graph_data = read_mat_data(os.path.join(data_folder, "Homo_sapiens.mat"))
  elif dataset == "wiki":
    embedding_file = "emb/wiki.emb"
    data_folder = "graph"
    embedding = read_embedding(embedding_file)
    graph_data = read_mat_data(os.path.join(data_folder, "POS.mat"))
  else:
    raise ValueError("Unsupported dataset!")

  # generate a split
  label_rate = 0.5
  num_nodes = embedding.shape[0]
  num_nodes_train = int(num_nodes * label_rate)
  num_nodes_test = num_nodes - num_nodes_train

  prng = np.random.RandomState(seed)
  perm_idx = prng.permutation(num_nodes)
  split_train = perm_idx[:num_nodes_train]
  split_test = perm_idx[num_nodes_train:]

  train_X = embedding[split_train]
  train_Y = graph_data["label"][split_train]
  test_X = embedding[split_test]
  test_Y = graph_data["label"][split_test]

  classifier = OneVsRestClassifier(
      LogisticRegression(solver='sag', max_iter=1000, random_state=seed))

  classifier.fit(train_X, train_Y)

  pred_test_Y = classifier.predict(test_X)

  micro_F1 = f1_score(test_Y, pred_test_Y, average='micro')
  macro_F1 = f1_score(test_Y, pred_test_Y, average='macro')

  print("Test Micro-F1 score = {}".format(micro_F1))
  print("Test Macro-F1 score = {}".format(macro_F1))


if __name__ == "__main__":
  main()