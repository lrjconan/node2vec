import os
import numpy as np
from data_reader import read_blog_data, read_mat_data, read_embedding
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

# TUNE_HYPER = True
# BATCH_RUN = False

TUNE_HYPER = False
BATCH_RUN = True


def rand_split_data(embedding, label, label_rate, seed=1234):

  num_nodes = embedding.shape[0]
  num_nodes_train = int(num_nodes * label_rate)
  num_nodes_test = num_nodes - num_nodes_train

  prng = np.random.RandomState(seed)
  perm_idx = prng.permutation(num_nodes)
  split_train = perm_idx[:num_nodes_train]
  split_test = perm_idx[num_nodes_train:]

  train_X = embedding[split_train]
  train_Y = label[split_train]
  test_X = embedding[split_test]
  test_Y = label[split_test]

  data = {}
  data["train_X"] = train_X
  data["train_Y"] = train_Y
  data["test_X"] = test_X
  data["test_Y"] = test_Y

  return data


def logistic_regression(data, l2_reg=1.0e-2):

  classifier = OneVsRestClassifier(
      LogisticRegression(
          C=1.0 / l2_reg,
          solver='liblinear',
          max_iter=100000,
          random_state=None))

  classifier.fit(data["train_X"], data["train_Y"])

  return classifier.predict(data["test_X"])


def search_hyper_param(embedding, label, label_rate, seed=1234):

  hyper_grid = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1, 5, 10]
  prng = np.random.RandomState(seed)
  num_nodes = embedding.shape[0]
  perm_idx = prng.permutation(num_nodes)
  num_nodes_val = int(num_nodes * label_rate)
  split_val = perm_idx[:num_nodes_val]
  embedding_val = embedding[split_val]
  label_val = label[split_val]

  K = 3  # K-fold cross validation
  best_micro_F1 = 0.0
  best_hyper = 0.0

  for hyper in hyper_grid:
    micro_F1 = 0.0

    for kk in xrange(K):
      label_rate_k = 1.0 - 1.0 / float(K)
      data = rand_split_data(embedding_val, label_val, label_rate_k, seed=seed)
      pred_test_Y = logistic_regression(data, l2_reg=hyper)
      micro_F1 += f1_score(data["test_Y"], pred_test_Y, average='micro')

    micro_F1 /= float(K)
    if micro_F1 > best_micro_F1:
      best_micro_F1 = micro_F1
      best_hyper = hyper

  print("best hyperparameter = {}".format(best_hyper))
  print("best micro F1 score = {}".format(best_micro_F1))

  return best_hyper


def bat_run_LR(seed_list, embedding, label, label_rate, l2_reg):

  micro_F1 = []
  macro_F1 = []

  for seed in seed_list:
    data = rand_split_data(embedding, label, label_rate, seed=seed)
    pred_test_Y = logistic_regression(data, l2_reg=l2_reg)

    micro_F1 += [f1_score(data["test_Y"], pred_test_Y, average='micro')]
    macro_F1 += [f1_score(data["test_Y"], pred_test_Y, average='macro')]

  return micro_F1, macro_F1


def main():
  # dataset = "blog"
  dataset = "ppi"
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

  # hyper parameter search  
  if TUNE_HYPER:
    label_rate = 0.1
    best_l2_reg = search_hyper_param(
        embedding, graph_data["label"], label_rate, seed=1234)

  # random split
  if BATCH_RUN:
    seed_list = range(1234, 1234 + 10)
    # l2_reg = 5.0e-2 # wiki
    l2_reg = 1.0e-2 # ppi
    label_rate = 0.9
    micro_F1, macro_F1 = bat_run_LR(seed_list, embedding, graph_data["label"],
                                    label_rate, l2_reg)
    print(
        "Micro-F1 score = {} +- {}".format(np.mean(micro_F1), np.std(micro_F1)))
    print(
        "Macro-F1 score = {} +- {}".format(np.mean(macro_F1), np.std(macro_F1)))


if __name__ == "__main__":
  main()
