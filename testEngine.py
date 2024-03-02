import csv
import json
import requests
from time import time
# url = 'http://35.232.59.3:8080'
# place the domain you got from ngrok or GCP IP below.
url = 'http://8ea7-34-134-26-103.ngrok-free.app'

def average_precision(true_list, predicted_list, k=40):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    precisions = []
    for i,doc_id in enumerate(predicted_list):
        if doc_id in true_set:
            prec = (len(precisions)+1) / (i+1)
            precisions.append(prec)
    if len(precisions) == 0:
        return 0.0
    return round(sum(precisions)/len(precisions),3)

def precision_at_k(true_list, predicted_list, k):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    if len(predicted_list) == 0:
        return 0.0
    return round(len([1 for doc_id in predicted_list if doc_id in true_set]) / len(predicted_list), 3)
def recall_at_k(true_list, predicted_list, k):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    if len(true_set) < 1:
        return 1.0
    return round(len([1 for doc_id in predicted_list if doc_id in true_set]) / len(true_set), 3)
def f1_at_k(true_list, predicted_list, k):
    p = precision_at_k(true_list, predicted_list, k)
    r = recall_at_k(true_list, predicted_list, k)
    if p == 0.0 or r == 0.0:
        return 0.0
    return round(2.0 / (1.0/p + 1.0/r), 3)
def results_quality(true_list, predicted_list):
    p5 = precision_at_k(true_list, predicted_list, 5)
    f1_30 = f1_at_k(true_list, predicted_list, 30)
    if p5 == 0.0 or f1_30 == 0.0:
        return 0.0
    return round(2.0 / (1.0/p5 + 1.0/f1_30), 3)


def main():
    with open('queries_train.json', 'rt') as f:
        queries = json.load(f)
    rq = None
    qs_res = []
    for q, true_wids in queries.items():
      duration, ap = None, None
      t_start = time()
      try:
        res = requests.get(url + '/search', {'query': q}, timeout=35)
        duration = time() - t_start
        if res.status_code == 200:
          pred_wids, _ = zip(*res.json())
          rq = results_quality(true_wids, pred_wids)
        else:
          rq = "Couldn't get results"
      except:
        rq = "Couldn't get results"

      qs_res.append((q, duration, rq))

    # Calculate and print metrics for each query
    for q, duration, rq in qs_res:
        print(f"Query: {q}, Duration: {duration}, Results Quality: {rq}")

if __name__ == '__main__':
    main()