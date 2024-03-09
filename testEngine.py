import json
import requests
from time import time
# place the domain you got from ngrok or GCP IP below.
url = 'http://34.72.210.232:8080' # gcp ip
# url = 'http://127.0.0.1:8080' # local

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


def metrics():
    with open('queries_train.json', 'rt') as f:
        queries = json.load(f)
    qs_res = []
    for q, true_wids in queries.items():
      duration, ap, rq = None, None, None
      p5, p10, p30, f1_30, r5, r10, r30 = None, None, None, None, None, None, None
      t_start = time()
      try:
        res = requests.get(url + '/search', {'query': q}, timeout=60)
        duration = time() - t_start
        if res.status_code == 200:
          pred_wids, _ = zip(*res.json())
          rq = results_quality(true_wids, pred_wids)
          p5 = precision_at_k(true_wids, pred_wids, 5)
          p10 = precision_at_k(true_wids, pred_wids, 10)
          p30 = precision_at_k(true_wids, pred_wids, 30)
          ap = average_precision(true_wids, pred_wids, 30)
          f1_30 = f1_at_k(true_wids, pred_wids, 30)
          r5 = recall_at_k(true_wids, pred_wids, 5)
          r10 = recall_at_k(true_wids, pred_wids, 10)
          r30 = recall_at_k(true_wids, pred_wids, 30)
        else:
          duration = "Couldn't get response"
      except:
        duration = "Timeout"

      qs_res.append((q, duration, rq, p5, p10, p30, ap, f1_30, r5, r10, r30))

    # write results to a csv file
    with open('metrics/gcp_final_exp3or1_100res.csv', 'wt') as f:
        f.write('query,duration,rq,precision@5,precision@10,precision@30,average precision,f1@30,recall@5,recall@10,recall@30\n')
        for q, duration, rq, p5, p10, p30, ap, f1_30, r5, r10, r30 in qs_res:
            f.write(f'{q},{duration},{rq},{p5},{p10},{p30},{ap},{f1_30},{r5},{r10},{r30}\n')

if __name__ == '__main__':
    metrics()