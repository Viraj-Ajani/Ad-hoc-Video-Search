import pandas as pd
import numpy as np
from collections import defaultdict

def load_ground_truth(file_path):
    df = pd.read_csv(file_path, sep=r'\s+', header=None)
    df.columns = ['query', 'junk', 'shotid', 'stratum', 'judgment']
    return df

def load_results(file_path):
    df = pd.read_csv(file_path, sep=r'\s+', header=None)
    df.columns = ['query', 'shotid']
    return df

def xinfAP_for_query(gt_df, result_df, query_id):
    gt_query = gt_df[gt_df['query'] == query_id]
    results_query = result_df[result_df['query'] == query_id]['shotid'].tolist()

    # Remove unjudged samples
    judged_gt = gt_query[gt_query['judgment'] != -1]
    if judged_gt.empty:
        return 0.0

    relevant_shots = set(judged_gt[judged_gt['judgment'] == 1]['shotid'])
    total_relevant = len(relevant_shots)

    if total_relevant == 0:
        return 0.0

    # Build the ranked list
    precisions = []
    num_correct = 0

    for rank, shotid in enumerate(results_query, start=1):
        if shotid not in judged_gt['shotid'].values:
            continue  # Skip unjudged

        if shotid in relevant_shots:
            num_correct += 1
            precisions.append(num_correct / rank)

    if len(precisions) == 0:
        return 0.0

    return sum(precisions) / total_relevant

def compute_mean_xinfAP(ground_truth_path, results_path):
    gt_df = load_ground_truth(ground_truth_path)
    result_df = load_results(results_path)

    query_ids = sorted(gt_df['query'].unique())
    ap_scores = []

    for query_id in query_ids:
        ap = xinfAP_for_query(gt_df, result_df, query_id)
        ap_scores.append(ap)

    mean_xinfap = np.mean(ap_scores)
    return mean_xinfap, dict(zip(query_ids, ap_scores))

# Example usage
ground_truth_file = 'ground_truth.txt'
results_file = 'cleaned_results.txt'

mean_xinfap, per_query_ap = compute_mean_xinfAP(ground_truth_file, results_file)

print(f"Mean xinfAP: {mean_xinfap:.4f}")
for query, ap in per_query_ap.items():
    print(f"Query {query}: xinfAP = {ap:.4f}")

