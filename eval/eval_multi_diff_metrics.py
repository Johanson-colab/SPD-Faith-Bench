#!/usr/bin/env python3
"""
Multi-Difference Detection Metrics Evaluator.
Computes DQR, DS, TF1, CF1 metrics for VLM spot-the-difference task.
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


class MultiDiffEvaluator:
    """Evaluator for multi-difference detection metrics."""

    def __init__(self):
        self.type_names = ['color', 'remove', 'position']

    def calculate_sample_metrics(self, gt_modifications: List[Dict],
                                  pred_structured_output: List[Dict]) -> Dict:
        """Calculate metrics for a single sample."""
        result = {
            'gt_num': len(gt_modifications),
            'pred_num': len(pred_structured_output),
            'num_recall': 0.0,
            'type_level': {},
            'category_level': {}
        }

        # DQR: Difference Quantity Recall
        if len(gt_modifications) > 0:
            result['num_recall'] = len(pred_structured_output) / len(gt_modifications)

        # Type-level metrics (TF1)
        gt_types = [m['type'] for m in gt_modifications]
        pred_types = [p.get('type', '') for p in pred_structured_output if p.get('type')]

        for type_name in self.type_names:
            gt_count = gt_types.count(type_name)
            pred_count = pred_types.count(type_name)
            matched = min(pred_count, gt_count) if gt_count > 0 else 0

            precision = matched / pred_count if pred_count > 0 else None
            recall = matched / gt_count if gt_count > 0 else None

            if precision is not None and recall is not None and (precision + recall) > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            result['type_level'][type_name] = {
                'gt_count': gt_count,
                'pred_count': pred_count,
                'matched': matched,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

        # Category-level metrics (CF1)
        gt_categories = [m['category'] for m in gt_modifications]
        pred_categories = [p.get('category', '') for p in pred_structured_output if p.get('category')]

        matched = sum(1 for pred_cat in pred_categories if pred_cat in gt_categories)
        precision = matched / len(pred_categories) if len(pred_categories) > 0 else None
        recall = matched / len(gt_categories) if len(gt_categories) > 0 else None

        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        result['category_level'] = {
            'gt_count': len(gt_categories),
            'pred_count': len(pred_categories),
            'matched': matched,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        return result

    def aggregate_results(self, sample_results: List[Dict]) -> Dict:
        """Aggregate results across all samples."""
        total_samples = len(sample_results)

        # Average number recall (DQR)
        num_recalls = [r['num_recall'] for r in sample_results]
        avg_num_recall = sum(num_recalls) / total_samples if total_samples > 0 else 0

        # Aggregate type-level metrics (TF1)
        type_aggregated = {}
        for type_name in self.type_names:
            total_gt = sum(r['type_level'][type_name]['gt_count'] for r in sample_results)
            total_pred = sum(r['type_level'][type_name]['pred_count'] for r in sample_results)
            total_matched = sum(r['type_level'][type_name]['matched'] for r in sample_results)

            precision = total_matched / total_pred if total_pred > 0 else None
            recall = total_matched / total_gt if total_gt > 0 else None

            if precision is not None and recall is not None and (precision + recall) > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0

            type_aggregated[type_name] = {
                'total_gt': total_gt,
                'total_pred': total_pred,
                'total_matched': total_matched,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }

        # Overall type metrics
        total_type_gt = sum(type_aggregated[t]['total_gt'] for t in self.type_names)
        total_type_pred = sum(type_aggregated[t]['total_pred'] for t in self.type_names)
        total_type_matched = sum(type_aggregated[t]['total_matched'] for t in self.type_names)

        overall_precision = total_type_matched / total_type_pred if total_type_pred > 0 else None
        overall_recall = total_type_matched / total_type_gt if total_type_gt > 0 else None

        if overall_precision is not None and overall_recall is not None and (overall_precision + overall_recall) > 0:
            overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        else:
            overall_f1 = 0.0

        type_aggregated['overall'] = {
            'total_gt': total_type_gt,
            'total_pred': total_type_pred,
            'total_matched': total_type_matched,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1
        }

        # Aggregate category-level metrics (CF1)
        total_cat_gt = sum(r['category_level']['gt_count'] for r in sample_results)
        total_cat_pred = sum(r['category_level']['pred_count'] for r in sample_results)
        total_cat_matched = sum(r['category_level']['matched'] for r in sample_results)

        cat_precision = total_cat_matched / total_cat_pred if total_cat_pred > 0 else None
        cat_recall = total_cat_matched / total_cat_gt if total_cat_gt > 0 else None

        if cat_precision is not None and cat_recall is not None and (cat_precision + cat_recall) > 0:
            cat_f1 = 2 * cat_precision * cat_recall / (cat_precision + cat_recall)
        else:
            cat_f1 = 0.0

        category_aggregated = {
            'total_gt': total_cat_gt,
            'total_pred': total_cat_pred,
            'total_matched': total_cat_matched,
            'precision': cat_precision,
            'recall': cat_recall,
            'f1': cat_f1
        }

        return {
            'total_samples': total_samples,
            'avg_num_recall': avg_num_recall,
            'type_level': type_aggregated,
            'category_level': category_aggregated
        }

    def evaluate_model(self, response_file: str) -> Tuple[Dict, List[Dict]]:
        """Evaluate model from response file."""
        print(f"\n{'='*60}")
        print(f"Evaluating: {Path(response_file).name}")
        print(f"{'='*60}\n")

        with open(response_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        model_name = data.get('model', 'unknown')
        responses = data.get('responses', [])

        print(f"Model: {model_name}")
        print(f"Samples: {len(responses)}\n")

        sample_results = []
        for resp in responses:
            gt = resp.get('ground_truth', {})
            parsed = resp.get('parsed_response', {})

            gt_modifications = gt.get('modifications', [])
            pred_structured = parsed.get('structured_output', [])

            if not parsed.get('has_structured_output', False):
                continue

            metrics = self.calculate_sample_metrics(gt_modifications, pred_structured)
            metrics['sample_id'] = resp.get('sample_id', 'unknown')
            sample_results.append(metrics)

        print(f"Valid samples: {len(sample_results)}\n")

        aggregated = self.aggregate_results(sample_results)
        aggregated['model'] = model_name

        return aggregated, sample_results

    def print_summary(self, aggregated: Dict):
        """Print evaluation summary."""
        print(f"\n{'='*60}")
        print(f"Evaluation Summary - {aggregated['model']}")
        print(f"{'='*60}\n")

        print(f"Total samples: {aggregated['total_samples']}")
        print(f"DQR (Avg Num Recall): {aggregated['avg_num_recall']:.4f}\n")

        print(f"{'─'*60}")
        print(f"Type-Level Metrics (TF1)")
        print(f"{'─'*60}")
        print(f"{'Type':<12} {'GT':<8} {'Pred':<8} {'Match':<8} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print(f"{'─'*60}")

        for type_name in self.type_names:
            m = aggregated['type_level'][type_name]
            precision = f"{m['precision']:.4f}" if m['precision'] is not None else "N/A"
            recall = f"{m['recall']:.4f}" if m['recall'] is not None else "N/A"
            print(f"{type_name:<12} {m['total_gt']:<8} {m['total_pred']:<8} "
                  f"{m['total_matched']:<8} {precision:<12} {recall:<12} {m['f1']:.4f}")

        print(f"{'─'*60}")
        o = aggregated['type_level']['overall']
        o_precision = f"{o['precision']:.4f}" if o['precision'] is not None else "N/A"
        o_recall = f"{o['recall']:.4f}" if o['recall'] is not None else "N/A"
        print(f"{'OVERALL':<12} {o['total_gt']:<8} {o['total_pred']:<8} "
              f"{o['total_matched']:<8} {o_precision:<12} {o_recall:<12} {o['f1']:.4f}")

        print(f"\n{'─'*60}")
        print(f"Category-Level Metrics (CF1)")
        print(f"{'─'*60}")
        cat = aggregated['category_level']
        cat_p = f"{cat['precision']:.4f}" if cat['precision'] is not None else "N/A"
        cat_r = f"{cat['recall']:.4f}" if cat['recall'] is not None else "N/A"
        print(f"GT: {cat['total_gt']}, Pred: {cat['total_pred']}, Matched: {cat['total_matched']}")
        print(f"Precision: {cat_p}, Recall: {cat_r}, F1: {cat['f1']:.4f}\n")

    def save_results(self, aggregated: Dict, sample_results: List[Dict], output_file: str):
        """Save results to JSON file."""
        output_data = {
            'summary': aggregated,
            'per_sample_results': sample_results
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"Results saved: {output_file}\n")


def main():
    parser = argparse.ArgumentParser(description='Multi-difference detection metrics evaluator')
    parser.add_argument('--input_dir', type=str, default='./eval_results_difference',
                        help='Input directory containing response JSON files')
    parser.add_argument('--output_dir', type=str, default='./eval_results_difference_metrics',
                        help='Output directory for metrics')
    parser.add_argument('--models', type=str, nargs='+',
                        help='Specific model files to evaluate')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    evaluator = MultiDiffEvaluator()

    input_dir = Path(args.input_dir)
    if args.models:
        response_files = [input_dir / model_file for model_file in args.models]
    else:
        response_files = list(input_dir.glob('*_responses.json'))

    if not response_files:
        print(f"No response files found in: {input_dir}")
        return

    print(f"\nFound {len(response_files)} response files")

    all_summaries = []
    for response_file in response_files:
        if not response_file.exists():
            print(f"File not found: {response_file}")
            continue

        try:
            aggregated, sample_results = evaluator.evaluate_model(str(response_file))
            evaluator.print_summary(aggregated)

            output_file = Path(args.output_dir) / f"{response_file.stem}_metrics.json"
            evaluator.save_results(aggregated, sample_results, str(output_file))
            all_summaries.append(aggregated)

        except Exception as e:
            print(f"Error processing: {response_file.name}")
            print(f"   Error: {e}\n")
            continue

    if all_summaries:
        comparison_file = Path(args.output_dir) / 'all_models_comparison.json'
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)
        print(f"{'='*60}")
        print(f"All models comparison saved: {comparison_file}")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
