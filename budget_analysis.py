"""Analyze the controlled seed-11 training-budget comparison after training."""
import json
from pathlib import Path
from experiment import analyze_checked, compare_equivalence
from plot import INCLUDED_RUN, compare_learning_curves

root = Path(__file__).resolve().parent
reference = root / 'runs' / INCLUDED_RUN
longer = root / 'runs' / 'budget_seed11_40000_20261102'
output = root / 'comparisons' / longer.name
output.mkdir(exist_ok=True)
a, b = analyze_checked(reference), analyze_checked(longer)
report = {
    'reference': a, 'longer': b,
    'common_prefix': compare_equivalence(reference, longer, prefix=True),
    'mean_return_difference': b['mean_evaluation_return'] - a['mean_evaluation_return'],
    'angle_mae_difference_deg': b['mean_angle_mae_last_2s_deg'] - a['mean_angle_mae_last_2s_deg'],
}
(output / 'budget_comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
compare_learning_curves([reference, longer], output, filename='budget_learning_curves.png')
print('Budget comparison:', output)
