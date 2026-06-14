# Seed-focused three-model ablation

## Selected Seed

- CUAP best seed from prior detail rows: `20260603`
- Source scenario: `unseen_20x20_4a_r05` / Unseen 20x20 / 4 agents / 5% obstacles
- CUAP source Coverage-AUC: `0.8932`
- Course-4 training obstacle seeds: `20260440, 20260441, 20260442, 20260443, 20260444, 20260445, 20260446, 20260447`

## Focused Summary

| Scenario | Arm | Ep. | AUC | T90 | T95 | C@100 | C@200 | C@300 | Repeat | Repeat90 | Final cov. | Cov done | Mission done |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CUAP best prior-eval seed 20260603 (Unseen 20x20 / 4 agents / 5% obstacles) | GAT-OFF | 1 | 0.8691 | 139 (100.0%) | 161 (100.0%) | 74.2% | 98.7% | 100.0% | 81.0% | 97.4% | 100.0% | 100.0% | 0.0% |
| CUAP best prior-eval seed 20260603 (Unseen 20x20 / 4 agents / 5% obstacles) | GAT-ON | 1 | 0.8779 | 130 (100.0%) | 169 (100.0%) | 81.8% | 99.7% | 100.0% | 58.5% | 90.3% | 100.0% | 100.0% | 100.0% |
| CUAP best prior-eval seed 20260603 (Unseen 20x20 / 4 agents / 5% obstacles) | GAT-CUAP | 1 | 0.8932 | 99 (100.0%) | 109 (100.0%) | 90.8% | 100.0% | 100.0% | 46.3% | 88.0% | 100.0% | 100.0% | 100.0% |
| Course-4 native config seed 20260431 | GAT-OFF | 1 | 0.8736 | 120 (100.0%) | 130 (100.0%) | 76.8% | 100.0% | 100.0% | 81.0% | 97.5% | 100.0% | 100.0% | 0.0% |
| Course-4 native config seed 20260431 | GAT-ON | 1 | 0.8945 | 97 (100.0%) | 106 (100.0%) | 92.4% | 100.0% | 100.0% | 36.2% | 81.4% | 100.0% | 100.0% | 100.0% |
| Course-4 native config seed 20260431 | GAT-CUAP | 1 | 0.8918 | 110 (100.0%) | 116 (100.0%) | 86.3% | 100.0% | 100.0% | 43.1% | 83.5% | 100.0% | 100.0% | 100.0% |
| Course-4 training obstacle seeds 20260440-20260447 | GAT-OFF | 8 | 0.8350 | 197 (100.0%) | 236 (100.0%) | 71.2% | 90.6% | 97.5% | 78.0% | 95.7% | 99.9% | 87.5% | 50.0% |
| Course-4 training obstacle seeds 20260440-20260447 | GAT-ON | 8 | 0.8873 | 112 (100.0%) | 124 (100.0%) | 85.2% | 100.0% | 100.0% | 51.6% | 87.7% | 100.0% | 100.0% | 87.5% |
| Course-4 training obstacle seeds 20260440-20260447 | GAT-CUAP | 8 | 0.8834 | 117 (100.0%) | 128 (100.0%) | 82.5% | 100.0% | 100.0% | 54.6% | 88.7% | 100.0% | 87.5% | 87.5% |

## Per-seed Focused Rows

| Scenario | Arm | Seed | AUC | T90 | T95 | C@100 | C@200 | C@300 | Repeat | Repeat90 | Final cov. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CUAP best prior-eval seed 20260603 (Unseen 20x20 / 4 agents / 5% obstacles) | GAT-OFF | 20260603 | 0.8691 | 139 | 161 | 74.2% | 98.7% | 100.0% | 81.0% | 97.4% | 100.0% |
| Course-4 native config seed 20260431 | GAT-OFF | 20260431 | 0.8736 | 120 | 130 | 76.8% | 100.0% | 100.0% | 81.0% | 97.5% | 100.0% |
| Course-4 training obstacle seeds 20260440-20260447 | GAT-OFF | 20260440 | 0.8736 | 120 | 130 | 76.8% | 100.0% | 100.0% | 81.0% | 97.5% | 100.0% |
| Course-4 training obstacle seeds 20260440-20260447 | GAT-OFF | 20260441 | 0.8323 | 176 | 224 | 63.7% | 93.2% | 100.0% | 72.9% | 94.5% | 100.0% |
| Course-4 training obstacle seeds 20260440-20260447 | GAT-OFF | 20260442 | 0.8420 | 166 | 269 | 74.7% | 93.2% | 98.9% | 81.2% | 97.5% | 98.9% |
| Course-4 training obstacle seeds 20260440-20260447 | GAT-OFF | 20260443 | 0.8802 | 113 | 130 | 82.6% | 98.7% | 100.0% | 72.6% | 95.9% | 100.0% |
| Course-4 training obstacle seeds 20260440-20260447 | GAT-OFF | 20260444 | 0.7712 | 288 | 309 | 62.1% | 70.5% | 92.4% | 77.4% | 92.8% | 100.0% |
| Course-4 training obstacle seeds 20260440-20260447 | GAT-OFF | 20260445 | 0.7988 | 318 | 391 | 70.0% | 87.1% | 89.5% | 81.0% | 94.8% | 100.0% |
| Course-4 training obstacle seeds 20260440-20260447 | GAT-OFF | 20260446 | 0.8477 | 176 | 195 | 70.3% | 95.0% | 99.5% | 76.3% | 95.8% | 100.0% |
| Course-4 training obstacle seeds 20260440-20260447 | GAT-OFF | 20260447 | 0.8346 | 219 | 244 | 69.2% | 87.4% | 100.0% | 81.0% | 96.6% | 100.0% |
| CUAP best prior-eval seed 20260603 (Unseen 20x20 / 4 agents / 5% obstacles) | GAT-ON | 20260603 | 0.8779 | 130 | 169 | 81.8% | 99.7% | 100.0% | 58.5% | 90.3% | 100.0% |
| Course-4 native config seed 20260431 | GAT-ON | 20260431 | 0.8945 | 97 | 106 | 92.4% | 100.0% | 100.0% | 36.2% | 81.4% | 100.0% |
| Course-4 training obstacle seeds 20260440-20260447 | GAT-ON | 20260440 | 0.8945 | 97 | 106 | 92.4% | 100.0% | 100.0% | 36.2% | 81.4% | 100.0% |
| Course-4 training obstacle seeds 20260440-20260447 | GAT-ON | 20260441 | 0.8720 | 147 | 152 | 74.5% | 100.0% | 100.0% | 54.3% | 85.4% | 100.0% |
| Course-4 training obstacle seeds 20260440-20260447 | GAT-ON | 20260442 | 0.8910 | 105 | 114 | 86.6% | 100.0% | 100.0% | 49.2% | 88.6% | 100.0% |
| Course-4 training obstacle seeds 20260440-20260447 | GAT-ON | 20260443 | 0.8885 | 108 | 129 | 87.4% | 100.0% | 100.0% | 49.7% | 88.1% | 100.0% |
| Course-4 training obstacle seeds 20260440-20260447 | GAT-ON | 20260444 | 0.8908 | 103 | 121 | 88.2% | 100.0% | 100.0% | 48.4% | 88.1% | 100.0% |
| Course-4 training obstacle seeds 20260440-20260447 | GAT-ON | 20260445 | 0.8847 | 112 | 123 | 82.4% | 100.0% | 100.0% | 81.0% | 97.6% | 100.0% |
| Course-4 training obstacle seeds 20260440-20260447 | GAT-ON | 20260446 | 0.8928 | 109 | 121 | 88.7% | 100.0% | 100.0% | 46.9% | 86.2% | 100.0% |
| Course-4 training obstacle seeds 20260440-20260447 | GAT-ON | 20260447 | 0.8838 | 113 | 122 | 81.6% | 100.0% | 100.0% | 46.9% | 85.8% | 100.0% |
| CUAP best prior-eval seed 20260603 (Unseen 20x20 / 4 agents / 5% obstacles) | GAT-CUAP | 20260603 | 0.8932 | 99 | 109 | 90.8% | 100.0% | 100.0% | 46.3% | 88.0% | 100.0% |
| Course-4 native config seed 20260431 | GAT-CUAP | 20260431 | 0.8918 | 110 | 116 | 86.3% | 100.0% | 100.0% | 43.1% | 83.5% | 100.0% |
| Course-4 training obstacle seeds 20260440-20260447 | GAT-CUAP | 20260440 | 0.8918 | 110 | 116 | 86.3% | 100.0% | 100.0% | 43.1% | 83.5% | 100.0% |
| Course-4 training obstacle seeds 20260440-20260447 | GAT-CUAP | 20260441 | 0.8750 | 127 | 144 | 83.4% | 100.0% | 100.0% | 57.4% | 90.0% | 100.0% |
| Course-4 training obstacle seeds 20260440-20260447 | GAT-CUAP | 20260442 | 0.8857 | 116 | 127 | 82.4% | 99.7% | 99.7% | 81.1% | 97.7% | 99.7% |
| Course-4 training obstacle seeds 20260440-20260447 | GAT-CUAP | 20260443 | 0.8865 | 112 | 123 | 85.0% | 100.0% | 100.0% | 51.3% | 89.0% | 100.0% |
| Course-4 training obstacle seeds 20260440-20260447 | GAT-CUAP | 20260444 | 0.8826 | 116 | 129 | 81.8% | 100.0% | 100.0% | 46.0% | 83.9% | 100.0% |
| Course-4 training obstacle seeds 20260440-20260447 | GAT-CUAP | 20260445 | 0.8802 | 125 | 138 | 79.7% | 100.0% | 100.0% | 56.4% | 89.7% | 100.0% |
| Course-4 training obstacle seeds 20260440-20260447 | GAT-CUAP | 20260446 | 0.8834 | 116 | 124 | 81.8% | 100.0% | 100.0% | 50.8% | 87.8% | 100.0% |
| Course-4 training obstacle seeds 20260440-20260447 | GAT-CUAP | 20260447 | 0.8819 | 117 | 123 | 79.5% | 100.0% | 100.0% | 50.8% | 87.7% | 100.0% |
