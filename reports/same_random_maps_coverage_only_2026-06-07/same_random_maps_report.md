# Same-random-maps coverage-only ablation

- Scenario: Same random maps 20x20 / 4 agents / 5%
- Seeds: 20261001-20261020 (20 maps)
- Evaluation mode: coverage-only. Return policy is not used.

## Summary

| Arm | Ep. | AUC | T90 | T95 | C@100 | C@200 | C@300 | Repeat | Repeat90 | Final cov. | Cov done | Steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GAT-OFF | 20 | 0.8332 | 182.5 | 212.2 | 70.9% | 91.4% | 96.5% | 72.5% | 94.2% | 99.8% | 70.0% | 377.1 |
| GAT-ON | 20 | 0.8646 | 131.4 | 135.1 | 79.2% | 96.6% | 98.8% | 46.8% | 82.7% | 99.7% | 95.0% | 191.7 |
| GAT-CUAP | 20 | 0.8619 | 133.8 | 137.0 | 78.0% | 96.6% | 98.5% | 46.3% | 81.9% | 99.6% | 95.0% | 192.9 |

## Per-seed Wins

- AUC wins: GAT-OFF: 0, GAT-ON: 13, GAT-CUAP: 7
- Coverage@100 wins: GAT-OFF: 2, GAT-ON: 11, GAT-CUAP: 7
- Lowest RepeatRatio wins: GAT-OFF: 1, GAT-ON: 11, GAT-CUAP: 8
