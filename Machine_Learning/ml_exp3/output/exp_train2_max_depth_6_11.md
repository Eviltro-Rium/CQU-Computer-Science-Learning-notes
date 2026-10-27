# California 房价 — CART 回归：max_depth 对比（验证集）

| max_depth | MAE | RMSE | R² |
|----------:|----:|-----:|---:|
| 6 | 49267.73 | 69354.65 | 0.6340 |
| 7 | 46557.21 | 66215.51 | 0.6663 |
| 8 | 42687.19 | 62850.68 | 0.6994 |
| 9 | 41378.45 | 61804.37 | **0.7093** |
| 10 | 41186.53 | 62387.36 | 0.7038 |
| 11 | **40711.57** | 62972.13 | 0.6982 |

- 单位：MAE / RMSE 与标签一致（美元量级）。  
- 本组结果中：**R² 最高为 depth=9**；**MAE 最低为 depth=11**（RMSE 在 depth=9 最低）。

---

## LaTeX（booktabs 三线表，可直接贴论文）

```latex
\begin{table}[htbp]
  \centering
  \caption{California housing: validation metrics vs.\ \texttt{max\_depth} (CART regressor)}
  \begin{tabular}{@{}rrrr@{}}
    \toprule
    max\_depth & MAE & RMSE & $R^2$ \\
    \midrule
    6  & 49267.73 & 69354.65 & 0.6340 \\
    7  & 46557.21 & 66215.51 & 0.6663 \\
    8  & 42687.19 & 62850.68 & 0.6994 \\
    9  & 41378.45 & 61804.37 & 0.7093 \\
    10 & 41186.53 & 62387.36 & 0.7038 \\
    11 & 40711.57 & 62972.13 & 0.6982 \\
    \bottomrule
  \end{tabular}
\end{table}
```

（需在导言区加入 `\usepackage{booktabs}`。）
