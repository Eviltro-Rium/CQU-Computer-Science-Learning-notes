# Boston 房价 — CART 回归：max_depth 对比（验证集）

| max_depth | MAE | RMSE | R² |
|----------:|----:|-----:|---:|
| 4 | 2.9712 | 3.8160 | 0.8478 |
| 5 | 2.5916 | **3.3747** | **0.8810** |
| 6 | 2.5555 | 3.4691 | 0.8742 |
| 7 | **2.5095** | 3.4156 | 0.8781 |
| 8 | 2.5909 | 3.5079 | 0.8714 |

- 标签 **MEDV** 单位为**千美元**；MAE / RMSE 与之一致。  
- 本组中：**RMSE、R² 最优均为 depth=5**；**MAE 最优为 depth=7**。

---

## LaTeX（booktabs 三线表）

```latex
\begin{table}[htbp]
  \centering
  \caption{Boston housing: validation metrics vs.\ \texttt{max\_depth} (CART regressor)}
  \begin{tabular}{@{}rrrr@{}}
    \toprule
    max\_depth & MAE & RMSE & $R^2$ \\
    \midrule
    4 & 2.9712 & 3.8160 & 0.8478 \\
    5 & 2.5916 & 3.3747 & 0.8810 \\
    6 & 2.5555 & 3.4691 & 0.8742 \\
    7 & 2.5095 & 3.4156 & 0.8781 \\
    8 & 2.5909 & 3.5079 & 0.8714 \\
    \bottomrule
  \end{tabular}
\end{table}
```

（需 `\usepackage{booktabs}`。）
