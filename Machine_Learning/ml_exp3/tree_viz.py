"""决策树可视化：Unicode 文本树、Graphviz DOT、matplotlib PNG。"""
from pathlib import Path
from typing import Any, List, Optional, Union


def _feat_name(j: int, feature_names: Optional[List[str]]) -> str:
    if feature_names is not None and 0 <= j < len(feature_names):
        return str(feature_names[j])
    return f"x[{j}]"


def _fmt_leaf_class(v: Any, id2str: Optional[dict]) -> str:
    if id2str is not None:
        try:
            return str(id2str.get(int(v), v))
        except (TypeError, ValueError):
            return str(v)
    return str(v)


def tree_lines(
    node: dict,
    *,
    feature_names: Optional[List[str]] = None,
    id2str: Optional[dict] = None,
    regression: bool = False,
) -> List[str]:
    """
    生成更易读的 Unicode 树形行（├ └ │ ─）。
    regression=True 时叶子为数值预测；否则为分类 id / id2str。
    """

    def lines_rec(n: dict, prefix: str, is_last: bool) -> List[str]:
        branch = "└── " if is_last else "├── "
        ext = "    " if is_last else "│   "
        if n["type"] == "leaf":
            if regression:
                text = f"预测 = {float(n['value']):.6g}"
            else:
                text = f"class = {_fmt_leaf_class(n['value'], id2str)}"
            return [prefix + branch + text]

        j, t = n["feature"], n["threshold"]
        cond = f"{_feat_name(j, feature_names)} ≤ {t:.6g} ?"
        out = [prefix + branch + cond]
        child_p = prefix + ext
        out.extend(lines_rec(n["left"], child_p, is_last=False))
        out.extend(lines_rec(n["right"], child_p, is_last=True))
        return out

    if node["type"] == "leaf":
        if regression:
            t = f"预测 = {float(node['value']):.6g}"
        else:
            t = f"class = {_fmt_leaf_class(node['value'], id2str)}"
        return [t]
    j, t = node["feature"], node["threshold"]
    root_line = f"{_feat_name(j, feature_names)} ≤ {t:.6g} ?"
    out = [root_line]
    out.extend(lines_rec(node["left"], "", is_last=False))
    out.extend(lines_rec(node["right"], "", is_last=True))
    return out


def print_tree_pretty(
    node: dict,
    *,
    feature_names: Optional[List[str]] = None,
    id2str: Optional[dict] = None,
    regression: bool = False,
    title: str = "Decision tree",
) -> None:
    print(title)
    print("\n".join(tree_lines(node, feature_names=feature_names, id2str=id2str, regression=regression)))
    print()


def tree_to_dot(
    node: dict,
    *,
    feature_names: Optional[List[str]] = None,
    id2str: Optional[dict] = None,
    regression: bool = False,
    graph_name: str = "Tree",
) -> str:
    """生成 Graphviz DOT 字符串（可保存为 .dot 后执行 dot -Tpng tree.dot -o tree.png）。"""
    lines: List[str] = [
        f'digraph "{graph_name}" {{',
        '  node [fontname="Helvetica", fontsize=10];',
    ]
    nid = [0]

    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    def rec(n: dict) -> int:
        i = nid[0]
        nid[0] += 1
        if n["type"] == "leaf":
            if regression:
                lab = esc(f"pred\\n{float(n['value']):.6g}")
            else:
                lab = esc(f"class\\n{_fmt_leaf_class(n['value'], id2str)}")
            lines.append(f'  n{i} [shape=box, style=filled, fillcolor="#E8F4E8", label="{lab}"];')
            return i
        j, t = n["feature"], n["threshold"]
        lab = esc(f"{_feat_name(j, feature_names)} ≤ {t:.6g}?")
        lines.append(f'  n{i} [shape=box, style=filled, fillcolor="#E8EEF8", label="{lab}"];')
        li = rec(n["left"])
        ri = rec(n["right"])
        lines.append(f'  n{i} -> n{li} [label="yes"];')
        lines.append(f'  n{i} -> n{ri} [label="no"];')
        return i

    rec(node)
    lines.append("}")
    return "\n".join(lines)


def resolve_dot_path(dot_path: Union[Path, str], png_path: Optional[Union[Path, str]]) -> Path:
    """
    若 DOT 与 PNG 指向同一文件，DOT 会覆盖 PNG。此时自动改用 .dot 后缀并打印说明。
    """
    d = Path(dot_path) if not isinstance(dot_path, Path) else dot_path
    if png_path:
        p = Path(png_path) if not isinstance(png_path, Path) else png_path
        try:
            if d.resolve() == p.resolve():
                alt = p.with_suffix(".dot")
                print(
                    f"Note: TREE_DOT_PATH 与 TREE_PNG_PATH 相同，已避免覆盖 PNG；"
                    f"DOT 改为写入: {alt}"
                )
                return alt
        except OSError:
            pass
    return d


def save_tree_dot(node: dict, path: Union[Path, str], **kwargs) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(tree_to_dot(node, **kwargs), encoding="utf-8")


def save_tree_png_matplotlib(
    node: dict,
    path: Union[Path, str],
    *,
    feature_names: Optional[List[str]] = None,
    id2str: Optional[dict] = None,
    regression: bool = False,
    title: str = "Decision tree",
    fig_dpi: int = 120,
) -> bool:
    """用 matplotlib 画树并保存 PNG。未安装 matplotlib 时返回 False。"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib import patches
    except ImportError:
        return False

    pos: dict[int, tuple[float, float, dict]] = {}

    def layout(n: dict, xmin: float, xmax: float, y: float) -> float:
        if n["type"] == "leaf":
            x = (xmin + xmax) / 2.0
            pos[id(n)] = (x, y, n)
            return x
        mid = (xmin + xmax) / 2.0
        xl = layout(n["left"], xmin, mid, y + 1.0)
        xr = layout(n["right"], mid, xmax, y + 1.0)
        x = (xl + xr) / 2.0
        pos[id(n)] = (x, y, n)
        return x

    layout(node, 0.0, 1.0, 0.0)
    max_y = max(p[1] for p in pos.values()) if pos else 0.0
    n_nodes = len(pos)
    fig_w = max(10.0, min(40.0, 1.2 * n_nodes))
    fig_h = max(4.0, min(30.0, (max_y + 1.5) * 0.65))

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(max_y + 0.6, -0.5)

    # 边
    children_map: dict[int, tuple[int, int]] = {}

    def collect_edges(n: dict) -> None:
        if n["type"] == "internal":
            i = id(n)
            li, ri = id(n["left"]), id(n["right"])
            children_map[i] = (li, ri)
            collect_edges(n["left"])
            collect_edges(n["right"])

    collect_edges(node)

    fs = max(5, min(9, 120 // max(n_nodes, 10)))

    for pid, (lid, rid) in children_map.items():
        xp, yp, _ = pos[pid]
        xl, yl, _ = pos[lid]
        xr, yr, _ = pos[rid]
        ax.plot([xp, xl], [yp, yl], color="#555", linewidth=1.0, zorder=1)
        ax.plot([xp, xr], [yp, yr], color="#555", linewidth=1.0, zorder=1)
        ax.text((xp + xl) / 2, (yp + yl) / 2 - 0.08, "yes", fontsize=7, ha="center", color="#2a6a2a")
        ax.text((xp + xr) / 2, (yp + yr) / 2 - 0.08, "no", fontsize=7, ha="center", color="#6a2a2a")

    for _, (x, y, n) in pos.items():
        if n["type"] == "leaf":
            if regression:
                lab = f"{float(n['value']):.4g}"
            else:
                lab = _fmt_leaf_class(n["value"], id2str)
            color = "#d8f0d8"
        else:
            j, t = n["feature"], n["threshold"]
            fn = _feat_name(j, feature_names)
            lab = f"{fn}\n<= {t:.4g}?"
            color = "#d8e4f8"
        ax.add_patch(
            patches.FancyBboxPatch(
                (x - 0.045, y - 0.18),
                0.09,
                0.32,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                linewidth=0.8,
                edgecolor="#333",
                facecolor=color,
                zorder=2,
            )
        )
        ax.text(x, y, lab, ha="center", va="center", fontsize=fs, zorder=3)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        fig.savefig(tmp, format="png", bbox_inches="tight", facecolor="white", dpi=fig_dpi)
    finally:
        plt.close(fig)
    tmp.replace(p)
    return True
