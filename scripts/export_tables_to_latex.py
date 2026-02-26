import os
import pandas as pd
import numpy as np


TABLE_DIR = "memoria/Tablas"
os.makedirs(TABLE_DIR, exist_ok=True)


def fmt_num(value, decimals=2):
    if pd.isna(value):
        return "--"
    return f"{value:.{decimals}f}"


def write_tabular(path, header, rows, align_spec):
    lines = []
    lines.append(f"\\begin{{tabular}}{{{align_spec}}}")
    lines.append("\\hline")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\hline")
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def export_experiment_a():
    path = "results/experiment_a_results.csv"
    if not os.path.exists(path):
        print(f"Missing {path}")
        return

    df = pd.read_csv(path).sort_values("Ticker")
    summary = df.groupby(["Model", "Ticker"])["MAPE"].mean().unstack()
    tickers = ["BTC-USD", "TSLA", "NVDA"]
    models = ["Naive", "ARIMA", "RandomForest", "XGBoost", "CNN", "LSTM", "GRU"]
    summary = summary.reindex(index=models, columns=tickers)
    summary["Media"] = summary.mean(axis=1)

    col_mins = summary.min(axis=0)

    display_names = {
        "RandomForest": "Random Forest"
    }

    rows = []
    for model in models:
        row = [display_names.get(model, model)]
        for col in tickers + ["Media"]:
            val = summary.loc[model, col]
            formatted = fmt_num(val)
            if np.isfinite(val) and np.isclose(val, col_mins[col]):
                formatted = f"\\textbf{{{formatted}}}"
            row.append(formatted)
        rows.append(row)

    header = [
        "\\textbf{Modelo}",
        "\\textbf{BTC-USD}",
        "\\textbf{TSLA}",
        "\\textbf{NVDA}",
        "\\textbf{Media}"
    ]
    write_tabular(
        os.path.join(TABLE_DIR, "experiment_a_mape.tex"),
        header,
        rows,
        "l|ccc|c"
    )


def export_experiment_b():
    path = "results/experiment_b_results.csv"
    if not os.path.exists(path):
        print(f"Missing {path}")
        return

    df = pd.read_csv(path)
    summary = df.groupby(["Model", "Horizon"])["MAPE"].mean().unstack()
    horizons = [1, 7, 30]
    models = ["CNN", "GRU", "LSTM"]
    summary = summary.reindex(index=models, columns=horizons)

    col_mins = summary.min(axis=0)
    rows = []
    for model in models:
        row = [model]
        for h in horizons:
            val = summary.loc[model, h]
            formatted = fmt_num(val)
            if np.isfinite(val) and np.isclose(val, col_mins[h]):
                formatted = f"\\textbf{{{formatted}}}"
            row.append(formatted)
        rows.append(row)

    header = [
        "\\textbf{Modelo}",
        "\\textbf{1 dia}",
        "\\textbf{7 dias}",
        "\\textbf{30 dias}"
    ]
    write_tabular(
        os.path.join(TABLE_DIR, "experiment_b_mape.tex"),
        header,
        rows,
        "l|ccc"
    )


def export_experiment_c():
    path = "results/experiment_c_results.csv"
    if not os.path.exists(path):
        print(f"Missing {path}")
        return

    df = pd.read_csv(path)
    summary = df.groupby(["Ticker", "Model", "Sentiment"])["MAPE"].mean().unstack()
    summary = summary.reset_index().sort_values(["Ticker", "Model"])

    rows = []
    for _, row_df in summary.iterrows():
        ticker = row_df["Ticker"]
        model = row_df["Model"]
        with_val = row_df.get("With", np.nan)
        without_val = row_df.get("Without", np.nan)

        delta_pct = np.nan
        helps = "--"
        if np.isfinite(with_val) and np.isfinite(without_val) and without_val != 0:
            # Mejora relativa al incluir sentimiento (positivo = mejora)
            delta_pct = (without_val - with_val) / without_val * 100
            helps = "Si" if delta_pct > 0 else "No"

        if np.isfinite(delta_pct):
            sign = f"{delta_pct:.1f}\\%"
            if helps == "Si":
                helps_fmt = f"\\textcolor{{green!60!black}}{{Si ({sign})}}"
            else:
                helps_fmt = f"\\textcolor{{red}}{{No ({sign})}}"
        else:
            helps_fmt = "--"

        rows.append([
            ticker,
            model,
            fmt_num(with_val),
            fmt_num(without_val),
            helps_fmt
        ])

    header = [
        "\\textbf{Activo}",
        "\\textbf{Modelo}",
        "\\textbf{Con Sent.}",
        "\\textbf{Sin Sent.}",
        "\\textbf{Ayuda?}"
    ]
    write_tabular(
        os.path.join(TABLE_DIR, "experiment_c_ablation.tex"),
        header,
        rows,
        "l|l|cc|c"
    )


def export_experiment_d():
    path = "results/experiment_d_results.csv"
    if not os.path.exists(path):
        print(f"Missing {path}")
        return

    df = pd.read_csv(path)
    period_order = ["Bull-2021", "Bear-2022", "Recent"]
    model_order = ["LSTM", "GRU", "CNN"]

    rows = []
    for period in period_order:
        pdf = df[df["Period"] == period]
        if pdf.empty:
            continue
        for model in model_order:
            mdf = pdf[pdf["Model"] == model]
            if mdf.empty:
                rows.append([period, model, "--", "--", "--"])
                continue
            rows.append([
                period,
                model,
                fmt_num(mdf["RMSE"].values[0], decimals=0),
                fmt_num(mdf["MAE"].values[0], decimals=0),
                fmt_num(mdf["MAPE"].values[0])
            ])

    header = [
        "\\textbf{Periodo}",
        "\\textbf{Modelo}",
        "\\textbf{RMSE (USD)}",
        "\\textbf{MAE (USD)}",
        "\\textbf{MAPE (\\%)}"
    ]
    write_tabular(
        os.path.join(TABLE_DIR, "experiment_d_periods.tex"),
        header,
        rows,
        "l|l|ccc"
    )


def export_directional_metrics():
    path = "results/directional_accuracy_results.csv"
    if not os.path.exists(path):
        print(f"Missing {path}")
        return

    df = pd.read_csv(path)
    cols = ["DirectionalAccuracy", "DirectionalF1", "DirectionalMCC"]
    summary = df.groupby(["Model", "Ticker"])[cols].mean().reset_index()
    tickers = ["BTC-USD", "TSLA", "NVDA"]
    models = ["Naive", "ARIMA", "RandomForest", "XGBoost", "CNN", "LSTM", "GRU"]

    formatted_rows = []
    for model in models:
        for metric, label in zip(cols, ["DA (\\%)", "F1 dir. (\\%)", "MCC"]):
            vals = []
            for ticker in tickers:
                val = summary[(summary["Model"] == model) & (summary["Ticker"] == ticker)][metric]
                vals.append(val.values[0] if len(val) > 0 else np.nan)
            avg = np.nanmean(vals) if len(vals) > 0 else np.nan
            formatted_rows.append([
                f"{model} - {label}",
                fmt_num(vals[0]),
                fmt_num(vals[1]),
                fmt_num(vals[2]),
                fmt_num(avg)
            ])

    header = [
        "\\textbf{Modelo/Metrica}",
        "\\textbf{BTC-USD}",
        "\\textbf{TSLA}",
        "\\textbf{NVDA}",
        "\\textbf{Media}"
    ]
    write_tabular(
        os.path.join(TABLE_DIR, "directional_metrics.tex"),
        header,
        formatted_rows,
        "l|ccc|c"
    )


def export_dm_test():
    path = "results/diebold_mariano_results.csv"
    if not os.path.exists(path):
        print(f"Missing {path}")
        return

    df = pd.read_csv(path)
    rows = []
    for _, row in df.iterrows():
        rows.append([
            row.get("Ticker", "--"),
            row.get("Model", "--"),
            fmt_num(row.get("dm_statistic"), decimals=2),
            fmt_num(row.get("p_value"), decimals=3),
            "Si" if row.get("significant") else "No"
        ])

    header = [
        "\\textbf{Activo}",
        "\\textbf{Modelo}",
        "\\textbf{Estadistico DM}",
        "\\textbf{p-value}",
        "\\textbf{Significativo?}"
    ]
    write_tabular(
        os.path.join(TABLE_DIR, "diebold_mariano.tex"),
        header,
        rows,
        "l|l|ccc"
    )


def export_ablation_tests():
    path = "results/experiment_c_ablation_tests.csv"
    if not os.path.exists(path):
        print(f"Missing {path}")
        return

    df = pd.read_csv(path)
    metric_order = ["MAPE", "DirectionalAccuracy", "DirectionalF1", "DirectionalMCC"]
    df = df[df["Metric"].isin(metric_order)].copy()
    df["Metric"] = pd.Categorical(df["Metric"], categories=metric_order, ordered=True)
    df = df.sort_values(["Metric", "Ticker", "Model"])

    metric_labels = {
        "MAPE": "MAPE",
        "DirectionalAccuracy": "DA",
        "DirectionalF1": "F1",
        "DirectionalMCC": "MCC",
    }

    rows = []
    for _, row in df.iterrows():
        metric = row.get("Metric", "--")
        rows.append([
            row.get("Ticker", "--"),
            row.get("Model", "--"),
            metric_labels.get(metric, metric),
            fmt_num(row.get("MeanDiff")),
            fmt_num(row.get("p_value"), decimals=3),
            "Si" if row.get("significant") else "No"
        ])

    header = [
        "\\textbf{Activo}",
        "\\textbf{Modelo}",
        "\\textbf{Metr.}",
        "\\textbf{Dif.}",
        "\\textbf{p}",
        "\\textbf{Sig.?}"
    ]
    write_tabular(
        os.path.join(TABLE_DIR, "ablation_tests.tex"),
        header,
        rows,
        "l|l|c|c|c|c"
    )


def main():
    export_experiment_a()
    export_experiment_b()
    export_experiment_c()
    export_experiment_d()
    export_directional_metrics()
    export_dm_test()
    export_ablation_tests()
    print(f"Tables exported to {TABLE_DIR}")


if __name__ == "__main__":
    main()
