# Gráficas de resumen para las conclusiones
import os

from src.utils.reproducibility import set_global_seed

SEED = 42
set_global_seed(SEED)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

RESULTS_DIR = "results/plots"

def plot_summary_best_models():
    # Mejor modelo por activo
    print("Generating: Summary - Best Models per Asset")
    
    df = pd.read_csv("results/experiment_a_results.csv")
    
    # Mejor modelo para cada activo
    summary = df.groupby(['Ticker', 'Model'])[['RMSE', 'MAE', 'MAPE']].mean()
    best_idx = summary.groupby('Ticker')['MAPE'].idxmin()
    best_models = summary.loc[best_idx].reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'BTC-USD': '#f39c12', 'TSLA': '#e74c3c', 'NVDA': '#27ae60'}
    
    bars = ax.bar(best_models['Ticker'], best_models['MAPE'], 
                  color=[colors[t] for t in best_models['Ticker']])
    
    ax.set_title('Resumen: Mejor Modelo por Activo', fontsize=14, fontweight='bold', pad=16)
    ax.set_xlabel('Activo')
    ax.set_ylabel('MAPE (%)')
    y_max = best_models['MAPE'].max()
    ax.set_ylim(0, y_max * 1.2)
    
    # Nombres de modelos en las barras
    for i, (bar, row) in enumerate(zip(bars, best_models.itertuples())):
        height = bar.get_height()
        ax.annotate(f'{row.Model}\n{height:.2f}%', 
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 5), textcoords="offset points", 
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/summary_best_models.png", dpi=150)
    plt.close()
    print(f"  Saved: summary_best_models.png")

def plot_summary_dl_vs_baselines():
    # Comparar DL vs baselines
    print("Generating: Summary - DL vs Baselines")
    
    df = pd.read_csv("results/experiment_a_results.csv")
    
    # Categorías
    baselines = ['Naive', 'ARIMA']
    ml_models = ['RandomForest', 'XGBoost']
    dl_models = ['CNN', 'LSTM', 'GRU']
    
    summary = df.groupby(['Ticker', 'Model'])['MAPE'].mean().unstack()
    
    # Promedios
    results = []
    for ticker in summary.index:
        results.append({
            'Ticker': ticker,
            'Baselines': summary.loc[ticker, baselines].mean(),
            'ML Clásico': summary.loc[ticker, ml_models].mean(),
            'Deep Learning': summary.loc[ticker, dl_models].mean()
        })
    
    result_df = pd.DataFrame(results)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(result_df))
    width = 0.25
    
    bars1 = ax.bar(x - width, result_df['Baselines'], width, label='Baselines', color='#95a5a6')
    bars2 = ax.bar(x, result_df['ML Clásico'], width, label='ML Clásico', color='#e74c3c')
    bars3 = ax.bar(x + width, result_df['Deep Learning'], width, label='Deep Learning', color='#3498db')
    
    ax.set_title('Resumen: Comparación por Familia de Modelos', fontsize=14, fontweight='bold')
    ax.set_xlabel('Activo')
    ax.set_ylabel('MAPE Promedio (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(result_df['Ticker'])
    ax.legend()
    
    # Añadir etiquetas de valor
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/summary_dl_vs_baselines.png", dpi=150)
    plt.close()
    print(f"  Saved: summary_dl_vs_baselines.png")

def plot_summary_sentiment_impact():
    # Resumen: Visualización del impacto general del sentimiento.
    print("Generating: Summary - Sentiment Impact")
    
    df = pd.read_csv("results/experiment_c_results.csv")
    
    # Calcular impacto global por ticker
    with_sent = df[df['Sentiment'] == 'With'].groupby('Ticker')['MAPE'].mean()
    without_sent = df[df['Sentiment'] == 'Without'].groupby('Ticker')['MAPE'].mean()
    
    # Mejora relativa al incluir sentimiento (positivo = mejora).
    # Se calcula respecto al escenario "sin sentimiento" para alinear con tablas/heatmap.
    improvement = ((without_sent - with_sent) / without_sent * 100).round(1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Izquierda: Comparación absoluta
    ax1 = axes[0]
    x = np.arange(len(with_sent))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, with_sent, width, label='Con Sentimiento', color='#2ecc71')
    bars2 = ax1.bar(x + width/2, without_sent, width, label='Sin Sentimiento', color='#e74c3c')
    
    ax1.set_title('Resumen: MAPE Con vs Sin Sentimiento', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Activo')
    ax1.set_ylabel('MAPE (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(with_sent.index)
    ax1.legend()
    
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    
    # Derecha: Porcentaje de mejora
    ax2 = axes[1]
    colors = ['#27ae60' if v > 0 else '#c0392b' for v in improvement]
    bars = ax2.bar(improvement.index, improvement, color=colors)
    
    ax2.set_title('Resumen: Mejora (%) al Incluir Sentimiento', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Activo')
    ax2.set_ylabel('Mejora (%)')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3 if height > 0 else -15), textcoords="offset points", 
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/summary_sentiment_impact.png", dpi=150)
    plt.close()
    print(f"  Saved: summary_sentiment_impact.png")

def plot_summary_all_experiments():
    # Resumen: Hallazgos clave de todos los experimentos en una figura.
    print("Generating: Summary - All Experiments Overview")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Mejor modelo por activo
    ax1 = axes[0, 0]
    df_a = pd.read_csv("results/experiment_a_results.csv")
    summary_a = df_a.groupby(['Ticker', 'Model'])['MAPE'].mean()
    best_idx = summary_a.groupby('Ticker').idxmin()
    best_models = summary_a.loc[best_idx].reset_index()
    
    colors = {'BTC-USD': '#f39c12', 'TSLA': '#e74c3c', 'NVDA': '#27ae60'}
    bars = ax1.bar(best_models['Ticker'], best_models['MAPE'], 
                   color=[colors[t] for t in best_models['Ticker']])
    ax1.set_title('A) Mejor Modelo por Activo', fontweight='bold')
    ax1.set_ylabel('MAPE (%)')
    for i, (bar, row) in enumerate(zip(bars, best_models.itertuples())):
        height = bar.get_height()
        ax1.annotate(f'{row.Model}\n{height:.2f}%', 
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Panel 2: Impacto del horizonte
    ax2 = axes[0, 1]
    df_b = pd.read_csv("results/experiment_b_results.csv")
    for model in ['GRU', 'LSTM', 'CNN']:
        model_df = df_b[df_b['Model'] == model]
        summary = model_df.groupby('Horizon')['MAPE'].mean()
        ax2.plot(summary.index, summary.values, marker='o', label=model, linewidth=2)
    ax2.set_title('B) Impacto del Horizonte (BTC-USD)', fontweight='bold')
    ax2.set_xlabel('Horizonte (días)')
    ax2.set_ylabel('MAPE (%)')
    ax2.set_xticks([1, 7, 30])
    ax2.legend()
    
    # Panel 3: Ablación de sentimiento
    ax3 = axes[1, 0]
    df_c = pd.read_csv("results/experiment_c_results.csv")
    gru_df = df_c[df_c['Model'] == 'GRU']
    with_sent = gru_df[gru_df['Sentiment'] == 'With'].groupby('Ticker')['MAPE'].mean()
    without_sent = gru_df[gru_df['Sentiment'] == 'Without'].groupby('Ticker')['MAPE'].mean()
    
    x = np.arange(len(with_sent))
    width = 0.35
    ax3.bar(x - width/2, with_sent, width, label='Con Sent.', color='#2ecc71')
    ax3.bar(x + width/2, without_sent, width, label='Sin Sent.', color='#e74c3c')
    ax3.set_title('C) Ablación de Sentimiento (GRU)', fontweight='bold')
    ax3.set_ylabel('MAPE (%)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(with_sent.index)
    ax3.legend()
    
    # Panel 4: Periodos de mercado
    ax4 = axes[1, 1]
    df_d = pd.read_csv("results/experiment_d_results.csv")
    if len(df_d) > 0:
        period_order = ['Bull-2021', 'Bear-2022', 'Recent']
        model_order_d = ['LSTM', 'GRU', 'CNN']
        colors_models = {'LSTM': '#e74c3c', 'GRU': '#3498db', 'CNN': '#2ecc71'}
        periods_in = [p for p in period_order if p in df_d['Period'].unique()]
        x = np.arange(len(periods_in))
        total_width = 0.7
        n_models = len(model_order_d)
        bar_w = total_width / max(n_models, 1)
        for i, model in enumerate(model_order_d):
            mdf = df_d[df_d['Model'] == model]
            vals = [mdf[mdf['Period'] == p]['MAPE'].values[0] if len(mdf[mdf['Period'] == p]) > 0 else 0 for p in periods_in]
            pos = x - total_width / 2 + (i + 0.5) * bar_w
            ax4.bar(pos, vals, bar_w, label=model, color=colors_models.get(model, '#95a5a6'))
        ax4.set_title('D) Rendimiento por Periodo (BTC-USD)', fontweight='bold')
        ax4.set_ylabel('MAPE (%)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(periods_in)
        ax4.legend(fontsize=9)
    
    plt.suptitle('Resumen de Experimentos: Hallazgos Principales', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/summary_all_experiments.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: summary_all_experiments.png")

def plot_summary_ranking():
    """Resumen: Ranking de modelos en todos los activos."""
    print("Generating: Summary - Model Ranking")
    
    df = pd.read_csv("results/experiment_a_results.csv")
    
    # Calcular promedio global por modelo
    overall = df.groupby('Model')['MAPE'].mean().sort_values()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(overall)))
    
    bars = ax.barh(overall.index, overall.values, color=colors)
    
    ax.set_title('Ranking de Modelos: MAPE Promedio (Todos los Activos)', fontsize=14, fontweight='bold')
    ax.set_xlabel('MAPE (%)')
    ax.invert_yaxis()  # Mejor arriba
    
    for bar in bars:
        width = bar.get_width()
        ax.annotate(f'{width:.2f}%', xy=(width, bar.get_y() + bar.get_height()/2),
                   xytext=(5, 0), textcoords="offset points", 
                   ha='left', va='center', fontsize=11, fontweight='bold')
    
    # Añadir números de ranking
    for i, (model, value) in enumerate(overall.items()):
        ax.annotate(f'#{i+1}', xy=(0.1, i),
                   ha='left', va='center', fontsize=12, fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/summary_model_ranking.png", dpi=150)
    plt.close()
    print(f"  Saved: summary_model_ranking.png")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GENERATING SUMMARY COMPARISON PLOTS")
    print("=" * 60)
    
    plot_summary_best_models()
    plot_summary_dl_vs_baselines()
    plot_summary_sentiment_impact()
    plot_summary_all_experiments()
    plot_summary_ranking()
    
    print("\n" + "=" * 60)
    print("COMPLETADO: 5 gráficas de resumen generadas")
    print("=" * 60)
