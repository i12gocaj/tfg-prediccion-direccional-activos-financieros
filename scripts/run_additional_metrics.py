# Calcula precisión direccional y test Diebold-Mariano
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.evaluation.validator import TimeSeriesValidator

# Estilo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def load_experiment_results():
    path = "results/experiment_a_results.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run main.py first.")
    return pd.read_csv(path)

def load_error_results():
    path = "results/experiment_a_errors.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run main.py first.")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    return df

def build_directional_results(results_df, models=None):
    if models is None:
        models = ["Naive", "ARIMA", "RandomForest", "XGBoost", "CNN", "LSTM", "GRU"]
    df = results_df[results_df["Model"].isin(models)].copy()
    df["Model"] = pd.Categorical(df["Model"], categories=models, ordered=True)
    return df

def calculate_diebold_mariano(errors_df, horizon=7):
    """Calcula test de Diebold-Mariano para Naive vs cada modelo."""
    validator = TimeSeriesValidator()
    dm_results = []

    tickers = sorted(errors_df["Ticker"].unique())
    models = sorted(errors_df["Model"].unique())

    for ticker in tickers:
        naive = errors_df[(errors_df["Ticker"] == ticker) & (errors_df["Model"] == "Naive")]
        naive = naive[["Date", "Error"]].rename(columns={"Error": "Error_naive"})
        if naive.empty:
            continue

        for model in models:
            if model == "Naive":
                continue
            model_df = errors_df[(errors_df["Ticker"] == ticker) & (errors_df["Model"] == model)]
            model_df = model_df[["Date", "Error"]].rename(columns={"Error": "Error_model"})
            if model_df.empty:
                continue
            merged = naive.merge(model_df, on="Date", how="inner")
            if merged.empty:
                continue

            dm = validator.diebold_mariano_test(
                merged["Error_model"].to_numpy(),
                merged["Error_naive"].to_numpy(),
                horizon=horizon
            )
            dm["Ticker"] = ticker
            dm["Model"] = model
            dm_results.append(dm)

    return pd.DataFrame(dm_results)

def generate_plots(results_df, output_dir='results/plots'):
    """Genera gráficas para Precisión Direccional."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Agregar por Modelo y Ticker
    agg = results_df.groupby(['Ticker', 'Model']).agg({
        'MAPE': 'mean',
        'DirectionalAccuracy': 'mean'
    }).reset_index()
    
    # Gráfica 1: Precisión Direccional por Modelo (barras agrupadas)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    model_order = ["Naive", "ARIMA", "RandomForest", "XGBoost", "CNN", "LSTM", "GRU"]
    models = [m for m in model_order if m in agg['Model'].unique()]
    tickers = ["BTC-USD", "TSLA", "NVDA"]
    x = np.arange(len(tickers))

    total_width = 0.8
    width = total_width / max(len(models), 1)

    palette = plt.get_cmap('tab10').colors
    colors = {model: palette[i % len(palette)] for i, model in enumerate(models)}
    
    for i, model in enumerate(models):
        model_data = agg[agg['Model'] == model]
        da_values = [model_data[model_data['Ticker'] == t]['DirectionalAccuracy'].values[0] 
                     if len(model_data[model_data['Ticker'] == t]) > 0 else 0 
                     for t in tickers]
        positions = x - (total_width / 2) + (i + 0.5) * width
        bars = ax.bar(positions, da_values, width, label=model, color=colors.get(model, '#3498db'))
        
        # Añadir etiquetas de valor
        for bar, val in zip(bars, da_values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, label='Azar (50%)')
    ax.set_xlabel('Activo')
    ax.set_ylabel('Precisión Direccional (%)')
    ax.set_title('Precisión Direccional por Modelo y Activo')
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.legend()
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/directional_accuracy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Gráfica 2: MAPE vs Precisión Direccional scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model in models:
        model_data = agg[agg['Model'] == model]
        ax.scatter(model_data['MAPE'], model_data['DirectionalAccuracy'], 
                  label=model, s=150, color=colors.get(model, '#3498db'), alpha=0.8)
        
        # Añadir etiquetas de ticker
        for _, row in model_data.iterrows():
            ax.annotate(row['Ticker'], (row['MAPE'], row['DirectionalAccuracy']),
                       xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('MAPE (%)')
    ax.set_ylabel('Precisión Direccional (%)')
    ax.set_title('MAPE vs Precisión Direccional')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/mape_vs_directional_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Gráficas guardadas en {output_dir}/")
    
    return agg

def main():
    print("="*60)
    print("CALCULATING DIRECTIONAL ACCURACY AND DIEBOLD-MARIANO TEST")
    print("="*60)
    
    # Cargar resultados ya generados
    experiment_df = load_experiment_results()
    errors_df = load_error_results()

    # Construir dataset direccional sin reentrenar
    results_df = build_directional_results(experiment_df)
    
    # Guardar
    results_df.to_csv('results/directional_accuracy_results.csv', index=False)
    print("\nResults saved to results/directional_accuracy_results.csv")
    
    # Summary table
    print("\n" + "="*60)
    print("DIRECTIONAL ACCURACY SUMMARY")
    print("="*60)
    summary_cols = ['MAPE', 'DirectionalAccuracy', 'DirectionalF1', 'DirectionalMCC']
    summary = results_df.groupby(['Ticker', 'Model'])[summary_cols].mean().round(2)
    print(summary.to_string())
    
    # Test Diebold-Mariano
    print("\n" + "="*60)
    print("DIEBOLD-MARIANO TEST (Naive vs all models)")
    print("="*60)
    dm_results = calculate_diebold_mariano(errors_df)
    print(dm_results.to_string(index=False))
    dm_results.to_csv('results/diebold_mariano_results.csv', index=False)
    
    # Generar gráficas
    agg_results = generate_plots(results_df)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED")
    print("="*60)
    
    return results_df, dm_results

if __name__ == "__main__":
    main()
