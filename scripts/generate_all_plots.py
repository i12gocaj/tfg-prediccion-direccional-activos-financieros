# Script para generar todas las gráficas del TFG
# Ejecutar después de run_all_experiments.py
import os

from src.utils.reproducibility import set_global_seed

SEED = 42
set_global_seed(SEED)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.models.dl_models import DLModels
from src.models.baselines import BaselineModels
from src.models.ml_models import MLModels
from src.evaluation.validator import TimeSeriesValidator

# Estilo de las gráficas
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

RESULTS_DIR = "results/plots"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_data(ticker):
    path = f"data/processed/{ticker.replace('-', '_')}_processed.csv"
    return pd.read_csv(path, index_col=0, parse_dates=True)

# Parte 1: pred vs real

def plot_prediction_vs_actual_comprehensive():
    # Genera pred vs real para cada activo y horizonte
    print("\n" + "=" * 60)
    print("GENERATING: Prediction vs Actual (All Combinations)")
    print("=" * 60)
    
    tickers = ['BTC-USD', 'TSLA', 'NVDA']
    horizons = [1, 7, 30]
    models = ['gru', 'lstm', 'cnn']
    
    for ticker in tickers:
        print(f"\n--- {ticker} ---")
        df = load_data(ticker)
        
        for horizon in horizons:
            print(f"  Horizon: {horizon} days")
            
            # Preparar datos
            model_wrapper = DLModels()
            window_size = 30
            
            train_size = int(len(df) * 0.85)
            train_df = df.iloc[:train_size]
            test_df = df.iloc[train_size:]
            
            try:
                train_X, train_y = model_wrapper.prepare_data(train_df, window_size=window_size, horizon=horizon)
                feature_cols = model_wrapper.feature_columns_
                
                test_input_df = pd.concat([train_df.iloc[-window_size:], test_df])
                scaler = model_wrapper.scaler
                target_scaler = model_wrapper.target_scaler
                
                test_data = test_input_df[feature_cols].copy().dropna()
                scaled_test = scaler.transform(test_data)
                
                X_test, y_test = [], []
                target_idx = test_data.columns.get_loc('Close')
                
                for i in range(window_size, len(scaled_test) - horizon + 1):
                    X_test.append(scaled_test[i-window_size:i])
                    y_test.append(scaled_test[i + horizon - 1, target_idx])
                
                if len(X_test) == 0:
                    print(f"    Skipping (no test samples)")
                    continue
                    
                X_test, y_test = np.array(X_test), np.array(y_test)
                y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1))
                target_dates = [
                    test_data.index[i + horizon - 1]
                    for i in range(window_size, len(scaled_test) - horizon + 1)
                ]
                
                # Gráfica con los 3 modelos juntos
                fig, ax = plt.subplots(figsize=(14, 7))
                ax.plot(target_dates, y_test_inv, label='Precio Real', color='black', linewidth=2)
                
                colors = {'gru': '#3498db', 'lstm': '#e74c3c', 'cnn': '#2ecc71'}
                
                for model_type in models:
                    pred = model_wrapper.train_predict(train_X, train_y, X_test, model_type=model_type, epochs=10)
                    ax.plot(target_dates, pred, label=f'Predicción {model_type.upper()}', 
                           color=colors[model_type], linestyle='--', linewidth=1.5, alpha=0.8)
                
                ax.set_title(f'{ticker}: Predicción vs Precio Real (Horizonte {horizon} días)', fontweight='bold')
                ax.set_xlabel('Fecha de test')
                ax.set_ylabel('Precio (USD)')
                locator = mdates.AutoDateLocator()
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
                fig.autofmt_xdate()
                ax.legend(loc='upper left')
                
                filename = f"{RESULTS_DIR}/pred_vs_actual_{ticker.replace('-', '_')}_h{horizon}.png"
                plt.tight_layout()
                plt.savefig(filename, dpi=150)
                plt.close()
                print(f"    Saved: {filename}")
                
            except Exception as e:
                print(f"    Error: {e}")

# Parte 2: comparativa de modelos

def plot_model_comparison_per_asset():
    # Barras comparando modelos para cada activo
    print("\n" + "=" * 60)
    print("GENERATING: Model Comparison per Asset")
    print("=" * 60)
    
    df = pd.read_csv("results/experiment_a_results.csv")
    tickers = df['Ticker'].unique()
    
    model_order = ['Naive', 'ARIMA', 'RandomForest', 'XGBoost', 'CNN', 'LSTM', 'GRU']
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(model_order)))
    
    for ticker in tickers:
        ticker_df = df[df['Ticker'] == ticker]
        summary = ticker_df.groupby('Model')[['RMSE', 'MAE', 'MAPE']].mean()
        summary = summary.reindex([m for m in model_order if m in summary.index])
        
        # MAPE Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(summary.index, summary['MAPE'], color=colors[:len(summary)])
        ax.set_title(f'{ticker}: Comparación de Modelos (MAPE %)', fontweight='bold')
        ax.set_xlabel('Modelo')
        ax.set_ylabel('MAPE (%)')
        ax.set_xticks(np.arange(len(summary.index)))
        ax.set_xticklabels(summary.index, rotation=45, ha='right')
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        filename = f"{RESULTS_DIR}/model_comparison_{ticker.replace('-', '_')}_mape.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Saved: {filename}")
        
        # RMSE Chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(summary.index, summary['RMSE'], color=colors[:len(summary)])
        ax.set_title(f'{ticker}: Comparación de Modelos (RMSE)', fontweight='bold')
        ax.set_xlabel('Modelo')
        ax.set_ylabel('RMSE (USD)')
        ax.set_xticks(np.arange(len(summary.index)))
        ax.set_xticklabels(summary.index, rotation=45, ha='right')
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        filename = f"{RESULTS_DIR}/model_comparison_{ticker.replace('-', '_')}_rmse.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Saved: {filename}")

def plot_model_comparison_mape_dispersion():
    # MAPE medio con dispersión por fold
    print("\n" + "=" * 60)
    print("GENERATING: MAPE Dispersion per Asset")
    print("=" * 60)

    df = pd.read_csv("results/experiment_a_results.csv")
    model_order = ['Naive', 'ARIMA', 'RandomForest', 'XGBoost', 'CNN', 'LSTM', 'GRU']
    tickers = [t for t in ['BTC-USD', 'TSLA', 'NVDA'] if t in df['Ticker'].unique()]
    if not tickers:
        print("No data for dispersion plot")
        return

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(model_order)))
    fig, axes = plt.subplots(1, len(tickers), figsize=(5 * len(tickers), 8), sharey=False)
    if len(tickers) == 1:
        axes = [axes]

    for ax, ticker in zip(axes, tickers):
        ticker_df = df[df['Ticker'] == ticker]
        summary = ticker_df.groupby('Model')['MAPE'].agg(['mean', 'std'])
        labels = [m for m in model_order if m in summary.index]
        summary = summary.reindex(labels)

        x = np.arange(len(labels))
        ax.bar(
            x,
            summary['mean'].values,
            yerr=summary['std'].values,
            capsize=4,
            color=colors[:len(labels)]
        )
        ax.set_title(f'{ticker}: MAPE medio +/- desv.', fontweight='bold')
        ax.set_xlabel('Modelo')
        ax.set_ylabel('MAPE (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.2)

    plt.tight_layout()
    filename = f"{RESULTS_DIR}/model_comparison_mape_dispersion.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

# Parte 3: impacto del horizonte

def plot_horizon_impact_per_model():
    # Cómo se degrada cada modelo al predecir más lejos
    print("\n" + "=" * 60)
    print("GENERATING: Horizon Impact per Model")
    print("=" * 60)
    
    df = pd.read_csv("results/experiment_b_results.csv")
    models = df['Model'].unique()
    
    colors = {'CNN': '#2ecc71', 'GRU': '#3498db', 'LSTM': '#e74c3c'}
    
    # Combined chart
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in models:
        model_df = df[df['Model'] == model]
        summary = model_df.groupby('Horizon')['MAPE'].mean()
        ax.plot(summary.index, summary.values, marker='o', label=model, 
               linewidth=2.5, markersize=10, color=colors.get(model, 'gray'))
    
    ax.set_title('BTC-USD: Impacto del Horizonte de Predicción por Modelo', fontweight='bold')
    ax.set_xlabel('Horizonte (días)')
    ax.set_ylabel('MAPE (%)')
    ax.set_xticks([1, 7, 30])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"{RESULTS_DIR}/horizon_impact_all_models.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")
    
    # Gráficas individuales por modelo
    for model in models:
        model_df = df[df['Model'] == model]
        summary = model_df.groupby('Horizon')[['RMSE', 'MAE', 'MAPE']].mean()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 8))
        
        for idx, (metric, ax) in enumerate(zip(['RMSE', 'MAE', 'MAPE'], axes)):
            ax.bar(summary.index.astype(str), summary[metric], color=colors.get(model, 'gray'))
            ax.set_title(f'{model}: {metric} por Horizonte', fontweight='bold')
            ax.set_xlabel('Horizonte (días)')
            ax.set_ylabel(metric)
            
            for i, v in enumerate(summary[metric]):
                ax.text(i, v + v*0.02, f'{v:.2f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        filename = f"{RESULTS_DIR}/horizon_impact_{model.lower()}_metrics.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Saved: {filename}")

# Parte 4: ablación (con/sin sentimiento)

def plot_ablation_comprehensive():
    # Comparativa con y sin sentimiento
    print("\n" + "=" * 60)
    print("GENERATING: Ablation Study (Comprehensive)")
    print("=" * 60)
    
    df = pd.read_csv("results/experiment_c_results.csv")
    tickers = df['Ticker'].unique()
    models = df['Model'].unique()
    
    # Por activo
    for ticker in tickers:
        ticker_df = df[df['Ticker'] == ticker]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(models))
        width = 0.35
        
        with_sent = ticker_df[ticker_df['Sentiment'] == 'With'].groupby('Model')['MAPE'].mean()
        without_sent = ticker_df[ticker_df['Sentiment'] == 'Without'].groupby('Model')['MAPE'].mean()
        
        # Reorder
        with_sent = with_sent.reindex(models)
        without_sent = without_sent.reindex(models)
        
        bars1 = ax.bar(x - width/2, with_sent, width, label='Con Sentimiento', color='#2ecc71')
        bars2 = ax.bar(x + width/2, without_sent, width, label='Sin Sentimiento', color='#e74c3c')
        
        ax.set_title(f'{ticker}: Estudio de Ablación por Modelo', fontweight='bold')
        ax.set_xlabel('Modelo')
        ax.set_ylabel('MAPE (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        
        # Etiquetas
        for bar in bars1:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        filename = f"{RESULTS_DIR}/ablation_{ticker.replace('-', '_')}_all_models.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Saved: {filename}")
    
    # Heatmap - ojo que valores negativos = empeora con sentimiento
    pivot_with = df[df['Sentiment'] == 'With'].groupby(['Ticker', 'Model'])['MAPE'].mean().unstack()
    pivot_without = df[df['Sentiment'] == 'Without'].groupby(['Ticker', 'Model'])['MAPE'].mean().unstack()
    improvement = ((pivot_without - pivot_with) / pivot_without * 100).round(1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(improvement.values, cmap='RdYlGn', aspect='auto')
    
    ax.set_xticks(np.arange(len(improvement.columns)))
    ax.set_yticks(np.arange(len(improvement.index)))
    ax.set_xticklabels(improvement.columns)
    ax.set_yticklabels(improvement.index)
    
    for i in range(len(improvement.index)):
        for j in range(len(improvement.columns)):
            val = improvement.values[i, j]
            if not np.isnan(val):
                text = ax.text(j, i, f'{val:.1f}%', ha='center', va='center', 
                              color='black' if abs(val) < 20 else 'white', fontsize=11, fontweight='bold')
    
    ax.set_title('Reducción del Error (%) al Incluir Sentimiento\n(Verde = Mejora, Rojo = Empeora)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Reducción (%)')
    
    plt.tight_layout()
    filename = f"{RESULTS_DIR}/ablation_improvement_heatmap.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")

def plot_ablation_by_model_overview():
    # Comparativa por modelo en una sola figura
    print("\n" + "=" * 60)
    print("GENERATING: Ablation by Model Overview")
    print("=" * 60)

    df = pd.read_csv("results/experiment_c_results.csv")
    models = [m for m in ['CNN', 'LSTM', 'GRU'] if m in df['Model'].unique()]
    tickers = [t for t in ['BTC-USD', 'TSLA', 'NVDA'] if t in df['Ticker'].unique()]
    if not models or not tickers:
        print("No data for ablation overview")
        return

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 8), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        model_df = df[df['Model'] == model]
        with_sent = model_df[model_df['Sentiment'] == 'With'].groupby('Ticker')['MAPE'].mean()
        without_sent = model_df[model_df['Sentiment'] == 'Without'].groupby('Ticker')['MAPE'].mean()

        with_sent = with_sent.reindex(tickers)
        without_sent = without_sent.reindex(tickers)

        x = np.arange(len(tickers))
        width = 0.35
        ax.bar(x - width / 2, with_sent.values, width, label='Con Sentimiento', color='#2ecc71')
        ax.bar(x + width / 2, without_sent.values, width, label='Sin Sentimiento', color='#e74c3c')
        ax.set_title(f'{model}: Ablación por Activo', fontweight='bold')
        ax.set_xlabel('Activo')
        ax.set_xticks(x)
        ax.set_xticklabels(tickers)
        ax.grid(axis='y', alpha=0.2)

    axes[0].set_ylabel('MAPE (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.05))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename = f"{RESULTS_DIR}/ablation_models_overview.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

# Parte 5: periodos de mercado

def plot_market_periods_extended():
    # Cómo varía el error en bull, bear, etc
    print("\n" + "=" * 60)
    print("GENERATING: Market Period Analysis (Extended)")
    print("=" * 60)
    
    df = pd.read_csv("results/experiment_d_results.csv")
    
    if len(df) == 0:
        print("No market period data available")
        return
    
    period_order = ['Bull-2021', 'Bear-2022', 'Recent']
    model_order = ['LSTM', 'GRU', 'CNN']
    model_colors = {'LSTM': '#e74c3c', 'GRU': '#3498db', 'CNN': '#2ecc71'}
    periods_in = [p for p in period_order if p in df['Period'].unique()]
    models_in = [m for m in model_order if m in df['Model'].unique()]

    # Gráfica de barras agrupadas por periodo y modelo
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    
    for idx, (metric, ax) in enumerate(zip(['RMSE', 'MAE', 'MAPE'], axes)):
        x = np.arange(len(periods_in))
        total_width = 0.7
        n_models = max(len(models_in), 1)
        bar_w = total_width / n_models

        for i, model in enumerate(models_in):
            mdf = df[df['Model'] == model]
            vals = [mdf[mdf['Period'] == p][metric].values[0] if len(mdf[mdf['Period'] == p]) > 0 else 0 for p in periods_in]
            pos = x - total_width / 2 + (i + 0.5) * bar_w
            bars = ax.bar(pos, vals, bar_w, label=model, color=model_colors.get(model, '#95a5a6'))

            for bar, val in zip(bars, vals):
                if val > 0:
                    label = f'{val:.1f}%' if metric == 'MAPE' else f'{val:.0f}'
                    ax.annotate(label, xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                               xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8)

        ax.set_title(f'BTC-USD: {metric} por Periodo', fontweight='bold')
        ax.set_xlabel('Periodo')
        ax.set_ylabel(metric if metric != 'MAPE' else f'{metric} (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(periods_in, rotation=15, ha='right')
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    filename = f"{RESULTS_DIR}/market_periods_all_metrics.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")

# Parte 6: comparar entre activos

def plot_cross_asset_comparison():
    # Heatmap y barras comparando BTC, TSLA, NVDA
    print("\n" + "=" * 60)
    print("GENERATING: Cross-Asset Comparison")
    print("=" * 60)
    
    df = pd.read_csv("results/experiment_a_results.csv")
    
    # Heatmap de todos los modelos x activos
    pivot = df.groupby(['Ticker', 'Model'])['MAPE'].mean().unstack()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto')
    
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax.set_yticklabels(pivot.index)
    
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text = ax.text(j, i, f'{val:.2f}%', ha='center', va='center', 
                              color='white' if val > 4 else 'black', fontsize=10, fontweight='bold')
    
    ax.set_title('MAPE (%) por Modelo y Activo\n(Colores más claros = mejor rendimiento)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='MAPE (%)')
    
    plt.tight_layout()
    filename = f"{RESULTS_DIR}/cross_asset_heatmap.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")
    
    # Comparación agrupada
    fig, ax = plt.subplots(figsize=(14, 7))
    pivot.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title('Comparación de Modelos entre Activos (MAPE %)', fontweight='bold')
    ax.set_xlabel('Activo')
    ax.set_ylabel('MAPE (%)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title='Modelo', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    filename = f"{RESULTS_DIR}/cross_asset_grouped_bar.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")

# Parte 7: boxplots

def plot_error_distributions():
    # Boxplots para ver variabilidad
    print("\n" + "=" * 60)
    print("GENERATING: Error Distribution Analysis")
    print("=" * 60)
    
    df = pd.read_csv("results/experiment_a_results.csv")
    
    # Boxplot general
    fig, ax = plt.subplots(figsize=(12, 6))
    models = ['Naive', 'ARIMA', 'RandomForest', 'XGBoost', 'CNN', 'LSTM', 'GRU']
    data = [df[df['Model'] == m]['MAPE'].values for m in models if m in df['Model'].unique()]
    valid_models = [m for m in models if m in df['Model'].unique()]
    
    try:
        bp = ax.boxplot(data, tick_labels=valid_models, patch_artist=True)
    except TypeError:
        bp = ax.boxplot(data, labels=valid_models, patch_artist=True)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(valid_models)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_title('Distribución del Error (MAPE) por Modelo', fontweight='bold')
    ax.set_xlabel('Modelo')
    ax.set_ylabel('MAPE (%)')
    
    plt.tight_layout()
    filename = f"{RESULTS_DIR}/error_distribution_boxplot.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")
    
def plot_significance_heatmaps():
    # Heatmaps de p-values para DM y Wilcoxon
    print("\n" + "=" * 60)
    print("GENERATING: Significance Heatmaps")
    print("=" * 60)

    dm_path = "results/diebold_mariano_results.csv"
    if os.path.exists(dm_path):
        df = pd.read_csv(dm_path)
        model_order = ['ARIMA', 'RandomForest', 'XGBoost', 'CNN', 'LSTM', 'GRU']
        tickers = [t for t in ['BTC-USD', 'TSLA', 'NVDA'] if t in df['Ticker'].unique()]
        pivot = df.pivot(index='Ticker', columns='Model', values='p_value')
        pivot = pivot.reindex(index=tickers, columns=[m for m in model_order if m in pivot.columns])

        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(pivot.values, cmap='YlOrRd_r', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
        ax.set_yticklabels(pivot.index)

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    color = 'white' if val < 0.2 else 'black'
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center', color=color, fontsize=9)

        ax.set_title('Diebold-Mariano: p-values (Naive vs modelo)', fontweight='bold')
        fig.colorbar(im, ax=ax, label='p-value')
        plt.tight_layout()
        filename = f"{RESULTS_DIR}/diebold_mariano_pvalues_heatmap.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
    else:
        print("Missing diebold_mariano_results.csv")

    wilcoxon_path = "results/experiment_c_ablation_tests.csv"
    if os.path.exists(wilcoxon_path):
        df = pd.read_csv(wilcoxon_path)
        metrics = ['MAPE', 'DirectionalAccuracy']
        model_order = ['CNN', 'LSTM', 'GRU']
        tickers = [t for t in ['BTC-USD', 'TSLA', 'NVDA'] if t in df['Ticker'].unique()]

        fig, axes = plt.subplots(1, len(metrics), figsize=(12, 4), sharey=True, layout="constrained")
        if len(metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            mdf = df[df['Metric'] == metric]
            pivot = mdf.pivot(index='Ticker', columns='Model', values='p_value')
            pivot = pivot.reindex(index=tickers, columns=[m for m in model_order if m in pivot.columns])

            im = ax.imshow(pivot.values, cmap='YlOrRd_r', aspect='auto', vmin=0, vmax=1)
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
            ax.set_yticklabels(pivot.index)
            ax.set_title(f'Wilcoxon: {metric}', fontweight='bold')

            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if np.isfinite(val):
                        color = 'white' if val < 0.2 else 'black'
                        ax.text(j, i, f'{val:.3f}', ha='center', va='center', color=color, fontsize=9)

        fig.colorbar(im, ax=axes, location="right", label='p-value')
        filename = f"{RESULTS_DIR}/wilcoxon_pvalues_heatmap.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
    else:
        print("Missing experiment_c_ablation_tests.csv")



if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("# COMPREHENSIVE VISUALIZATION GENERATOR")
    print("#" * 70)
    
    # Generar todas las gráficas
    plot_model_comparison_per_asset()
    plot_model_comparison_mape_dispersion()
    plot_horizon_impact_per_model()
    plot_ablation_comprehensive()
    plot_ablation_by_model_overview()
    plot_significance_heatmaps()
    plot_market_periods_extended()
    plot_cross_asset_comparison()
    plot_error_distributions()
    plot_prediction_vs_actual_comprehensive()  # Esta tarda más
    
    # Contar archivos generados
    files = os.listdir(RESULTS_DIR)
    png_files = [f for f in files if f.endswith('.png')]
    
    print("\n" + "#" * 70)
    print(f"# COMPLETADO: Generadas {len(png_files)} gráficas")
    print(f"# Ubicación: {RESULTS_DIR}/")
    print("#" * 70)
    
    print("\nArchivos generados:")
    for f in sorted(png_files):
        print(f"  - {f}")
