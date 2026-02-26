# Ejecuta todos los experimentos A-D y guarda resultados
import os
import json
from datetime import datetime

from src.utils.reproducibility import set_global_seed

SEED = 42
set_global_seed(SEED)

import pandas as pd
import numpy as np
from scipy import stats

# Importar nuestros módulos
from src.models.baselines import BaselineModels
from src.models.ml_models import MLModels
from src.models.dl_models import DLModels
from src.evaluation.validator import TimeSeriesValidator

class ExperimentRunner:
    def __init__(self, results_dir="results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.all_results = []
        self.error_records = []

    def _record_errors(self, ticker, model, horizon, fold_idx, dates, errors):
        for dt, err in zip(dates, errors):
            self.error_records.append({
                'Ticker': ticker,
                'Model': model,
                'Horizon': horizon,
                'Fold': fold_idx,
                'Date': dt,
                'Error': err
            })
    
    def load_data(self, ticker):
        # Carga datos procesados
        path = f"data/processed/{ticker.replace('-', '_')}_processed.csv"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data not found: {path}")
        return pd.read_csv(path, index_col=0, parse_dates=True)
    
    def run_baselines(self, df, ticker, horizon=7, record_errors=False):
        # Ejecuta Naive y ARIMA
        results = []
        validator = TimeSeriesValidator(n_splits=12, test_size=30)
        baselines = BaselineModels()
        full_close = df['Close']
        future_close = full_close.shift(-horizon)
        
        for fold_idx, (train_df, test_df) in enumerate(validator.split(df)):
            test_idx = test_df.index
            y_true = future_close.loc[test_idx]
            y_prev = full_close.loc[test_idx]

            valid_mask = y_true.notna()
            y_true = y_true[valid_mask]
            y_prev = y_prev[valid_mask]

            if len(y_true) == 0:
                continue
            
            # Naive
            pred_naive = y_prev.values.copy()
            metrics_naive = validator.evaluate(y_true.values, pred_naive, y_prev=y_prev.values)
            metrics_naive.update({
                'Model': 'Naive',
                'Ticker': ticker,
                'Horizon': horizon,
                'Fold': fold_idx
            })
            results.append(metrics_naive)
            if record_errors:
                self._record_errors(
                    ticker, 'Naive', horizon, fold_idx, y_true.index, y_true.values - pred_naive
                )
            
            # ARIMA
            pred_arima = []
            model_fit = baselines.fit_arima(train_df['Close'], order=(5, 1, 0))
            arima_state = model_fit

            for ts in y_true.index:
                current_close = full_close.loc[ts]

                if arima_state is None or not np.isfinite(current_close):
                    pred_arima.append(current_close)
                    continue

                try:
                    if hasattr(arima_state, "append"):
                        obs = pd.Series([current_close], index=[ts])
                        arima_state = arima_state.append(obs, refit=False)
                    forecast = baselines.forecast_from_state(arima_state, steps=horizon)
                    pred_arima.append(forecast[-1] if len(forecast) > 0 else current_close)
                except Exception:
                    pred_arima.append(current_close)

            pred_arima = np.array(pred_arima, dtype=float)
            metrics_arima = validator.evaluate(y_true.values, pred_arima, y_prev=y_prev.values)
            metrics_arima.update({
                'Model': 'ARIMA',
                'Ticker': ticker,
                'Horizon': horizon,
                'Fold': fold_idx
            })
            results.append(metrics_arima)
            if record_errors:
                self._record_errors(
                    ticker, 'ARIMA', horizon, fold_idx, y_true.index, y_true.values - pred_arima
                )
        
        return results
    
    def run_ml_models(self, df, ticker, horizon=7, record_errors=False):
        # Ejecuta RF y XGBoost
        results = []
        ml_models = MLModels()
        
        validator = TimeSeriesValidator(n_splits=12, test_size=30)

        # Preparar features manteniendo el índice original para alinear folds con
        # baselines y DL (mismas fechas de test por fold).
        X_all, y_all, ref_price_all = ml_models.prepare_features(
            df, window_size=10, horizon=horizon, dropna=False
        )
        combined = pd.concat(
            [X_all, y_all.rename('Target'), ref_price_all.rename('Ref_Close')],
            axis=1
        )
        
        for fold_idx, (train_raw, test_raw) in enumerate(validator.split(df)):
            # Evitar fuga de información: en el train no deben existir muestras cuyo target
            # dependa de precios dentro del bloque de test (t+h fuera del train).
            train_idx = train_raw.index
            test_idx = test_raw.index
            safe_train_idx = train_idx[:-horizon] if horizon > 0 else train_idx
            if len(safe_train_idx) == 0:
                continue

            safe_train_block = combined.loc[safe_train_idx].dropna()
            test_block = combined.loc[test_idx].dropna()
            if safe_train_block.empty or test_block.empty:
                continue

            train_X = safe_train_block[X_all.columns]
            train_y = safe_train_block['Target']
            test_X = test_block[X_all.columns]
            test_y_target = test_block['Target']
            test_ref_price = test_block['Ref_Close']
            
            actual_price = test_ref_price * np.exp(test_y_target)
            
            for model_type in ['rf', 'xgb']:
                pred_return = ml_models.train_predict(train_X, train_y, test_X, model_type=model_type)
                pred_price = test_ref_price * np.exp(pred_return)
                
                metrics = validator.evaluate(actual_price.values, pred_price.values, y_prev=test_ref_price.values)
                metrics.update({
                    'Model': 'RandomForest' if model_type == 'rf' else 'XGBoost',
                    'Ticker': ticker,
                    'Horizon': horizon,
                    'Fold': fold_idx
                })
                results.append(metrics)
                if record_errors:
                    model_name = 'RandomForest' if model_type == 'rf' else 'XGBoost'
                    self._record_errors(
                        ticker, model_name, horizon, fold_idx,
                        test_ref_price.index, actual_price.values - pred_price.values
                    )
        
        return results
    
    def run_dl_models(self, df, ticker, horizon=7, include_sentiment=True, record_errors=False):
        # Ejecuta LSTM, GRU, CNN
        results = []
        
        # Sin sentimiento para ablación
        if not include_sentiment:
            sentiment_cols = [c for c in df.columns if c.startswith('Sentiment')]
            if sentiment_cols:
                df = df.drop(columns=sentiment_cols)
        
        full_close = df['Close']
        future_close = full_close.shift(-horizon)

        validator = TimeSeriesValidator(n_splits=12, test_size=30)
        model_offsets = {'lstm': 0, 'gru': 1, 'cnn': 2}
        
        for fold_idx, (train_df, test_df) in enumerate(validator.split(df)):
            window_size = 30
            model_wrapper = DLModels()
            
            train_X, train_y = model_wrapper.prepare_data(train_df, window_size=window_size, horizon=horizon)
            feature_cols = model_wrapper.feature_columns_
            
            # Preparar test: una predicción por cada día del bloque de test.
            # Para el primer día del test, necesitamos window_size-1 días previos del train.
            history_df = train_df.iloc[-(window_size - 1):]
            test_input_df = pd.concat([history_df, test_df])[feature_cols].copy().dropna()
            scaler = model_wrapper.scaler
            scaled_test = scaler.transform(test_input_df)

            n_test = len(test_df)
            X_test = np.array([scaled_test[i:i + window_size] for i in range(n_test)])

            test_idx = test_df.index
            y_true = future_close.loc[test_idx]
            y_prev = full_close.loc[test_idx]

            valid_mask = y_true.notna()
            y_true = y_true[valid_mask]
            y_prev = y_prev[valid_mask]
            X_test = X_test[valid_mask.to_numpy()]

            if len(y_true) == 0:
                continue
            
            for model_type in ['lstm', 'gru', 'cnn']:
                set_global_seed(SEED + fold_idx * 10 + model_offsets[model_type])
                pred = model_wrapper.train_predict(train_X, train_y, X_test, model_type=model_type, epochs=10)
                metrics = validator.evaluate(y_true.values, pred.flatten(), y_prev=y_prev.values)
                metrics.update({
                    'Model': model_type.upper(),
                    'Ticker': ticker,
                    'Horizon': horizon,
                    'Fold': fold_idx,
                    'Sentiment': 'With' if include_sentiment else 'Without'
                })
                results.append(metrics)
                if record_errors:
                    self._record_errors(
                        ticker, model_type.upper(), horizon, fold_idx,
                        y_true.index, y_true.values - pred.flatten()
                    )
        
        return results

    def compute_ablation_significance(self, results_df, metrics=None):
        if metrics is None:
            metrics = ['MAPE', 'DirectionalAccuracy', 'DirectionalF1', 'DirectionalMCC']

        if results_df.empty:
            return pd.DataFrame()

        with_df = results_df[results_df['Sentiment'] == 'With']
        without_df = results_df[results_df['Sentiment'] == 'Without']

        merged = with_df.merge(
            without_df,
            on=['Ticker', 'Model', 'Fold'],
            suffixes=('_with', '_without')
        )

        results = []
        for metric in metrics:
            col_with = f"{metric}_with"
            col_without = f"{metric}_without"
            if col_with not in merged.columns or col_without not in merged.columns:
                continue

            for (ticker, model), group in merged.groupby(['Ticker', 'Model']):
                diffs = group[col_without] - group[col_with]
                diffs = diffs.replace([np.inf, -np.inf], np.nan).dropna()
                n = len(diffs)

                if n < 2 or np.allclose(diffs, 0):
                    stat = np.nan
                    p_value = np.nan
                else:
                    try:
                        stat, p_value = stats.wilcoxon(diffs, zero_method='wilcox')
                    except ValueError:
                        stat = np.nan
                        p_value = np.nan

                results.append({
                    'Ticker': ticker,
                    'Model': model,
                    'Metric': metric,
                    'N': n,
                    'MeanDiff': diffs.mean() if n > 0 else np.nan,
                    'MedianDiff': diffs.median() if n > 0 else np.nan,
                    'p_value': p_value,
                    'significant': bool(p_value < 0.05) if np.isfinite(p_value) else False
                })

        return pd.DataFrame(results)
    
    def experiment_a_model_comparison(self):
        # Exp A: comparar modelos en todos los activos
        print("=" * 60)
        print("EXPERIMENT A: Model Comparison Across Assets")
        print("=" * 60)
        
        tickers = ['BTC-USD', 'TSLA', 'NVDA']
        all_results = []
        self.error_records = []
        
        for ticker in tickers:
            print(f"\n--- Processing {ticker} ---")
            try:
                df = self.load_data(ticker)
                
                # Baselines
                print(f"  Running Baselines...")
                results_baseline = self.run_baselines(df, ticker, horizon=7, record_errors=True)
                all_results.extend(results_baseline)
                
                # ML
                print(f"  Running ML Models...")
                results_ml = self.run_ml_models(df, ticker, horizon=7, record_errors=True)
                all_results.extend(results_ml)
                
                # DL
                print(f"  Running DL Models...")
                results_dl = self.run_dl_models(
                    df, ticker, horizon=7, include_sentiment=True, record_errors=True
                )
                all_results.extend(results_dl)
                
            except Exception as e:
                print(f"  Error: {e}")
        
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f"{self.results_dir}/experiment_a_results.csv", index=False)
        if self.error_records:
            errors_df = pd.DataFrame(self.error_records)
            errors_df["Date"] = pd.to_datetime(errors_df["Date"]).dt.strftime("%Y-%m-%d")
            errors_df.to_csv(f"{self.results_dir}/experiment_a_errors.csv", index=False)
        
        # Agregar por modelo y ticker
        summary = results_df.groupby(['Ticker', 'Model'])[['RMSE', 'MAE', 'MAPE']].mean().round(2)
        print("\n" + "=" * 60)
        print("EXPERIMENT A SUMMARY")
        print("=" * 60)
        print(summary.to_string())
        
        return results_df
    
    def experiment_b_horizons(self):
        # Exp B: impacto del horizonte
        print("\n" + "=" * 60)
        print("EXPERIMENT B: Prediction Horizon Impact")
        print("=" * 60)
        
        horizons = [1, 7, 30]
        ticker = 'BTC-USD'  # Solo BTC para este exp
        all_results = []
        
        df = self.load_data(ticker)
        
        for horizon in horizons:
            print(f"\n--- Horizon: {horizon} days ---")
            
            # Solo DL
            print(f"  Running DL Models...")
            results_dl = self.run_dl_models(df, ticker, horizon=horizon, include_sentiment=True)
            all_results.extend(results_dl)
        
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f"{self.results_dir}/experiment_b_results.csv", index=False)
        
        summary = results_df.groupby(['Horizon', 'Model'])[['RMSE', 'MAE', 'MAPE']].mean().round(2)
        print("\n" + "=" * 60)
        print("EXPERIMENT B SUMMARY")
        print("=" * 60)
        print(summary.to_string())
        
        return results_df
    
    def experiment_c_ablation(self):
        # Exp C: ablación con/sin sentimiento
        print("\n" + "=" * 60)
        print("EXPERIMENT C: Sentiment Ablation Study")
        print("=" * 60)
        
        tickers = ['BTC-USD', 'TSLA', 'NVDA']
        all_results = []
        
        for ticker in tickers:
            print(f"\n--- Processing {ticker} ---")
            try:
                df = self.load_data(ticker)
                
                # Con sentimiento
                print(f"  Running WITH sentiment...")
                results_with = self.run_dl_models(df, ticker, horizon=7, include_sentiment=True)
                all_results.extend(results_with)
                
                # Without sentiment
                print(f"  Running WITHOUT sentiment...")
                results_without = self.run_dl_models(df, ticker, horizon=7, include_sentiment=False)
                all_results.extend(results_without)
                
            except Exception as e:
                print(f"  Error: {e}")
        
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f"{self.results_dir}/experiment_c_results.csv", index=False)

        # Test estadistico para ablacion
        ablation_tests = self.compute_ablation_significance(results_df)
        if not ablation_tests.empty:
            ablation_tests.to_csv(f"{self.results_dir}/experiment_c_ablation_tests.csv", index=False)
            print("\n" + "=" * 60)
            print("EXPERIMENT C: ABLATION SIGNIFICANCE (Wilcoxon)")
            print("=" * 60)
            print(ablation_tests.to_string(index=False))
        
        summary = results_df.groupby(['Ticker', 'Model', 'Sentiment'])[['RMSE', 'MAE', 'MAPE']].mean().round(2)
        print("\n" + "=" * 60)
        print("EXPERIMENT C SUMMARY")
        print("=" * 60)
        print(summary.to_string())
        
        return results_df
    
    def experiment_d_market_periods(self):
        # Exp D: por periodo de mercado
        print("\n" + "=" * 60)
        print("EXPERIMENT D: Market Period Analysis")
        print("=" * 60)
        
        ticker = 'BTC-USD'
        df = self.load_data(ticker)
        
        # Definir periodos
        periods = {
            'Bull-2021': ('2021-01-01', '2021-12-01'),
            'Bear-2022': ('2022-01-01', '2022-12-01'),
            'Recent': ('2023-01-01', '2025-11-01')
        }

        dl_model_types = ['lstm', 'gru', 'cnn']
        model_offsets = {'lstm': 0, 'gru': 1, 'cnn': 2}
        
        all_results = []
        
        for period_idx, (period_name, (start, end)) in enumerate(periods.items()):
            print(f"\n--- Periodo: {period_name} ({start} a {end}) ---")
            try:
                period_df = df.loc[start:end]
                if len(period_df) < 100:
                    print(f"  Saltando (datos insuficientes: {len(period_df)} filas)")
                    continue
                
                model_wrapper = DLModels()
                window_size = 30
                horizon = 7
                
                train_size = int(len(period_df) * 0.8)
                train_df = period_df.iloc[:train_size]
                test_df = period_df.iloc[train_size:]
                
                if len(train_df) < window_size + horizon or len(test_df) < horizon:
                    print(f"  Saltando (datos insuficientes para partición)")
                    continue
                
                set_global_seed(SEED + 100 + period_idx)
                train_X, train_y = model_wrapper.prepare_data(train_df, window_size=window_size, horizon=horizon)
                feature_cols = model_wrapper.feature_columns_
                
                scaler = model_wrapper.scaler
                validator = TimeSeriesValidator()

                full_close = period_df['Close']
                future_close = full_close.shift(-horizon)

                history_df = train_df.iloc[-(window_size - 1):]
                test_input_df = pd.concat([history_df, test_df])[feature_cols].copy().dropna()
                scaled_test = scaler.transform(test_input_df)

                n_test = len(test_df)
                X_test = np.array([scaled_test[i:i + window_size] for i in range(n_test)])

                test_idx = test_df.index
                y_true = future_close.loc[test_idx]
                y_prev = full_close.loc[test_idx]

                valid_mask = y_true.notna()
                y_true = y_true[valid_mask]
                y_prev = y_prev[valid_mask]
                X_test = X_test[valid_mask.to_numpy()]

                if len(y_true) == 0:
                    print(f"  Saltando (no hay muestras de test válidas)")
                    continue

                for model_type in dl_model_types:
                    set_global_seed(SEED + 100 + period_idx * 10 + model_offsets[model_type])
                    pred = model_wrapper.train_predict(train_X, train_y, X_test, model_type=model_type, epochs=10)
                    metrics = validator.evaluate(y_true.values, pred.flatten(), y_prev=y_prev.values)
                    metrics.update({
                        'Period': period_name,
                        'Model': model_type.upper(),
                        'Samples': len(y_true)
                    })
                    all_results.append(metrics)
                    print(f"  {model_type.upper()} MAPE: {metrics['MAPE']:.2f}%")
                
            except Exception as e:
                print(f"  Error: {e}")
        
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(f"{self.results_dir}/experiment_d_results.csv", index=False)
        
        print("\n" + "=" * 60)
        print("EXPERIMENT D SUMMARY")
        print("=" * 60)
        print(results_df.to_string(index=False))
        
        return results_df


if __name__ == "__main__":
    runner = ExperimentRunner()
    
    print("\n" + "#" * 70)
    print("# TFG COMPREHENSIVE EXPERIMENT SUITE")
    print("#" * 70)
    
    # Exp A
    exp_a = runner.experiment_a_model_comparison()
    
    # Exp B
    exp_b = runner.experiment_b_horizons()
    
    # Exp C
    exp_c = runner.experiment_c_ablation()
    
    # Exp D
    exp_d = runner.experiment_d_market_periods()
    
    print("\n" + "#" * 70)
    print("# TODOS LOS EXPERIMENTOS COMPLETADOS")
    print(f"# Resultados guardados en: {runner.results_dir}/")
    print("#" * 70)
