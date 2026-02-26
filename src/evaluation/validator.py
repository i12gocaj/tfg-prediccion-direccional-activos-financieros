import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats

class TimeSeriesValidator:
    def __init__(self, n_splits=5, test_size=30):
        # n_splits = folds, test_size = días de test
        self.n_splits = n_splits
        self.test_size = test_size

    def split(self, df):
        # Genera particiones train/test walk-forward
        n_samples = len(df)
        if n_samples < self.test_size * (self.n_splits + 1):
            raise ValueError("No hay datos suficientes.")

        fold_size = n_samples // (self.n_splits + 1)

        for i in range(1, self.n_splits + 1):
            train_end = fold_size * i
            test_end = min(train_end + self.test_size, n_samples)
            if test_end <= train_end:
                break

            train_df = df.iloc[:train_end]
            test_df = df.iloc[train_end:test_end]

            yield train_df, test_df

    def evaluate(self, y_true, y_pred, y_prev=None):
        # Calcula RMSE, MAE, MAPE y métricas direccionales
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # MAPE
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        # Precisión direccional (sube o baja)
        if y_prev is not None and len(y_prev) == len(y_true):
            actual_change = y_true - y_prev
            pred_change = y_pred - y_prev
        else:
            if len(y_true) > 1:
                actual_change = np.diff(y_true)
                pred_change = np.diff(y_pred)
            else:
                actual_change = np.array([])
                pred_change = np.array([])

        if len(actual_change) > 0:
            mask = np.isfinite(actual_change) & np.isfinite(pred_change)
            actual_change = actual_change[mask]
            pred_change = pred_change[mask]
            direction_mask = actual_change != 0
            actual_direction = (actual_change[direction_mask] > 0).astype(int)
            pred_direction = (pred_change[direction_mask] > 0).astype(int)
        else:
            actual_direction = np.array([])
            pred_direction = np.array([])

        if len(actual_direction) > 0:
            dir_accuracy = np.mean(actual_direction == pred_direction) * 100
            # Métricas binarias sin warnings cuando solo aparece una clase en una ventana.
            tp = int(np.sum((actual_direction == 1) & (pred_direction == 1)))
            tn = int(np.sum((actual_direction == 0) & (pred_direction == 0)))
            fp = int(np.sum((actual_direction == 0) & (pred_direction == 1)))
            fn = int(np.sum((actual_direction == 1) & (pred_direction == 0)))

            denom_f1 = (2 * tp + fp + fn)
            dir_f1 = (2 * tp / denom_f1) * 100 if denom_f1 > 0 else 0.0

            denom_mcc = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            dir_mcc = (tp * tn - fp * fn) / denom_mcc if denom_mcc > 0 else 0.0
        else:
            dir_accuracy = np.nan
            dir_f1 = np.nan
            dir_mcc = np.nan
        
        return {
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "DirectionalAccuracy": dir_accuracy,
            "DirectionalF1": dir_f1,
            "DirectionalMCC": dir_mcc
        }
    
    def diebold_mariano_test(self, errors_1, errors_2, horizon=1):
        """
        Test de Diebold-Mariano para comparar precisión predictiva de dos modelos.
        H0: ambos modelos tienen igual precisión.
        
        Args:
            errors_1, errors_2: Arrays de errores (y_true - y_pred)
            horizon: Horizonte de predicción h
            
        Returns:
            dict con dm_statistic, p_value, significant, better_model
        """
        # Diferencia de pérdidas (errores cuadrados)
        d = errors_1**2 - errors_2**2
        d = d[np.isfinite(d)]

        n = len(d)
        if n < 2:
            return {"dm_statistic": np.nan, "p_value": np.nan, 
                    "significant": False, "better_model": None}
        
        mean_d = np.mean(d)
        
        # Varianza de largo plazo con corrección Newey-West
        # gamma_0 + 2*sum(gamma_k) para k=1..h-1
        gamma_0 = np.sum((d - mean_d)**2) / n
        
        gamma_sum = 0
        for k in range(1, min(horizon, n)):
            gamma_k = np.sum((d[:-k] - mean_d) * (d[k:] - mean_d)) / n
            gamma_sum += gamma_k
        
        # Varianza de la media muestral
        var_mean_d = (gamma_0 + 2 * gamma_sum) / n
        
        if var_mean_d <= 0 or not np.isfinite(var_mean_d):
            # Sin corrección si falla
            var_mean_d = np.var(d, ddof=1) / n
            
        if var_mean_d <= 0:
            return {"dm_statistic": np.nan, "p_value": np.nan,
                    "significant": False, "better_model": None}
        
        dm_stat = mean_d / np.sqrt(var_mean_d)
        
        # Usar t-Student para muestras finitas (Harvey, Leybourne, Newbold 1997)
        # Grados de libertad: n - 1
        p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=n-1))
        
        return {
            "dm_statistic": dm_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "better_model": 1 if mean_d < 0 else 2  # 1 si modelo 1 tiene menor error
        }
