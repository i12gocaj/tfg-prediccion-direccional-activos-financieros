# TFG - Prediccion direccional de activos financieros con aprendizaje automatico y datos alternativos

Este repositorio contiene el codigo y los artefactos del Trabajo de Fin de Grado orientado a la prediccion de direccion del mercado en varios activos financieros, combinando:

- Indicadores tecnicos
- Modelos baseline y de machine learning
- Modelos deep learning
- Datos alternativos de sentimiento (GDELT)

## Autor

- Javier Gonzalez Casares

## Objetivo

Comparar el rendimiento de diferentes enfoques de modelado para predecir la evolucion de activos financieros (`BTC-USD`, `TSLA`, `NVDA`) en varios horizontes temporales, y evaluar el impacto del sentimiento de noticias como variable explicativa.

## Stack tecnologico

- Python 3.x
- pandas, numpy, scipy
- scikit-learn, xgboost, statsmodels
- tensorflow
- yfinance, requests
- matplotlib, seaborn
- ta

## Estructura del proyecto

```text
.
├── main.py                      # Pipeline completo (descarga, preproceso, experimentos, metricas, graficas)
├── requirements.txt
├── data/
│   ├── raw/                     # Datos descargados
│   └── processed/               # Datos procesados para modelado
├── src/
│   ├── data/                    # Descarga y preprocesamiento
│   ├── models/                  # Baselines, ML y DL
│   ├── evaluation/              # Validacion temporal y metricas
│   └── utils/                   # Reproducibilidad
├── scripts/                     # Ejecucion de experimentos y generacion de salidas
├── results/                     # CSV de resultados y graficas
└── memoria/                     # Memoria escrita (LaTeX), figuras y anexos
```

## Instalacion

1. Crear entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Ejecucion

Para ejecutar el flujo completo:

```bash
python main.py
```

El pipeline realiza:

1. Descarga de datos financieros y de sentimiento
2. Preprocesamiento y construccion de variables
3. Ejecucion de experimentos A-D
4. Calculo de metricas adicionales
5. Generacion de graficas
6. Export de tablas para LaTeX

## Resultados

Los artefactos principales se generan en:

- `results/`: tablas y metricas en CSV
- `results/plots/`: visualizaciones
- `memoria/Imagenes/`: sincronizacion de figuras para la memoria

## Reproducibilidad

- Semilla global fijada en `42` en los scripts de experimentacion.
- El rango temporal de descarga por defecto se define en `src/data/downloader.py`.

## Memoria del TFG

La documentacion academica y anexos estan en la carpeta `memoria/`.

## Nota

Proyecto con fines academicos.
