import os
import sys
import shutil
from pathlib import Path


def sync_plots_to_memoria(src_dir="results/plots", dst_dir="memoria/Imagenes"):
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    if not src_path.exists():
        return

    dst_path.mkdir(parents=True, exist_ok=True)

    for plot_path in src_path.glob("*.png"):
        shutil.copy2(plot_path, dst_path / plot_path.name)


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    if existing_pythonpath:
        os.environ["PYTHONPATH"] = project_root + os.pathsep + existing_pythonpath
    else:
        os.environ["PYTHONPATH"] = project_root

    # Evitar generar artefactos __pycache__ en ejecuciones completas.
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    print("TFG: Predicción Financiera - Ejecución Completa", flush=True)
    
    # 1. Descarga de Datos
    print("\n[1/6] Descargando datos...", flush=True)
    if os.system(f"{sys.executable} src/data/downloader.py") != 0:
        raise SystemExit("Error en la descarga de datos.")
    
    # 2. Preprocesamiento
    print("\n[2/6] Procesando datos...", flush=True)
    if os.system(f"{sys.executable} src/data/preprocessor.py") != 0:
        raise SystemExit("Error en el preprocesamiento.")
    
    # 3. Ejecutar Experimentos Principales
    print("\n[3/6] Ejecutando experimentos (A-D)...", flush=True)
    if os.system(f"{sys.executable} scripts/run_all_experiments.py") != 0:
        raise SystemExit("Error en los experimentos.")
    
    # 4. Métricas adicionales
    print("\n[4/6] Calculando métricas direccionales y test Diebold-Mariano...", flush=True)
    if os.system(f"{sys.executable} scripts/run_additional_metrics.py") != 0:
        raise SystemExit("Error calculando metricas adicionales.")
    
    # 5. Generar gráficas
    print("\n[5/6] Generando gráficas (y sincronizando para LaTeX)...", flush=True)
    if os.system(f"{sys.executable} scripts/generate_all_plots.py") != 0:
        raise SystemExit("Error generando graficas.")
    if os.system(f"{sys.executable} scripts/generate_summary_plots.py") != 0:
        raise SystemExit("Error generando graficas de resumen.")
    sync_plots_to_memoria()

    # 6. Exportar tablas LaTeX
    print("\n[6/6] Exportando tablas a LaTeX...", flush=True)
    if os.system(f"{sys.executable} scripts/export_tables_to_latex.py") != 0:
        raise SystemExit("Error exportando tablas LaTeX.")
    
    print("\nEjecución Finalizada", flush=True)
    print("Revisa results/ para los CSV y results/plots/ para las figuras.", flush=True)
    print("Para compilar la memoria, las figuras también quedan en memoria/Imagenes/.", flush=True)

if __name__ == "__main__":
    main()
