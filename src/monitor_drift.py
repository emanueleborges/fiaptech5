import argparse
import json
import os
from datetime import datetime

import pandas as pd


def load_train_stats(train_stats_path: str) -> dict:
    if not os.path.exists(train_stats_path):
        raise FileNotFoundError(f"Arquivo de estatisticas de treino nao encontrado: {train_stats_path}")
    with open(train_stats_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_current_stats(input_path: str) -> pd.Series:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Arquivo de dados de entrada nao encontrado: {input_path}")

    df = pd.read_csv(input_path)
    # Considera apenas colunas numericas para comparacao de medias
    return df.select_dtypes(include=["int64", "float64", "int32", "float32"]).mean()


def compare_stats(train_stats: dict, current_stats: pd.Series, threshold: float = 0.3) -> dict:
    drifts = {}
    for col, train_mean in train_stats.items():
        if col not in current_stats.index:
            continue
        current_mean = float(current_stats[col])
        if train_mean == 0:
            continue
        diff = abs(current_mean - train_mean)
        ratio = diff / abs(train_mean)
        if ratio > threshold:
            drifts[col] = {
                "train_mean": train_mean,
                "current_mean": current_mean,
                "relative_diff": ratio,
                "status": "DRIFT DETECTED",
            }
    return drifts


def generate_report(input_path: str, train_stats_path: str, output_dir: str = "logs") -> str:
    os.makedirs(output_dir, exist_ok=True)

    train_stats = load_train_stats(train_stats_path)
    current_stats = compute_current_stats(input_path)
    drifts = compare_stats(train_stats, current_stats)

    report = {
        "timestamp": datetime.now().isoformat(),
        "input_path": input_path,
        "train_stats_path": train_stats_path,
        "features_compared": list(train_stats.keys()),
        "drifts": drifts,
    }

    report_path = os.path.join(
        output_dir,
        f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compara estatisticas de dados de entrada com as estatisticas de treino "
            "e gera um relatorio de drift em JSON."
        )
    )
    parser.add_argument(
        "input_path",
        help="Caminho para um CSV com dados recentes de entrada (ex.: amostra de requests)",
    )
    parser.add_argument(
        "--train-stats-path",
        default="models/train_stats.json",
        help="Caminho para o arquivo JSON com estatisticas de treino (default: models/train_stats.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="logs",
        help="Diretorio onde o relatorio de drift sera salvo (default: logs)",
    )

    args = parser.parse_args()

    report_path = generate_report(
        input_path=args.input_path,
        train_stats_path=args.train_stats_path,
        output_dir=args.output_dir,
    )

    print(f"Relatorio de drift gerado em: {report_path}")


if __name__ == "__main__":
    main()
