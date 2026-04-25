"""
Contagem de leitões por imagem a partir **somente dos arquivos de label YOLO** em
`pig/train/labels` e `pig/val/labels` (gabarito anotado). Não carrega pesos, não
executa detecção e não gera previsões — apenas estatística descritiva dos rótulos.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import csv

import matplotlib.pyplot as plt


def contar_objetos_por_imagem(labels_dir: Path) -> list[int]:
    """Conta quantos objetos (leitões) existem em cada arquivo de label YOLO."""
    contagens: list[int] = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        linhas_validas = 0
        for linha in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            partes = linha.strip().split()
            if len(partes) >= 5:
                linhas_validas += 1
        contagens.append(linhas_validas)
    return contagens


def salvar_csv_resumo(contagens_train: list[int], contagens_val: list[int], output_dir: Path) -> None:
    output_csv = output_dir / "resumo_leitoes_por_imagem.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["grupo", "total_imagens", "min_leitoes", "max_leitoes", "media_leitoes"])
        writer.writerow(
            [
                "train",
                len(contagens_train),
                min(contagens_train) if contagens_train else 0,
                max(contagens_train) if contagens_train else 0,
                round(sum(contagens_train) / len(contagens_train), 4) if contagens_train else 0,
            ]
        )
        writer.writerow(
            [
                "val",
                len(contagens_val),
                min(contagens_val) if contagens_val else 0,
                max(contagens_val) if contagens_val else 0,
                round(sum(contagens_val) / len(contagens_val), 4) if contagens_val else 0,
            ]
        )


def salvar_csv_tabela_marginais(contagens_train: list[int], contagens_val: list[int], output_dir: Path) -> None:
    """Resumo agregado: imagens, densidade média, faixa (min–max) por conjunto."""
    path = output_dir / "tabela_marginais_train_val.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["grupo", "total_imagens", "densidade_media_objetos_por_imagem", "min_objetos_por_imagem", "max_objetos_por_imagem"])
        for nome, cont in [("train", contagens_train), ("val", contagens_val)]:
            n = len(cont)
            w.writerow(
                [
                    nome,
                    n,
                    round(sum(cont) / n, 4) if n else 0,
                    min(cont) if cont else 0,
                    max(cont) if cont else 0,
                ]
            )


def salvar_csv_distribuicao_por_quantidade(contagens_train: list[int], contagens_val: list[int], output_dir: Path) -> None:
    """
    Para cada quantidade k de leitões por imagem: frequência absoluta e % dentro do conjunto.
    Inclui todos os k presentes em treino ou validação (faixa unificada).
    """
    freq_t = Counter(contagens_train)
    freq_v = Counter(contagens_val)
    n_t, n_v = len(contagens_train), len(contagens_val)
    todos_k = sorted(set(freq_t) | set(freq_v))
    rows: list[list[object]] = []
    for k in todos_k:
        ft, fv = freq_t.get(k, 0), freq_v.get(k, 0)
        pt = round(100.0 * ft / n_t, 4) if n_t else 0.0
        pv = round(100.0 * fv / n_v, 4) if n_v else 0.0
        rows.append([k, ft, pt, fv, pv])

    for path, header in (
        (
            output_dir / "tabela_frequencia_por_quantidade_leitoes.csv",
            [
                "quantidade_leitoes_por_imagem",
                "frequencia_train",
                "porcentagem_train",
                "frequencia_val",
                "porcentagem_val",
            ],
        ),
        (
            output_dir / "tabela_distribuicao_frequencias_por_k_train_val.csv",
            [
                "k_leitoes",
                "frequencia_treino",
                "percentual_treino",
                "frequencia_val",
                "percentual_val",
            ],
        ),
    ):
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)


def salvar_csv_tabela_metricas_pivot(
    contagens_train: list[int], contagens_val: list[int], output_dir: Path
) -> None:
    """
    Tabela “larga” no estilo Métrica | Train | Val para colar em artigo ou slides.
    Valores numéricos de min/max também ficam em linhas separadas (além da faixa em texto).
    """
    path = output_dir / "tabela_metricas_train_val_pivot.csv"
    n_t, n_v = len(contagens_train), len(contagens_val)
    med_t = round(sum(contagens_train) / n_t, 4) if n_t else 0.0
    med_v = round(sum(contagens_val) / n_v, 4) if n_v else 0.0
    min_t = min(contagens_train) if contagens_train else 0
    max_t = max(contagens_train) if contagens_train else 0
    min_v = min(contagens_val) if contagens_val else 0
    max_v = max(contagens_val) if contagens_val else 0
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metrica", "train", "val"])
        w.writerow(["numero_de_imagens", n_t, n_v])
        w.writerow(["densidade_media_objetos_por_imagem", med_t, med_v])
        w.writerow(["min_objetos_por_imagem", min_t, min_v])
        w.writerow(["max_objetos_por_imagem", max_t, max_v])
        w.writerow(
            [
                "faixa_texto_min_max (use em tabelas tipo artigo)",
                f"{min_t}-{max_t}",
                f"{min_v}-{max_v}",
            ]
        )

    # Versão com rótulos legíveis em português (mesmos dados)
    path_br = output_dir / "tabela_metricas_train_val_pivot_pt.csv"
    with path_br.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metrica", "train", "val"])
        w.writerow(["Número de imagens", n_t, n_v])
        w.writerow(["Densidade média de objetos por imagem", med_t, med_v])
        w.writerow(["Mínimo de objetos por imagem", min_t, min_v])
        w.writerow(["Máximo de objetos por imagem", max_t, max_v])
        w.writerow(["Min / max por imagem (texto)", f"{min_t}-{max_t}", f"{min_v}-{max_v}"])


def gerar_histogramas(contagens_train: list[int], contagens_val: list[int], output_dir: Path) -> None:
    # Define bins inteiros compartilhados entre train/val para comparação justa.
    todos = contagens_train + contagens_val
    if not todos:
        raise ValueError("Nenhuma contagem encontrada para gerar histogramas.")
    minimo = min(todos)
    maximo = max(todos)
    bins = [x - 0.5 for x in range(minimo, maximo + 2)]

    # Histograma treino.
    plt.figure(figsize=(8, 5))
    plt.hist(contagens_train, bins=bins, color="#1f77b4", edgecolor="black", alpha=0.85)
    plt.title("Histograma - Quantidade de leitões por imagem (Train)")
    plt.xlabel("Quantidade de leitões por imagem")
    plt.ylabel("Frequência")
    plt.xticks(range(minimo, maximo + 1))
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "histograma_leitoes_train.png", dpi=170)
    plt.close()

    # Histograma validação.
    plt.figure(figsize=(8, 5))
    plt.hist(contagens_val, bins=bins, color="#ff7f0e", edgecolor="black", alpha=0.85)
    plt.title("Histograma - Quantidade de leitões por imagem (Val)")
    plt.xlabel("Quantidade de leitões por imagem")
    plt.ylabel("Frequência")
    plt.xticks(range(minimo, maximo + 1))
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "histograma_leitoes_val.png", dpi=170)
    plt.close()

    # Figura comparativa em subplots.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    axes[0].hist(contagens_train, bins=bins, color="#1f77b4", edgecolor="black", alpha=0.85)
    axes[0].set_title("Train")
    axes[0].set_xlabel("Leitões por imagem")
    axes[0].set_ylabel("Frequência")
    axes[0].set_xticks(range(minimo, maximo + 1))
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].hist(contagens_val, bins=bins, color="#ff7f0e", edgecolor="black", alpha=0.85)
    axes[1].set_title("Val")
    axes[1].set_xlabel("Leitões por imagem")
    axes[1].set_xticks(range(minimo, maximo + 1))
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Comparação: quantidade de leitões por imagem")
    fig.tight_layout()
    fig.savefig(output_dir / "histograma_leitoes_train_val_comparativo.png", dpi=170)
    plt.close(fig)


def main() -> None:
    dataset_root = Path("pig")
    train_labels = dataset_root / "train" / "labels"
    val_labels = dataset_root / "val" / "labels"
    output_dir = Path("outputs") / "analise_leitoes_por_imagem"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_labels.exists() or not val_labels.exists():
        raise FileNotFoundError("Não foi possível localizar as pastas de labels em pig/train/labels e pig/val/labels.")

    contagens_train = contar_objetos_por_imagem(train_labels)
    contagens_val = contar_objetos_por_imagem(val_labels)

    gerar_histogramas(contagens_train, contagens_val, output_dir)
    salvar_csv_resumo(contagens_train, contagens_val, output_dir)
    salvar_csv_tabela_marginais(contagens_train, contagens_val, output_dir)
    salvar_csv_distribuicao_por_quantidade(contagens_train, contagens_val, output_dir)

    print(f"[OK] Análise concluída. Arquivos salvos em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
