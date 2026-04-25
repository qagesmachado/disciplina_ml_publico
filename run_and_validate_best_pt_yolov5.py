import csv
import inspect
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import ImageFont


def read_yolo_file(path: Path) -> list[tuple[int, float, float, float, float]]:
    # Lê arquivo de labels YOLO e retorna linhas no formato:
    # (class_id, x_center, y_center, width, height), todos normalizados [0,1].
    rows: list[tuple[int, float, float, float, float]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        rows.append(
            (
                int(float(parts[0])),
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
            )
        )
    return rows


def yolo_to_xyxy(x_center: float, y_center: float, width: float, height: float) -> tuple[float, float, float, float]:
    # Converte caixa YOLO (centro + tamanho) para canto superior/esquerdo e inferior/direito.
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    return x1, y1, x2, y2


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    # Calcula IoU entre duas caixas (já em xyxy normalizado).
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def match_counts(
    ground_truth_rows: list[tuple[int, float, float, float, float]],
    prediction_rows: list[tuple[int, float, float, float, float]],
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    # Faz pareamento greedy entre predições e ground truth:
    # retorna (true_positives, false_positives, false_negatives).
    used_ground_truth = [False] * len(ground_truth_rows)
    true_positives = 0
    false_positives = 0

    for prediction in prediction_rows:
        prediction_class, px, py, pw, ph = prediction
        prediction_box = yolo_to_xyxy(px, py, pw, ph)
        best_iou = 0.0
        best_index = -1
        for index, ground_truth in enumerate(ground_truth_rows):
            if used_ground_truth[index]:
                continue
            ground_truth_class, gx, gy, gw, gh = ground_truth
            if prediction_class != ground_truth_class:
                continue
            current_iou = iou(prediction_box, yolo_to_xyxy(gx, gy, gw, gh))
            if current_iou > best_iou:
                best_iou = current_iou
                best_index = index

        if best_index >= 0 and best_iou >= iou_threshold:
            used_ground_truth[best_index] = True
            true_positives += 1
        else:
            false_positives += 1

    false_negatives = used_ground_truth.count(False)
    return true_positives, false_positives, false_negatives


def extract_main_metrics(val_result: object) -> tuple[float, float, float, float]:
    # Extrai as métricas globais retornadas pelo yolov5.val.run():
    # precision, recall, mAP@0.50, mAP@0.50:0.95
    # O retorno do yolov5 pode variar por versão (dict, tuple, ndarray),
    # então este parser tenta cobrir os formatos comuns.
    def _flatten_numbers(value: object) -> list[float]:
        numbers: list[float] = []
        if isinstance(value, (int, float)):
            numbers.append(float(value))
        elif isinstance(value, np.ndarray):
            try:
                numbers.extend([float(v) for v in value.flatten().tolist()])
            except Exception:
                pass
        elif isinstance(value, (list, tuple)):
            for item in value:
                numbers.extend(_flatten_numbers(item))
        return numbers

    if isinstance(val_result, dict):
        precision = float(val_result.get("mp", 0.0))
        recall = float(val_result.get("mr", 0.0))
        map_50 = float(val_result.get("map50", 0.0))
        map_50_95 = float(val_result.get("map", 0.0))
        return precision, recall, map_50, map_50_95

    flattened = _flatten_numbers(val_result)
    if len(flattened) >= 4:
        return flattened[0], flattened[1], flattened[2], flattened[3]

    return 0.0, 0.0, 0.0, 0.0


def write_html_report(
    output_dir: Path,
    summary_row: dict[str, object],
    per_image_csv_path: Path,
    summary_csv_path: Path,
    detect_output_dir: Path,
) -> None:
    # Gera relatório HTML consolidando:
    # - resumo de métricas
    # - interpretação textual
    # - glossário
    # - links/caminhos dos artefatos gerados
    html_path = output_dir / "results.html"
    html = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <title>Relatório de Validação YOLOv5 - best.pt</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    h1, h2 {{ margin: 0 0 12px 0; }}
    table {{ border-collapse: collapse; margin-top: 12px; }}
    th, td {{ border: 1px solid #ccc; padding: 8px 10px; text-align: left; }}
    th {{ background: #f3f3f3; }}
    code {{ background: #f6f6f6; padding: 2px 6px; }}
    .metric-card {{ margin-top: 14px; padding: 12px; border: 1px solid #ddd; border-radius: 8px; }}
    .metric-card p {{ margin: 6px 0; }}
  </style>
</head>
<body>
  <h1>Validação YOLOv5 - best.pt</h1>
  <p>Este relatório foi gerado pelo script <code>run_and_validate_best_pt_yolov5.py</code>.</p>

  <h2>Resumo Executivo das Métricas</h2>
  <table>
    <tr>
      <th>Classe Avaliada</th>
      <th>Total de Imagens</th>
      <th>Total de Instâncias (Porcos)</th>
      <th>Precisão (Precision)</th>
      <th>Revocação (Recall)</th>
      <th>mAP@0.50</th>
      <th>mAP@0.50:0.95</th>
    </tr>
    <tr>
      <td>{summary_row["class_name"]}</td>
      <td>{summary_row["total_images"]}</td>
      <td>{summary_row["total_instances"]}</td>
      <td>{summary_row["precision"]}</td>
      <td>{summary_row["recall"]}</td>
      <td>{summary_row["mean_average_precision_iou_0_50"]}</td>
      <td>{summary_row["mean_average_precision_iou_0_50_to_0_95"]}</td>
    </tr>
  </table>

  <h2>Como interpretar estes valores</h2>
  <div class="metric-card">
    <p><strong>Precisão (Precision):</strong> entre as detecções que o modelo fez, quantas estavam corretas.
    Valor alto indica menos falsos positivos.</p>
    <p><strong>Revocação (Recall):</strong> entre todos os porcos reais anotados, quantos o modelo conseguiu detectar.
    Valor alto indica menos falsos negativos.</p>
    <p><strong>mAP@0.50:</strong> qualidade média das detecções com critério IoU >= 0.50.
    É uma visão mais tolerante da qualidade de localização das caixas.</p>
    <p><strong>mAP@0.50:0.95:</strong> média do AP em múltiplos limiares de IoU (0.50 até 0.95).
    É um critério mais rigoroso e mostra quão bem as caixas estão ajustadas com precisão fina.</p>
    <p><strong>Leitura prática para apresentação:</strong> se Recall está alto e Precision moderada,
    o modelo encontra a maioria dos porcos, porém ainda com detecções excedentes.
    Se mAP@0.50:0.95 está bem abaixo de mAP@0.50, há espaço para melhorar o ajuste fino das caixas.</p>
  </div>

  <h2>Glossário de abreviações</h2>
  <div class="metric-card">
    <p><strong>TP (True Positive):</strong> detecção correta (modelo detectou porco e era porco).</p>
    <p><strong>FP (False Positive):</strong> falso positivo (modelo detectou porco onde não deveria).</p>
    <p><strong>FN (False Negative):</strong> falso negativo (havia porco e o modelo não detectou).</p>
    <p><strong>TN (True Negative):</strong> verdadeiro negativo (não havia porco e o modelo não detectou; pouco usado diretamente em detecção).</p>
    <p><strong>IoU (Intersection over Union):</strong> medida de sobreposição entre caixa predita e caixa real.</p>
    <p><strong>AP (Average Precision):</strong> área sob a curva Precisão x Recall para uma classe.</p>
    <p><strong>mAP (mean Average Precision):</strong> média dos APs.</p>
    <p><strong>mAP@0.50:</strong> mAP com limiar IoU de 0.50.</p>
    <p><strong>mAP@0.50:0.95:</strong> média de AP em múltiplos limiares de IoU (0.50 a 0.95).</p>
  </div>

  <h2>Arquivos gerados</h2>
  <ul>
    <li>Resultados por imagem: <code>{per_image_csv_path}</code></li>
    <li>Resumo consolidado: <code>{summary_csv_path}</code></li>
    <li>Inferência com log por imagem: <code>{detect_output_dir}</code></li>
  </ul>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")


def main() -> None:
    # Uso:
    # python run_and_validate_best_pt_yolov5.py
    # python run_and_validate_best_pt_yolov5.py "pig/val/images" 640 16 0.25
    # ====== 0) ENTRADAS DO SCRIPT ======
    # source: imagem/pasta para inferência detect
    # image_size: resolução de inferência/validação
    # batch_size: batch da validação
    # confidence: limiar de confiança para detecção
    source = sys.argv[1] if len(sys.argv) > 1 else "pig/val/images"
    image_size = int(sys.argv[2]) if len(sys.argv) > 2 else 640
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    confidence = float(sys.argv[4]) if len(sys.argv) > 4 else 0.25

    weights_path = Path("pig") / "best.pt"
    data_yaml_path = Path("pig_yolov5_data.yaml")
    validation_labels_dir = Path("pig") / "val" / "labels"
    validation_images_dir = Path("pig") / "val" / "images"
    output_dir = Path("outputs") / "best_pt_yolov5_val"
    detect_output_name = "best_pt_yolov5_detect_log"

    # ====== 1) VALIDAÇÃO DE ARQUIVOS DE ENTRADA ======
    if not weights_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {weights_path}")
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Arquivo data.yaml não encontrado: {data_yaml_path}")

    # ====== 2) PATCHES DE COMPATIBILIDADE (ambiente legado YOLOv5) ======
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    # Compatibilidade NumPy 2.x para código legado do YOLOv5.
    if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
        np.trapz = np.trapezoid  # type: ignore[attr-defined]

    # Compatibilidade Pillow >=10 para código legado do YOLOv5.
    if not hasattr(ImageFont.FreeTypeFont, "getsize"):
        def _compat_getsize(self, text):  # type: ignore[no-redef]
            left, top, right, bottom = self.getbbox(text)
            return right - left, bottom - top

        ImageFont.FreeTypeFont.getsize = _compat_getsize  # type: ignore[attr-defined]

    # ====== 3) INFERÊNCIA (DETECT) ======
    # Essa etapa imprime logs por imagem no terminal ("N Pigs") e salva imagens/labels.
    detect_command = [
        sys.executable,
        "-m",
        "yolov5.detect",
        "--weights",
        str(weights_path),
        "--source",
        source,
        "--img",
        str(image_size),
        "--conf",
        str(confidence),
        "--project",
        "outputs",
        "--name",
        detect_output_name,
        "--exist-ok",
        "--save-txt",
    ]
    print("[INFO] Executando inferência por imagem:", " ".join(detect_command))
    subprocess.run(detect_command, check=True, env=os.environ.copy())

    # ====== 4) VALIDAÇÃO (VAL) ======
    # Essa etapa calcula métricas globais oficiais: P, R, mAP@0.50, mAP@0.50:0.95.
    from yolov5.val import run as val_run

    requested_kwargs = {
        "data": str(data_yaml_path),
        "weights": str(weights_path),
        "imgsz": image_size,
        "batch_size": batch_size,
        "project": "outputs",
        "name": "best_pt_yolov5_val",
        "exist_ok": True,
        "save_txt": True,
        "save_json": False,
        "verbose": False,
        "conf_thres": confidence,
    }
    valid_parameters = set(inspect.signature(val_run).parameters.keys())
    filtered_kwargs = {k: v for k, v in requested_kwargs.items() if k in valid_parameters}
    if "plots" in valid_parameters:
        filtered_kwargs["plots"] = False

    print("[INFO] Executando validação YOLOv5 com argumentos:", filtered_kwargs)
    validation_result = val_run(**filtered_kwargs)
    # <- AQUI as métricas são extraídas do retorno da validação.
    precision, recall, map_50, map_50_95 = extract_main_metrics(validation_result)

    # Para manter coerência com o log "image ... N Pigs", usamos as predições
    # geradas pelo detect (mesma etapa que imprime esse log).
    prediction_labels_dir = Path("outputs") / detect_output_name / "labels"
    image_paths = sorted(
        [p for p in validation_images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    )

    # ====== 5) MÉTRICAS POR IMAGEM ======
    # Usa labels do detect para manter coerência com o log por imagem.
    # Para cada imagem: conta GT, predições e calcula TP/FP/FN + status.
    per_image_rows: list[dict[str, object]] = []
    total_instances = 0

    for image_path in image_paths:
        stem = image_path.stem
        ground_truth_rows = read_yolo_file(validation_labels_dir / f"{stem}.txt")
        prediction_rows = read_yolo_file(prediction_labels_dir / f"{stem}.txt")
        total_instances += len(ground_truth_rows)

        true_positives, false_positives, false_negatives = match_counts(
            ground_truth_rows, prediction_rows, iou_threshold=0.5
        )

        status = "ok"
        if false_positives > 0 and false_negatives > 0:
            status = "false_positives_and_false_negatives"
        elif false_positives > 0:
            status = "false_positives_only"
        elif false_negatives > 0:
            status = "false_negatives_only"

        per_image_rows.append(
            {
                "image_name": image_path.name,
                "ground_truth_count": len(ground_truth_rows),
                "predicted_count": len(prediction_rows),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "status": status,
                "detection_count_difference": len(prediction_rows) - len(ground_truth_rows),
            }
        )

    # ====== 6) GERAÇÃO DOS CSVs ======
    # CSV 1: resultados individuais por imagem.
    per_image_csv_path = output_dir / "individual_image_results.csv"
    summary_csv_path = output_dir / "summary_results.csv"

    per_image_fieldnames = [
        "image_name",
        "ground_truth_count",
        "predicted_count",
        "true_positives",
        "false_positives",
        "false_negatives",
        "status",
        "detection_count_difference",
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    with per_image_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=per_image_fieldnames)
        writer.writeheader()
        writer.writerows(per_image_rows)

    # CSV 2: resumo global das métricas.
    # <- AQUI os valores de mAP entram no CSV final.
    summary_row = {
        "class_name": "all",
        "total_images": len(image_paths),
        "total_instances": total_instances,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "mean_average_precision_iou_0_50": round(map_50, 6),
        "mean_average_precision_iou_0_50_to_0_95": round(map_50_95, 6),
    }

    summary_fieldnames = [
        "class_name",
        "total_images",
        "total_instances",
        "precision",
        "recall",
        "mean_average_precision_iou_0_50",
        "mean_average_precision_iou_0_50_to_0_95",
    ]
    with summary_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerow(summary_row)

    # ====== 7) RELATÓRIO HTML ======
    detect_output_dir = Path("outputs") / detect_output_name
    write_html_report(
        output_dir=output_dir,
        summary_row=summary_row,
        per_image_csv_path=per_image_csv_path,
        summary_csv_path=summary_csv_path,
        detect_output_dir=detect_output_dir,
    )

    print(f"[OK] CSV por imagem gerado em: {per_image_csv_path}")
    print(f"[OK] CSV resumo gerado em: {summary_csv_path}")
    print(f"[OK] HTML atualizado em: {output_dir / 'results.html'}")


if __name__ == "__main__":
    main()
