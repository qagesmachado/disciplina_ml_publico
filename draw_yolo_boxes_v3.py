from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Versão 3: desenha boxes YOLO e salva outputs em pastas separadas "
            "(imagens anotadas + relatórios CSV)."
        )
    )
    parser.add_argument("--label-file", type=Path, help="Arquivo de label YOLO (.txt).")
    parser.add_argument("--labels-dir", type=Path, help="Diretório com labels YOLO (.txt).")
    parser.add_argument(
        "--image-file",
        type=Path,
        help="Imagem correspondente ao --label-file (opcional se houver --images-dir).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        help="Diretório com imagens para casar com os labels.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs") / "yolo_boxes_v3_runs",
        help="Pasta raiz das execuções. Cada execução vai para uma subpasta própria.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Nome da execução. Se vazio, usa timestamp.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Espessura da linha da bounding box.",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="255,0,0",
        help="Cor RGB no formato R,G,B.",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Gera CSV consolidado com todas as boxes.",
    )
    parser.add_argument(
        "--write-per-image-csv",
        action="store_true",
        help="Gera também um CSV por imagem.",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="boxes_report.csv",
        help="Nome do CSV consolidado.",
    )
    return parser.parse_args()


def parse_color(color_str: str) -> tuple[int, int, int]:
    # Converte texto "R,G,B" para tupla de inteiros validada.
    parts = [p.strip() for p in color_str.split(",")]
    if len(parts) != 3:
        raise ValueError("Cor inválida. Use formato R,G,B.")
    rgb = tuple(int(v) for v in parts)
    if any(v < 0 or v > 255 for v in rgb):
        raise ValueError("Valores RGB devem estar entre 0 e 255.")
    return rgb  # type: ignore[return-value]


def build_run_dirs(output_root: Path, run_name: str) -> tuple[Path, Path, Path]:
    # Cria a estrutura por execução (run) para organizar imagens e relatórios.
    if not run_name:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_name
    images_out = run_dir / "annotated_images"
    reports_out = run_dir / "reports"
    images_out.mkdir(parents=True, exist_ok=True)
    reports_out.mkdir(parents=True, exist_ok=True)
    return run_dir, images_out, reports_out


def iter_label_files(label_file: Path | None, labels_dir: Path | None) -> Iterable[Path]:
    # Permite processar arquivo único ou diretório completo de labels.
    if label_file:
        yield label_file
    if labels_dir:
        yield from sorted(labels_dir.glob("*.txt"))


def find_image_for_label(label_path: Path, image_file: Path | None, images_dir: Path | None) -> Path:
    if image_file:
        return image_file
    if images_dir is None:
        raise FileNotFoundError("Informe --image-file ou --images-dir.")

    # Primeiro tenta correspondência exata por stem.
    for ext in IMAGE_EXTENSIONS:
        exact = images_dir / f"{label_path.stem}{ext}"
        if exact.exists():
            return exact

    # Fallback por prefixo anterior ao ".rf." para casos com hashes distintos.
    prefix = label_path.name.split(".rf.")[0]
    for ext in IMAGE_EXTENSIONS:
        matches = sorted(images_dir.glob(f"{prefix}*.{ext.lstrip('.')}"))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"Imagem não encontrada para label: {label_path.name}")


def read_yolo_rows(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    # Lê linhas no padrão YOLO:
    # class_id x_center y_center width height.
    rows: list[tuple[int, float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
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


def yolo_to_pixel_box(
    x_center: float, y_center: float, width: float, height: float, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    # Converte coordenadas normalizadas para pixel.
    bw = width * img_w
    bh = height * img_h
    cx = x_center * img_w
    cy = y_center * img_h
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    return x1, y1, x2, y2


def write_csv(rows: list[dict], csv_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_label(
    label_path: Path,
    image_path: Path,
    image_out_path: Path,
    color: tuple[int, int, int],
    line_width: int,
) -> list[dict]:
    rows = read_yolo_rows(label_path)
    report_rows: list[dict] = []

    with Image.open(image_path) as im:
        img = im.convert("RGB")
        draw = ImageDraw.Draw(img)
        img_w, img_h = img.size

        for idx, (class_id, x_center, y_center, width, height) in enumerate(rows, start=1):
            # Desenha box e registra dados para relatórios consolidados e por imagem.
            x1, y1, x2, y2 = yolo_to_pixel_box(x_center, y_center, width, height, img_w, img_h)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
            draw.text((x1 + 3, max(0, y1 - 12)), f"id:{class_id}", fill=color)

            report_rows.append(
                {
                    "label_file": label_path.name,
                    "image_file": image_path.name,
                    "box_index": idx,
                    "class_id": class_id,
                    "x_center_norm": x_center,
                    "y_center_norm": y_center,
                    "width_norm": width,
                    "height_norm": height,
                    "x1_px": round(x1, 2),
                    "y1_px": round(y1, 2),
                    "x2_px": round(x2, 2),
                    "y2_px": round(y2, 2),
                    "box_w_px": round(x2 - x1, 2),
                    "box_h_px": round(y2 - y1, 2),
                    "img_w_px": img_w,
                    "img_h_px": img_h,
                }
            )

        img.save(image_out_path)

    return report_rows


def main() -> None:
    args = parse_args()
    if not args.label_file and not args.labels_dir:
        raise ValueError("Informe --label-file ou --labels-dir.")

    # Fluxo principal:
    # 1) prepara run-dir, 2) processa labels, 3) salva imagens, 4) salva relatórios CSV.
    color = parse_color(args.color)
    label_files = list(iter_label_files(args.label_file, args.labels_dir))
    if not label_files:
        raise FileNotFoundError("Nenhum label encontrado.")

    run_dir, images_out, reports_out = build_run_dirs(args.output_root, args.run_name)
    all_rows: list[dict] = []

    for label_path in label_files:
        if not label_path.exists():
            print(f"[AVISO] Label não encontrado: {label_path}")
            continue

        image_path = find_image_for_label(label_path, args.image_file, args.images_dir)
        output_image_name = f"{label_path.stem}_boxed{image_path.suffix.lower()}"
        output_image_path = images_out / output_image_name

        rows = process_label(
            label_path=label_path,
            image_path=image_path,
            image_out_path=output_image_path,
            color=color,
            line_width=args.line_width,
        )
        all_rows.extend(rows)
        print(f"[OK] Imagem anotada: {output_image_path}")

        if args.write_per_image_csv:
            per_image_csv = reports_out / f"{label_path.stem}_report.csv"
            write_csv(rows, per_image_csv)
            print(f"[OK] CSV por imagem: {per_image_csv}")

    if args.write_csv:
        consolidated = reports_out / args.csv_name
        write_csv(all_rows, consolidated)
        print(f"[OK] CSV consolidado: {consolidated}")

    print(f"[OK] Execução concluída em: {run_dir}")


if __name__ == "__main__":
    main()
