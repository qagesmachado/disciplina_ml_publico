from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Versão 2: desenha bounding boxes YOLO e exporta CSV com coordenadas "
            "normalizadas e em pixels."
        )
    )
    parser.add_argument("--label-file", type=Path, help="Arquivo .txt de label YOLO.")
    parser.add_argument("--labels-dir", type=Path, help="Diretório com labels YOLO.")
    parser.add_argument(
        "--image-file",
        type=Path,
        help="Imagem correspondente ao --label-file (opcional se houver --images-dir).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        help="Diretório de imagens para localizar automaticamente as correspondências.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "yolo_boxes_drawn_v2",
        help="Diretório de saída para imagens anotadas e CSV.",
    )
    parser.add_argument(
        "--line-width", type=int, default=2, help="Espessura da linha da bounding box."
    )
    parser.add_argument(
        "--color",
        type=str,
        default="255,0,0",
        help="Cor RGB no formato R,G,B. Exemplo: 255,0,0",
    )
    parser.add_argument(
        "--write-csv",
        action="store_true",
        help="Se informado, gera CSV por imagem com coordenadas em pixel.",
    )
    parser.add_argument(
        "--csv-name",
        type=str,
        default="boxes_report.csv",
        help="Nome do CSV consolidado (quando --write-csv for usado).",
    )
    return parser.parse_args()


def parse_color(color_str: str) -> tuple[int, int, int]:
    # Converte texto "R,G,B" para tupla de inteiros validada.
    parts = [p.strip() for p in color_str.split(",")]
    if len(parts) != 3:
        raise ValueError("A cor deve estar no formato R,G,B.")
    rgb = tuple(int(p) for p in parts)
    if any(v < 0 or v > 255 for v in rgb):
        raise ValueError("Cada valor RGB deve estar entre 0 e 255.")
    return rgb  # type: ignore[return-value]


def iter_label_files(label_file: Path | None, labels_dir: Path | None) -> Iterable[Path]:
    # Permite processar um único label ou um diretório inteiro.
    if label_file:
        yield label_file
    if labels_dir:
        yield from sorted(labels_dir.glob("*.txt"))


def find_image_for_label(label_path: Path, image_file: Path | None, images_dir: Path | None) -> Path:
    if image_file:
        return image_file
    if images_dir is None:
        raise FileNotFoundError("Informe --image-file ou --images-dir.")

    # Primeiro tenta casar exatamente pelo stem do label.
    for ext in IMAGE_EXTENSIONS:
        exact = images_dir / f"{label_path.stem}{ext}"
        if exact.exists():
            return exact

    # Fallback por prefixo anterior ao ".rf." para datasets com hash diferente.
    prefix = label_path.name.split(".rf.")[0]
    for ext in IMAGE_EXTENSIONS:
        matches = sorted(images_dir.glob(f"{prefix}*.{ext.lstrip('.')}"))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"Imagem correspondente não encontrada para {label_path.name}.")


def read_yolo_label_rows(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    # Lê linhas YOLO no formato:
    # class_id x_center y_center width height (normalizado em [0,1]).
    rows: list[tuple[int, float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        values = line.strip().split()
        if len(values) != 5:
            continue
        rows.append(
            (
                int(float(values[0])),
                float(values[1]),
                float(values[2]),
                float(values[3]),
                float(values[4]),
            )
        )
    return rows


def yolo_to_pixel_box(
    x_center: float, y_center: float, width: float, height: float, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    # Converte coordenadas YOLO normalizadas para coordenadas em pixel.
    box_w = width * img_w
    box_h = height * img_h
    cx = x_center * img_w
    cy = y_center * img_h
    x1 = cx - box_w / 2
    y1 = cy - box_h / 2
    x2 = cx + box_w / 2
    y2 = cy + box_h / 2
    return x1, y1, x2, y2


def draw_and_collect(
    image_path: Path,
    label_rows: list[tuple[int, float, float, float, float]],
    output_path: Path,
    color: tuple[int, int, int],
    line_width: int,
) -> list[dict]:
    with Image.open(image_path) as image:
        img = image.convert("RGB")
        draw = ImageDraw.Draw(img)
        img_w, img_h = img.size
        report_rows: list[dict] = []

        for idx, (class_id, x_center, y_center, width, height) in enumerate(label_rows, start=1):
            # Desenha box e também acumula dados para relatório CSV.
            x1, y1, x2, y2 = yolo_to_pixel_box(
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
                img_w=img_w,
                img_h=img_h,
            )
            draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
            draw.text((x1 + 3, max(0, y1 - 12)), f"id:{class_id}", fill=color)

            report_rows.append(
                {
                    "image_name": image_path.name,
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
                    "box_width_px": round(x2 - x1, 2),
                    "box_height_px": round(y2 - y1, 2),
                    "image_width_px": img_w,
                    "image_height_px": img_h,
                }
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)

    return report_rows


def write_csv(rows: list[dict], csv_path: Path) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.label_file and not args.labels_dir:
        raise ValueError("Informe --label-file ou --labels-dir.")

    # Fluxo principal:
    # 1) parse args/cor, 2) varre labels, 3) desenha boxes, 4) opcionalmente grava CSV.
    color = parse_color(args.color)
    label_files = list(iter_label_files(args.label_file, args.labels_dir))
    if not label_files:
        raise FileNotFoundError("Nenhum label .txt encontrado.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_report_rows: list[dict] = []

    for label_path in label_files:
        if not label_path.exists():
            print(f"[AVISO] Label não encontrado: {label_path}")
            continue

        image_path = find_image_for_label(label_path, args.image_file, args.images_dir)
        label_rows = read_yolo_label_rows(label_path)
        output_name = f"{label_path.stem}_boxed{image_path.suffix.lower()}"
        output_image_path = args.output_dir / output_name

        report_rows = draw_and_collect(
            image_path=image_path,
            label_rows=label_rows,
            output_path=output_image_path,
            color=color,
            line_width=args.line_width,
        )
        all_report_rows.extend(report_rows)
        print(f"[OK] Imagem anotada: {output_image_path}")

    if args.write_csv:
        csv_path = args.output_dir / args.csv_name
        write_csv(all_report_rows, csv_path)
        print(f"[OK] CSV gerado: {csv_path}")


if __name__ == "__main__":
    main()
