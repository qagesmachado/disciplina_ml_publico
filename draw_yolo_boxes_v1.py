from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Desenha bounding boxes a partir de labels YOLO e salva imagens anotadas "
            "em um diretório de saída."
        )
    )
    parser.add_argument(
        "--label-file",
        type=Path,
        help="Caminho para um arquivo .txt de label no formato YOLO.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        help="Diretório com múltiplos arquivos .txt de labels YOLO.",
    )
    parser.add_argument(
        "--image-file",
        type=Path,
        help="Imagem correspondente ao --label-file (opcional se houver --images-dir).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        help="Diretório de imagens para localizar automaticamente a imagem correspondente.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "yolo_boxes_drawn",
        help="Diretório de saída para imagens anotadas.",
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
        help="Cor RGB da bounding box no formato 'R,G,B'. Exemplo: 255,0,0",
    )
    return parser.parse_args()


def parse_color(color_str: str) -> tuple[int, int, int]:
    # Converte texto "R,G,B" para tupla de inteiros validada.
    parts = [p.strip() for p in color_str.split(",")]
    if len(parts) != 3:
        raise ValueError("A cor deve estar no formato R,G,B.")
    rgb = tuple(int(p) for p in parts)
    if any(v < 0 or v > 255 for v in rgb):
        raise ValueError("Cada componente RGB deve estar entre 0 e 255.")
    return rgb  # type: ignore[return-value]


def iter_label_files(label_file: Path | None, labels_dir: Path | None) -> Iterable[Path]:
    # Permite processar um único arquivo ou uma pasta inteira de labels.
    if label_file:
        yield label_file
    if labels_dir:
        yield from sorted(labels_dir.glob("*.txt"))


def find_image_for_label(label_path: Path, image_file: Path | None, images_dir: Path | None) -> Path:
    if image_file:
        return image_file
    if images_dir is None:
        raise FileNotFoundError(
            "Informe --image-file ou --images-dir para localizar a imagem correspondente."
        )

    # Primeiro tenta correspondência exata por nome-base (stem).
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{label_path.stem}{ext}"
        if candidate.exists():
            return candidate

    # Fallback para datasets em que label/imagem compartilham prefixo,
    # mas diferem no hash após ".rf.".
    prefix = label_path.name.split(".rf.")[0]
    for ext in IMAGE_EXTENSIONS:
        matches = sorted(images_dir.glob(f"{prefix}*.{ext.lstrip('.')}"))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"Não foi encontrada imagem correspondente para {label_path.name} em {images_dir}"
    )


def read_yolo_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    # Lê cada linha YOLO no formato:
    # class_id x_center y_center width height (valores normalizados em [0,1]).
    labels: list[tuple[int, float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        values = line.strip().split()
        if len(values) != 5:
            continue
        class_id = int(float(values[0]))
        x_center = float(values[1])
        y_center = float(values[2])
        width = float(values[3])
        height = float(values[4])
        labels.append((class_id, x_center, y_center, width, height))
    return labels


def draw_boxes(
    image_path: Path,
    labels: list[tuple[int, float, float, float, float]],
    output_path: Path,
    color: tuple[int, int, int],
    line_width: int,
) -> None:
    with Image.open(image_path) as image:
        img = image.convert("RGB")
        draw = ImageDraw.Draw(img)
        img_w, img_h = img.size

        for class_id, x_center, y_center, width, height in labels:
            # Converte coordenadas YOLO normalizadas para pixels.
            box_w = width * img_w
            box_h = height * img_h
            center_x = x_center * img_w
            center_y = y_center * img_h

            # Canto superior esquerdo (x1, y1) e inferior direito (x2, y2).
            x1 = center_x - box_w / 2
            y1 = center_y - box_h / 2
            x2 = center_x + box_w / 2
            y2 = center_y + box_h / 2

            draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
            draw.text((x1 + 3, max(0, y1 - 12)), f"id:{class_id}", fill=color)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)


def main() -> None:
    args = parse_args()

    if not args.label_file and not args.labels_dir:
        raise ValueError("Informe pelo menos --label-file ou --labels-dir.")

    # Fluxo principal:
    # 1) resolve cor, 2) coleta labels, 3) encontra imagem, 4) desenha, 5) salva output.
    rgb_color = parse_color(args.color)
    label_files = list(iter_label_files(args.label_file, args.labels_dir))
    if not label_files:
        raise FileNotFoundError("Nenhum arquivo de label .txt foi encontrado.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for label_path in label_files:
        if not label_path.exists():
            print(f"[AVISO] Label não encontrado: {label_path}")
            continue

        image_path = find_image_for_label(label_path, args.image_file, args.images_dir)
        labels = read_yolo_labels(label_path)
        output_name = f"{label_path.stem}_boxed{image_path.suffix.lower()}"
        output_path = args.output_dir / output_name

        draw_boxes(
            image_path=image_path,
            labels=labels,
            output_path=output_path,
            color=rgb_color,
            line_width=args.line_width,
        )
        print(f"[OK] Gerado: {output_path}")


if __name__ == "__main__":
    main()
