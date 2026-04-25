# Explicação detalhada do `draw_yolo_boxes_v1.py` (v1)

## Objetivo
Ler labels YOLO, encontrar a imagem correspondente, desenhar as bounding boxes e salvar a imagem anotada.

## Entrada esperada
- Label YOLO por arquivo (`--label-file`) ou por pasta (`--labels-dir`)
- Imagem correspondente via `--image-file` ou busca em `--images-dir`

## Formato YOLO
`class_id x_center y_center width height` (normalizado em `[0,1]`)

## Fluxo
1. Lê argumentos.
2. Valida cor RGB.
3. Lista labels.
4. Para cada label:
   - encontra a imagem,
   - lê as linhas YOLO,
   - converte coordenadas para pixel,
   - desenha caixas e `id:classe`,
   - salva arquivo `_boxed`.

## Conversão para pixel
- `box_w = width * W`, `box_h = height * H`
- `center_x = x_center * W`, `center_y = y_center * H`
- `x1 = center_x - box_w/2`, `y1 = center_y - box_h/2`
- `x2 = center_x + box_w/2`, `y2 = center_y + box_h/2`

## Limitações da v1
- Não gera CSV.
- Não separa outputs por execução.

## Exemplo
```powershell
python "draw_yolo_boxes_v1.py" `
  --labels-dir "pig\train\labels" `
  --images-dir "pig\train\images" `
  --output-dir "outputs\label_box_preview_v1_train"
```
