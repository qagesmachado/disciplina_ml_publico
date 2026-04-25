# Explicação sobre conversão de dados dos labels

## 1) Formato dos labels YOLO

Cada linha do `.txt`:

`class_id x_center y_center width height`

Onde:

- `class_id`: id da classe (ex.: `0` para porco).
- `x_center`, `y_center`: centro da caixa, normalizado em `[0,1]`.
- `width`, `height`: largura e altura da caixa, normalizadas em `[0,1]`.

## 2) Conversão de YOLO para pixel

Se a imagem tem largura `W` e altura `H`, então:

- `box_w = width * W`
- `box_h = height * H`
- `center_x = x_center * W`
- `center_y = y_center * H`
- `x1 = center_x - box_w / 2`
- `y1 = center_y - box_h / 2`
- `x2 = center_x + box_w / 2`
- `y2 = center_y + box_h / 2`

As coordenadas `(x1, y1, x2, y2)` são usadas para desenhar o retângulo da bounding box.