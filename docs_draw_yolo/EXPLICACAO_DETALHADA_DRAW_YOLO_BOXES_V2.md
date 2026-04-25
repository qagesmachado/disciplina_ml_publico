# Explicação detalhada do `draw_yolo_boxes_v2.py` (v2)

## O que muda em relação à v1
A v2 mantém o desenho de caixas e adiciona relatório tabular em CSV.

## Funcionalidades principais
- Desenha bounding boxes como na v1.
- Coleta metadados por caixa:
  - coordenadas normalizadas,
  - coordenadas em pixel (`x1_px`, `y1_px`, `x2_px`, `y2_px`),
  - dimensões da box e da imagem.
- Salva CSV consolidado quando `--write-csv` está ativo.

## Fluxo
1. Lê argumentos e valida entradas.
2. Lista labels a processar.
3. Para cada label:
   - encontra imagem,
   - desenha boxes,
   - acumula linhas para relatório.
4. Se `--write-csv`, grava CSV consolidado.

## Parâmetros novos da v2
- `--write-csv`: habilita gravação do CSV.
- `--csv-name`: nome do CSV de saída.

## Estrutura de saída típica
- Imagens anotadas: `outputs/.../*.jpg`
- CSV consolidado: `outputs/.../boxes_report.csv`

## Exemplo
```powershell
python "draw_yolo_boxes_v2.py" `
  --labels-dir "pig\train\labels" `
  --images-dir "pig\train\images" `
  --output-dir "outputs\label_box_preview_v2_train" `
  --write-csv `
  --csv-name "train_boxes_report.csv"
```
