# Explicação detalhada do `draw_yolo_boxes_v3.py` (v3)

## O que muda em relação à v2
A v3 adiciona organização por execução (run) e relatório por imagem opcional.

## Funcionalidades principais
- Desenho de boxes (igual v1/v2).
- CSV consolidado opcional (`--write-csv`).
- CSV por imagem opcional (`--write-per-image-csv`).
- Estrutura por execução:
  - `annotated_images/`
  - `reports/`

## Organização por execução
A função `build_run_dirs` cria:
- `output_root/run_name/annotated_images`
- `output_root/run_name/reports`

Se `--run-name` não for informado, é usado timestamp.

## Fluxo
1. Lê argumentos e valida entradas.
2. Cria diretórios da execução.
3. Processa cada label:
   - encontra imagem,
   - desenha boxes,
   - coleta metadados.
4. Se ativado:
   - grava CSV por imagem,
   - grava CSV consolidado.

## Parâmetros exclusivos da v3
- `--output-root`: raiz das execuções.
- `--run-name`: nome da execução.
- `--write-per-image-csv`: CSV individual por imagem.

## Exemplo
```powershell
python "draw_yolo_boxes_v3.py" `
  --labels-dir "pig\train\labels" `
  --images-dir "pig\train\images" `
  --output-root "outputs\yolo_boxes_v3_runs" `
  --run-name "train_full" `
  --write-csv `
  --write-per-image-csv
```
