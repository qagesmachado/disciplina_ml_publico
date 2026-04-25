# Como rodar os scripts `draw_yolo_boxes_v*.py`

Este guia mostra como executar os 3 scripts criados na raiz do projeto:

- `draw_yolo_boxes_v1.py` (v1)
- `draw_yolo_boxes_v2.py` (v2)
- `draw_yolo_boxes_v3.py` (v3)

Navegação rápida:
- Índice geral: `INDEX_GERAL.md`
- Documentação detalhada por versão: `docs_draw_yolo/`
- Diagramas de fluxo: `DIAGRAMAS_MERMAID_DRAW_YOLO_BOXES.md`

Importante:
- Os scripts **apenas leem** dados da pasta `pig`.
- As saídas são gravadas em `outputs/...`.
- A pasta `pig` permanece como **source of truth**.

---

## 0) Pré-requisitos

No PowerShell, na raiz do projeto:

```powershell
cd "c:\repositories_mestrado\pig_detection_algorithm"
python -m pip install pillow
```

---

## 1) Script v1: `draw_yolo_boxes_v1.py`

### O que faz
- Lê labels YOLO (`.txt`)
- Desenha as caixas na imagem correspondente
- Salva imagem anotada na pasta de saída

### Exemplo (1 arquivo)
```powershell
python "draw_yolo_boxes_v1.py" `
  --label-file "pig\train\labels\vlcsnap-2022-01-08-18h26m24s718_png.rf.6d71b521ce80b3938d7ee6b8c0ea9760.txt" `
  --images-dir "pig\train\images" `
  --output-dir "outputs\label_box_preview_v1"
```

### Exemplo (pasta inteira)
```powershell
python "draw_yolo_boxes_v1.py" `
  --labels-dir "pig\train\labels" `
  --images-dir "pig\train\images" `
  --output-dir "outputs\label_box_preview_v1_train"
```

### Parâmetros principais
- `--label-file`: um label `.txt`
- `--labels-dir`: pasta com labels `.txt`
- `--image-file`: imagem específica (opcional)
- `--images-dir`: pasta de imagens para casar com labels
- `--output-dir`: pasta de saída
- `--line-width`: espessura da caixa (padrão `2`)
- `--color`: cor RGB (`R,G,B`), ex: `255,0,0`

---

## 2) Script v2: `draw_yolo_boxes_v2.py`

### O que faz
- Tudo da v1
- Gera CSV consolidado com:
  - coordenadas normalizadas
  - coordenadas em pixel (`x1_px`, `y1_px`, `x2_px`, `y2_px`)
  - dimensões da box e da imagem

### Exemplo (1 arquivo + CSV)
```powershell
python "draw_yolo_boxes_v2.py" `
  --label-file "pig\train\labels\vlcsnap-2022-01-08-18h26m24s718_png.rf.6d71b521ce80b3938d7ee6b8c0ea9760.txt" `
  --images-dir "pig\train\images" `
  --output-dir "outputs\label_box_preview_v2" `
  --write-csv
```

### Exemplo (pasta inteira + CSV customizado)
```powershell
python "draw_yolo_boxes_v2.py" `
  --labels-dir "pig\train\labels" `
  --images-dir "pig\train\images" `
  --output-dir "outputs\label_box_preview_v2_train" `
  --write-csv `
  --csv-name "train_boxes_report.csv"
```

### Parâmetros extras da v2
- `--write-csv`: habilita CSV consolidado
- `--csv-name`: nome do CSV consolidado

---

## 3) Script v3: `draw_yolo_boxes_v3.py`

### O que faz
- Tudo da v2
- Organiza cada execução em pasta própria:
  - `annotated_images/`
  - `reports/`
- Suporta CSV consolidado e CSV por imagem

### Exemplo (1 arquivo, execução nomeada)
```powershell
python "draw_yolo_boxes_v3.py" `
  --label-file "pig\train\labels\vlcsnap-2022-01-08-18h26m24s718_png.rf.6d71b521ce80b3938d7ee6b8c0ea9760.txt" `
  --images-dir "pig\train\images" `
  --output-root "outputs\yolo_boxes_v3_runs" `
  --run-name "single_example" `
  --write-csv `
  --write-per-image-csv
```

### Exemplo (pasta inteira)
```powershell
python "draw_yolo_boxes_v3.py" `
  --labels-dir "pig\train\labels" `
  --images-dir "pig\train\images" `
  --output-root "outputs\yolo_boxes_v3_runs" `
  --run-name "train_full" `
  --write-csv
```

### Parâmetros extras da v3
- `--output-root`: raiz das execuções
- `--run-name`: nome da execução (se vazio, usa timestamp)
- `--write-per-image-csv`: gera CSV por imagem
- `--write-csv`: gera CSV consolidado
- `--csv-name`: nome do CSV consolidado

---

## 4) Dicas rápidas de uso

- Para ver ajuda de qualquer script:
```powershell
python "draw_yolo_boxes_v3.py" --help
```

- Se quiser trocar cor/espessura:
```powershell
--color "0,255,0" --line-width 3
```

- Se ocorrer erro de correspondência entre label e imagem, informe explicitamente:
```powershell
--image-file "caminho\da\imagem.jpg"
```

---

## 5) Estrutura de saída esperada

Exemplo da v3:

- `outputs\yolo_boxes_v3_runs\train_full\annotated_images\...`
- `outputs\yolo_boxes_v3_runs\train_full\reports\boxes_report.csv`
- `outputs\yolo_boxes_v3_runs\train_full\reports\*_report.csv` (se `--write-per-image-csv`)

