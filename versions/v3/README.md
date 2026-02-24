# v3 (color tracker com experimentos CNN/ONNX)

## Visão geral
A v3 conserva todo o pipeline herdado da v2-legacy — o `start.py` aciona `cone_tracker.App`, que combina o `ColorDetector`, o `MultiConeTracker`, o `Visualizer`, o `RunCSVLogger` e o `ReasonsWriter`; o `cone_config.yaml` controla câmeras, thresholds HSV, logging e hot-reload. Além disso, v3 amplia o repositório com experimentos baseados em ONNX/CNN dentro de `CNN/` e scripts de preparação em `geracnn/`.

## Execução principal (color tracker)
1. Instale as dependências adicionais: `pip install -r requirements.txt` (adiciona `onnxruntime`, `albumentations`, `tqdm`).
2. Edite `cone_config.yaml` para apontar para o vídeo/câmera desejados e habilitar logs/mascaras/debug.
3. Rode o pipeline:

```bash
python3 start.py
```

Como na v2, o App monitora `cone_config.yaml`, recarrega quaisquer mudanças, registra FPS e grava CSVs/JSONL quando habilitado. O loop principal desenha tracks confirmadas e rejeitos em janelas OpenCV.

## Experimentos CNN/ONNX (`CNN/`)
- `CNN/best.onnx` é o peso principal usado pelas demos na pasta `CNN/`.
- `CNN/start.py` abre uma sessão ONNX Runtime (CPU) e processa a imagem/folders definidos nas constantes `MODEL_PATH`, `IMAGE_PATH` e `OUTPUT_*`. Ele extrai caixas, aplica limiar de confiança + NMS e escreve imagens anotadas com retângulos verdes.
- `CNN/start2.py` é uma versão de depuração: imprime stats do tensor de saída (`conf`, `coords`) e converte heurísticamente qualquer formato (`xyxy`, `cxcywh`, normalizado ou absoluto) em coordenadas de pixel para desenhar.
- `CNN/test.py` apenas inspeciona as entradas/saídas do ONNX para entender shapes antes de ajustar scripts de inferência.
- O diretório `CNN/dataset/` já contém imagens de referência que servem para testar os scripts acima.

## Geração de dados e validação (`geracnn/`)
- `geracnn/aug.py` usa `albumentations` + `tqdm` para criar aumentações (flip, brightness, rotate, shift/scale) de cada imagem em `dataset_raw/train/images` e copia os rótulos `yolo` para `dataset_aug/`.
- `geracnn/best.onnx` pode ser usado com os testes acima ou exportado para validação adicional.
- `geracnn/onnxtest.py` garante que o ONNX carregue com o backend OpenCV DNN.
- `geracnn/yolo.cpp` é um exemplo em C++ que lê o mesmo `best.onnx`, cria um blob 320x320 e mostra quantos outputs o modelo entrega. Pode ser útil para portabilidade fora do Python.

## Processamento em lote e depuração
- `run_batch_detection.sh` existe com a mesma intenção da v2: cria `DATASET`, conta imagens JPEG/PNG, chama `batch_detect_images.py` (copie o `batch_detect_images.py` da v2 se necessário) e produz `out_*.png`, `detection_report_*.json`/`.txt`.
- `scripts/export_sample_csv.py` está disponível dentro de `versions/v3/scripts` e funciona como na v2 para gerar logs CSV rápidos com janelas fechadas.

## Testes e dependências
- Instale: `pip install -r requirements.txt` (additions: `onnxruntime`, `albumentations`, `tqdm`).
- A suíte `pytest tests` cobre trackers, detectores, logger e integrações que também existem na v2.

## Observações
- A versão principal ainda depende da pasta `cone_tracker/`, então quaisquer melhorias nessa camada beneficiam todas as versões.
- Os scripts CNN/geracnn são independentes e podem ser usados como ponto de partida para converter pesos em ONNX ou para testar inferência offline.
