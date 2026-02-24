# v2-legacy (legacy color tracker)

## Visão geral
A versão v2-legacy traz o pipeline completo de detecção e tracking de cones baseado em segmentação por cor + validação geométrica e log detalhado. O entrypoint `start.py` inicializa o `cone_tracker.App`, que monta o `ColorDetector`, o `MultiConeTracker`, o `Visualizer`, o logger CSV e o `ReasonsWriter` e fica em loop capturando frames até o usuário fechar.

## Componentes principais
- `cone_tracker/config.py` fornece o `DEFAULT_CONFIG`, faz merge com o `cone_config.yaml` local e expõe `watch_config` para hot-reload (qualquer alteração em `cone_config.yaml` é aplicada em tempo de execução).
- `cone_tracker/detector.py` combina limiares HSV, fallback Lab, opcional `rg chromaticity` e projeção inversa se houver histórico, aplica CLAHE + morfologia e agrupa partes para validar cones.
- `cone_tracker/tracker.py` mantém vários rastros, confirma cones após várias frames, gera ids e calcula scores médias. Os resultados passam pelo `Visualizer` para desenho e pelo `RunCSVLogger` + `ReasonsWriter` para exportar logs `.csv`/`.jsonl` e relatórios de rejeições.
- `cone_tracker/app.py` também calcula FPS reais, atualiza máscaras, captura eventos de config reload e acaba liberando recursos (janela OpenCV + captura).

## Como executar
1. Instale as dependências específicas: `pip install -r requirements.txt`.
2. Ajuste `cone_config.yaml` para apontar para câmeras, vídeos, thresholds de cor, flags de debug e diretórios de log.
3. Execute o pipeline principal:

```bash
python3 start.py
```

O `App` respeita os settings em `cone_config.yaml` e reloaded automático quando o arquivo muda (via `watch_config`). Use os flags `debug` para habilitar/ desabilitar máscaras (`show_mask`), overlays (`show_windows`) e CSV/logs.

## Processamento em lote
`run_batch_detection.sh` cria os diretórios `DATASET` e `BATCH_OUTPUT` por padrão, conta imagens suportadas e chama `batch_detect_images.py` (mesmo diretório). O script de batch:

- lê todas as imagens do dataset, redimensiona para a resolução de processamento e executa detecção repetida por 1+ segundo para acumular tracker e rejeições.
- gera imagens anotadas (`out_*.png`), relatórios JSON (`detection_report_*.json`) e TXT (`detection_report_*.txt`).
- coleta rejeições únicas e suspeitos para referência manual.

Você pode inspecionar o `.txt` mais recente com `cat $(ls -t BATCH_OUTPUT/detection_report_*.txt | head -1)` após rodar.

## Depuração e testes
- Use `scripts/export_sample_csv.py` (passando vídeo) para gerar logs CSV de um número limitado de frames com janelas escondidas e exportar path via stdout.
- A suíte de testes `pytest tests` cobre trackers, detectores, logger, streamlit, integração e utilitários para validar mudanças.

## Observações
- Dependências principais estão em `requirements.txt` e incluem `numpy`, `opencv-python`, `PyYAML`, `pytest`, `streamlit` e `pyserial`.
- Os artefatos gerados (logs, imagens) devem ser limpos manualmente ou usando scripts da raiz.
