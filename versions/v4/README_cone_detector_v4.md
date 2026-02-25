# cone_detector_v4.py

Script de detecção de cones usando YOLO ONNX da `v4` + OpenCV.

## Modelos usados

Diretório padrão: `v4/yolov26n`

- `aug/normal.onnx` (640x640)
- `aug/simplified.onnx` (640x640)
- `noaug/normal.onnx` (960x960)
- `noaug/simplified.onnx` (960x960)

## Uso rápido

Listar modelos detectados:

```bash
python3 v4/cone_detector_v4.py --source v3/CNN/dataset/0_image1.png --list-models
```

Imagem única:

```bash
python3 v4/cone_detector_v4.py \
  --source v3/CNN/dataset/0_image1.png \
  --output /tmp/cone_out.png \
  --profile balanced
```

Pasta de imagens:

```bash
python3 v4/cone_detector_v4.py \
  --source v3/CNN/dataset \
  --output /tmp/cone_dir_out \
  --profile balanced
```

Exemplo com dataset da v4:

```bash
python3 v4/cone_detector_v4.py \
  --source v4/dataset \
  --output v4/outputs_example \
  --profile balanced \
  --family aug \
  --variant simplified
```

Vídeo:

```bash
python3 v4/cone_detector_v4.py \
  --source /caminho/video.mp4 \
  --output /tmp/video_out.mp4 \
  --profile fast
```

## GUI e front-end (planejamento)

- Plano de GUI: `v4/GUI_PLAN.md`
- Prompt pronto para gerar base front-end no Lovable: `v4/LOVABLE_PROMPT.md`

Câmera:

```bash
python3 v4/cone_detector_v4.py \
  --source 0 \
  --output /tmp/cam_out.mp4 \
  --show \
  --profile fast \
  --camera-width 640 \
  --camera-height 480 \
  --buffer-size 1 \
  --drop-grabs 1
```

## Novidades recentes

- O comando agora gera `v4/outputs/<source>_detected` automaticamente quando `--output` não é fornecido, para evitar sobrescrever arquivos existentes.
- É possível limitar a região de interesse com `--roi`, forçar ou desativar o filtro HSV global (`--prefilter`, `--no-prefilter`, `--prefilter-min-ratio`) e validar cada caixa com `--min-box-orange-ratio` antes da inferência.
- Preferência de visualização/registro expandida: `--show-orange-ratio` imprime a razão de pixels laranja no rótulo, `--save-txt` grava logs simples e `--list-models` mostra todos os ONNX carregáveis.
- O modo de seguimento (`--follow`) agora calcula erro angular/distância, saturações de `v/w` com ganho P/D e grava telemetria em JSONL com `--follow-jsonl` e os parâmetros `--camera-hfov-deg/--camera-fx-px`, `--cone-height-m` etc.
- Controles de câmera (`--camera-width`, `--camera-height`, `--buffer-size`, `--drop-grabs`) e ONNX Runtime (`--threads`, `--graph-opt`, `--enable-spinning`) mantêm o pipeline rápido em hardware limitado.

## Presets

- `fast`: menor custo de CPU, usa `aug/simplified`, `det_every=2`.
- `balanced`: equilíbrio entre velocidade e qualidade, usa `aug/simplified`.
- `quality`: mais qualidade, usa `noaug/normal`.

## Parâmetros de execução

| Parâmetro | Valor padrão | O que faz |
| --- | --- | --- |
| `--source` | _--- (obrigatório)_ | Caminho para imagem, vídeo, pasta ou índice de câmera. Determina automaticamente se o output é arquivo, pasta ou captura de câmera. |
| `--output` | `outputs/<source>_detected` | Arquivo ou pasta de destino. Quando omitido, o script cria `v4/outputs/<source>_detected` para evitar sobrescrever. |
| `--model-root` | `yolov26n` | Raiz local onde o detector busca os modelos ONNX. Use para apontar para outros pesos. |
| `--list-models` | `false` | Imprime todos os modelos encontrados sob `--model-root` e encerra. |
| `--profile` | `balanced` | Define `family/variant`, thresholds e `det_every` via presets (`fast`, `balanced`, `quality`). |
| `--family` | `auto` | Força `aug` ou `noaug`. `auto` escolhe com base no preset ou disponibilidade. |
| `--variant` | `auto` | Força `simplified` ou `normal`. `auto` tenta `simplified`, depois `normal`. |
| `--conf` | depende do preset | Override do threshold de confiança (`0.25-0.35`). |
| `--iou` | depende do preset | Override do IoU utilizado no NMS. |
| `--max-det` | depende do preset | Override do máximos `detections` por frame. |
| `--det-every` | depende do preset | Rodar inferência a cada N frames (valor mais alto reduz carga). |
| `--threads` | depende do preset | `intra_op_num_threads` do ONNX Runtime (`0` = auto, `-1` desativa mudança). |
| `--graph-opt` | `all` | Nível de otimização do grafo ONNX (`basic`, `extended`, `all`). |
| `--enable-spinning` | `false` | Permite que o ONNX Runtime mantenha threads rodando entre inferências para latência menor em alguns sistemas. |
| `--prefilter` / `--no-prefilter` | preset controla | Liga/desliga o filtro HSV global. Útil para pular inferência quando não há laranja dominando o frame. |
| `--prefilter-min-ratio` | preset controla | Razão mínima de pixels laranja detectados no frame para permitir a inferência. |
| `--min-box-orange-ratio` | preset controla | Razão mínima dentro de cada caixa (validada após inferência) para reduzir falsos positivos. |
| `--roi` | `''` | ROI normalizado `x1,y1,x2,y2` ([0,1]). Limita inferência a uma sub-região do frame. |
| `--camera-width` / `--camera-height` | `0` | Solicita resolução à câmera. Quando `0`, o default do dispositivo é mantido. |
| `--buffer-size` | `1` | Ajusta `cv.CAP_PROP_BUFFERSIZE` para controlar quantos frames são capturados por loop. |
| `--drop-grabs` | `0` | Gravações extra (`cap.grab()`) antes de `read()` para reduzir latência em câmeras USB. |
| `--show` | `false` | Mostra preview em janela OpenCV. Pressione `q` para sair em vídeos/câmeras. |
| `--show-orange-ratio` | `false` | Inclui o valor calculado de razão laranja no rótulo desenhado. |
| `--save-txt` | `''` | Grava lista simples de logs/HW stats em texto. |
| `--follow` | `false` | Ativa cálculo de erro de seguimento (ângulo/distância) e comandos `v/w`. |
| `--follow-jsonl` | `''` | Caminho de saída para telemetria JSONL (inclui parâmetros e `follow_state`). |
| `--camera-hfov-deg` | `70.0` | Campo de visão horizontal usado quando `fx` não é fornecido. |
| `--camera-fx-px` | `0.0` | Foco em pixels (override do HFOV) para estimativa de distância. |
| `--camera-fy-px` | `0.0` | Foco vertical em pixels; usa `fx` se omitido. |
| `--cone-height-m` | `0.45` | Altura física do cone em metros para estimativa métrica. |
| `--target-distance-m` | `1.50` | Distância desejada ao cone. |
| `--target-box-height-ratio` | `0.18` | Altura alvo da bounding box relativa à imagem; fallback sem distância métrica. |
| `--error-ema-alpha` | `0.45` | EMA para suavizar o erro angular do seguimento. |
| `--follow-kp-ang` | `1.80` | Ganho proporcional angular. |
| `--follow-kd-ang` | `0.15` | Ganho derivativo angular. |
| `--follow-kp-dist` | `0.90` | Ganho proporcional linear. |
| `--follow-max-w` | `1.30` | Saturação da velocidade angular (`rad/s`). |
| `--follow-max-v` | `0.70` | Saturação da velocidade linear (`m/s` ou proxy). |
| `--follow-deadband-deg` | `1.0` | Deadband angular (graus) onde o controle angular é zerado. |
| `--follow-deadband-dist` | `0.05` | Deadband de distância. |
| `--follow-v-slowdown-deg` | `25.0` | Reduz `v_cmd` conforme o erro angular absoluto cresce além desse valor. |

## Erro para robô autônomo (follow)

Ative `--follow` para gerar erro de controle por frame:

- erro angular (`heading_error_deg`, `heading_error_rad`)
- erro de distância (`distance_error_m` se métrico, ou `distance_proxy_error`)
- comandos recomendados (`v_cmd`, `w_cmd`)
- erro combinado (`combined_error`)

Salvar telemetria JSONL:

```bash
python3 v4/cone_detector_v4.py \
  --source v4/dataset \
  --output v4/outputs_follow \
  --profile balanced \
  --follow \
  --follow-jsonl v4/outputs_follow/follow_log.jsonl \
  --cone-height-m 0.45 \
  --target-distance-m 1.5 \
  --camera-hfov-deg 70
```

Parâmetros de controle:

- `--follow-kp-ang`, `--follow-kd-ang`: ganhos angular P/D
- `--follow-kp-dist`: ganho linear P
- `--follow-max-v`, `--follow-max-w`: saturações
- `--follow-deadband-deg`, `--follow-deadband-dist`: zonas mortas
- `--error-ema-alpha`: suavização do erro angular

## Dica para hardware fraco

Comece com:

```bash
python3 v4/cone_detector_v4.py \
  --source 0 \
  --profile fast \
  --show \
  --camera-width 640 \
  --camera-height 480 \
  --buffer-size 1 \
  --drop-grabs 1 \
  --det-every 2
```
