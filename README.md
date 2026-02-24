# Trekking-DragonBotZv2

    Segunda versão do robô autônomo "Root", desenvolvido para navegação outdoor em terrenos irregulares. O projeto é focado na categoria Trekking Pro (Robocore) e utiliza ROS, fusão de sensores (GPS + IMU + encoder) e controle PID para percorrer waypoints de forma precisa e confiável.


## Estrutura atual
```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   └── cleanup_generated.sh
└── versions/
    ├── v2-legacy/
    ├── v3/
    └── v4/
```

## Versões (links para os READMEs)
- [v2-legacy](versions/v2-legacy/README.md): pipeline color tracker com `cone_tracker` modular, detecção HSV+CLAHE, `MultiConeTracker`, logs CSV/TXT, scripts de batch e exportação de CSV.
- [v3](versions/v3/README.md): mantém o tracker da v2 e adiciona experimentos ONNX/CNN em `CNN/` e scripts de geração/validação em `geracnn/`.
- [v4](versions/v4/README_cone_detector_v4.md): detector YOLO ONNX (`cone_detector_v4.py`), presets `fast/balanced/quality`, telemetria `--follow`, filtros HSV pré-inferência e geração automática de outputs.

## Scripts auxiliares
- `scripts/cleanup_generated.sh`: limpa `versions/v3/BATCH_OUTPUT`, `versions/v3/CNN/outputs`, `versions/v3/geracnn/yolo` e todos os `__pycache__` em `versions/`.

## Dependências e instalação
- Instale as bibliotecas base com `pip install -r requirements.txt`.
- Cada versão também expõe seu `requirements.txt` quando precisa de extras (v3 adiciona `onnxruntime`, `albumentations`, `tqdm`).
- `versions/v2-legacy/cone_config.yaml` já contém os overrides usados por essa versão; as outras versões usam os defaults enquanto não houver um `cone_config.yaml` no próprio diretório, mas você pode criar o seu para personalizar câmeras, vídeos e thresholds.

## Testes
- Execute `pytest versions/v2-legacy/tests` ou `pytest versions/v3/tests` para validar trackers, detectores e integrações em cada versão.

## Observações
- A estrutura centralizada facilita comparar implementações e copiar experimentos para novas branches.
- Use os READMEs por versão para ver detalhes de execução, scripts auxiliares e requisitos específicos.
