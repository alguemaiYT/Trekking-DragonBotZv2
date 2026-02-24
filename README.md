# Trekking-DragonBotZv2

Repositorio reorganizado por versao, com scripts centralizados e limpeza de artefatos gerados.

## Estrutura atual

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── run_version.sh
│   └── cleanup_generated.sh
└── versions/
    ├── v2-legacy/
    │   ├── cone_tracker/
    │   ├── tests/
    │   ├── batch_detect_images.py
    │   ├── run_batch_detection.sh
    │   ├── start.py
    │   └── requirements.txt
    ├── v3/
    │   ├── cone_tracker/
    │   ├── CNN/
    │   ├── geracnn/
    │   ├── tests/
    │   ├── start.py
    │   ├── run_batch_detection.sh
    │   └── requirements.txt
    └── v4/
        └── yolov26n/
            ├── aug/
            └── noaug/
```

## Versoes

- `v2-legacy`: pipeline completo de deteccao e tracking por cor (`cone_tracker`) com execucao principal em `start.py`.
- `v3`: evolucao com modulo `cone_tracker` + experimentos ONNX/CNN em `CNN/` e `geracnn/`.
- `v4`: apenas pesos/modelos YOLO (`.pt` e `.onnx`) separados em `aug/` e `noaug/`.

## Scripts centralizados (raiz)

- `scripts/run_version.sh`: executa rapidamente cada versao.
- `scripts/cleanup_generated.sh`: remove arquivos gerados para manter repositorio limpo.

Uso:

```bash
# Executar v2 (pipeline principal)
./scripts/run_version.sh v2

# Executar batch da v2 (usa DATASET e BATCH_OUTPUT por padrao)
./scripts/run_version.sh v2-batch

# Executar v3 (pipeline principal)
./scripts/run_version.sh v3

# Executar inferencia ONNX da v3/CNN
./scripts/run_version.sh v3-cnn

# Listar modelos da v4
./scripts/run_version.sh v4-models
```

## Instalacao de dependencias

Opcao 1 (base do projeto):

```bash
pip install -r requirements.txt
```

Opcao 2 (por versao):

```bash
pip install -r versions/v2-legacy/requirements.txt
pip install -r versions/v3/requirements.txt
```

## Limpeza de artefatos

Para remover saidas geradas e binarios compilados:

```bash
./scripts/cleanup_generated.sh
```

O script remove:

- `versions/v3/BATCH_OUTPUT/`
- `versions/v3/CNN/outputs/`
- `versions/v3/geracnn/yolo`
- `__pycache__/` dentro de `versions/`

## Testes

Exemplo para rodar testes de cada versao:

```bash
pytest versions/v2-legacy/tests
pytest versions/v3/tests
```

## Observacoes

- Os diretorios antigos no topo (`Trekkingv2(legacy)`, `Trekkingv3`, `Trekkingv4`) foram consolidados em `versions/`.
- A estrutura agora separa melhor codigo, modelos e scripts operacionais.
