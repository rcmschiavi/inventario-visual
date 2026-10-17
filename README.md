# inventario-visual

Servidor local que recebe **vídeo ao vivo da câmera do celular** (pelo navegador,
sem app nativo) via **WebRTC** e salva automaticamente **frames-chave**, ou seja,
os momentos em que a câmera está parada e a imagem está nítida. Esses frames
ficam em disco para o processamento posterior (YOLO-World + modelo de
visão-linguagem), que se conecta por uma interface plugável: o YOLO-World
encontra e conta os objetos, e um modelo de visão-linguagem (Qwen3-VL) preenche
uma ficha para cada um (nome, marca, modelo, estado, texto visível), gerando um
`inventario.csv`.

```
 Celular (Chrome/Safari)                       PC com Linux (este projeto)
┌──────────────────────┐   HTTPS (página+SDP)  ┌──────────────────────────────────────────┐
│ getUserMedia         │ ───────────────────▶  │ FastAPI  ── /api/offer ──▶ aiortc         │
│ (câmera traseira)    │                       │                              │            │
│ RTCPeerConnection    │ ══ WebRTC (vídeo) ══▶ │  amostragem ▶ KeyframeSelector            │
│ DataChannel ◀────────┼── eventos/status ──── │   (movimento + nitidez + novidade)        │
└──────────────────────┘                       │          │                                │
                                               │          ▼                                │
                                               │  data/keyframes/<sessão>/*.jpg + .jsonl   │
                                               │          │                                │
                                               │          ▼                                │
                                               │  ProcessorPipeline ▶ FrameProcessor(s)    │
                                               │  (YOLO-World -> VLM -> inventario.csv)    │
                                               └──────────────────────────────────────────┘
```

## Estrutura

```
inventario-visual/
├── pyproject.toml / requirements*.txt
├── .env.example                 # todos os parâmetros configuráveis
├── config/
│   └── classes.txt              # o que o YOLO-World deve procurar
├── scripts/
│   └── gerar_certificado.py     # mkcert, com fallback para CA própria
├── deploy/
│   └── inventario-visual.service  # modelo de serviço systemd
├── src/inventario_visual/
│   ├── __main__.py              # `inventario-visual` (sobe o servidor HTTPS)
│   ├── app.py                   # rotas FastAPI
│   ├── webrtc.py                # sessões aiortc, consumo do vídeo
│   ├── keyframes.py             # seletor de frames-chave (puro, sem WebRTC)
│   ├── storage.py               # gravação JPEG + metadados JSONL
│   ├── processors.py            # FrameProcessor (ABC) e pipeline assíncrono
│   ├── vision/                  # YOLO-World (detecção) e VLM (descrição dos itens)
│   ├── offline.py               # `inventario-offline`: seletor (+ processadores) em um vídeo
│   ├── config.py                # Settings (variáveis INVENTARIO_*)
│   └── static/                  # index.html, app.js, style.css (JS puro)
└── tests/                       # pytest com imagens e vídeos sintéticos
```

---

## 1. Instalação no Linux (com GPU NVIDIA)

> Os comandos abaixo são para **Ubuntu/Debian**; em outras distribuições, troque
> o `apt` pelo gerenciador de pacotes equivalente.
>
> O **streaming e o seletor de frames rodam na CPU** e funcionam em qualquer
> máquina. A GPU NVIDIA só será usada pela etapa de visão (YOLO-World + VLM),
> que será plugada depois. Mesmo assim, já deixamos o ambiente pronto para ela.
> Uma máquina sem monitor (servidor headless) funciona normalmente.

### 1.1 Pré-requisitos

1. **uv** (recomendado; ele mesmo baixa o Python 3.11+ se o do sistema for antigo):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Driver NVIDIA**. Confira com:
   ```bash
   nvidia-smi
   ```
   Se o comando não existir, instale o driver recomendado e reinicie:
   ```bash
   sudo ubuntu-drivers install && sudo reboot
   ```
   Não é preciso instalar o CUDA Toolkit: o PyTorch já traz as bibliotecas CUDA,
   só o driver é necessário.
3. **mkcert** (opcional, mas recomendado):
   ```bash
   sudo apt install mkcert libnss3-tools
   ```
   Sem o mkcert, o script gera uma CA própria usando a biblioteca `cryptography`.

### 1.2 Dependências do projeto

Com **uv**:

```bash
git clone https://github.com/rcmschiavi/inventario-visual.git
cd inventario-visual
uv sync            # cria .venv e instala tudo (inclui pytest/mypy do grupo dev)
```

Ou com **pip**:

```bash
sudo apt install python3-venv     # Python 3.11+ (Ubuntu 24.04 / Debian 12 já têm)
cd inventario-visual
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

> O `aiortc`, o `av` (PyAV) e o `opencv-python-headless` têm wheels prontos para
> Linux: não é preciso compilar nada, nem instalar FFmpeg ou bibliotecas gráficas.

### 1.3 (Para a etapa de visão) PyTorch com CUDA

Ainda não é necessário para o streaming, mas quando for plugar o YOLO-World:

```bash
uv add torch torchvision      # no Linux, o pacote do PyPI já vem com CUDA
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Deve imprimir `True` e o nome da sua placa. Use `uv add` (e não `uv pip install`)
para o pacote ficar registrado no `pyproject.toml`; senão o próximo `uv sync`
o remove.

---

## 2. HTTPS na rede local (obrigatório)

Os navegadores **só liberam a câmera (`getUserMedia`) em contexto seguro**:
`https://` ou `http://localhost`. Como o celular acessa o PC pelo IP da rede
(ex.: `https://192.168.0.50:8443`), precisamos de um certificado válido para
esse IP.

### 2.1 Gerar o certificado

```bash
uv run python scripts/gerar_certificado.py
```

O script detecta os IPs do PC e gera em `certs/`:

| Arquivo | Uso |
|---|---|
| `cert.pem`, `key.pem` | certificado do servidor (usado pelo uvicorn) |
| `rootCA.crt` / `rootCA.pem` | **CA a instalar no celular** |
| `rootCA-key.pem` | chave da CA (só no fallback) — **secreta, não compartilhe** |

- Se o IP detectado estiver errado (VPN, várias placas de rede), informe-o:
  `uv run python scripts/gerar_certificado.py --ip 192.168.0.50`
- **Se o IP do PC mudar, rode o script de novo.** A CA é reaproveitada, então
  não é preciso reinstalá-la no celular. Dica: reserve um IP fixo para o PC no
  roteador (reserva DHCP).
- Para forçar o fallback mesmo com mkcert instalado: `--sem-mkcert`.

Descubra o IP do PC com `hostname -I` ou `ip -4 addr` (interface Wi-Fi/Ethernet, ex.: `192.168.0.50`).

### 2.2 Instalar a CA no Android

1. Suba o servidor (seção 3) e, no Chrome do celular, abra
   `https://IP-DO-PC:8443/ca.crt`. Como a CA ainda não é confiável, o Chrome
   mostrará um aviso: toque em **Avançado → Continuar para … (não seguro)**.
   O arquivo `inventario-visual-ca.crt` será baixado.
   (Alternativa: copie `certs/rootCA.crt` para o celular por cabo/Drive.)
2. Abra **Configurações → Segurança e privacidade → Mais configurações de
   segurança → Criptografia e credenciais → Instalar um certificado →
   Certificado de CA** (o caminho varia por fabricante; procure por
   "certificado" na busca das Configurações).
3. Confirme o aviso **Instalar mesmo assim** e selecione o arquivo baixado.
4. Feche e reabra o Chrome. O cadeado deve aparecer sem avisos.

> O Chrome para Android confia em CAs instaladas pelo usuário. Se preferir não
> instalar nada, dá para simplesmente aceitar o aviso de segurança
> (*Avançado → Continuar*) a cada acesso: a câmera funciona do mesmo jeito.

### 2.3 Instalar a CA no iPhone / iPad

É preciso usar o **Safari** para baixar o perfil.

1. No Safari, abra `https://IP-DO-PC:8443/ca.crt`. No aviso de certificado,
   toque em **Mostrar Detalhes → visitar este site**. Toque em **Permitir**
   para baixar o perfil de configuração.
2. Abra **Ajustes → Geral → VPN e Gerenciamento de Dispositivos** (ou toque em
   *Perfil Transferido* no topo dos Ajustes), selecione
   *inventario-visual CA* (ou *mkcert …*) e toque em **Instalar**.
3. **Passo que todos esquecem:** vá em **Ajustes → Geral → Sobre → Ajustes de
   Confiança de Certificados** e **ative a confiança total** para a CA.
4. Feche o Safari (deslize para fechar a aba) e acesse `https://IP-DO-PC:8443/`.

> Requer iOS 14.3+ para WebRTC com câmera fora do Safari; no Safari funciona
> desde o iOS 11.

**Segurança:** quem tiver a chave da CA (`rootCA-key.pem` ou a CA do mkcert em
`mkcert -CAROOT`) consegue emitir certificados que o seu celular aceitará.
Mantenha-a só no PC e remova o perfil/certificado do celular quando não
precisar mais.

---

## 3. Rodando o servidor

```bash
uv run inventario-visual
#   ou:  uv run python -m inventario_visual --port 8443
```

O terminal mostra o endereço a abrir no celular, por exemplo:

```
  Abra no celular:  https://192.168.0.50:8443/
```

### 3.1 Firewall

O celular precisa alcançar o PC em **TCP 8443** (página) e em **portas UDP
dinâmicas** (mídia WebRTC; o aiortc escolhe portas aleatórias). Se o firewall
estiver desligado (padrão no Ubuntu desktop), não há nada a fazer. Confira com
`sudo ufw status`.

Com **ufw** (Ubuntu), troque `192.168.0.0/24` pela faixa da sua rede:

```bash
sudo ufw allow 8443/tcp
sudo ufw allow from 192.168.0.0/24 proto udp   # mídia WebRTC, só da rede local
```

Com **firewalld** (Fedora, RHEL…):

```bash
sudo firewall-cmd --permanent --add-port=8443/tcp
sudo firewall-cmd --permanent --zone=trusted --add-source=192.168.0.0/24
sudo firewall-cmd --reload
```

### 3.2 No celular

1. Conecte o celular **na mesma rede Wi-Fi** do PC (redes de *convidados* e
   roteadores com *isolamento de AP/cliente* bloqueiam a comunicação entre
   aparelhos — desative ou use a rede principal).
2. Abra `https://IP-DO-PC:8443/` no Chrome (Android) ou Safari (iPhone).
3. Toque em **Iniciar câmera** e permita o acesso. A câmera traseira é
   solicitada com `facingMode: "environment"`.
4. Aponte para os itens e **segure parado ~1 segundo** em cada um. Quando um
   frame-chave é salvo, a tela pisca, o celular vibra (Android) e o contador no
   canto aumenta. A barra inferior mostra o que o seletor está vendo:
   *Em movimento*, *Estabilizando*, *Desfocado*, *Cena já capturada*…

### 3.3 Rodar como serviço (systemd)

Para o servidor subir sozinho com a máquina, use o modelo em
`deploy/inventario-visual.service`:

```bash
# ajuste User= e os caminhos no arquivo antes de copiar
sudo cp deploy/inventario-visual.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now inventario-visual
journalctl -u inventario-visual -f        # acompanhar os logs
```

O serviço lê o `.env` da pasta do projeto, como na execução manual.

### 3.4 Onde ficam os frames

```
data/keyframes/
  20261017_143005_ab12cd/                 # uma pasta por sessão de câmera
    20261017_173012_481233_0001.jpg       # timestamp UTC + sequência
    20261017_173015_102344_0002.jpg
    keyframes.jsonl                       # metadados, uma linha por frame
```

Cada linha do `keyframes.jsonl`:

```json
{"index": 1, "file": "20261017_173012_481233_0001.jpg", "captured_at": "2026-10-17T17:30:12.481233+00:00",
 "width": 1920, "height": 1080, "sharpness": 412.7, "motion": 1.83, "novelty": null}
```

Estado ao vivo: `https://IP-DO-PC:8443/api/status`.

---

## 4. Seletor de frames-chave

Para cada frame **amostrado** (`SAMPLE_FPS` por segundo; os demais são
descartados antes mesmo da conversão de cor):

1. Converte para cinza e reduz para `ANALYSIS_WIDTH` px de largura.
2. **Movimento** = diferença absoluta média (0–255) em relação à amostra
   anterior, após um leve desfoque para ignorar ruído do sensor.
3. **Nitidez** = variância do Laplaciano.
4. Após `STABLE_SAMPLES` amostras seguidas com movimento ≤ `MOTION_THRESHOLD`,
   pega o **frame mais nítido dessa janela** e o salva se:
   - nitidez ≥ `SHARPNESS_THRESHOLD`;
   - passou `MIN_KEYFRAME_INTERVAL_S` desde o último frame-chave;
   - difere do último frame-chave em ≥ `NOVELTY_THRESHOLD` (evita salvar a
     mesma prateleira várias vezes enquanto você continua parado nela).

O frame salvo é o **original em resolução cheia**, não a versão reduzida.

### Parâmetros

Defina no ambiente ou num arquivo `.env` (veja `.env.example`), sempre com o
prefixo `INVENTARIO_`:

| Variável | Padrão | Efeito |
|---|---|---|
| `SAMPLE_FPS` | `5` | amostras analisadas por segundo |
| `ANALYSIS_WIDTH` | `640` | largura da imagem usada nas métricas |
| `MOTION_THRESHOLD` | `6.0` | ↑ aceita mais tremor da mão |
| `STABLE_SAMPLES` | `3` | ↑ exige ficar parado por mais tempo (3 @ 5 fps ≈ 0,6 s) |
| `SHARPNESS_THRESHOLD` | `80.0` | ↑ mais exigente com foco; depende de `ANALYSIS_WIDTH` e da cena |
| `NOVELTY_THRESHOLD` | `12.0` | ↑ exige cenas mais diferentes entre frames-chave |
| `MIN_KEYFRAME_INTERVAL_S` | `1.0` | intervalo mínimo entre frames-chave |
| `JPEG_QUALITY` | `92` | qualidade do JPEG salvo |
| `OUTPUT_DIR` | `data/keyframes` | pasta de saída |
| `PORT` / `HOST` | `8443` / `0.0.0.0` | servidor |
| `PROCESSORS` | `inventario_visual.processors:LogProcessor` | processadores plugados |

### Calibração

Os valores de nitidez dependem muito da cena e da câmera. Grave um vídeo
curto com o celular passando pelos itens e rode:

```bash
uv run inventario-offline meu_video.mp4 -v --saida data/calibracao
```

A saída lista movimento e nitidez de cada amostra e a decisão tomada; ajuste
`--motion` / `--sharpness` até os frames salvos ficarem bons e então leve os
valores para o `.env`. A barra inferior da página no celular também mostra
`mov=` e `nitidez=` em tempo real.

---

## 5. Detecção de objetos (YOLO-World)

Cada frame-chave pode passar pelo **YOLO-World**, um detector de *vocabulário
aberto*: em vez de um conjunto fixo de classes, você escreve a lista de objetos
que quer encontrar. Para cada frame ele marca onde está cada objeto, conta
quantos há de cada tipo e recorta cada um (os recortes serão a entrada do
modelo de visão-linguagem, na próxima etapa).

### 5.1 Instalar

```bash
uv sync --extra visao      # ultralytics + PyTorch (com CUDA) + CLIP; ~3 GB
#   ou:  pip install -r requirements-visao.txt
```

Na primeira execução são baixados automaticamente:
- os pesos do modelo (`models/yolov8l-worldv2.pt`, ~90 MB);
- o CLIP (`~/.cache/clip`, ~340 MB), usado para entender os nomes das classes.

### 5.2 Ativar

No `.env`:

```
INVENTARIO_PROCESSORS=inventario_visual.processors:LogProcessor,inventario_visual.vision:YoloWorldProcessor
```

Ao subir o servidor, o modelo é carregado na GPU antes de aceitar frames
(aparece `YOLO-World carregado (..., dispositivo=cuda:0)` no log). Sem GPU ele
roda na CPU, mais devagar; os frames-chave enfileiram, mas o vídeo não trava.

### 5.3 Escolher o que detectar: `config/classes.txt`

```
# rótulo no catálogo: prompt em inglês[, outro prompt]
monitor: computer monitor
cadeira: chair, office chair
furadeira: power drill, cordless drill
```

- O **rótulo** (antes do `:`) é o que aparece nos resultados, em português.
- Os **prompts** (depois do `:`) vão para o modelo. Use **inglês** e termos
  concretos e visuais: o modelo entende inglês bem melhor.
- Vários prompts por rótulo ajudam; todos viram o mesmo rótulo na saída.
- Uma lista **enxuta** com só o que você quer catalogar dá menos confusão.
- O arquivo inicial tem ~45 itens de escritório, casa e oficina; edite-o à
  vontade e reinicie o servidor.

### 5.4 O que é gerado

```
data/keyframes/<sessão>/
  20261017_173012_481233_0001.jpg
  detections.jsonl                              # uma linha por frame-chave
  crops/20261017_173012_481233_0001_01_monitor.jpg
  crops/20261017_173012_481233_0001_02_cadeira.jpg
  annotated/20261017_173012_481233_0001.jpg     # frame com as caixas desenhadas
```

```json
{"index": 1, "file": "20261017_173012_481233_0001.jpg", "captured_at": "...",
 "counts": {"cadeira": 2, "monitor": 1},
 "detections": [{"label": "monitor", "prompt": "computer monitor", "confidence": 0.81,
                 "bbox": [10, 20, 210, 170], "crop_file": "crops/..._01_monitor.jpg"}, ...]}
```

No celular, abaixo das métricas, aparece o resumo de cada frame-chave
(ex.: `#3: 2× cadeira, 1× monitor`).

### 5.5 Parâmetros (`INVENTARIO_YOLO_*`)

| Variável | Padrão | Efeito |
|---|---|---|
| `MODEL` | `yolov8l-worldv2.pt` | `s` / `m` / `l` / `x`: maior = mais preciso e mais lento |
| `MODELS_DIR` | `models` | onde os pesos são guardados |
| `CLASSES_FILE` | `config/classes.txt` | vazio = 80 classes padrão do COCO, em inglês |
| `DEVICE` | `auto` | `auto` (GPU se houver), `cpu`, `cuda:0` |
| `CONF` | `0.15` | confiança mínima; ↑ menos falsos positivos, ↓ detecta mais |
| `IOU` | `0.5` | sobreposição para fundir caixas duplicadas |
| `IMGSZ` | `640` | resolução da inferência; `1280` ajuda com objetos pequenos |
| `HALF` | `true` | FP16 na GPU (mais rápido, mesma qualidade na prática) |
| `AGNOSTIC_NMS` | `true` | funde caixas sobrepostas mesmo de rótulos diferentes |
| `SAVE_CROPS` / `SAVE_ANNOTATED` | `true` | grava recortes / imagem anotada |

### 5.6 Testar sem celular

Grave um vídeo do ambiente e rode a mesma cadeia (seletor + detecção):

```bash
uv run inventario-offline video.mp4 --processar
# resultados em data/offline/video/  (abra annotated/ para conferir)
```

Use isso para ajustar a lista de classes e o `CONF` antes de ir a campo.

### 5.7 Limitações conhecidas

- **Contagem por frame, não por item único:** o mesmo objeto visto em dois
  frames-chave é contado duas vezes. Juntar as detecções num inventário sem
  duplicatas é uma etapa futura.
- Não distingue **marca/modelo** nem lê texto: isso fica para o modelo de
  visão-linguagem.
- Objetos pequenos, distantes, transparentes ou parcialmente escondidos têm
  confiança menor; aproxime a câmera ou aumente `IMGSZ`.

### 5.8 Criando outros processadores

O streaming não conhece a visão: ele só entrega cada frame-chave a uma lista de
`FrameProcessor`, que roda numa **thread de trabalho separada**, em ordem, com
fila limitada. Os processadores rodam na ordem de `INVENTARIO_PROCESSORS`, e
cada um enxerga o que os anteriores deixaram em `keyframe.extras`. Por exemplo,
as detecções do YOLO-World ficam em `keyframe.extras["detections"]`.

```python
# meu_pacote/descricao.py
from inventario_visual import FrameProcessor, Keyframe


class DescreveItens(FrameProcessor):
    name = "vlm"

    def setup(self) -> None:
        ...  # carregue o modelo aqui (chamado uma vez)

    def process(self, keyframe: Keyframe) -> None:
        for det in keyframe.extras.get("detections", []):
            x1, y1, x2, y2 = det.bbox
            recorte = keyframe.image[y1:y2, x1:x2]
            ...  # descrever o recorte
        keyframe.send_event({"type": "info", "texto": "..."})  # opcional: avisa o celular
```

```
INVENTARIO_PROCESSORS=inventario_visual.vision:YoloWorldProcessor,meu_pacote.descricao:DescreveItens
```

O `Keyframe` traz: `session_id`, `index`, `captured_at`, `image` (BGR, numpy),
`path` (JPEG salvo), `sharpness`, `motion`, `novelty` e `extras` (dict livre).
Erros de um processador são registrados no log e não afetam os demais nem o
streaming. Se a visão ficar para trás, os frames mais antigos da fila são
descartados (continuam salvos em disco para reprocessar depois).

---

## 6. Descrição dos itens (modelo de visão-linguagem)

O YOLO-World diz *o que* e *onde* ("cadeira", nesta caixa). O modelo de
visão-linguagem (VLM) olha cada recorte e preenche a **ficha do item**: nome,
categoria, marca, modelo, cor, estado, texto visível (etiquetas, números de
série) e se o recorte realmente é o que o detector disse (`confere`), o que
ajuda a descartar falsos positivos.

### 6.1 Modelo escolhido: Qwen3-VL-4B em 4 bits

Para uma **GPU de 6 GB**, o padrão é o
[`Qwen/Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
quantizado em 4 bits (NF4, via bitsandbytes):

- ~3 GB de VRAM, deixando espaço para o YOLO-World na mesma placa;
- bom em ler texto (OCR) e em responder em português;
- segue bem a instrução de responder em JSON.

Alternativas, trocando só `INVENTARIO_VLM_MODEL`:

| Modelo | VRAM aprox. (4 bits) | Quando usar |
|---|---|---|
| `Qwen/Qwen3-VL-4B-Instruct` | ~3 GB | **padrão** |
| `Qwen/Qwen3-VL-2B-Instruct` | ~1,5 GB | se faltar memória ou precisar de mais velocidade (lê texto pior) |
| `Qwen/Qwen3.5-4B` | ~3 GB | mais novo e melhor em raciocínio nos benchmarks; não testado neste projeto, vale comparar |

> Os números de VRAM são estimativas; o log mostra o uso real
> (`VLM carregado: X GB de VRAM`).

### 6.2 Instalar

```bash
uv sync --extra visao --extra vlm
#   ou:  pip install -r requirements-vlm.txt
```

Na primeira execução o modelo (~8 GB de download, ~3 GB na GPU depois de
quantizado) vai para `~/.cache/huggingface`. Para outro disco, use
`INVENTARIO_VLM_CACHE_DIR`.

### 6.3 Como usar: em lote (recomendado para 6 GB) ou ao vivo

O VLM leva alguns segundos por item. Por isso há dois modos.

**Em lote (recomendado):** filme com só o YOLO-World ao vivo (resposta rápida no
celular) e descreva tudo depois:

```bash
# .env
INVENTARIO_PROCESSORS=inventario_visual.vision:YoloWorldProcessor

# depois de filmar:
uv run inventario-descrever data/keyframes/20261017_143005_ab12cd
```

O lote pode ser interrompido (Ctrl+C) e rodado de novo: ele **retoma** de onde
parou. `--refazer` apaga as descrições e começa do zero (útil depois de mudar o
prompt ou o modelo). Também aceita várias sessões de uma vez.

**Ao vivo:** o VLM roda logo depois do YOLO-World, e o celular mostra os nomes
dos itens (ex.: `#3: Monitor Dell; Cadeira (?)`; o `(?)` indica que o VLM
discordou do detector):

```bash
INVENTARIO_PROCESSORS=inventario_visual.vision:YoloWorldProcessor,inventario_visual.vision:VlmProcessor
```

Funciona bem andando devagar. Se você gerar frames-chave mais rápido do que o
VLM descreve, a fila enche e os frames mais antigos ficam sem descrição. Nesse
caso, rode o `inventario-descrever` depois na mesma sessão: ele completa só o
que faltou.

### 6.4 O que é gerado

```
data/keyframes/<sessão>/
  descriptions.jsonl     # registro completo, incluindo a resposta bruta do modelo
  inventario.csv         # uma linha por item: abre direto no LibreOffice/Excel
```

Colunas do `inventario.csv`: `frame`, `indice_frame`, `capturado_em`,
`rotulo_detector`, `confianca_detector`, `confere`, `nome`, `categoria`,
`marca`, `modelo`, `cor`, `estado`, `texto_visivel`, `descricao`, `erro`,
`recorte`, `bbox`.

Se o modelo responder algo que não é JSON, a linha é gravada mesmo assim, com
a coluna `erro` preenchida e a resposta bruta no `descriptions.jsonl`.

### 6.5 Parâmetros (`INVENTARIO_VLM_*`)

| Variável | Padrão | Efeito |
|---|---|---|
| `MODEL` | `Qwen/Qwen3-VL-4B-Instruct` | qualquer modelo compatível com `AutoModelForImageTextToText` |
| `QUANTIZATION` | `4bit` | `4bit`, `8bit` ou `none` (só GPU; na CPU roda sem quantizar) |
| `DEVICE` | `auto` | `auto`, `cuda:0`, `cpu` |
| `MAX_SIDE` | `768` | lado maior do recorte enviado: ↑ lê textos menores, usa mais VRAM |
| `MIN_CROP_PX` | `40` | ignora recortes menores que isso |
| `MAX_ITEMS` | `6` | máximo de itens descritos por frame-chave (maior confiança primeiro) |
| `MAX_NEW_TOKENS` | `200` | tamanho máximo da resposta |
| `PROMPT_FILE` | (vazio) | prompt próprio; use `{rotulo}` onde entra o rótulo detectado |
| `CACHE_DIR` | (vazio) | pasta dos pesos (padrão: `~/.cache/huggingface`) |

### 6.6 Se faltar memória na GPU (CUDA out of memory)

Em ordem, do que menos afeta a qualidade para o que mais afeta:

1. Use o **modo em lote**: o VLM roda sozinho, sem disputar a GPU com o vídeo.
2. `INVENTARIO_VLM_MAX_SIDE=512`.
3. `INVENTARIO_YOLO_MODEL=yolov8m-worldv2.pt` (detector menor).
4. `INVENTARIO_VLM_MODEL=Qwen/Qwen3-VL-2B-Instruct`.

Feche outros programas que usam a GPU (navegador com aceleração, jogos) e
confira o consumo com `nvidia-smi`.

---

## 7. Testes

```bash
uv run pytest
uv run mypy src        # checagem de tipos (modo estrito)
```

Os testes usam **imagens e vídeos sintéticos** (texturas aleatórias, desfoque
gaussiano, translações simulando a câmera andando, ruído de sensor) para
verificar: métricas de nitidez/movimento, emissão única em cena parada,
rejeição de movimento e de imagens borradas, escolha do frame mais nítido,
duplicatas, intervalo mínimo, amostragem, vídeo MJPG gravado em disco, o
pipeline de processadores, a gravação com timestamp, as rotas HTTP e um teste
**ponta a ponta com WebRTC real** (cliente aiortc enviando vídeo ao servidor).

Os testes do YOLO-World e do VLM usam modelos falsos, então rodam sem PyTorch.
Com o extra `visao` instalado, um teste adicional roda o YOLO-World real
(`yolov8s`) na imagem de exemplo do ultralytics; para rodar só ele:
`uv run pytest -m visao`.

---

## 8. Solução de problemas

| Sintoma | Causa provável / solução |
|---|---|
| "Câmera indisponível: abra a página via HTTPS" | Página aberta em `http://`. Use `https://IP:8443`. |
| Aviso de certificado mesmo após instalar a CA | iPhone: falta ativar a *confiança total* (2.3, passo 3). IP mudou: rode o script de novo. |
| Página abre, mas fica em "Conectando…" / "Falha na conexão WebRTC" | Firewall bloqueando UDP (3.1); rede de convidados/isolamento de AP; VPN ativa no PC ou no celular. |
| Página nem abre no celular | Celular em outra rede (ou dados móveis); porta 8443 bloqueada no firewall; IP errado. |
| Abre a câmera frontal | Alguns aparelhos ignoram `facingMode`; feche outras apps de câmera e recarregue. |
| Nenhum frame salvo | Veja `nitidez=` na barra inferior e reduza `SHARPNESS_THRESHOLD`; ou aumente `MOTION_THRESHOLD`. |
| Muitos frames repetidos | Aumente `NOVELTY_THRESHOLD` e/ou `MIN_KEYFRAME_INTERVAL_S`. |
| `Pacote 'ultralytics' não instalado` | Rode `uv sync --extra visao`. |
| Log mostra `dispositivo=cpu` tendo GPU | Confira `nvidia-smi` e `uv run python -c "import torch; print(torch.cuda.is_available())"`. |
| Detecta objetos errados / demais | Aumente `INVENTARIO_YOLO_CONF`, enxugue `config/classes.txt`, use prompts mais específicos. |
| Não detecta objetos pequenos | Aproxime a câmera ou use `INVENTARIO_YOLO_IMGSZ=1280`. |
| `CUDA out of memory` | Veja a seção 6.6. |
| `inventario.csv` com muitos `erro` | O modelo não respondeu em JSON: confira o campo `raw` no `descriptions.jsonl`; aumente `INVENTARIO_VLM_MAX_NEW_TOKENS` se a resposta estiver cortada. |
| Marca/modelo vazios | O VLM só preenche o que consegue ler: aproxime a câmera da etiqueta ou aumente `INVENTARIO_VLM_MAX_SIDE`. |
| Imagem com resolução baixa | O navegador reduz a resolução quando a rede está ruim; aproxime-se do roteador (5 GHz ajuda). |

---

## Apêndice: rodando no Windows

O código é o mesmo. Diferenças:

- Instale o uv com `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
  e o mkcert com `winget install FiloSottile.mkcert`.
- Deixe a rede Wi-Fi como **Rede privada** e, na primeira execução, permita o
  Python no firewall. Se a pergunta não aparecer, no PowerShell como administrador:
  ```powershell
  New-NetFirewallRule -DisplayName "inventario-visual HTTPS" -Direction Inbound `
      -Protocol TCP -LocalPort 8443 -Action Allow -Profile Private
  $py = uv run python -c "import sys; print(sys._base_executable)"
  New-NetFirewallRule -DisplayName "inventario-visual WebRTC" -Direction Inbound `
      -Program $py -Protocol UDP -Action Allow -Profile Private
  ```
