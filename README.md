# inventario-visual

Servidor local que recebe **vídeo ao vivo da câmera do celular** (pelo navegador,
sem app nativo) via **WebRTC** e salva automaticamente **frames-chave**, ou seja,
os momentos em que a câmera está parada e a imagem está nítida. Esses frames
ficam em disco para o processamento posterior (YOLO-World + modelo de
visão-linguagem), que se conecta por uma interface plugável.

```
 Celular (Chrome/Safari)                       PC com Windows (este projeto)
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
                                               │  (LogProcessor hoje; YOLO-World/VLM depois)│
                                               └──────────────────────────────────────────┘
```

## Estrutura

```
inventario-visual/
├── pyproject.toml / requirements*.txt
├── .env.example                 # todos os parâmetros configuráveis
├── scripts/
│   └── gerar_certificado.py     # mkcert, com fallback para CA própria
├── src/inventario_visual/
│   ├── __main__.py              # `inventario-visual` (sobe o servidor HTTPS)
│   ├── app.py                   # rotas FastAPI
│   ├── webrtc.py                # sessões aiortc, consumo do vídeo
│   ├── keyframes.py             # seletor de frames-chave (puro, sem WebRTC)
│   ├── storage.py               # gravação JPEG + metadados JSONL
│   ├── processors.py            # FrameProcessor (ABC) e pipeline assíncrono
│   ├── offline.py               # `inventario-offline`: roda o seletor em um vídeo
│   ├── config.py                # Settings (variáveis INVENTARIO_*)
│   └── static/                  # index.html, app.js, style.css (JS puro)
└── tests/                       # pytest com imagens e vídeos sintéticos
```

---

## 1. Instalação no Windows (com GPU NVIDIA)

> O **streaming e o seletor de frames rodam na CPU** e funcionam em qualquer PC.
> A GPU NVIDIA só será usada pela etapa de visão (YOLO-World + VLM), que será
> plugada depois. Mesmo assim, já deixamos o ambiente pronto para ela.

### 1.1 Pré-requisitos

1. **Python 3.11+** — <https://www.python.org/downloads/> (marque *Add python.exe to PATH*).
2. **uv** (recomendado) — no PowerShell:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
3. **Driver NVIDIA** atualizado. Confira com:
   ```powershell
   nvidia-smi
   ```
   A coluna *CUDA Version* mostra a versão máxima de CUDA suportada pelo driver.
4. **mkcert** (opcional, mas recomendado):
   ```powershell
   winget install FiloSottile.mkcert
   ```
   (ou `choco install mkcert` / `scoop install mkcert`). Sem o mkcert, o script
   gera uma CA própria usando a biblioteca `cryptography`.

### 1.2 Dependências do projeto

Com **uv**:

```powershell
cd inventario-visual
uv sync            # cria .venv e instala tudo (inclui pytest/mypy do grupo dev)
```

Ou com **pip**:

```powershell
cd inventario-visual
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
```

> O `aiortc` e o `av` (PyAV) têm wheels prontos para Windows; não é preciso
> compilar nada nem instalar o FFmpeg separadamente.

### 1.3 (Para a etapa de visão) PyTorch com CUDA

Ainda não é necessário para o streaming, mas quando for plugar o YOLO-World:

```powershell
# Escolha o índice compatível com seu driver em https://pytorch.org/get-started/locally/
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Deve imprimir `True` e o nome da sua placa.

---

## 2. HTTPS na rede local (obrigatório)

Os navegadores **só liberam a câmera (`getUserMedia`) em contexto seguro**:
`https://` ou `http://localhost`. Como o celular acessa o PC pelo IP da rede
(ex.: `https://192.168.0.50:8443`), precisamos de um certificado válido para
esse IP.

### 2.1 Gerar o certificado

```powershell
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

Descubra o IP do PC com `ipconfig` (campo *Endereço IPv4* do adaptador Wi-Fi/Ethernet).

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

```powershell
uv run inventario-visual
#   ou:  uv run python -m inventario_visual --port 8443
```

O terminal mostra o endereço a abrir no celular, por exemplo:

```
  Abra no celular:  https://192.168.0.50:8443/
```

### 3.1 Firewall do Windows

O celular precisa alcançar o PC em **TCP 8443** (página) e em **portas UDP
dinâmicas** (mídia WebRTC).

- Garanta que a rede Wi-Fi esteja como **Rede privada**
  (*Configurações → Rede e Internet → Wi-Fi → propriedades*).
- Na primeira execução o Windows pergunta se o **Python** pode se comunicar em
  redes privadas: clique em **Permitir**.
- Se não apareceu a pergunta ou o vídeo não conecta, crie as regras manualmente
  (PowerShell **como administrador**, dentro da pasta do projeto):
  ```powershell
  New-NetFirewallRule -DisplayName "inventario-visual HTTPS" -Direction Inbound `
      -Protocol TCP -LocalPort 8443 -Action Allow -Profile Private
  $py = uv run python -c "import sys; print(sys._base_executable)"
  New-NetFirewallRule -DisplayName "inventario-visual WebRTC" -Direction Inbound `
      -Program $py -Protocol UDP -Action Allow -Profile Private
  ```
  (A regra UDP é por programa porque o WebRTC usa portas aleatórias. Usamos o
  executável *base* do Python porque o `python.exe` do `.venv` é só um lançador.)

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

### 3.3 Onde ficam os frames

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

```powershell
uv run inventario-offline meu_video.mp4 -v --saida data/calibracao
```

A saída lista movimento e nitidez de cada amostra e a decisão tomada; ajuste
`--motion` / `--sharpness` até os frames salvos ficarem bons e então leve os
valores para o `.env`. A barra inferior da página no celular também mostra
`mov=` e `nitidez=` em tempo real.

---

## 5. Plugando a etapa de visão (YOLO-World + VLM)

O streaming não conhece a visão. Ele só entrega cada frame-chave a uma lista de
`FrameProcessor`, que roda numa **thread de trabalho separada** (pode bloquear
e usar a GPU sem travar o vídeo), em ordem, com fila limitada.

```python
# meu_pacote/visao.py
from inventario_visual import FrameProcessor, Keyframe


class YoloWorldVlm(FrameProcessor):
    name = "yolo-world+vlm"

    def setup(self) -> None:
        # Chamado uma vez, antes do primeiro frame: carregue os modelos na GPU aqui.
        from ultralytics import YOLOWorld
        self.det = YOLOWorld("yolov8l-worldv2.pt").to("cuda")
        self.det.set_classes(["caixa", "garrafa", "ferramenta"])

    def process(self, keyframe: Keyframe) -> None:
        resultado = self.det.predict(keyframe.image, verbose=False)[0]
        keyframe.extras["deteccoes"] = resultado  # visível para processadores seguintes
        # ... recortes -> modelo de visão-linguagem -> gravar inventário ...

    def close(self) -> None:
        ...
```

Ative pelo `.env`, sem tocar no código do servidor (vários separados por vírgula,
executados na ordem):

```
INVENTARIO_PROCESSORS=inventario_visual.processors:LogProcessor,meu_pacote.visao:YoloWorldVlm
```

O `Keyframe` traz: `session_id`, `index`, `captured_at`, `image` (BGR, numpy),
`path` (JPEG salvo), `sharpness`, `motion`, `novelty` e `extras` (dict livre).
Erros de um processador são registrados no log e não afetam os demais nem o
streaming. Se a visão ficar para trás, os frames mais antigos da fila são
descartados (continuam salvos em disco para reprocessar depois).

---

## 6. Testes

```powershell
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

---

## 7. Solução de problemas

| Sintoma | Causa provável / solução |
|---|---|
| "Câmera indisponível: abra a página via HTTPS" | Página aberta em `http://`. Use `https://IP:8443`. |
| Aviso de certificado mesmo após instalar a CA | iPhone: falta ativar a *confiança total* (2.3, passo 3). IP mudou: rode o script de novo. |
| Página abre, mas fica em "Conectando…" / "Falha na conexão WebRTC" | Firewall bloqueando UDP do Python (3.1); rede de convidados/isolamento de AP; VPN ativa no PC ou no celular. |
| Página nem abre no celular | Celular em outra rede (ou dados móveis); porta 8443 bloqueada no firewall; IP errado. |
| Abre a câmera frontal | Alguns aparelhos ignoram `facingMode`; feche outras apps de câmera e recarregue. |
| Nenhum frame salvo | Veja `nitidez=` na barra inferior e reduza `SHARPNESS_THRESHOLD`; ou aumente `MOTION_THRESHOLD`. |
| Muitos frames repetidos | Aumente `NOVELTY_THRESHOLD` e/ou `MIN_KEYFRAME_INTERVAL_S`. |
| Imagem com resolução baixa | O navegador reduz a resolução quando a rede está ruim; aproxime-se do roteador (5 GHz ajuda). |
