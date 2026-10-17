// Cliente WebRTC do celular: captura a câmera traseira e envia ao servidor.
// JavaScript puro, sem dependências.
"use strict";

const video = document.getElementById("preview");
const estado = document.getElementById("estado");
const metricas = document.getElementById("metricas");
const deteccoes = document.getElementById("deteccoes");
const badge = document.getElementById("badge");
const flash = document.getElementById("flash");
const btnIniciar = document.getElementById("btn-iniciar");
const btnParar = document.getElementById("btn-parar");

let pc = null;
let stream = null;
let sessionId = null;

// Textos amigáveis para as decisões do seletor (ver keyframes.Decision).
const DECISOES = {
  moving: "Em movimento — segure parado",
  stabilizing: "Estabilizando…",
  blurry: "Parado, mas desfocado",
  cooldown: "Aguardando intervalo",
  duplicate: "Cena já capturada — mova para o próximo item",
  keyframe: "Frame-chave salvo!",
};

function setEstado(texto, classe = "") {
  estado.textContent = texto;
  estado.className = classe;
}

function piscar() {
  flash.classList.add("on");
  requestAnimationFrame(() => requestAnimationFrame(() => flash.classList.remove("on")));
  if (navigator.vibrate) navigator.vibrate(60);
}

// Aguarda a coleta completa de candidatos ICE (o servidor não usa trickle ICE).
function aguardarIce(conexao) {
  if (conexao.iceGatheringState === "complete") return Promise.resolve();
  return new Promise((resolve) => {
    const verificar = () => {
      if (conexao.iceGatheringState === "complete") {
        conexao.removeEventListener("icegatheringstatechange", verificar);
        resolve();
      }
    };
    conexao.addEventListener("icegatheringstatechange", verificar);
    // Na rede local a coleta é rápida; o timeout evita travar se algo falhar.
    setTimeout(resolve, 3000);
  });
}

async function iniciar() {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    setEstado("Câmera indisponível: abra a página via HTTPS.", "estado-blurry");
    return;
  }
  btnIniciar.disabled = true;
  try {
    setEstado("Pedindo acesso à câmera…");
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: "environment" }, // câmera traseira
        width: { ideal: 1920 },
        height: { ideal: 1080 },
        frameRate: { ideal: 30, max: 30 },
      },
    });
    video.srcObject = stream;

    // Sem iceServers: celular e PC estão na mesma rede local.
    pc = new RTCPeerConnection({ iceServers: [] });
    pc.addEventListener("connectionstatechange", () => {
      const s = pc ? pc.connectionState : "closed";
      if (s === "connected") setEstado("Conectado — transmitindo", "estado-keyframe");
      else if (s === "failed") { setEstado("Falha na conexão WebRTC (firewall?)", "estado-blurry"); parar(); }
      else setEstado(`Conexão: ${s}`);
    });

    // Canal de dados para o servidor avisar sobre frames-chave e métricas.
    const canal = pc.createDataChannel("eventos");
    canal.onmessage = (ev) => tratarEvento(JSON.parse(ev.data));

    const [trilha] = stream.getVideoTracks();
    trilha.contentHint = "detail"; // prioriza nitidez em vez de fluidez
    const sender = pc.addTrack(trilha, stream);
    await ajustarQualidade(sender);

    const oferta = await pc.createOffer();
    await pc.setLocalDescription(oferta);
    await aguardarIce(pc);

    setEstado("Conectando ao servidor…");
    const resp = await fetch("/api/offer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type }),
    });
    if (!resp.ok) throw new Error(`servidor respondeu ${resp.status}`);
    const resposta = await resp.json();
    sessionId = resposta.session_id;
    await pc.setRemoteDescription({ sdp: resposta.sdp, type: resposta.type });

    badge.textContent = "0";
    badge.classList.remove("hidden");
    btnParar.disabled = false;
  } catch (err) {
    console.error(err);
    setEstado(`Erro: ${err.message || err}`, "estado-blurry");
    await parar();
  }
}

// Pede mais bitrate e mantém a resolução (melhor para frames-chave nítidos).
async function ajustarQualidade(sender) {
  try {
    const params = sender.getParameters();
    if (!params.encodings || params.encodings.length === 0) params.encodings = [{}];
    params.encodings[0].maxBitrate = 6_000_000;
    params.degradationPreference = "maintain-resolution";
    await sender.setParameters(params);
  } catch (e) {
    // Nem todo navegador permite ajustar antes da negociação; não é crítico.
    console.debug("setParameters ignorado:", e);
  }
}

function tratarEvento(msg) {
  if (msg.type === "keyframe") {
    badge.textContent = String(msg.index);
    piscar();
  } else if (msg.type === "detections") {
    // Resultado do YOLO-World para o frame-chave (chega logo após o "keyframe").
    const itens = Object.entries(msg.counts).map(([rotulo, n]) => `${n}× ${rotulo}`);
    deteccoes.textContent = `#${msg.index}: ` + (itens.length ? itens.join(", ") : "nada detectado");
    deteccoes.classList.remove("hidden");
  } else if (msg.type === "status") {
    setEstado(DECISOES[msg.decision] || msg.decision, `estado-${msg.decision}`);
    const nit = msg.sharpness == null ? "-" : msg.sharpness.toFixed(0);
    const mov = msg.motion == null ? "-" : msg.motion.toFixed(1);
    metricas.textContent = `${msg.resolution}  mov=${mov}  nitidez=${nit}  salvos=${msg.keyframes}`;
  }
}

async function parar() {
  btnParar.disabled = true;
  if (sessionId) {
    fetch(`/api/sessions/${sessionId}`, { method: "DELETE" }).catch(() => {});
    sessionId = null;
  }
  if (pc) { pc.close(); pc = null; }
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  video.srcObject = null;
  btnIniciar.disabled = false;
  if (!estado.className.includes("blurry")) setEstado("Desconectado");
}

btnIniciar.addEventListener("click", iniciar);
btnParar.addEventListener("click", parar);
window.addEventListener("pagehide", parar);
