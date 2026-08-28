import { useEffect, useRef, useState } from 'react';
import { LoggerWithoutDebug, Wllama } from '@wllama/wllama';
import WasmFromCDN from './wasm-from-cdn.js';

const MODEL = {
  repo: 'mradermacher/Qwen3.5-0.8B-abliterated-GGUF',
  quant: 'Q4_K_S',
};

// Conservative mobile profile for Expo Go + Android WebView.
// The goal here is stability first; you can tune these later in VS Code.
const MODEL_LOAD_CONFIG = {
  n_ctx: 1280,
  n_batch: 32,
  n_ubatch: 16,
  n_gpu_layers: 0,
  n_threads: 1,
  n_parallel: 1,
  kv_unified: true,
  cache_type_k: 'q8_0',
  cache_type_v: 'q8_0',
  flash_attn: false,
  offload_kqv: false,
  cont_batching: false,
  warmup: false,
  reasoning: false,
  useCache: true,
};

const GENERATION_CONFIG = {
  max_tokens: 192,
  temperature: 0.7,
  top_p: 0.9,
};

const HISTORY_KEY = 'qwen_mobile_history_v1';
const MEMORY_KEY = 'qwen_mobile_memories_v1';
const PENDING_KEY = 'qwen_mobile_pending_v1';

function readJSON(key, fallback) {
  try {
    const value = localStorage.getItem(key);
    return value ? JSON.parse(value) : fallback;
  } catch {
    return fallback;
  }
}

function writeJSON(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Storage can be unavailable on some WebView configurations.
  }
}

function removeStored(key) {
  try {
    localStorage.removeItem(key);
  } catch {
    // Ignore storage cleanup failures.
  }
}

function clipText(value, maxChars) {
  const text = String(value || '');
  if (text.length <= maxChars) return text;
  return `${text.slice(0, maxChars)}…`;
}

function postToMobile(payload) {
  const message = JSON.stringify(payload);
  if (window.ReactNativeWebView?.postMessage) {
    window.ReactNativeWebView.postMessage(message);
  }
}

function rememberIfRequested(text) {
  const match = text.trim().match(/^remember(?:\s+that)?\s+(.+)/i);
  if (!match) return;

  const memory = clipText(match[1].trim().replace(/[.!?]+$/, ''), 180);
  if (!memory) return;

  const memories = readJSON(MEMORY_KEY, []);
  if (!memories.some((item) => item.toLowerCase() === memory.toLowerCase())) {
    memories.push(memory);
    writeJSON(MEMORY_KEY, memories.slice(-20));
  }
}

function buildSystemPrompt() {
  const memories = readJSON(MEMORY_KEY, []).slice(-5);
  const memoryBlock = memories.length
    ? memories.map((item) => `- ${clipText(item, 120)}`).join('\n')
    : '- None stored yet.';

  return `You are a concise conversational assistant running locally on the user's device.
Use the current conversation as context.
Persistent user memories are authoritative facts previously supplied by the user.
If a memory directly answers a question, answer from it instead of guessing.
Keep answers reasonably concise on mobile.

Persistent memories:\n${memoryBlock}`;
}

function isFatalRuntimeError(error) {
  const message = `${error?.name || ''} ${error?.message || String(error)}`;
  return /abort|unreachable|out of memory|\boom\b|runtimeerror/i.test(message);
}

export default function App() {
  const engineRef = useRef(null);
  const historyRef = useRef(readJSON(HISTORY_KEY, []));
  const busyRef = useRef(false);
  const lastInboundRef = useRef({ value: '', time: 0 });

  const [status, setStatus] = useState('Preparing local Qwen engine…');
  const [ready, setReady] = useState(false);

  const sendStatus = (text) => {
    setStatus(text);
    postToMobile({ type: 'status', text });
  };

  useEffect(() => {
    let cancelled = false;

    async function startEngine() {
      try {
        sendStatus('Initializing local AI runtime in stability mode…');

        const wllama = new Wllama(WasmFromCDN, {
          logger: LoggerWithoutDebug,
          parallelDownloads: 2,
        });
        engineRef.current = wllama;

        let lastProgress = -1;
        await wllama.loadModelFromHF(MODEL, {
          ...MODEL_LOAD_CONFIG,
          progressCallback: ({ loaded, total }) => {
            if (!total) return;
            const percent = Math.floor((loaded / total) * 100);
            if (percent === lastProgress) return;
            if (percent % 2 !== 0 && percent !== 100) return;
            lastProgress = percent;
            sendStatus(`Downloading Qwen model… ${percent}%`);
          },
        });

        if (cancelled) return;

        setReady(true);
        sendStatus('Qwen loaded. CPU/WASM stability mode ready.');
        postToMobile({ type: 'ready' });

        const pending = readJSON(PENDING_KEY, null);
        if (pending?.text && pending?.retries === 1) {
          sendStatus('Runtime recovered. Retrying your message once…');
          setTimeout(() => handleChat(pending.text, true), 700);
        }
      } catch (error) {
        const message = error?.message || String(error);
        removeStored(PENDING_KEY);
        sendStatus(`Engine error: ${message}`);
        postToMobile({ type: 'error', text: message });
      }
    }

    async function handleChat(text, recovering = false) {
      if (!engineRef.current || busyRef.current) return;
      const userText = clipText(String(text || '').trim(), 1000);
      if (!userText) return;

      busyRef.current = true;
      rememberIfRequested(userText);

      if (!recovering) {
        writeJSON(PENDING_KEY, { text: userText, retries: 0 });
      }

      const recentHistory = historyRef.current.slice(-4).map((item) => ({
        role: item.role,
        content: clipText(item.content, 350),
      }));

      const messages = [
        { role: 'system', content: buildSystemPrompt() },
        ...recentHistory,
        { role: 'user', content: userText },
      ];

      try {
        postToMobile({ type: 'reply', text: '' });

        const stream = await engineRef.current.createChatCompletion({
          messages,
          ...GENERATION_CONFIG,
          stream: true,
        });

        let answer = '';
        for await (const chunk of stream) {
          const piece = chunk?.choices?.[0]?.delta?.content || '';
          if (!piece) continue;
          answer += piece;
          postToMobile({ type: 'reply', text: answer, streaming: true });
        }

        const finalAnswer = answer.trim();
        historyRef.current = [
          ...historyRef.current,
          { role: 'user', content: userText },
          { role: 'assistant', content: finalAnswer },
        ].slice(-20);
        writeJSON(HISTORY_KEY, historyRef.current);
        removeStored(PENDING_KEY);

        postToMobile({ type: 'reply', text: finalAnswer, streaming: false });
      } catch (error) {
        const pending = readJSON(PENDING_KEY, {
          text: userText,
          retries: recovering ? 1 : 0,
        });

        if (isFatalRuntimeError(error) && (pending?.retries || 0) < 1) {
          writeJSON(PENDING_KEY, {
            text: userText,
            retries: 1,
          });
          postToMobile({
            type: 'status',
            text: 'Local runtime reset triggered. Retrying automatically…',
          });

          setTimeout(() => window.location.reload(), 500);
          return;
        }

        removeStored(PENDING_KEY);
        const message = error?.message || String(error);
        postToMobile({
          type: 'reply',
          text: `Generation error after stability retry: ${message}`,
        });
      } finally {
        busyRef.current = false;
      }
    }

    function onBridgeMessage(event) {
      const raw = event?.data;
      if (typeof raw !== 'string') return;

      const now = Date.now();
      if (
        raw === lastInboundRef.current.value &&
        now - lastInboundRef.current.time < 150
      ) {
        return;
      }
      lastInboundRef.current = { value: raw, time: now };

      try {
        const data = JSON.parse(raw);
        if (data?.type === 'chat') {
          handleChat(data.message);
        }
        if (data?.type === 'clear') {
          historyRef.current = [];
          writeJSON(HISTORY_KEY, []);
          removeStored(PENDING_KEY);
          postToMobile({ type: 'status', text: 'Conversation cleared.' });
        }
      } catch {
        // Ignore non-JSON bridge traffic.
      }
    }

    window.addEventListener('message', onBridgeMessage);
    document.addEventListener('message', onBridgeMessage);
    startEngine();

    return () => {
      cancelled = true;
      window.removeEventListener('message', onBridgeMessage);
      document.removeEventListener('message', onBridgeMessage);
    };
  }, []);

  return (
    <main className="shell">
      <section className="panel">
        <p className="eyebrow">Qwen Mobile Engine</p>
        <h1>{ready ? 'Local model ready' : 'Loading local model'}</h1>
        <p className="status">{status}</p>
        <p className="detail">
          Qwen3.5-0.8B abliterated · Q4_K_S · CPU/WASM mobile stability profile.
        </p>
      </section>
    </main>
  );
}
