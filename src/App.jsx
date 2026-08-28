import { useEffect, useRef, useState } from 'react';
import { LoggerWithoutDebug, Wllama } from '@wllama/wllama';
import WasmFromCDN from '@wllama/wllama/esm/wasm-from-cdn.js';

const MODEL = {
  repo: 'mradermacher/Qwen3.5-0.8B-abliterated-GGUF',
  quant: 'Q4_K_S',
};

const HISTORY_KEY = 'qwen_mobile_history_v1';
const MEMORY_KEY = 'qwen_mobile_memories_v1';

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

function postToMobile(payload) {
  const message = JSON.stringify(payload);
  if (window.ReactNativeWebView?.postMessage) {
    window.ReactNativeWebView.postMessage(message);
  }
}

function rememberIfRequested(text) {
  const match = text.trim().match(/^remember(?:\s+that)?\s+(.+)/i);
  if (!match) return;

  const memory = match[1].trim().replace(/[.!?]+$/, '');
  if (!memory) return;

  const memories = readJSON(MEMORY_KEY, []);
  if (!memories.some((item) => item.toLowerCase() === memory.toLowerCase())) {
    memories.push(memory);
    writeJSON(MEMORY_KEY, memories.slice(-40));
  }
}

function buildSystemPrompt() {
  const memories = readJSON(MEMORY_KEY, []);
  const memoryBlock = memories.length
    ? memories.map((item) => `- ${item}`).join('\n')
    : '- None stored yet.';

  return `You are a concise conversational assistant running locally on the user's device.
Use the current conversation as context.
Persistent user memories are authoritative facts previously supplied by the user.
If a memory directly answers a question, answer from it instead of guessing.

Persistent memories:\n${memoryBlock}`;
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
        sendStatus('Initializing local AI runtime…');

        const wllama = new Wllama(WasmFromCDN, {
          logger: LoggerWithoutDebug,
          parallelDownloads: 3,
        });
        engineRef.current = wllama;

        let lastProgress = -1;
        await wllama.loadModelFromHF(MODEL, {
          n_ctx: 2048,
          useCache: true,
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

        const mode = wllama.isSupportWebGPU() ? 'WebGPU/WASM ready' : 'WASM ready';
        setReady(true);
        sendStatus(`Qwen loaded. ${mode}.`);
        postToMobile({ type: 'ready' });
      } catch (error) {
        const message = error?.message || String(error);
        sendStatus(`Engine error: ${message}`);
        postToMobile({ type: 'error', text: message });
      }
    }

    async function handleChat(text) {
      if (!engineRef.current || busyRef.current) return;
      const userText = String(text || '').trim();
      if (!userText) return;

      busyRef.current = true;
      rememberIfRequested(userText);

      const recentHistory = historyRef.current.slice(-10);
      const messages = [
        { role: 'system', content: buildSystemPrompt() },
        ...recentHistory,
        { role: 'user', content: userText },
      ];

      try {
        postToMobile({ type: 'reply', text: '' });

        const stream = await engineRef.current.createChatCompletion({
          messages,
          max_tokens: 384,
          temperature: 0.7,
          top_p: 0.9,
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
        ].slice(-24);
        writeJSON(HISTORY_KEY, historyRef.current);

        postToMobile({ type: 'reply', text: finalAnswer, streaming: false });
      } catch (error) {
        const message = error?.message || String(error);
        postToMobile({ type: 'reply', text: `Generation error: ${message}` });
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
          Model: Qwen3.5-0.8B abliterated · Q4_K_S. Inference runs on this device.
        </p>
      </section>
    </main>
  );
}
