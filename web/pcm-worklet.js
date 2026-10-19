// VoxDesk mic capture: float32 @ context rate -> PCM16 mono @ 16 kHz, framed in 512-sample blocks.
// 512 samples = 32 ms, the chunk size the server-side Silero VAD expects.

const TARGET_RATE = 16000;
const FRAME_SIZE = 512;

class PCM16Resampler extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options?.processorOptions || {};
    this.targetRate = opts.targetRate || TARGET_RATE;
    this.frameSize = opts.frameSize || FRAME_SIZE;
    this.ratio = sampleRate / this.targetRate; // input samples per output sample
    this.pos = 0;      // read position inside [prev, ...chunk]
    this.prev = 0;     // last input sample from the previous chunk (interpolation continuity)
    this.out = new Int16Array(this.frameSize);
    this.outLen = 0;
    this.muted = false;
    this.started = false;
    this.port.onmessage = (event) => {
      const data = event.data || {};
      if (data.type === 'mute') this.muted = !!data.muted;
    };
  }

  process(inputs) {
    const channels = inputs[0];
    const chunk = channels && channels[0];
    if (!chunk || chunk.length === 0) return true;
    if (!this.started) {
      this.prev = chunk[0];
      this.started = true;
    }

    const n = chunk.length + 1;
    const buf = new Float32Array(n);
    buf[0] = this.prev;
    buf.set(chunk, 1);

    let p = this.pos;
    while (true) {
      const i0 = Math.floor(p);
      if (i0 + 1 >= n) break;
      const frac = p - i0;
      const sample = buf[i0] + (buf[i0 + 1] - buf[i0]) * frac;
      this.out[this.outLen++] = this.clamp16(sample);
      p += this.ratio;
      if (this.outLen === this.frameSize) this.flush();
    }
    this.prev = chunk[chunk.length - 1];
    this.pos = p - (n - 1);
    return true;
  }

  clamp16(x) {
    if (x >= 1) return 32767;
    if (x <= -1) return -32768;
    return Math.round(x * 32767);
  }

  flush() {
    if (this.outLen > 0 && !this.muted) {
      const payload = this.out.slice(0, this.outLen);
      this.port.postMessage(payload.buffer, [payload.buffer]);
    }
    this.outLen = 0;
  }
}

registerProcessor('pcm16-resampler', PCM16Resampler);
