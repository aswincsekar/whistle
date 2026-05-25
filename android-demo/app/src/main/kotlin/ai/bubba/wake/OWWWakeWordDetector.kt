package ai.bubba.wake

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.nio.FloatBuffer

/**
 * Streaming "Hey Bubba" detector that mirrors openWakeWord's reference
 * implementation. The trained classifier was trained on a rolling 16-frame
 * embedding buffer, not a one-shot 2 s window — so we must reproduce that
 * streaming pattern exactly or the classifier sees out-of-distribution input
 * and produces ~0 for everything.
 *
 * Pipeline per 80 ms audio chunk:
 *   1) Append to audio ring buffer.
 *   2) Run melspectrogram.onnx on the audio ring -> mel frames.
 *      Take only the *newest* 8 mel frames and push into the mel deque.
 *   3) If we have 76+ mel frames buffered, run embedding_model.onnx on the
 *      latest 76-frame window -> one 96-dim embedding. Push into the
 *      embedding deque.
 *   4) When the embedding deque has 16 entries, run the classifier on
 *      (1, 16, 96) and emit a score.
 *
 * Steady-state ops per chunk: one mel call (cheap), one embedding call,
 * one classifier call. Latency a few ms on a modern phone.
 */
class OWWWakeWordDetector(
    context: Context,
    override val sampleRate: Int = 16_000,
    override val windowSamples: Int = 32_640,      // 2.04 s audio history for mel context
    override val hopSamples: Int = 1_280,          // 80 ms (openWakeWord native hop)
    override var threshold: Float = 0.5f,
    override var minDbfs: Float = -45f,
    override var cooldownMs: Long = 1_500L,
    private val smoothN: Int = 3,
    melAsset: String = "melspectrogram.onnx",
    embAsset: String = "embedding_model.onnx",
    classifierAsset: String = "hey_bubba_v0.1.onnx",
) : WakeDetector {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val melSession: OrtSession
    private val embSession: OrtSession
    private val clsSession: OrtSession
    private val melInputName: String
    private val embInputName: String
    private val clsInputName: String

    // Audio ring buffer (windowSamples).
    private val audioRing = FloatArray(windowSamples)

    // Mel frame deque: each entry is 32 floats. We cap at MEL_BUF_LEN; older drops out.
    private val melBuf = ArrayDeque<FloatArray>(MEL_BUF_LEN)

    // Embedding deque: each entry is 96 floats. Classifier needs 16.
    private val embBuf = ArrayDeque<FloatArray>(N_EMBS)

    private val smoothQueue = ArrayDeque<Float>(smoothN)
    private var lastFireMs: Long = 0L

    companion object {
        // 76-frame window with stride 8 => need 76 frames buffered before first emb
        private const val EMB_WIN = 76
        private const val EMB_STRIDE = 8
        private const val N_EMBS = 16
        private const val MEL_BINS = 32
        private const val EMB_DIM = 96
        // Keep ~196 mel frames (covers the 16 embedding windows). A few extra is harmless.
        private const val MEL_BUF_LEN = EMB_WIN + (N_EMBS - 1) * EMB_STRIDE  // = 196
    }

    init {
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
            setInterOpNumThreads(1)
        }
        fun load(asset: String): OrtSession =
            env.createSession(context.assets.open(asset).use { it.readBytes() }, opts)
        melSession = load(melAsset)
        embSession = load(embAsset)
        clsSession = load(classifierAsset)
        melInputName = melSession.inputNames.iterator().next()
        embInputName = embSession.inputNames.iterator().next()
        clsInputName = clsSession.inputNames.iterator().next()
    }

    override fun ingest(chunk: FloatArray, nowMs: Long): WakeStep {
        check(chunk.size == hopSamples) {
            "expected ${hopSamples} samples, got ${chunk.size}"
        }

        // 1) Slide audio ring + append new chunk
        System.arraycopy(audioRing, hopSamples, audioRing, 0, windowSamples - hopSamples)
        System.arraycopy(chunk, 0, audioRing, windowSamples - hopSamples, hopSamples)

        // Energy gate on the most recent ~1 s of audio
        var sumSq = 0.0
        for (v in audioRing) sumSq += v.toDouble() * v
        val rms = kotlin.math.sqrt(sumSq / audioRing.size)
        val dbfs = (20.0 * kotlin.math.log10(rms + 1e-9)).toFloat()
        if (dbfs < minDbfs) {
            pushSmooth(0f)
            return WakeStep(smoothed(), 0f, dbfs, fired = false, skipped = true)
        }

        // 2) Mel for the full audio ring; take the LAST 8 frames as the new ones.
        // CRITICAL: openWakeWord's melspectrogram.onnx expects audio in int16
        // magnitude range (float32 values in [-32768, 32767]), NOT the normal
        // [-1, 1] float PCM scale. AudioRecord ENCODING_PCM_FLOAT gives us
        // [-1, 1], so we scale up before sending in.
        val scaled = FloatArray(windowSamples)
        for (i in 0 until windowSamples) scaled[i] = audioRing[i] * 32767f
        val audioIn = OnnxTensor.createTensor(env, FloatBuffer.wrap(scaled),
                                              longArrayOf(1, windowSamples.toLong()))
        val newMelFrames: Array<FloatArray>
        try {
            val out = melSession.run(mapOf(melInputName to audioIn))
            @Suppress("UNCHECKED_CAST")
            val melArr = out[0].value as Array<Array<Array<FloatArray>>>  // (1, 1, T, 32)
            val T = melArr[0][0].size
            val take = minOf(8, T)
            // Copy last `take` rows AND apply openWakeWord's mel post-transform:
            //   mel_out = mel_out / 10 + 2
            // This makes the ONNX melspec match the original TF speech_embedding
            // training distribution. Without it the embedding model sees OOD
            // input and the classifier produces ~0 for everything.
            newMelFrames = Array(take) { row ->
                val src = melArr[0][0][T - take + row]
                val dst = FloatArray(MEL_BINS)
                for (b in 0 until MEL_BINS) dst[b] = src[b] / 10f + 2f
                dst
            }
            out.close()
        } finally { audioIn.close() }

        for (f in newMelFrames) {
            if (melBuf.size == MEL_BUF_LEN) melBuf.removeFirst()
            melBuf.addLast(f)
        }

        // 3) If we have 76+ mel frames, compute the NEW embedding from the latest 76.
        if (melBuf.size >= EMB_WIN) {
            val embInBuf = FloatArray(EMB_WIN * MEL_BINS)
            val melList = melBuf.toList()
            val start = melList.size - EMB_WIN
            for (i in 0 until EMB_WIN) {
                System.arraycopy(melList[start + i], 0, embInBuf, i * MEL_BINS, MEL_BINS)
            }
            val embIn = OnnxTensor.createTensor(
                env, FloatBuffer.wrap(embInBuf),
                longArrayOf(1, EMB_WIN.toLong(), MEL_BINS.toLong(), 1L),
            )
            val newEmb: FloatArray
            try {
                val out = embSession.run(mapOf(embInputName to embIn))
                @Suppress("UNCHECKED_CAST")
                val embArr = out[0].value as Array<Array<Array<FloatArray>>>  // (1, 1, 1, 96)
                newEmb = embArr[0][0][0].copyOf()
                out.close()
            } finally { embIn.close() }

            if (embBuf.size == N_EMBS) embBuf.removeFirst()
            embBuf.addLast(newEmb)
        }

        // 4) Classifier — only when we have a full embedding buffer
        if (embBuf.size < N_EMBS) {
            pushSmooth(0f)
            return WakeStep(smoothed(), 0f, dbfs, fired = false, skipped = false)
        }
        val clsInBuf = FloatArray(N_EMBS * EMB_DIM)
        var n = 0
        for (e in embBuf) {
            System.arraycopy(e, 0, clsInBuf, n * EMB_DIM, EMB_DIM)
            n++
        }
        val clsIn = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(clsInBuf),
            longArrayOf(1L, N_EMBS.toLong(), EMB_DIM.toLong()),
        )
        val raw: Float = try {
            val out = clsSession.run(mapOf(clsInputName to clsIn))
            val value = out[0].value
            val score = when (value) {
                is Array<*> -> when (val row = value[0]) {
                    is FloatArray -> row[0]
                    is Array<*> -> ((row as Array<*>)[0] as FloatArray)[0]
                    else -> 0f
                }
                else -> 0f
            }
            out.close()
            score
        } finally { clsIn.close() }

        pushSmooth(raw)
        val sm = smoothed()
        val fired = sm >= threshold && (nowMs - lastFireMs) >= cooldownMs
        if (fired) lastFireMs = nowMs
        return WakeStep(sm, raw, dbfs, fired, skipped = false)
    }

    override fun close() {
        try { melSession.close() } catch (_: Throwable) {}
        try { embSession.close() } catch (_: Throwable) {}
        try { clsSession.close() } catch (_: Throwable) {}
    }

    private fun pushSmooth(v: Float) {
        if (smoothQueue.size == smoothN) smoothQueue.removeFirst()
        smoothQueue.addLast(v)
    }

    private fun smoothed(): Float {
        if (smoothQueue.isEmpty()) return 0f
        var s = 0f
        for (v in smoothQueue) s += v
        return s / smoothQueue.size
    }
}
