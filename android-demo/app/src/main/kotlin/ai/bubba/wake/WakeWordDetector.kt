package ai.bubba.wake

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.nio.FloatBuffer

/**
 * Mirror of `src/whistle/stream.py:Detector` — slide a 1.2 s window over 16 kHz
 * mono float32 PCM, score with the bundled ONNX model, fire when the smoothed
 * positive probability crosses [threshold] and the cooldown has elapsed.
 *
 * Mobile-specific extras:
 *  - Energy gate ([minDbfs]): silent windows skip inference entirely. Without
 *    this, per-utterance normalization amplifies the mic noise floor into
 *    feature space and the classifier fires on nothing.
 *  - Smoothing over [smoothN] consecutive windows protects against single-window
 *    spikes that don't correspond to a wake word.
 */
class WakeWordDetector(
    context: Context,
    modelAsset: String = "wakeword_audio.int8.onnx",
    val sampleRate: Int = 16_000,
    val windowSamples: Int = 19_200,    // 1.2 s @ 16 kHz
    val hopSamples: Int = 1_600,        // 0.1 s @ 16 kHz
    // v4 (UrbanSound8K fine-tune): FP32 op point 0.47, INT8 runtime default 0.35.
    // The score distribution tightened a bit after fine-tuning — keeping threshold
    // moderately conservative still gives FRR < 0.5% at the chosen FAR.
    var threshold: Float = 0.35f,
    var minDbfs: Float = -45f,
    var cooldownMs: Long = 1_500,
    private val smoothN: Int = 3,
) {
    data class StepResult(
        val smoothedProb: Float,
        val rawProb: Float,
        val dbfs: Float,
        val fired: Boolean,
        val skipped: Boolean,
    )

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val ring = FloatArray(windowSamples)
    private val smoothQueue = ArrayDeque<Float>(smoothN)
    // Sentinel meaning "no fire yet". Has to be 0 (or any small positive) — not
    // Long.MIN_VALUE — because (now - MIN_VALUE) overflows to a *negative* number
    // that's always < cooldownMs, suppressing every fire.
    private var lastFireMs: Long = 0L

    init {
        val modelBytes = context.assets.open(modelAsset).use { it.readBytes() }
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
            setInterOpNumThreads(1)
        }
        session = env.createSession(modelBytes, opts)
    }

    /** Append [chunk] (exactly [hopSamples] float32 samples in [-1, 1]) and step the detector. */
    fun ingest(chunk: FloatArray, nowMs: Long = System.currentTimeMillis()): StepResult {
        check(chunk.size == hopSamples) {
            "ingest expects chunks of exactly hopSamples ($hopSamples); got ${chunk.size}"
        }
        // shift ring left by hopSamples, append new chunk to the end
        System.arraycopy(ring, hopSamples, ring, 0, windowSamples - hopSamples)
        System.arraycopy(chunk, 0, ring, windowSamples - hopSamples, hopSamples)

        // RMS dBFS on the full 1.2 s window
        var sumSq = 0.0
        for (v in ring) sumSq += v.toDouble() * v
        val rms = kotlin.math.sqrt(sumSq / ring.size)
        val dbfs = (20.0 * kotlin.math.log10(rms + 1e-9)).toFloat()

        if (dbfs < minDbfs) {
            pushSmooth(0f)
            return StepResult(smoothed(), 0f, dbfs, fired = false, skipped = true)
        }

        // Run model
        val input = OnnxTensor.createTensor(env, FloatBuffer.wrap(ring), longArrayOf(1, windowSamples.toLong()))
        val raw = try {
            val out = session.run(mapOf("audio" to input))
            try {
                @Suppress("UNCHECKED_CAST")
                val logits = (out.get(0).value as Array<FloatArray>)[0]
                sigmoid(logits[1] - logits[0])
            } finally {
                out.close()
            }
        } finally {
            input.close()
        }

        pushSmooth(raw)
        val sm = smoothed()
        val fired = sm >= threshold && (nowMs - lastFireMs) >= cooldownMs
        if (fired) lastFireMs = nowMs
        return StepResult(sm, raw, dbfs, fired, skipped = false)
    }

    fun close() {
        session.close()
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

    private fun sigmoid(x: Float): Float = (1.0 / (1.0 + kotlin.math.exp(-x.toDouble()))).toFloat()
}
