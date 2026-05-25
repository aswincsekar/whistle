package ai.bubba.wake

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.nio.FloatBuffer

/**
 * "Hey Bubba" wake-word detector built on openWakeWord's pretrained
 * speech embedding + a classifier trained via openWakeWord's native pipeline
 * (see models/baseline-v5/).
 *
 * Three ONNX models chained at runtime:
 *   1. melspectrogram.onnx    : (1, samples)        -> (1, 1, T, 32)
 *   2. embedding_model.onnx   : (16, 76, 32, 1)     -> (16, 1, 1, 96)
 *   3. hey_bubba_v0.1.onnx    : (1, 16, 96)         -> (1, 1) sigmoid
 *
 * Window math: 16 sliding 76-frame mel windows with stride 8 → 16*8 + (76-8) = 196 mel
 * frames ≈ 1.96 s of audio. We use a 2.04 s ring buffer (32640 samples) so the
 * leading-edge mel frames have room to settle before being sampled.
 */
class OWWWakeWordDetector(
    context: Context,
    override val sampleRate: Int = 16_000,
    override val windowSamples: Int = 32_640,        // 2.04 s @ 16 kHz
    override val hopSamples: Int = 1_280,            // 80 ms (openWakeWord native hop)
    override var threshold: Float = 0.5f,
    override var minDbfs: Float = -45f,
    override var cooldownMs: Long = 1_500L,
    private val smoothN: Int = 3,
    melAsset: String = "melspectrogram.onnx",
    embAsset: String = "embedding_model.onnx",
    classifierAsset: String = "hey_bubba_v0.1.onnx",
) : WakeDetector {

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val mel: OrtSession
    private val emb: OrtSession
    private val cls: OrtSession
    private val melInputName: String
    private val embInputName: String
    private val clsInputName: String

    private val ring = FloatArray(windowSamples)
    private val smoothQueue = ArrayDeque<Float>(smoothN)
    private var lastFireMs: Long = 0L

    // Window constants matching the v5 training pipeline.
    private val embWinFrames = 76
    private val embStride = 8
    private val nEmbeddings = 16
    private val melTargetT = embWinFrames + (nEmbeddings - 1) * embStride   // 196
    private val melBins = 32
    private val embDim = 96

    init {
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
            setInterOpNumThreads(1)
        }
        fun load(asset: String): OrtSession =
            env.createSession(context.assets.open(asset).use { it.readBytes() }, opts)
        mel = load(melAsset)
        emb = load(embAsset)
        cls = load(classifierAsset)
        melInputName = mel.inputNames.iterator().next()
        embInputName = emb.inputNames.iterator().next()
        clsInputName = cls.inputNames.iterator().next()
    }

    override fun ingest(chunk: FloatArray, nowMs: Long): WakeStep {
        check(chunk.size == hopSamples) {
            "expected ${hopSamples} samples, got ${chunk.size}"
        }
        System.arraycopy(ring, hopSamples, ring, 0, windowSamples - hopSamples)
        System.arraycopy(chunk, 0, ring, windowSamples - hopSamples, hopSamples)

        var sumSq = 0.0
        for (v in ring) sumSq += v.toDouble() * v
        val rms = kotlin.math.sqrt(sumSq / ring.size)
        val dbfs = (20.0 * kotlin.math.log10(rms + 1e-9)).toFloat()
        if (dbfs < minDbfs) {
            pushSmooth(0f)
            return WakeStep(smoothed(), 0f, dbfs, fired = false, skipped = true)
        }

        // 1. mel
        val audioIn = OnnxTensor.createTensor(env, FloatBuffer.wrap(ring),
                                              longArrayOf(1, windowSamples.toLong()))
        val melArr: Array<Array<Array<FloatArray>>>
        try {
            val out = mel.run(mapOf(melInputName to audioIn))
            @Suppress("UNCHECKED_CAST")
            melArr = out[0].value as Array<Array<Array<FloatArray>>>
            out.close()
        } finally { audioIn.close() }

        val rawT = melArr[0][0].size
        val melBuf = FloatArray(melTargetT * melBins)
        val copyT = minOf(rawT, melTargetT)
        for (t in 0 until copyT) {
            System.arraycopy(melArr[0][0][t], 0, melBuf, t * melBins, melBins)
        }
        if (rawT < melTargetT) {
            val lastRow = melArr[0][0][rawT - 1]
            for (t in rawT until melTargetT) {
                System.arraycopy(lastRow, 0, melBuf, t * melBins, melBins)
            }
        }

        // 2. embedding batch over 16 sliding windows
        val embInBuf = FloatArray(nEmbeddings * embWinFrames * melBins)
        for (n in 0 until nEmbeddings) {
            val startFrame = n * embStride
            System.arraycopy(
                melBuf, startFrame * melBins,
                embInBuf, n * embWinFrames * melBins,
                embWinFrames * melBins,
            )
        }
        val embIn = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(embInBuf),
            longArrayOf(nEmbeddings.toLong(), embWinFrames.toLong(),
                        melBins.toLong(), 1L),
        )
        val embArr: Array<Array<Array<FloatArray>>>
        try {
            val out = emb.run(mapOf(embInputName to embIn))
            @Suppress("UNCHECKED_CAST")
            embArr = out[0].value as Array<Array<Array<FloatArray>>>
            out.close()
        } finally { embIn.close() }

        val clsInBuf = FloatArray(nEmbeddings * embDim)
        for (n in 0 until nEmbeddings) {
            System.arraycopy(embArr[n][0][0], 0, clsInBuf, n * embDim, embDim)
        }

        // 3. classifier — outputs a single sigmoid score (already 0..1)
        val clsIn = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(clsInBuf),
            longArrayOf(1L, nEmbeddings.toLong(), embDim.toLong()),
        )
        val raw: Float = try {
            val out = cls.run(mapOf(clsInputName to clsIn))
            @Suppress("UNCHECKED_CAST")
            val arr = out[0].value
            // openWakeWord classifier may return either [[score]] or [score]; handle both
            val s = when (arr) {
                is Array<*> -> when (val a0 = arr[0]) {
                    is FloatArray -> a0[0]
                    is Array<*> -> ((a0 as Array<*>)[0] as FloatArray)[0]
                    else -> throw IllegalStateException("unexpected classifier output shape")
                }
                else -> throw IllegalStateException("unexpected classifier output type")
            }
            out.close()
            s
        } finally { clsIn.close() }

        pushSmooth(raw)
        val sm = smoothed()
        val fired = sm >= threshold && (nowMs - lastFireMs) >= cooldownMs
        if (fired) lastFireMs = nowMs
        return WakeStep(sm, raw, dbfs, fired, skipped = false)
    }

    override fun close() {
        try { mel.close() } catch (_: Throwable) {}
        try { emb.close() } catch (_: Throwable) {}
        try { cls.close() } catch (_: Throwable) {}
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
