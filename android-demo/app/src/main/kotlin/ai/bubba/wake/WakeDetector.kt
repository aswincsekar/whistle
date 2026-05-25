package ai.bubba.wake

/** One run of the streaming detector over the current ring buffer. */
data class WakeStep(
    val smoothedProb: Float,
    val rawProb: Float,
    val dbfs: Float,
    val fired: Boolean,
    val skipped: Boolean,
)

/** Common surface for the single-model BC-ResNet detector and the 3-stage
 *  openWakeWord-chain detector. AudioPipeline only needs this. */
interface WakeDetector {
    val sampleRate: Int
    val windowSamples: Int
    val hopSamples: Int
    var threshold: Float
    var minDbfs: Float
    var cooldownMs: Long
    fun ingest(chunk: FloatArray, nowMs: Long = System.currentTimeMillis()): WakeStep
    fun close()
}
