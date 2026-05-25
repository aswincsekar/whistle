package ai.bubba.wake

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import kotlin.concurrent.thread

/**
 * Captures 16 kHz mono float PCM via AudioRecord and feeds the detector in
 * [hopSamples]-sized chunks. The capture buffer is generous (4× the hop) so
 * the OS audio pump won't underrun if the model takes longer than expected.
 */
enum class AudioSourceMode(val androidSrc: Int, val label: String) {
    /** Raw / unprocessed mic. May be silently broken on Huawei + some Samsung. */
    RAW(MediaRecorder.AudioSource.UNPROCESSED, "UNPROCESSED"),
    /** Tuned for ASR. Adds AGC + noise suppression. Reliable across vendors. */
    PROCESSED(MediaRecorder.AudioSource.VOICE_RECOGNITION, "VOICE_RECOGNITION"),
}

class AudioPipeline(
    private val detector: WakeDetector,
    private val onResult: (WakeStep) -> Unit,
    private val onSource: (String) -> Unit = {},
    private val sourceMode: AudioSourceMode = AudioSourceMode.PROCESSED,
) {
    private val sampleRate = detector.sampleRate
    private val hopSamples = detector.hopSamples
    @Volatile private var running = false
    private var thread: Thread? = null
    private var record: AudioRecord? = null

    fun start() {
        if (running) return
        running = true

        val minBuf = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_FLOAT,
        )
        // hold a few hops worth of samples so the model can lag a little
        val bufBytes = maxOf(minBuf, hopSamples * 4 * Float.SIZE_BYTES)

        val rec = AudioRecord.Builder()
            .setAudioSource(sourceMode.androidSrc)
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                    .build()
            )
            .setBufferSizeInBytes(bufBytes)
            .build()
        rec.startRecording()
        Log.i(TAG, "audio source = ${sourceMode.label}")
        onSource(sourceMode.label)
        record = rec

        thread = thread(name = "wake-audio", start = true) {
            val chunk = FloatArray(hopSamples)
            try {
                while (running) {
                    var off = 0
                    while (off < hopSamples) {
                        val n = record?.read(chunk, off, hopSamples - off, AudioRecord.READ_BLOCKING)
                            ?: break
                        if (n < 0) {
                            Log.w(TAG, "AudioRecord.read error $n")
                            break
                        }
                        off += n
                    }
                    if (off == hopSamples) {
                        val r = detector.ingest(chunk)
                        onResult(r)
                    }
                }
            } catch (e: Throwable) {
                Log.e(TAG, "audio thread crashed", e)
            } finally {
                try { record?.stop() } catch (_: Throwable) {}
                try { record?.release() } catch (_: Throwable) {}
                record = null
            }
        }
    }

    fun stop() {
        running = false
        thread?.join(500)
        thread = null
    }

    companion object { private const val TAG = "AudioPipeline" }
}
