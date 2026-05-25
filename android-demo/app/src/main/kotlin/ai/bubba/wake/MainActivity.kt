package ai.bubba.wake

import android.Manifest
import android.animation.ArgbEvaluator
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.RadioGroup
import android.widget.SeekBar
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import java.util.Locale

class MainActivity : AppCompatActivity() {

    private lateinit var indicator: View
    private lateinit var fireCount: TextView
    private lateinit var probValue: TextView
    private lateinit var micBar: ProgressBar
    private lateinit var micValue: TextView
    private lateinit var thrLabel: TextView
    private lateinit var thrSeek: SeekBar
    private lateinit var dbLabel: TextView
    private lateinit var dbSeek: SeekBar
    private lateinit var srcGroup: RadioGroup
    private lateinit var srcLabel: TextView
    private lateinit var startBtn: Button
    private lateinit var buildInfo: TextView

    private var detector: WakeDetector? = null
    private var audio: AudioPipeline? = null
    private val ui = Handler(Looper.getMainLooper())
    private val argb = ArgbEvaluator()
    private var fires = 0
    private var lastFireMs = 0L

    private var threshold: Float = 0.5f
    private var minDbfs: Float = -45f
    private var sourceMode: AudioSourceMode = AudioSourceMode.PROCESSED

    private val requestMic = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) startListening()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        indicator = findViewById(R.id.indicator)
        fireCount = findViewById(R.id.fireCount)
        probValue = findViewById(R.id.probValue)
        micBar = findViewById(R.id.micBar)
        micValue = findViewById(R.id.micValue)
        thrLabel = findViewById(R.id.thrLabel)
        thrSeek = findViewById(R.id.thrSeek)
        dbLabel = findViewById(R.id.dbLabel)
        dbSeek = findViewById(R.id.dbSeek)
        srcGroup = findViewById(R.id.srcGroup)
        srcLabel = findViewById(R.id.srcLabel)
        startBtn = findViewById(R.id.startBtn)
        buildInfo = findViewById(R.id.buildInfo)

        val pkg = packageManager.getPackageInfo(packageName, 0)
        buildInfo.text = "v${pkg.versionName} (code ${pkg.longVersionCode})"

        thrSeek.setOnSeekBarChangeListener(simple { p ->
            threshold = p / 100f
            thrLabel.text = String.format(Locale.US, "threshold: %.2f", threshold)
            detector?.threshold = threshold
        })
        dbSeek.setOnSeekBarChangeListener(simple { p ->
            minDbfs = -60f + p.toFloat()
            dbLabel.text = String.format(Locale.US, "energy gate: %.0f dBFS", minDbfs)
            detector?.minDbfs = minDbfs
        })
        srcGroup.setOnCheckedChangeListener { _, checkedId ->
            sourceMode = if (checkedId == R.id.srcRaw) AudioSourceMode.RAW
                         else AudioSourceMode.PROCESSED
            if (audio != null) { stopListening(); startListening() }
            else srcLabel.text = "source: ${sourceMode.label} (not active)"
        }

        startBtn.setOnClickListener {
            if (audio == null) ensurePermissionAndStart() else stopListening()
        }
    }

    private fun simple(onChange: (Int) -> Unit) = object : SeekBar.OnSeekBarChangeListener {
        override fun onProgressChanged(seekBar: SeekBar?, p: Int, fromUser: Boolean) { onChange(p) }
        override fun onStartTrackingTouch(seekBar: SeekBar?) {}
        override fun onStopTrackingTouch(seekBar: SeekBar?) {}
    }

    private fun ensurePermissionAndStart() {
        val ok = ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED
        if (ok) startListening() else requestMic.launch(Manifest.permission.RECORD_AUDIO)
    }

    private fun startListening() {
        if (audio != null) return
        try {
            val d = OWWWakeWordDetector(applicationContext).apply {
                this.threshold = this@MainActivity.threshold
                this.minDbfs = this@MainActivity.minDbfs
            }
            detector = d
            audio = AudioPipeline(
                detector = d,
                onResult = { r -> ui.post { renderStep(r) } },
                onSource = { src -> ui.post { srcLabel.text = "source: $src" } },
                sourceMode = sourceMode,
            ).also { it.start() }
            startBtn.text = "Stop"
        } catch (e: Throwable) {
            buildInfo.text = "init failed: ${e.message}"
        }
    }

    private fun stopListening() {
        audio?.stop(); audio = null
        detector?.close(); detector = null
        startBtn.text = "Start listening"
        setIndicatorByProb(0f)
    }

    private fun renderStep(r: WakeStep) {
        val now = System.currentTimeMillis()
        if (r.fired) {
            fires += 1
            fireCount.text = fires.toString()
            lastFireMs = now
        }
        if (now - lastFireMs < 1500) {
            indicator.setBackgroundResource(R.drawable.bg_indicator_active)
        } else {
            setIndicatorByProb(r.smoothedProb)
        }
        probValue.text = String.format(Locale.US, "p %.2f", r.smoothedProb)

        val dbDisp = if (r.dbfs < -90f || r.dbfs.isInfinite()) "-∞" else String.format(Locale.US, "%+.0f", r.dbfs)
        micValue.text = "mic $dbDisp dBFS"
        val micFrac = ((r.dbfs + 60f) / 50f).coerceIn(0f, 1f)
        micBar.progress = (micFrac * 1000f).toInt()
    }

    private fun setIndicatorByProb(p: Float) {
        val from = Color.parseColor("#22232E")
        val to = Color.parseColor("#2EC4B6")
        val color = argb.evaluate(p.coerceIn(0f, 1f), from, to) as Int
        val bg = ContextCompat.getDrawable(this, R.drawable.bg_indicator_idle)?.mutate()
        if (bg is GradientDrawable) {
            bg.setColor(color)
            indicator.background = bg
        } else {
            indicator.setBackgroundColor(color)
        }
    }

    override fun onPause() {
        super.onPause()
        stopListening()
    }
}
