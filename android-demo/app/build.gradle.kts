plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "ai.bubba.wake"
    compileSdk = 35

    defaultConfig {
        applicationId = "ai.bubba.wake"
        minSdk = 24
        targetSdk = 35
        versionCode = 16
        versionName = "0.16.0-mel-fix"
        ndk {
            // Universal build for teammates on any device (Chromebook, older
            // 32-bit ARM phones, emulators).
            abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64", "x86")
        }
    }

    buildTypes {
        debug { isMinifyEnabled = false }
        release { isMinifyEnabled = false }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }

    androidResources {
        noCompress += listOf("onnx", "bin")
    }

    sourceSets {
        getByName("main") { kotlin.srcDirs("src/main/kotlin") }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.6")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.19.2")
}
