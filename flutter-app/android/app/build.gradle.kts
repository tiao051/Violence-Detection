//
// THIS IS THE FIX: We must import the 'Properties' class
//
import java.util.Properties

//
// Apply plugins by ID at the top
//
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    // Activate the Google Services plugin for the app module
    id("com.google.gms.google-services")
    // Activate the Flutter plugin
    id("dev.flutter.flutter-gradle-plugin")
}

//
// THIS IS THE FIX: We must use Kotlin (val) syntax, NOT Groovy (def)
//
val localProperties = Properties() // Now it understands 'Properties'
val localPropertiesFile = rootProject.file("local.properties")
if (localPropertiesFile.exists()) {
    // Use Kotlin's file reader
    localPropertiesFile.bufferedReader().use { reader ->
        localProperties.load(reader)
    }
}

// Use Kotlin "val" and elvis operator (?:) for defaults
val flutterVersionCode = localProperties.getProperty("flutter.versionCode") ?: "1"
val flutterVersionName = localProperties.getProperty("flutter.versionName") ?: "1.0"


android {
    namespace = "com.example.security_app"
    compileSdk = 36 // Updated to meet plugin requirements
    ndkVersion = "27.0.12077973" // Use the NDK version Flutter prefers

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    kotlinOptions {
        jvmTarget = "1.8"
    }

    sourceSets {
        // Use Kotlin syntax
        getByName("main").java.srcDirs("src/main/kotlin")
    }

    defaultConfig {
        applicationId = "com.example.security_app"
        minSdk = flutter.minSdkVersion // Keep at 21 for broad compatibility
        targetSdk = 34
        versionCode = flutterVersionCode.toInt()
        versionName = flutterVersionName
    }

    buildTypes {
        release {
            //
            // THIS IS THE FIX: Use .getByName("debug") for Kotlin
            //
            signingConfig = signingConfigs.getByName("debug") // TODO: Change this for production
        }
    }
}

flutter {
    source = "../.."
}

dependencies {
    // We recommend using the Firebase BOM (Bill of Materials)
    // This manages versions for all Firebase libraries
    implementation(platform("com.google.firebase:firebase-bom:33.1.1"))

    // Add any other Firebase dependencies here (e.g., Analytics)
    // implementation("com.google.firebase:firebase-analytics")
}
