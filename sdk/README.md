# Miyraa SDK

Mobile client libraries for iOS and Android to integrate with Miyraa Emotion Analysis API.

## Features

- ✅ Emotion analysis with 8+ emotions
- ✅ Multi-language support (100+ languages via XLM-R)
- ✅ Batch processing for multiple texts
- ✅ Emotion explanation with keyword highlighting
- ✅ Text summarization
- ✅ In-memory caching with TTL
- ✅ Automatic retry with exponential backoff
- ✅ Type-safe API with modern language features
- ✅ Async/await, coroutines, and callback support

---

## iOS SDK (Swift)

### Requirements
- iOS 13.0+
- Swift 5.5+
- Xcode 13.0+

### Installation

#### Swift Package Manager

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/miyraa/sdk-ios.git", from: "1.0.0")
]
```

#### Manual Installation

1. Copy `MiyraaSDK.swift` to your Xcode project
2. Add to your target

### Quick Start

```swift
import MiyraaSDK

// Initialize SDK
let config = MiyraaConfig(
    baseURL: "https://api.miyraa.com",
    apiKey: "your-api-key",
    timeout: 30.0,
    enableCache: true
)
let sdk = MiyraaSDK(config: config)

// Analyze emotion (async/await)
Task {
    do {
        let result = try await sdk.analyze(text: "I'm so happy today!")
        print("Emotion: \(result.predictions.first?.label ?? "unknown")")
        print("Emoji: \(result.predictions.first?.emoji ?? "😐")")
    } catch {
        print("Error: \(error)")
    }
}

// Analyze emotion (completion handler)
sdk.analyze(text: "I'm feeling great!") { result in
    switch result {
    case .success(let emotion):
        print("Emotion: \(emotion.predictions.first?.label ?? "unknown")")
    case .failure(let error):
        print("Error: \(error)")
    }
}

// Batch analysis
Task {
    let texts = ["I love this!", "This is terrible", "Feeling okay"]
    let batch = try await sdk.analyzeBatch(texts: texts)
    for result in batch.results {
        print("\(result.text) -> \(result.predictions.first?.label ?? "unknown")")
    }
}

// Get explanation
Task {
    let explanation = try await sdk.explain(text: "I'm so happy!", emotion: "joy")
    print("Keywords: \(explanation.keywords.map { $0.word }.joined(separator: ", "))")
}

// Summarize text
Task {
    let summary = try await sdk.summarize(text: longText, ratio: 0.3)
    print("Summary: \(summary.summary)")
}
```

### Configuration Options

```swift
let config = MiyraaConfig(
    baseURL: "https://api.miyraa.com",  // API base URL
    apiKey: "your-api-key",              // Optional API key
    timeout: 30.0,                       // Request timeout (seconds)
    retryAttempts: 3,                    // Number of retry attempts
    retryDelay: 1.0,                     // Delay between retries (seconds)
    enableCache: true,                   // Enable in-memory cache
    cacheTTL: 3600                       // Cache TTL (seconds)
)
```

### SwiftUI Example

```swift
struct EmotionView: View {
    @State private var text = ""
    @State private var emotion = ""
    @State private var emoji = "😐"
    
    let sdk = MiyraaSDK(config: MiyraaConfig(baseURL: "https://api.miyraa.com"))
    
    var body: some View {
        VStack {
            TextField("Enter text", text: $text)
                .textFieldStyle(.roundedBorder)
                .padding()
            
            Button("Analyze") {
                Task {
                    do {
                        let result = try await sdk.analyze(text: text)
                        emotion = result.predictions.first?.label ?? "unknown"
                        emoji = result.predictions.first?.emoji ?? "😐"
                    } catch {
                        print("Error: \(error)")
                    }
                }
            }
            
            Text(emoji)
                .font(.system(size: 100))
            
            Text(emotion)
                .font(.title)
        }
        .padding()
    }
}
```

---

## Android SDK (Kotlin)

### Requirements
- Android API 21+
- Kotlin 1.9+
- kotlinx.coroutines
- kotlinx.serialization

### Installation

#### Gradle

Add to your `build.gradle.kts`:

```kotlin
dependencies {
    implementation("com.miyraa:sdk-android:1.0.0")
    
    // Required dependencies
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.0")
}
```

Add kotlinx.serialization plugin:

```kotlin
plugins {
    kotlin("plugin.serialization") version "1.9.0"
}
```

#### Manual Installation

1. Copy `MiyraaSDK.kt` to your project
2. Add required dependencies above

### Quick Start

```kotlin
import com.miyraa.sdk.*

// Initialize SDK
val config = MiyraaConfig(
    baseURL = "https://api.miyraa.com",
    apiKey = "your-api-key",
    timeout = 30000L,
    enableCache = true
)
val sdk = MiyraaSDK(config)

// Analyze emotion (coroutines)
lifecycleScope.launch {
    try {
        val result = sdk.analyze("I'm so happy today!")
        println("Emotion: ${result.predictions.firstOrNull()?.label ?: "unknown"}")
        println("Emoji: ${result.predictions.firstOrNull()?.emoji ?: "😐"}")
    } catch (e: MiyraaException) {
        println("Error: ${e.message}")
    }
}

// Analyze emotion (callback)
sdk.analyze("I'm feeling great!") { result ->
    result.onSuccess { emotion ->
        println("Emotion: ${emotion.predictions.firstOrNull()?.label ?: "unknown"}")
    }.onFailure { error ->
        println("Error: ${error.message}")
    }
}

// Batch analysis
lifecycleScope.launch {
    val texts = listOf("I love this!", "This is terrible", "Feeling okay")
    val batch = sdk.analyzeBatch(texts)
    batch.results.forEach { result ->
        println("${result.text} -> ${result.predictions.firstOrNull()?.label ?: "unknown"}")
    }
}

// Get explanation
lifecycleScope.launch {
    val explanation = sdk.explain("I'm so happy!", emotion = "joy")
    println("Keywords: ${explanation.keywords.joinToString { it.word }}")
}

// Summarize text
lifecycleScope.launch {
    val summary = sdk.summarize(longText, ratio = 0.3)
    println("Summary: ${summary.summary}")
}
```

### Configuration Options

```kotlin
val config = MiyraaConfig(
    baseURL = "https://api.miyraa.com",  // API base URL
    apiKey = "your-api-key",              // Optional API key
    timeout = 30000L,                     // Request timeout (milliseconds)
    retryAttempts = 3,                    // Number of retry attempts
    retryDelay = 1000L,                   // Delay between retries (milliseconds)
    enableCache = true,                   // Enable in-memory cache
    cacheTTL = 3600000L                   // Cache TTL (milliseconds)
)
```

### Jetpack Compose Example

```kotlin
@Composable
fun EmotionScreen() {
    var text by remember { mutableStateOf("") }
    var emotion by remember { mutableStateOf("") }
    var emoji by remember { mutableStateOf("😐") }
    
    val scope = rememberCoroutineScope()
    val sdk = remember { MiyraaSDK(MiyraaConfig(baseURL = "https://api.miyraa.com")) }
    
    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        TextField(
            value = text,
            onValueChange = { text = it },
            label = { Text("Enter text") },
            modifier = Modifier.fillMaxWidth()
        )
        
        Spacer(modifier = Modifier.height(16.dp))
        
        Button(onClick = {
            scope.launch {
                try {
                    val result = sdk.analyze(text)
                    emotion = result.predictions.firstOrNull()?.label ?: "unknown"
                    emoji = result.predictions.firstOrNull()?.emoji ?: "😐"
                } catch (e: MiyraaException) {
                    println("Error: ${e.message}")
                }
            }
        }) {
            Text("Analyze")
        }
        
        Spacer(modifier = Modifier.height(32.dp))
        
        Text(
            text = emoji,
            fontSize = 100.sp
        )
        
        Text(
            text = emotion,
            fontSize = 24.sp
        )
    }
}
```

---

## API Reference

### Analyze Emotion

Analyze emotion in a single text.

**iOS:**
```swift
func analyze(text: String, task: String = "emotion", language: String? = nil) async throws -> EmotionResult
```

**Android:**
```kotlin
suspend fun analyze(text: String, task: String = "emotion", language: String? = null): EmotionResult
```

**Parameters:**
- `text`: Input text to analyze
- `task`: Task type (`"emotion"` or `"sentiment"`)
- `language`: Language code (optional, auto-detect if nil)

**Returns:** `EmotionResult` with predictions and metadata

---

### Batch Analysis

Analyze multiple texts in a single request.

**iOS:**
```swift
func analyzeBatch(texts: [String], task: String = "emotion", language: String? = nil) async throws -> BatchResponse
```

**Android:**
```kotlin
suspend fun analyzeBatch(texts: List<String>, task: String = "emotion", language: String? = null): BatchResponse
```

**Parameters:**
- `texts`: List of input texts
- `task`: Task type
- `language`: Language code

**Returns:** `BatchResponse` with list of results

---

### Explain Emotion

Get explanation for emotion prediction with keywords.

**iOS:**
```swift
func explain(text: String, emotion: String) async throws -> Explanation
```

**Android:**
```kotlin
suspend fun explain(text: String, emotion: String): Explanation
```

**Parameters:**
- `text`: Input text
- `emotion`: Emotion to explain

**Returns:** `Explanation` with keywords and scores

---

### Summarize Text

Extract summary from long text.

**iOS:**
```swift
func summarize(text: String, numSentences: Int? = nil, ratio: Double? = nil) async throws -> SummaryResult
```

**Android:**
```kotlin
suspend fun summarize(text: String, numSentences: Int? = null, ratio: Double? = null): SummaryResult
```

**Parameters:**
- `text`: Input text
- `numSentences`: Number of sentences in summary (optional)
- `ratio`: Compression ratio 0-1 (optional)

**Returns:** `SummaryResult` with summary and metadata

---

## Error Handling

### iOS

```swift
do {
    let result = try await sdk.analyze(text: "Hello")
} catch MiyraaError.unauthorized {
    print("Invalid API key")
} catch MiyraaError.rateLimitExceeded {
    print("Rate limit exceeded")
} catch MiyraaError.serverError(let code, let message) {
    print("Server error \(code): \(message)")
} catch {
    print("Error: \(error)")
}
```

### Android

```kotlin
try {
    val result = sdk.analyze("Hello")
} catch (e: MiyraaException.Unauthorized) {
    println("Invalid API key")
} catch (e: MiyraaException.RateLimitExceeded) {
    println("Rate limit exceeded")
} catch (e: MiyraaException.ServerError) {
    println("Server error ${e.code}: ${e.message}")
} catch (e: MiyraaException) {
    println("Error: ${e.message}")
}
```

---

## Cache Management

Both SDKs include in-memory caching with configurable TTL.

**iOS:**
```swift
// Clear cache
sdk.clearCache()
```

**Android:**
```kotlin
// Clear cache
sdk.clearCache()

// Get cache size
val size = sdk.getCacheSize()
```

---

## Best Practices

1. **Reuse SDK instance**: Create one instance and reuse across your app
2. **Enable caching**: Reduces API calls and improves performance
3. **Handle errors**: Always wrap API calls in try-catch
4. **Batch processing**: Use `analyzeBatch` for multiple texts
5. **Timeout configuration**: Adjust timeout based on network conditions
6. **API key security**: Store API key securely (Keychain/KeyStore)

---

## Support

- **Documentation**: https://docs.miyraa.com
- **API Reference**: https://api.miyraa.com/docs
- **Issues**: https://github.com/miyraa/sdk/issues
- **Email**: support@miyraa.com

---

## License

MIT License - see LICENSE file for details
