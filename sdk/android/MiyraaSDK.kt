// MiyraaSDK.kt
// Android SDK for Miyraa Emotion Analysis API
// Supports Kotlin 1.9+, Android API 21+

package com.miyraa.sdk

import kotlinx.coroutines.*
import kotlinx.serialization.*
import kotlinx.serialization.json.*
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.util.concurrent.ConcurrentHashMap
import kotlin.math.min

// MARK: - Models

/**
 * Emotion analysis result
 */
@Serializable
data class EmotionResult(
    val text: String,
    val predictions: List<Prediction>,
    val language: String? = null,
    @SerialName("language_name") val languageName: String? = null,
    val explanation: Explanation? = null,
    @SerialName("processing_time") val processingTime: Double? = null
)

/**
 * Prediction label and score
 */
@Serializable
data class Prediction(
    val label: String,
    val score: Double
) {
    val emoji: String
        get() = when (label.lowercase()) {
            "joy", "happy" -> "😊"
            "sadness", "sad" -> "😢"
            "anger", "angry" -> "😠"
            "fear" -> "😨"
            "surprise" -> "😲"
            "disgust" -> "🤢"
            "love" -> "😍"
            else -> "😐"
        }
}

/**
 * Emotion explanation with keywords
 */
@Serializable
data class Explanation(
    val keywords: List<Keyword>,
    val emotion: String,
    val confidence: Double,
    @SerialName("has_negation") val hasNegation: Boolean,
    @SerialName("has_intensifier") val hasIntensifier: Boolean,
    val explanation: String
)

/**
 * Keyword with score
 */
@Serializable
data class Keyword(
    val word: String,
    val score: Double,
    val span: List<Int>
)

/**
 * Text summary result
 */
@Serializable
data class SummaryResult(
    val summary: String,
    @SerialName("original_length") val originalLength: Int,
    @SerialName("summary_length") val summaryLength: Int,
    @SerialName("compression_ratio") val compressionRatio: Double,
    @SerialName("num_sentences") val numSentences: Int
)

/**
 * Batch analysis request
 */
@Serializable
data class BatchRequest(
    val texts: List<String>,
    val task: String? = null,
    val language: String? = null
)

/**
 * Batch analysis response
 */
@Serializable
data class BatchResponse(
    val results: List<EmotionResult>,
    @SerialName("total_processing_time") val totalProcessingTime: Double? = null
)

// MARK: - Configuration

/**
 * SDK Configuration
 */
data class MiyraaConfig(
    val baseURL: String = "http://localhost:8000",
    val apiKey: String? = null,
    val timeout: Long = 30000L, // milliseconds
    val retryAttempts: Int = 3,
    val retryDelay: Long = 1000L, // milliseconds
    val enableCache: Boolean = true,
    val cacheTTL: Long = 3600000L // milliseconds (1 hour)
)

// MARK: - Cache

/**
 * Simple in-memory cache
 */
class Cache<K, V>(private val ttl: Long = 3600000L) {
    private data class CacheEntry<V>(
        val value: V,
        val timestamp: Long
    )

    private val storage = ConcurrentHashMap<K, CacheEntry<V>>()

    fun get(key: K): V? {
        val entry = storage[key] ?: return null

        // Check expiration
        if (System.currentTimeMillis() - entry.timestamp > ttl) {
            storage.remove(key)
            return null
        }

        return entry.value
    }

    fun set(key: K, value: V) {
        storage[key] = CacheEntry(value, System.currentTimeMillis())
    }

    fun clear() {
        storage.clear()
    }

    fun size(): Int = storage.size
}

// MARK: - Errors

/**
 * SDK Exceptions
 */
sealed class MiyraaException(message: String, cause: Throwable? = null) : Exception(message, cause) {
    class InvalidURL(message: String = "Invalid API URL") : MiyraaException(message)
    class NetworkError(cause: Throwable) : MiyraaException("Network error: ${cause.message}", cause)
    class InvalidResponse(message: String = "Invalid server response") : MiyraaException(message)
    class DecodingError(cause: Throwable) : MiyraaException("Failed to decode response: ${cause.message}", cause)
    class ServerError(val code: Int, message: String) : MiyraaException("Server error ($code): $message")
    class Timeout(message: String = "Request timeout") : MiyraaException(message)
    class RateLimitExceeded(message: String = "Rate limit exceeded") : MiyraaException(message)
    class Unauthorized(message: String = "Unauthorized: Invalid API key") : MiyraaException(message)
}

// MARK: - SDK

/**
 * Miyraa SDK Client
 */
class MiyraaSDK(private val config: MiyraaConfig = MiyraaConfig()) {

    private val json = Json {
        ignoreUnknownKeys = true
        isLenient = true
    }

    private val cache = Cache<String, EmotionResult>(ttl = config.cacheTTL)

    // MARK: - Coroutine API

    /**
     * Analyze emotion in text
     */
    suspend fun analyze(
        text: String,
        task: String = "emotion",
        language: String? = null
    ): EmotionResult = withContext(Dispatchers.IO) {
        // Check cache
        val cacheKey = "$text:$task:${language ?: "auto"}"
        if (config.enableCache) {
            cache.get(cacheKey)?.let { return@withContext it }
        }

        // Build URL
        val params = buildString {
            append("text=${URLEncoder.encode(text, "UTF-8")}")
            append("&task=${URLEncoder.encode(task, "UTF-8")}")
            language?.let { append("&language=${URLEncoder.encode(it, "UTF-8")}") }
        }
        val url = "${config.baseURL}/analyze?$params"

        // Execute request
        val result = executeWithRetry<EmotionResult>(url, method = "GET")

        // Cache result
        if (config.enableCache) {
            cache.set(cacheKey, result)
        }

        result
    }

    /**
     * Batch analyze multiple texts
     */
    suspend fun analyzeBatch(
        texts: List<String>,
        task: String = "emotion",
        language: String? = null
    ): BatchResponse = withContext(Dispatchers.IO) {
        val url = "${config.baseURL}/batch"
        val batchRequest = BatchRequest(texts, task, language)
        val body = json.encodeToString(batchRequest)

        executeWithRetry(url, method = "POST", body = body)
    }

    /**
     * Get emotion explanation
     */
    suspend fun explain(
        text: String,
        emotion: String
    ): Explanation = withContext(Dispatchers.IO) {
        val params = buildString {
            append("text=${URLEncoder.encode(text, "UTF-8")}")
            append("&emotion=${URLEncoder.encode(emotion, "UTF-8")}")
        }
        val url = "${config.baseURL}/explain?$params"

        executeWithRetry(url, method = "GET")
    }

    /**
     * Summarize text
     */
    suspend fun summarize(
        text: String,
        numSentences: Int? = null,
        ratio: Double? = null
    ): SummaryResult = withContext(Dispatchers.IO) {
        val params = buildString {
            append("text=${URLEncoder.encode(text, "UTF-8")}")
            numSentences?.let { append("&num_sentences=$it") }
            ratio?.let { append("&ratio=$it") }
        }
        val url = "${config.baseURL}/summarize?$params"

        executeWithRetry(url, method = "GET")
    }

    // MARK: - Callback API

    /**
     * Analyze emotion with callback
     */
    fun analyze(
        text: String,
        task: String = "emotion",
        language: String? = null,
        callback: (Result<EmotionResult>) -> Unit
    ) {
        CoroutineScope(Dispatchers.Main).launch {
            try {
                val result = analyze(text, task, language)
                callback(Result.success(result))
            } catch (e: Exception) {
                callback(Result.failure(e))
            }
        }
    }

    /**
     * Batch analyze with callback
     */
    fun analyzeBatch(
        texts: List<String>,
        task: String = "emotion",
        language: String? = null,
        callback: (Result<BatchResponse>) -> Unit
    ) {
        CoroutineScope(Dispatchers.Main).launch {
            try {
                val result = analyzeBatch(texts, task, language)
                callback(Result.success(result))
            } catch (e: Exception) {
                callback(Result.failure(e))
            }
        }
    }

    // MARK: - Utilities

    /**
     * Clear cache
     */
    fun clearCache() {
        cache.clear()
    }

    /**
     * Get cache size
     */
    fun getCacheSize(): Int = cache.size()

    // MARK: - Private

    private suspend inline fun <reified T> executeWithRetry(
        url: String,
        method: String = "GET",
        body: String? = null
    ): T {
        var lastException: Exception? = null

        repeat(config.retryAttempts) { attempt ->
            try {
                return execute(url, method, body)
            } catch (e: Exception) {
                lastException = e

                // Don't retry on client errors
                when (e) {
                    is MiyraaException.Unauthorized -> throw e
                    is MiyraaException.ServerError -> if (e.code < 500) throw e
                }

                // Wait before retry
                if (attempt < config.retryAttempts - 1) {
                    delay(config.retryDelay)
                }
            }
        }

        throw lastException ?: MiyraaException.NetworkError(IOException("Unknown error"))
    }

    private inline fun <reified T> execute(
        url: String,
        method: String = "GET",
        body: String? = null
    ): T {
        val connection = URL(url).openConnection() as HttpURLConnection

        try {
            connection.requestMethod = method
            connection.connectTimeout = config.timeout.toInt()
            connection.readTimeout = config.timeout.toInt()
            connection.setRequestProperty("Accept", "application/json")

            if (body != null) {
                connection.setRequestProperty("Content-Type", "application/json")
                connection.doOutput = true
            }

            config.apiKey?.let {
                connection.setRequestProperty("Authorization", "Bearer $it")
            }

            // Write body if present
            body?.let {
                connection.outputStream.use { os ->
                    os.write(it.toByteArray())
                }
            }

            // Get response
            val responseCode = connection.responseCode

            when (responseCode) {
                in 200..299 -> {
                    val response = connection.inputStream.bufferedReader().use { it.readText() }
                    return json.decodeFromString(response)
                }
                401 -> throw MiyraaException.Unauthorized()
                429 -> throw MiyraaException.RateLimitExceeded()
                in 400..499 -> {
                    val message = connection.errorStream?.bufferedReader()?.use { it.readText() } ?: "Client error"
                    throw MiyraaException.ServerError(responseCode, message)
                }
                in 500..599 -> {
                    val message = connection.errorStream?.bufferedReader()?.use { it.readText() } ?: "Server error"
                    throw MiyraaException.ServerError(responseCode, message)
                }
                else -> throw MiyraaException.InvalidResponse()
            }

        } catch (e: IOException) {
            throw MiyraaException.NetworkError(e)
        } catch (e: SerializationException) {
            throw MiyraaException.DecodingError(e)
        } finally {
            connection.disconnect()
        }
    }
}

// MARK: - Example Usage

/*
// Initialize SDK
val config = MiyraaConfig(
    baseURL = "https://api.miyraa.com",
    apiKey = "your-api-key",
    timeout = 30000L,
    enableCache = true
)
val sdk = MiyraaSDK(config)

// Analyze with coroutines
lifecycleScope.launch {
    try {
        val result = sdk.analyze("I'm so happy today!", task = "emotion")
        println("Emotion: ${result.predictions.firstOrNull()?.label ?: "unknown"}")
        println("Score: ${result.predictions.firstOrNull()?.score ?: 0.0}")
        println("Emoji: ${result.predictions.firstOrNull()?.emoji ?: "😐"}")
    } catch (e: MiyraaException) {
        println("Error: ${e.message}")
    }
}

// Analyze with callback
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

// Explain emotion
lifecycleScope.launch {
    val explanation = sdk.explain("I'm so happy!", emotion = "joy")
    println("Keywords: ${explanation.keywords.joinToString { it.word }}")
    println("Confidence: ${explanation.confidence}")
}

// Summarize
lifecycleScope.launch {
    val summary = sdk.summarize(longText, ratio = 0.3)
    println("Summary: ${summary.summary}")
    println("Compression: ${summary.compressionRatio}")
}

// Clear cache
sdk.clearCache()
println("Cache size: ${sdk.getCacheSize()}")
*/
