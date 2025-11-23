// MiyraaSDK.swift
// iOS SDK for Miyraa Emotion Analysis API
// Supports Swift 5.5+, iOS 13+

import Foundation
import Combine

// MARK: - Models

/// Emotion analysis result
public struct EmotionResult: Codable {
    public let text: String
    public let predictions: [Prediction]
    public let language: String?
    public let languageName: String?
    public let explanation: Explanation?
    public let processingTime: Double?
    
    enum CodingKeys: String, CodingKey {
        case text, predictions, language
        case languageName = "language_name"
        case explanation
        case processingTime = "processing_time"
    }
}

/// Prediction label and score
public struct Prediction: Codable {
    public let label: String
    public let score: Double
    
    public var emoji: String {
        switch label.lowercased() {
        case "joy", "happy": return "😊"
        case "sadness", "sad": return "😢"
        case "anger", "angry": return "😠"
        case "fear": return "😨"
        case "surprise": return "😲"
        case "disgust": return "🤢"
        case "love": return "😍"
        default: return "😐"
        }
    }
}

/// Emotion explanation with keywords
public struct Explanation: Codable {
    public let keywords: [Keyword]
    public let emotion: String
    public let confidence: Double
    public let hasNegation: Bool
    public let hasIntensifier: Bool
    public let explanation: String
    
    enum CodingKeys: String, CodingKey {
        case keywords, emotion, confidence
        case hasNegation = "has_negation"
        case hasIntensifier = "has_intensifier"
        case explanation
    }
}

/// Keyword with score
public struct Keyword: Codable {
    public let word: String
    public let score: Double
    public let span: [Int]
}

/// Text summary result
public struct SummaryResult: Codable {
    public let summary: String
    public let originalLength: Int
    public let summaryLength: Int
    public let compressionRatio: Double
    public let numSentences: Int
    
    enum CodingKeys: String, CodingKey {
        case summary
        case originalLength = "original_length"
        case summaryLength = "summary_length"
        case compressionRatio = "compression_ratio"
        case numSentences = "num_sentences"
    }
}

/// Batch analysis request
public struct BatchRequest: Codable {
    public let texts: [String]
    public let task: String?
    public let language: String?
    
    public init(texts: [String], task: String? = nil, language: String? = nil) {
        self.texts = texts
        self.task = task
        self.language = language
    }
}

/// Batch analysis response
public struct BatchResponse: Codable {
    public let results: [EmotionResult]
    public let totalProcessingTime: Double?
    
    enum CodingKeys: String, CodingKey {
        case results
        case totalProcessingTime = "total_processing_time"
    }
}

// MARK: - Configuration

/// SDK Configuration
public struct MiyraaConfig {
    public let baseURL: String
    public let apiKey: String?
    public let timeout: TimeInterval
    public let retryAttempts: Int
    public let retryDelay: TimeInterval
    public let enableCache: Bool
    public let cacheTTL: TimeInterval
    
    public init(
        baseURL: String = "http://localhost:8000",
        apiKey: String? = nil,
        timeout: TimeInterval = 30.0,
        retryAttempts: Int = 3,
        retryDelay: TimeInterval = 1.0,
        enableCache: Bool = true,
        cacheTTL: TimeInterval = 3600
    ) {
        self.baseURL = baseURL
        self.apiKey = apiKey
        self.timeout = timeout
        self.retryAttempts = retryAttempts
        self.retryDelay = retryDelay
        self.enableCache = enableCache
        self.cacheTTL = cacheTTL
    }
}

// MARK: - Cache

/// Simple in-memory cache
class Cache<Key: Hashable, Value> {
    private var storage: [Key: (value: Value, timestamp: Date)] = [:]
    private let lock = NSLock()
    private let ttl: TimeInterval
    
    init(ttl: TimeInterval = 3600) {
        self.ttl = ttl
    }
    
    func get(_ key: Key) -> Value? {
        lock.lock()
        defer { lock.unlock() }
        
        guard let entry = storage[key] else {
            return nil
        }
        
        // Check expiration
        if Date().timeIntervalSince(entry.timestamp) > ttl {
            storage.removeValue(forKey: key)
            return nil
        }
        
        return entry.value
    }
    
    func set(_ key: Key, value: Value) {
        lock.lock()
        defer { lock.unlock() }
        
        storage[key] = (value, Date())
    }
    
    func clear() {
        lock.lock()
        defer { lock.unlock() }
        
        storage.removeAll()
    }
}

// MARK: - Errors

/// SDK Errors
public enum MiyraaError: Error, LocalizedError {
    case invalidURL
    case networkError(Error)
    case invalidResponse
    case decodingError(Error)
    case serverError(Int, String)
    case timeout
    case rateLimitExceeded
    case unauthorized
    
    public var errorDescription: String? {
        switch self {
        case .invalidURL:
            return "Invalid API URL"
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        case .invalidResponse:
            return "Invalid server response"
        case .decodingError(let error):
            return "Failed to decode response: \(error.localizedDescription)"
        case .serverError(let code, let message):
            return "Server error (\(code)): \(message)"
        case .timeout:
            return "Request timeout"
        case .rateLimitExceeded:
            return "Rate limit exceeded"
        case .unauthorized:
            return "Unauthorized: Invalid API key"
        }
    }
}

// MARK: - SDK

/// Miyraa SDK Client
public class MiyraaSDK {
    
    private let config: MiyraaConfig
    private let session: URLSession
    private let cache: Cache<String, EmotionResult>
    
    /// Initialize SDK with configuration
    public init(config: MiyraaConfig = MiyraaConfig()) {
        self.config = config
        
        let sessionConfig = URLSessionConfiguration.default
        sessionConfig.timeoutIntervalForRequest = config.timeout
        self.session = URLSession(configuration: sessionConfig)
        
        self.cache = Cache(ttl: config.cacheTTL)
    }
    
    // MARK: - Async/Await API
    
    /// Analyze emotion in text
    @available(iOS 13.0, *)
    public func analyze(
        text: String,
        task: String = "emotion",
        language: String? = nil
    ) async throws -> EmotionResult {
        // Check cache
        let cacheKey = "\(text):\(task):\(language ?? "auto")"
        if config.enableCache, let cached = cache.get(cacheKey) {
            return cached
        }
        
        // Build request
        let endpoint = "/analyze"
        var components = URLComponents(string: config.baseURL + endpoint)
        components?.queryItems = [
            URLQueryItem(name: "text", value: text),
            URLQueryItem(name: "task", value: task)
        ]
        if let language = language {
            components?.queryItems?.append(URLQueryItem(name: "language", value: language))
        }
        
        guard let url = components?.url else {
            throw MiyraaError.invalidURL
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        
        if let apiKey = config.apiKey {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        
        // Execute with retry
        let result: EmotionResult = try await executeWithRetry(request: request)
        
        // Cache result
        if config.enableCache {
            cache.set(cacheKey, value: result)
        }
        
        return result
    }
    
    /// Batch analyze multiple texts
    @available(iOS 13.0, *)
    public func analyzeBatch(
        texts: [String],
        task: String = "emotion",
        language: String? = nil
    ) async throws -> BatchResponse {
        let endpoint = "/batch"
        guard let url = URL(string: config.baseURL + endpoint) else {
            throw MiyraaError.invalidURL
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        if let apiKey = config.apiKey {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        
        let batchRequest = BatchRequest(texts: texts, task: task, language: language)
        request.httpBody = try JSONEncoder().encode(batchRequest)
        
        return try await executeWithRetry(request: request)
    }
    
    /// Get emotion explanation
    @available(iOS 13.0, *)
    public func explain(
        text: String,
        emotion: String
    ) async throws -> Explanation {
        let endpoint = "/explain"
        var components = URLComponents(string: config.baseURL + endpoint)
        components?.queryItems = [
            URLQueryItem(name: "text", value: text),
            URLQueryItem(name: "emotion", value: emotion)
        ]
        
        guard let url = components?.url else {
            throw MiyraaError.invalidURL
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        
        if let apiKey = config.apiKey {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        
        return try await executeWithRetry(request: request)
    }
    
    /// Summarize text
    @available(iOS 13.0, *)
    public func summarize(
        text: String,
        numSentences: Int? = nil,
        ratio: Double? = nil
    ) async throws -> SummaryResult {
        let endpoint = "/summarize"
        var components = URLComponents(string: config.baseURL + endpoint)
        var queryItems = [URLQueryItem(name: "text", value: text)]
        
        if let numSentences = numSentences {
            queryItems.append(URLQueryItem(name: "num_sentences", value: String(numSentences)))
        }
        if let ratio = ratio {
            queryItems.append(URLQueryItem(name: "ratio", value: String(ratio)))
        }
        
        components?.queryItems = queryItems
        
        guard let url = components?.url else {
            throw MiyraaError.invalidURL
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        
        if let apiKey = config.apiKey {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        
        return try await executeWithRetry(request: request)
    }
    
    // MARK: - Combine API
    
    /// Analyze emotion with Combine
    @available(iOS 13.0, *)
    public func analyzePublisher(
        text: String,
        task: String = "emotion",
        language: String? = nil
    ) -> AnyPublisher<EmotionResult, Error> {
        return Future<EmotionResult, Error> { promise in
            Task {
                do {
                    let result = try await self.analyze(text: text, task: task, language: language)
                    promise(.success(result))
                } catch {
                    promise(.failure(error))
                }
            }
        }.eraseToAnyPublisher()
    }
    
    // MARK: - Completion Handler API
    
    /// Analyze emotion with completion handler
    public func analyze(
        text: String,
        task: String = "emotion",
        language: String? = nil,
        completion: @escaping (Result<EmotionResult, Error>) -> Void
    ) {
        Task {
            do {
                let result = try await analyze(text: text, task: task, language: language)
                completion(.success(result))
            } catch {
                completion(.failure(error))
            }
        }
    }
    
    // MARK: - Utilities
    
    /// Clear cache
    public func clearCache() {
        cache.clear()
    }
    
    // MARK: - Private
    
    @available(iOS 13.0, *)
    private func executeWithRetry<T: Decodable>(request: URLRequest) async throws -> T {
        var lastError: Error?
        
        for attempt in 0..<config.retryAttempts {
            do {
                let (data, response) = try await session.data(for: request)
                
                guard let httpResponse = response as? HTTPURLResponse else {
                    throw MiyraaError.invalidResponse
                }
                
                // Handle HTTP errors
                switch httpResponse.statusCode {
                case 200...299:
                    break
                case 401:
                    throw MiyraaError.unauthorized
                case 429:
                    throw MiyraaError.rateLimitExceeded
                case 400...499:
                    let message = String(data: data, encoding: .utf8) ?? "Client error"
                    throw MiyraaError.serverError(httpResponse.statusCode, message)
                case 500...599:
                    let message = String(data: data, encoding: .utf8) ?? "Server error"
                    throw MiyraaError.serverError(httpResponse.statusCode, message)
                default:
                    throw MiyraaError.invalidResponse
                }
                
                // Decode response
                do {
                    let decoder = JSONDecoder()
                    return try decoder.decode(T.self, from: data)
                } catch {
                    throw MiyraaError.decodingError(error)
                }
                
            } catch {
                lastError = error
                
                // Don't retry on client errors
                if case MiyraaError.unauthorized = error {
                    throw error
                }
                if case MiyraaError.serverError(let code, _) = error, code < 500 {
                    throw error
                }
                
                // Wait before retry
                if attempt < config.retryAttempts - 1 {
                    try await Task.sleep(nanoseconds: UInt64(config.retryDelay * 1_000_000_000))
                }
            }
        }
        
        throw lastError ?? MiyraaError.networkError(NSError(domain: "Unknown", code: -1))
    }
}

// MARK: - Example Usage

/*
 // Initialize SDK
 let config = MiyraaConfig(
     baseURL: "https://api.miyraa.com",
     apiKey: "your-api-key",
     timeout: 30.0,
     enableCache: true
 )
 let sdk = MiyraaSDK(config: config)
 
 // Analyze with async/await
 Task {
     do {
         let result = try await sdk.analyze(text: "I'm so happy today!", task: "emotion")
         print("Emotion: \(result.predictions.first?.label ?? "unknown")")
         print("Score: \(result.predictions.first?.score ?? 0)")
     } catch {
         print("Error: \(error)")
     }
 }
 
 // Analyze with completion handler
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
 
 // Explain emotion
 Task {
     let explanation = try await sdk.explain(text: "I'm so happy!", emotion: "joy")
     print("Keywords: \(explanation.keywords.map { $0.word }.joined(separator: ", "))")
 }
 
 // Summarize
 Task {
     let summary = try await sdk.summarize(text: longText, ratio: 0.3)
     print("Summary: \(summary.summary)")
     print("Compression: \(summary.compressionRatio)")
 }
 */
