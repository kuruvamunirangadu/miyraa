"""
Caching layer for inference results.
Supports in-memory LRU cache and Redis (optional).
"""

from typing import Optional, Any, Dict, Callable
import hashlib
import json
import time
from functools import wraps
from collections import OrderedDict
import threading


class LRUCache:
    """Thread-safe LRU (Least Recently Used) cache"""

    def __init__(self, max_size: int = 1000, ttl: Optional[int] = 3600):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of items in cache
            ttl: Time-to-live in seconds (None = no expiration)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        self.timestamps = {}
        self.lock = threading.RLock()

        # Stats
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None

            # Check TTL
            if self.ttl is not None:
                timestamp = self.timestamps.get(key, 0)
                if time.time() - timestamp > self.ttl:
                    # Expired
                    del self.cache[key]
                    del self.timestamps[key]
                    self.misses += 1
                    return None

            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]

    def set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        with self.lock:
            # Update existing key
            if key in self.cache:
                self.cache.move_to_end(key)
                self.cache[key] = value
                self.timestamps[key] = time.time()
                return

            # Add new key
            self.cache[key] = value
            self.timestamps[key] = time.time()

            # Evict oldest if over capacity
            if len(self.cache) > self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
                self.evictions += 1

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                if key in self.timestamps:
                    del self.timestamps[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()

    def size(self) -> int:
        """Get current cache size"""
        with self.lock:
            return len(self.cache)

    def stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

            return {
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_rate": hit_rate,
            }


class RedisCache:
    """Redis-based cache (requires redis-py)"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        ttl: int = 3600,
        key_prefix: str = "miyraa:",
    ):
        """
        Initialize Redis cache.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            ttl: Time-to-live in seconds
            key_prefix: Prefix for all keys
        """
        try:
            import redis
            self.redis = redis.Redis(
                host=host, port=port, db=db, decode_responses=True
            )
            self.redis.ping()  # Test connection
            self.available = True
        except (ImportError, Exception) as e:
            print(f"Redis not available: {e}")
            self.redis = None
            self.available = False

        self.ttl = ttl
        self.key_prefix = key_prefix

        # Stats
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis"""
        if not self.available:
            return None

        try:
            full_key = self.key_prefix + key
            value = self.redis.get(full_key)

            if value is None:
                self.misses += 1
                return None

            self.hits += 1
            return json.loads(value)
        except Exception as e:
            print(f"Redis get error: {e}")
            self.misses += 1
            return None

    def set(self, key: str, value: Any) -> None:
        """Set value in Redis"""
        if not self.available:
            return

        try:
            full_key = self.key_prefix + key
            serialized = json.dumps(value)
            self.redis.setex(full_key, self.ttl, serialized)
        except Exception as e:
            print(f"Redis set error: {e}")

    def delete(self, key: str) -> bool:
        """Delete key from Redis"""
        if not self.available:
            return False

        try:
            full_key = self.key_prefix + key
            result = self.redis.delete(full_key)
            return result > 0
        except Exception as e:
            print(f"Redis delete error: {e}")
            return False

    def clear(self) -> None:
        """Clear all keys with prefix"""
        if not self.available:
            return

        try:
            pattern = self.key_prefix + "*"
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
        except Exception as e:
            print(f"Redis clear error: {e}")

    def stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "available": self.available,
        }


class CacheManager:
    """Unified cache manager supporting both LRU and Redis"""

    def __init__(
        self,
        backend: str = "lru",
        max_size: int = 1000,
        ttl: int = 3600,
        redis_config: Optional[Dict] = None,
    ):
        """
        Initialize cache manager.

        Args:
            backend: Cache backend ('lru' or 'redis')
            max_size: Max size for LRU cache
            ttl: Time-to-live in seconds
            redis_config: Redis configuration dict
        """
        self.backend = backend

        if backend == "redis":
            redis_config = redis_config or {}
            self.cache = RedisCache(ttl=ttl, **redis_config)
            if not self.cache.available:
                print("Falling back to LRU cache")
                self.cache = LRUCache(max_size=max_size, ttl=ttl)
                self.backend = "lru"
        else:
            self.cache = LRUCache(max_size=max_size, ttl=ttl)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        return self.cache.get(key)

    def set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        self.cache.set(key, value)

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        return self.cache.delete(key)

    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = self.cache.stats()
        stats["backend"] = self.backend
        return stats


def generate_cache_key(text: str, **kwargs) -> str:
    """
    Generate cache key from text and parameters.

    Args:
        text: Input text
        **kwargs: Additional parameters (model, task, etc.)

    Returns:
        Cache key (SHA256 hash)
    """
    # Normalize text
    text = text.strip().lower()

    # Include relevant parameters in key
    key_data = {"text": text}
    key_data.update(kwargs)

    # Generate hash
    key_str = json.dumps(key_data, sort_keys=True)
    key_hash = hashlib.sha256(key_str.encode()).hexdigest()[:16]

    return key_hash


def cached_inference(cache_manager: CacheManager):
    """
    Decorator for caching inference results.

    Usage:
        @cached_inference(cache_manager)
        def predict(text, model):
            return model(text)
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from first argument (text) and kwargs
            if args:
                text = args[0]
                cache_key = generate_cache_key(text, **kwargs)

                # Try to get from cache
                cached_result = cache_manager.get(cache_key)
                if cached_result is not None:
                    return cached_result

                # Compute result
                result = func(*args, **kwargs)

                # Store in cache
                cache_manager.set(cache_key, result)

                return result
            else:
                # No caching if no arguments
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Global cache instance
_global_cache = None


def get_cache(
    backend: str = "lru",
    max_size: int = 1000,
    ttl: int = 3600,
    redis_config: Optional[Dict] = None,
) -> CacheManager:
    """Get or create global cache instance"""
    global _global_cache

    if _global_cache is None:
        _global_cache = CacheManager(
            backend=backend,
            max_size=max_size,
            ttl=ttl,
            redis_config=redis_config,
        )

    return _global_cache


if __name__ == "__main__":
    # Demo
    print("=== LRU Cache Demo ===")
    lru_cache = CacheManager(backend="lru", max_size=3, ttl=None)

    # Add items
    lru_cache.set("key1", {"result": "value1"})
    lru_cache.set("key2", {"result": "value2"})
    lru_cache.set("key3", {"result": "value3"})

    print(f"Cache size: {lru_cache.stats()['size']}")

    # Get items
    print(f"Get key1: {lru_cache.get('key1')}")
    print(f"Get key2: {lru_cache.get('key2')}")

    # Add one more (should evict key3)
    lru_cache.set("key4", {"result": "value4"})
    print(f"Get key3 (should be evicted): {lru_cache.get('key3')}")

    # Stats
    print(f"\nCache stats: {lru_cache.stats()}")

    # Cache key generation
    print("\n=== Cache Key Generation ===")
    key1 = generate_cache_key("Hello world", model="xlm-roberta")
    key2 = generate_cache_key("Hello world", model="xlm-roberta")
    key3 = generate_cache_key("Hello world", model="bert")

    print(f"Key1: {key1}")
    print(f"Key2: {key2}")
    print(f"Key1 == Key2: {key1 == key2}")
    print(f"Key1 == Key3: {key1 == key3}")
