import time
import random
import logging
from typing import Callable
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

logger = logging.getLogger(__name__)

def safe_llm_request(
    request_fn: Callable[[], requests.Response], 
    max_retries: int = 5, 
    base_delay: float = 1.0, 
    max_delay: float = 20.0
) -> requests.Response:
    """
    Executes a requests.Response returning function with exponential backoff, jitter, and rate-limit handling.
    
    Thread-Safety: 
    This function is completely thread-safe for concurrent multi-agent execution. It only modifies 
    local loop state. Multi-agent safety is guaranteed provided the `request_fn` itself does not 
    mutate shared global state unsafely (e.g., standard isolated requests.post calls are safe).
    """
    retries = 0
    last_exception = None
    
    while retries <= max_retries:
        try:
            # Execute the request. The request_fn should handle its own per-request timeouts.
            response = request_fn()
            
            # If successful (2xx), return immediately
            response.raise_for_status()
            return response
            
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else None
            last_exception = e
            
            if status_code == 429:
                # Handle Rate Limiting: Respect Retry-After header if present
                retry_after = e.response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    delay = float(retry_after)
                else:
                    delay = min(max_delay, base_delay * (2 ** retries))
            elif status_code and 500 <= status_code < 600:
                # Handle Server Errors
                delay = min(max_delay, base_delay * (2 ** retries))
            else:
                # Fatal Client Error (4xx other than 429) -> do not retry
                raise e
                
        except (Timeout, ConnectionError, RequestException) as e:
            # Handle Network Timeouts and Connection Errors
            last_exception = e
            delay = min(max_delay, base_delay * (2 ** retries))
            
        if retries == max_retries:
            break
            
        # Add random jitter (0 to 50% of the delay) to prevent thundering herd in concurrent setups
        jitter = random.uniform(0, 0.5 * delay)
        sleep_time = min(max_delay, delay + jitter)
        
        logger.warning(
            f"Request failed: {last_exception}. "
            f"Retrying in {sleep_time:.2f}s (Attempt {retries + 1}/{max_retries})"
        )
        time.sleep(sleep_time)
        retries += 1
        
    raise Exception(
        f"safe_llm_request failed after {max_retries} retries. "
        f"Last error: {last_exception}"
    ) from last_exception

if __name__ == "__main__":
    pass
