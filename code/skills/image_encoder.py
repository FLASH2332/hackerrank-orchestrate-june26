import os
import io
import base64
from PIL import Image

def encode_image(image_path: str, cache: dict) -> str:
    """
    Encodes an image to a base64 string.
    Checks the cache dictionary before loading/encoding.
    Caches and returns the encoded string.
    """
    # 1. Check cache first
    if image_path in cache:
        return cache[image_path]
        
    # 2. Cache miss: load image with Pillow
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")
        
    with Image.open(image_path) as img:
        img_format = img.format or "JPEG"
        buffer = io.BytesIO()
        img.save(buffer, format=img_format)
        img_bytes = buffer.getvalue()
        
    # 3. Convert to base64 and store in cache
    encoded = base64.b64encode(img_bytes).decode("utf-8")
    cache[image_path] = encoded
    
    return encoded


if __name__ == "__main__":
    import tempfile
    
    print("Running image_encoder tests...")
    
    # Create a temporary image file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        # Create a tiny 10x10 red image and save it to the temp path
        img_red = Image.new("RGB", (10, 10), color="red")
        img_red.save(tmp_path, format="PNG")
        
        cache = {}
        
        # Test Case 1: First encode (Cache Miss)
        encoded_1 = encode_image(tmp_path, cache)
        print("Test 1 (Cache Miss): Image successfully encoded.")
        assert tmp_path in cache
        assert cache[tmp_path] == encoded_1
        assert len(encoded_1) > 0
        
        # Modify the image on disk to be blue.
        # If the encoder uses cache, it should NOT load the blue image.
        img_blue = Image.new("RGB", (10, 10), color="blue")
        img_blue.save(tmp_path, format="PNG")
        
        # Test Case 2: Second encode (Cache Hit)
        encoded_2 = encode_image(tmp_path, cache)
        print("Test 2 (Cache Hit): Retreived from cache directly.")
        assert encoded_2 == encoded_1
        
    finally:
        # Clean up the temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    print("All image_encoder tests passed!")
