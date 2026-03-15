# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Image utility functions for processing and converting images
"""

import base64
import io
from pathlib import Path
from PIL import Image


def convert_png_b64_to_jpg_b64(png_b64_str: str) -> str:
    """
    Convert a PNG base64 string to a JPG base64 string.
    
    Args:
        png_b64_str: Base64 encoded PNG image string
        
    Returns:
        Base64 encoded JPG image string, or None if conversion fails
    """
    try:
        if not png_b64_str or len(png_b64_str) < 10:
            print(f"⚠️  Invalid base64 string (too short): {png_b64_str[:50] if png_b64_str else 'None'}")
            return None
            
        img = Image.open(io.BytesIO(base64.b64decode(png_b64_str))).convert("RGB")
        out_io = io.BytesIO()
        img.save(out_io, format="JPEG", quality=95)
        return base64.b64encode(out_io.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"❌ Error converting image: {e}")
        print(f"   Input preview: {png_b64_str[:100] if png_b64_str else 'None'}")
        return None


def base64_to_image(b64_str: str) -> Image.Image | None:
    """
    Decode a base64 image string into a PIL image.

    Args:
        b64_str: Base64 image payload, with or without a data URL prefix

    Returns:
        PIL Image instance, or None if decoding fails
    """
    try:
        if not b64_str:
            return None

        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]

        image = Image.open(io.BytesIO(base64.b64decode(b64_str)))
        image.load()
        return image
    except Exception as e:
        print(f"❌ Error decoding base64 image: {e}")
        return None


def save_base64_image_as_png(b64_str: str, output_path: str | Path) -> Path | None:
    """
    Save a base64 image payload as a PNG file.

    Args:
        b64_str: Base64-encoded image payload
        output_path: Destination file path

    Returns:
        Path to the saved file, or None if saving fails
    """
    try:
        image = base64_to_image(b64_str)
        if image is None:
            return None

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(output_path, format="PNG")
        return output_path
    except Exception as e:
        print(f"❌ Error saving image to {output_path}: {e}")
        return None
