"""
视觉处理核心 - 使用 MiniCPM-V API 处理图像并生成描述
"""

import base64
import io
import json
import os
from typing import Optional

import httpx
from PIL import Image

from logger_local import app_logger as logger


class VisionProcessor:
    """视觉处理器，用于调用 MiniCPM-V API 获取图像描述"""

    def __init__(self, api_key: Optional[str] = None, api_base_url: str = "http://localhost:8008/v1"):
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("MINICPM_API_KEY", "")
        self.model = os.getenv("MINICPM_MODEL", "openbmb/MiniCPM-V-4.6")
        self.timeout = int(os.getenv("MINICPM_TIMEOUT", "30"))
        logger.info(f"VisionProcessor 初始化完成，模型: {self.model}")

    def _encode_image(self, image_data: bytes, max_size: tuple = (1024, 1024), quality: int = 85) -> str:
        try:
            img = Image.open(io.BytesIO(image_data))
            if img.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality, optimize=True)
            encoded_bytes = buffer.getvalue()
            logger.debug(f"图像压缩: 原始大小 {len(image_data)} -> 压缩后 {len(encoded_bytes)}")
            return base64.b64encode(encoded_bytes).decode("utf-8")
        except Exception as e:
            logger.error(f"图像编码失败: {e}")
            return base64.b64encode(image_data).decode("utf-8")

    def describe_image(self, image_data: bytes, prompt: str = "请详细描述这张图片的内容。") -> Optional[str]:
        try:
            image_base64 = self._encode_image(image_data)
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        ],
                    }
                ],
            }

            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.api_base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

            return data["choices"][0]["message"]["content"]
        except httpx.HTTPError as e:
            logger.error(f"MiniCPM-V HTTP错误: {e}")
            return None
        except Exception as e:
            logger.error(f"获取图像描述失败: {type(e).__name__}: {e}")
            return None

    def describe_image_with_context(self, image_data: bytes, user_query: str) -> Optional[str]:
        prompt = f"""用户想要查询以下问题: "{user_query}"

请详细描述图片中的内容，特别注意：
1. 图片中显示的标志、标识、认证、标准号等关键信息
2. 图片中的产品型号、规格参数
3. 图片中的文字内容

描述要简洁准确，便于后续与用户的问题结合。"""
        return self.describe_image(image_data, prompt)

    def combine_query_with_description(self, user_query: str, image_description: str) -> str:
        combined_query = f"图中显示{image_description}，{user_query}"
        logger.info(f"组合查询: {combined_query}")
        return combined_query


_vision_processor: Optional[VisionProcessor] = None


def get_vision_processor() -> VisionProcessor:
    global _vision_processor
    if _vision_processor is None:
        _vision_processor = VisionProcessor()
    return _vision_processor
