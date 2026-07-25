#!/usr/bin/env python3
"""
LongCat 视觉识别工具
通过 LongCat API（OpenAI 兼容格式）分析图片内容。

用法:
  python longcat_vision.py <图片路径> [提示词] [--model 模型名]

环境变量:
  LONGCAT_API_KEY  - API Key（必需）
  LONGCAT_BASE_URL - API 地址（默认 https://api.longcat.chat/openai/v1）
"""

import base64
import json
import os
import sys
import urllib.request
import urllib.error

API_KEY = os.environ.get("LONGCAT_API_KEY")
BASE_URL = (os.environ.get("LONGCAT_BASE_URL")
            or "https://api.longcat.chat/openai/v1/chat/completions")
DEFAULT_MODEL = "gpt-4o"


def encode_image(image_path: str) -> tuple[str, str]:
    """读取图片并返回 base64 和 MIME 类型"""
    ext = image_path.lower().rsplit(".", 1)[-1] if "." in image_path else "png"
    mime_map = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
        "bmp": "image/bmp",
    }
    mime = mime_map.get(ext, "image/png")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return b64, mime


def analyze_image(image_path: str, prompt: str = "请详细描述这张图片的内容",
                  model: str = DEFAULT_MODEL) -> str:
    """调用 LongCat API 分析图片"""
    if not API_KEY:
        return "错误：未设置 LONGCAT_API_KEY 环境变量"

    if not os.path.isfile(image_path):
        return f"错误：找不到图片文件 {image_path}"

    b64, mime = encode_image(image_path)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{b64}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 2048
    }

    req = urllib.request.Request(
        BASE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            info = ""
            if usage:
                info = (
                    f"\n\n---\n"
                    f"模型: {result.get('model', model)} | "
                    f"输入 tokens: {usage.get('prompt_tokens', '?')} | "
                    f"输出 tokens: {usage.get('completion_tokens', '?')}"
                )
            return content + info
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return f"API 请求失败 (HTTP {e.code}):\n{body[:2000]}"
    except urllib.error.URLError as e:
        return f"网络错误: {e.reason}"
    except json.JSONDecodeError as e:
        return f"JSON 解析失败: {e}"
    except Exception as e:
        return f"未知错误: {type(e).__name__}: {e}"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LongCat 图片识别")
    parser.add_argument("image", help="图片文件路径")
    parser.add_argument("prompt", nargs="?", default="请详细描述这张图片的内容",
                        help="识别提示词（可选）")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"模型名称（默认: {DEFAULT_MODEL}）")
    args = parser.parse_args()

    result = analyze_image(args.image, args.prompt, args.model)
    print(result)


if __name__ == "__main__":
    main()
