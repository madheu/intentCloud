#!/usr/bin/env python3
"""
视觉识别工具 — 调用 Gemini API 分析图片内容。

用法:
  python vision.py <图片路径> [提示词]

环境变量:
  GEMINI_API_KEY  - Gemini API Key（必需）
"""

import base64
import json
import os
import sys
import urllib.request
import urllib.error
import urllib.parse

API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash"  # 免费额度，支持多模态


def encode_image(image_path: str) -> tuple[str, str]:
    """读取图片并返回 base64 和 MIME 类型"""
    ext = image_path.lower().rsplit(".", 1)[-1] if "." in image_path else "png"
    mime_map = {
        "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "gif": "image/gif", "webp": "image/webp", "bmp": "image/bmp",
    }
    mime = mime_map.get(ext, "image/png")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return b64, mime


def analyze_image(image_path: str, prompt: str = "请详细描述这张图片的内容") -> str:
    """调用 Gemini API 分析图片"""
    if not API_KEY:
        return "错误：未设置 GEMINI_API_KEY 环境变量"

    if not os.path.isfile(image_path):
        return f"错误：找不到图片文件 {image_path}"

    b64, mime = encode_image(image_path)

    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{GEMINI_MODEL}:generateContent?key={urllib.parse.quote(API_KEY)}")

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime, "data": b64}}
            ]
        }],
        "safetySettings": [
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
        ]
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))

            # 检查是否被安全过滤
            if "promptFeedback" in result and result["promptFeedback"].get("blockReason"):
                return f"图片被安全过滤: {result['promptFeedback']['blockReason']}"

            candidates = result.get("candidates", [])
            if not candidates:
                return f"API 返回空结果: {json.dumps(result, ensure_ascii=False)[:500]}"

            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)

            usage = result.get("usageMetadata", {})
            info = ""
            if usage:
                info = (
                    f"\n\n---\n"
                    f"模型: {GEMINI_MODEL} | "
                    f"输入 tokens: {usage.get('promptTokenCount', '?')} | "
                    f"输出 tokens: {usage.get('candidatesTokenCount', '?')}"
                )
            return text + info

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        # 尝试提取更友好的错误信息
        try:
            err_json = json.loads(body)
            msg = err_json.get("error", {}).get("message", body[:500])
        except json.JSONDecodeError:
            msg = body[:500]
        return f"API 请求失败 (HTTP {e.code}):\n{msg}"
    except urllib.error.URLError as e:
        return f"网络错误: {e.reason}"
    except json.JSONDecodeError as e:
        return f"JSON 解析失败: {e}"
    except Exception as e:
        return f"未知错误: {type(e).__name__}: {e}"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Gemini 图片识别")
    parser.add_argument("image", help="图片文件路径")
    parser.add_argument("prompt", nargs="?", default="请详细描述这张图片的内容",
                        help="识别提示词（可选）")
    args = parser.parse_args()
    result = analyze_image(args.image, args.prompt)
    print(result)


if __name__ == "__main__":
    main()
