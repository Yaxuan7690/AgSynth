import base64
import io
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, OpenAI, RateLimitError
from PIL import Image

load_dotenv(override=False)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("LLMUtils")


API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("OPENAI_BASE_URL")
DEFAULT_MODEL = os.environ.get("AGENT_DEFAULT_MODEL")


AGENT_MODELS = {
    name: os.environ.get(f"AGENT_{name.upper()}_MODEL")
    for name in (
        "concept_designer",
        "visual_specifier",
        "painter",
        "qa_generator",
        "auditor",
        "final_verifier_a",
        "final_verifier_b",
    )
}

FINAL_VERIFIER_MODELS = (
    AGENT_MODELS.get("final_verifier_a"),
    AGENT_MODELS.get("final_verifier_b"),
)

_image_cache: Dict[str, Tuple[float, str]] = {}


def validate_runtime_config() -> None:
    """Fail before generation when required runtime configuration is absent."""
    missing = []
    if not API_KEY:
        missing.append("OPENAI_API_KEY")
    if not DEFAULT_MODEL:
        missing.append("AGENT_DEFAULT_MODEL")
    if not FINAL_VERIFIER_MODELS[0]:
        missing.append("AGENT_FINAL_VERIFIER_A_MODEL")
    if not FINAL_VERIFIER_MODELS[1]:
        missing.append("AGENT_FINAL_VERIFIER_B_MODEL")

    if missing:
        names = ", ".join(missing)
        raise RuntimeError(
            f"Missing required environment variables: {names}. "
            "Copy .env.example to .env and configure the values before generating data."
        )

    if FINAL_VERIFIER_MODELS[0] == FINAL_VERIFIER_MODELS[1]:
        raise RuntimeError(
            "AGENT_FINAL_VERIFIER_A_MODEL and AGENT_FINAL_VERIFIER_B_MODEL must differ."
        )


def get_client():
    """Create the client from environment-only configuration."""
    client_options = {}
    if API_KEY:
        client_options["api_key"] = API_KEY
    if BASE_URL:
        client_options["base_url"] = BASE_URL
    return OpenAI(timeout=90.0, max_retries=2, **client_options)


def _model_supports_temperature(model_name: str) -> bool:
    """Internal component of the AgSynth generation pipeline."""
    if not model_name:
        return True
    return "claude" not in model_name.lower()


def _safe_json_parse(content: str) -> Dict[str, Any]:
    """Internal component of the AgSynth generation pipeline."""
    if not content:
        return {}

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    clean_content = content.strip()

    if clean_content.startswith("```"):
        lines = clean_content.split("\n")
        if len(lines) > 1:
            clean_content = "\n".join(lines[1:])
        if clean_content.endswith("```"):
            clean_content = clean_content[:-3]
        clean_content = clean_content.strip()

    protected_escapes = {
        r"\\n": "\x00NEWLINE\x00",
        r"\\t": "\x00TAB\x00",
        r"\\r": "\x00RETURN\x00",
        r'\\"': "\x00QUOTE\x00",
        r"\\\\": "\x00BACKSLASH\x00",
        r"\\\/": "\x00SLASH\x00",
        r"\\b": "\x00BACKSPACE\x00",
        r"\\f": "\x00FORMFEED\x00",
    }

    for escape_seq, placeholder in protected_escapes.items():
        clean_content = clean_content.replace(escape_seq, placeholder)

    clean_content = clean_content.replace("\\", "\\\\")

    for escape_seq, placeholder in protected_escapes.items():
        clean_content = clean_content.replace(placeholder, escape_seq)

    try:
        return json.loads(clean_content)
    except json.JSONDecodeError:
        pass

    def extract_json_recursive(text: str) -> Optional[dict]:
        """Internal component of the AgSynth generation pipeline."""
        stack = []
        start_idx = -1

        for i, char in enumerate(text):
            if char == "{":
                if not stack:
                    start_idx = i
                stack.append(char)
            elif char == "}":
                if stack:
                    stack.pop()
                    if not stack and start_idx >= 0:
                        json_str = text[start_idx : i + 1]
                        try:
                            result = json.loads(json_str)
                            if isinstance(result, dict):
                                return result
                        except json.JSONDecodeError:
                            pass
                        start_idx = -1
        return None

    extracted = extract_json_recursive(clean_content)
    if extracted:
        logger.info("Parsed JSON through recursive extraction")
        return extracted

    try:
        code_match = re.search(r'"python_code"\s*:\s*"((?:[^"\\]|\\.)*)"', clean_content, re.DOTALL)
        explanation_match = re.search(
            r'"code_explanation"\s*:\s*"((?:[^"\\]|\\.)*)"', clean_content, re.DOTALL
        )

        if code_match:
            result = {
                "python_code": code_match.group(1)
                .replace('\\"', '"')
                .replace("\\n", "\n")
                .replace("\\\\", "\\"),
                "code_explanation": explanation_match.group(1)
                if explanation_match
                else "Code generated",
            }
            logger.info("Parsed JSON through field extraction")
            return result
    except Exception as e:
        logger.warning(f"Field extraction failed: {e}")

    logger.error(f"JSON parsing failed after all attempts. First 200 characters: {content[:200]}")
    return {"error": "json_parse_failed", "raw_content": content[:500], "content": content}


def encode_image_to_base64(
    image_path: str, use_cache: bool = True, max_size_kb: int = 512
) -> Optional[str]:
    """Internal component of the AgSynth generation pipeline."""
    try:
        current_mtime = os.path.getmtime(image_path)
    except OSError:
        current_mtime = -1.0
    if use_cache and image_path in _image_cache:
        cached_mtime, cached_value = _image_cache[image_path]
        if cached_mtime == current_mtime:
            return cached_value

    try:
        Image.MAX_IMAGE_PIXELS = 500_000_000

        img = Image.open(image_path)
        original_size = os.path.getsize(image_path) / 1024
        original_pixels = img.width * img.height

        max_pixels = 5_000_000
        if original_pixels > max_pixels:
            logger.warning(
                f"⚠️ Image has too many pixels ({original_pixels:,} > {max_pixels:,})，scaling down to a safe size..."
            )
            scale = (max_pixels / original_pixels) ** 0.5
            new_width = int(img.width * scale)
            new_height = int(img.height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.info(
                f"✅ pixel pre-scaling: {original_pixels:,}px → {new_width}×{new_height} ({img.width * img.height:,}px)"
            )

        max_long_edge = 1024
        long_edge = max(img.width, img.height)
        if long_edge > max_long_edge:
            scale = max_long_edge / long_edge
            new_width = int(img.width * scale)
            new_height = int(img.height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.info(
                f"✅ long-edge pre-scaling: long edge {long_edge}px → {max(new_width, new_height)}px ({new_width}×{new_height})"
            )

        if img.mode in ("RGBA", "P", "LA"):
            rgba = img.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            background.alpha_composite(rgba)
            img = background.convert("RGB")
        elif img.mode == "L":
            img = img.convert("RGB")

        logger.info(
            f"Starting aggressive compression (original file {original_size:.1f}KB，target ≤{max_size_kb}KB)..."
        )

        compression_steps = [
            (70, 1.0),
            (60, 0.8),
            (50, 0.65),
            (40, 0.5),
            (30, 0.4),
            (20, 0.3),
        ]

        for quality, scale in compression_steps:
            if scale < 1.0:
                new_width = max(1, int(img.width * scale))
                new_height = max(1, int(img.height * scale))
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                resized_img = img

            buffer = io.BytesIO()
            resized_img.save(buffer, format="JPEG", quality=quality, optimize=True)
            compressed_size = len(buffer.getvalue()) / 1024

            if compressed_size <= max_size_kb:
                logger.info(
                    f"✅ Compression succeeded: {original_size:.1f}KB → {compressed_size:.1f}KB (quality={quality}, scale={scale:.0%}, size={resized_img.width}×{resized_img.height})"
                )
                encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
                if use_cache:
                    _image_cache[image_path] = (current_mtime, encoded)
                return encoded

        logger.warning("⚠️ Using final fallback compression（long edge 512px + quality 20）")
        fallback_long_edge = 512
        current_long_edge = max(img.width, img.height)
        fallback_scale = fallback_long_edge / current_long_edge
        fallback_width = max(1, int(img.width * fallback_scale))
        fallback_height = max(1, int(img.height * fallback_scale))
        fallback_img = img.resize((fallback_width, fallback_height), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        fallback_img.save(buffer, format="JPEG", quality=20, optimize=True)
        final_size = len(buffer.getvalue()) / 1024
        logger.info(
            f"Final fallback result: {final_size:.1f}KB ({fallback_width}×{fallback_height})"
        )
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        if use_cache:
            _image_cache[image_path] = (current_mtime, encoded)
        return encoded

    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}")
        return None
    except Exception as e:
        logger.error(f"Image encoding failed: {e}")
        return None


def get_image_mime_type_for_encoded(image_path: str) -> str:
    """Internal component of the AgSynth generation pipeline."""
    return "image/jpeg"


def chat_completion(
    system_prompt: str,
    user_prompt: str,
    json_mode: bool = True,
    timeout: float = 120.0,
    temperature: float = 1.0,
    max_tokens: int = 8192,
    model: str = None,
) -> Union[Dict[str, Any], str]:
    """Internal component of the AgSynth generation pipeline."""
    client = get_client()

    model_name = model or DEFAULT_MODEL

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **({"temperature": temperature} if _model_supports_temperature(model_name) else {}),
            timeout=timeout,
            max_tokens=max_tokens,
            extra_body={"model": model_name},
        )

        content = None

        if hasattr(response, "choices") and response.choices and len(response.choices) > 0:
            if hasattr(response.choices[0], "message") and hasattr(
                response.choices[0].message, "content"
            ):
                content = response.choices[0].message.content

        if content is None and hasattr(response, "content") and response.content:
            if isinstance(response.content, list) and len(response.content) > 0:
                first_item = response.content[0]
                if isinstance(first_item, dict) and "text" in first_item:
                    content = first_item["text"]
                elif isinstance(first_item, str):
                    content = first_item

        if content is None:
            logger.error(f"Model returned empty content: {response}")
            return {"error": "empty_response", "message": "API returned empty content"}

        if json_mode:
            return _safe_json_parse(content)
        else:
            return content

    except (APIConnectionError, APIError, RateLimitError) as e:
        logger.error(f"Model call failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


def vision_completion(
    system_prompt: str,
    image_path: str,
    user_prompt: str,
    timeout: float = 120.0,
    temperature: float = 1.0,
    max_tokens: int = 16384,
    json_mode: bool = True,
    model: str = None,
) -> Union[dict, str]:
    """Internal component of the AgSynth generation pipeline."""
    client = get_client()

    model_name = model or DEFAULT_MODEL

    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        raise ValueError(f"Unable to read image: {image_path}")

    mime_type = get_image_mime_type_for_encoded(image_path)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
            **({"temperature": temperature} if _model_supports_temperature(model_name) else {}),
            timeout=timeout,
            max_tokens=max_tokens,
            extra_body={"model": model_name},
        )

        content = None

        if hasattr(response, "choices") and response.choices and len(response.choices) > 0:
            if hasattr(response.choices[0], "message") and hasattr(
                response.choices[0].message, "content"
            ):
                content = response.choices[0].message.content

        if content is None and hasattr(response, "content") and response.content:
            if isinstance(response.content, list) and len(response.content) > 0:
                first_item = response.content[0]
                if isinstance(first_item, dict) and "text" in first_item:
                    content = first_item["text"]
                elif isinstance(first_item, str):
                    content = first_item

        if content is None:
            logger.error(f"Model returned empty content: {response}")
            return {"error": "empty_response", "message": "API returned empty content"}

        if json_mode:
            result = _safe_json_parse(content)
            if not isinstance(result, dict):
                logger.error(f"_safe_json_parse returned a non-dict type: {type(result)}")
                return {"error": "invalid_response_type", "content": str(result)}
            return result
        else:
            return content

    except (APIConnectionError, APIError, RateLimitError) as e:
        logger.error(f"Model call failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


def multi_vision_completion(
    system_prompt: str,
    image_paths: List[str],
    user_prompt: str,
    timeout: float = 120.0,
    temperature: float = 1.0,
    max_tokens: int = 16384,
    json_mode: bool = True,
    model: str = None,
) -> Union[dict, str]:
    """Internal component of the AgSynth generation pipeline."""
    client = get_client()

    model_name = model or DEFAULT_MODEL

    content_list = []

    if not image_paths:
        raise ValueError("At least one image is required for a vision request.")

    for image_path in image_paths:
        base64_image = encode_image_to_base64(image_path)
        if not base64_image:
            raise ValueError(f"Unable to read required image: {image_path}")

        mime_type = get_image_mime_type_for_encoded(image_path)
        content_list.append(
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
        )

    content_list.append({"type": "text", "text": user_prompt})

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_list},
            ],
            **({"temperature": temperature} if _model_supports_temperature(model_name) else {}),
            timeout=timeout,
            max_tokens=max_tokens,
            extra_body={"model": model_name},
        )

        content = None

        if hasattr(response, "choices") and response.choices and len(response.choices) > 0:
            if hasattr(response.choices[0], "message") and hasattr(
                response.choices[0].message, "content"
            ):
                content = response.choices[0].message.content

        if content is None and hasattr(response, "content") and response.content:
            if isinstance(response.content, list) and len(response.content) > 0:
                first_item = response.content[0]
                if isinstance(first_item, dict) and "text" in first_item:
                    content = first_item["text"]
                elif isinstance(first_item, str):
                    content = first_item

        if content is None:
            logger.error(f"Model returned empty content: {response}")
            return {"error": "empty_response", "message": "API returned empty content"}

        if json_mode:
            result = _safe_json_parse(content)
            if not isinstance(result, dict):
                logger.error(f"_safe_json_parse returned a non-dict type: {type(result)}")
                return {"error": "invalid_response_type", "content": str(result)}
            return result
        else:
            return content

    except (APIConnectionError, APIError, RateLimitError) as e:
        logger.error(f"Model call failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise


TEXT_MASKING_REWRITE_SYSTEM_PROMPT = """You are an expert math problem editor. Your task is to analyze a math problem and its accompanying figure, then either rewrite it or confirm it already requires the figure.

First, assess whether the original problem ALREADY requires the figure to be solved - meaning key numerical values, coordinates, geometric data, or labels are missing from the text and can only be obtained by looking at the figure.

If the original problem ALREADY depends on the figure (i.e., it cannot be solved from text alone), respond with this JSON:
{"already_figure_dependent": true, "rewritten_question": ""}

If the problem does NOT yet depend on the figure, rewrite it so that all specific numerical values, coordinates, ratios, lengths, angles, or labels that appear in the figure are REMOVED from the text and replaced with figure references.

Rewriting rules:
1. Any specific numerical values, coordinates, ratios, lengths, angles, or labels that appear in the figure must be removed from the text and replaced with figure references.
2. The problem must remain logically consistent and solvable ONLY by looking at the figure.
3. The final answer and the core mathematical question must remain unchanged.
4. Do NOT change the language of the problem (keep it in the same language as the original).
5. Use natural phrases like "as shown in the figure", "from the figure", "as indicated in the diagram", "refer to the figure" to replace removed information.
6. Keep the problem structure and solution steps logically intact.

When rewriting is needed, respond with this JSON:
{"already_figure_dependent": false, "rewritten_question": "<the full rewritten problem text here>"}

Only respond with the JSON object, nothing else."""

TEXT_MASKING_VALIDATE_SYSTEM_PROMPT = """You are a strict math problem quality checker. You will be given:
1. An original math problem
2. A modified version of the problem
3. The actual generated figure and an optional image description

Your task is to verify that the modified problem:
1. Cannot be solved using ONLY the text (without looking at the figure) - key data must be missing from text
2. Remains logically consistent and the final answer is still achievable
3. The core mathematical question and final answer are unchanged
4. References to the figure are natural and appropriate

Respond with a JSON object:
{"is_valid": true/false, "reason": "brief explanation of why it passes or fails"}

Only respond with the JSON object, nothing else."""

MASKING_REWRITE_MODEL = os.environ.get("AGENT_MASKING_REWRITE_MODEL")
MASKING_VALIDATE_MODEL = os.environ.get("AGENT_MASKING_VALIDATE_MODEL")
MASKING_MAX_RETRIES = 5


def rewrite_question_for_figure_dependency(question_text: str, image_path: str) -> dict:
    """Rewrite question to be figure-dependent.

    Returns dict with keys:
      - already_figure_dependent (bool)
      - rewritten_question (str)
    """
    user_text = (
        f"Here is the original math problem:\n\n{question_text}\n\n"
        "First assess whether this problem ALREADY requires the figure to solve it. "
        'If yes, respond with {"already_figure_dependent": true, "rewritten_question": ""}. '
        "If no, rewrite it so all specific values visible in the figure are removed and replaced "
        "with figure references, then respond with "
        '{"already_figure_dependent": false, "rewritten_question": "<rewritten text>"}. '
        "Return ONLY the JSON object."
    )

    result = vision_completion(
        system_prompt=TEXT_MASKING_REWRITE_SYSTEM_PROMPT,
        image_path=image_path,
        user_prompt=user_text,
        json_mode=True,
        model=MASKING_REWRITE_MODEL,
        temperature=0.3,
        max_tokens=8192,
    )

    if isinstance(result, dict):
        return {
            "already_figure_dependent": bool(result.get("already_figure_dependent", False)),
            "rewritten_question": result.get("rewritten_question", ""),
        }
    raise ValueError(f"Text masking rewrite returned non-dict: {result}")


def validate_masked_question(
    original_question: str,
    rewritten_question: str,
    image_composition: str,
    image_path: str = None,
) -> tuple:
    """Validate that the rewritten question requires the figure to solve.

    Returns (is_valid: bool, reason: str)
    """
    user_text = (
        f"Original problem:\n{original_question}\n\n"
        f"Modified problem:\n{rewritten_question}\n\n"
        f"Figure description (what is visible in the image):\n{image_composition}\n\n"
        "Please verify if the modified problem meets all the requirements."
    )

    if image_path:
        result = vision_completion(
            system_prompt=TEXT_MASKING_VALIDATE_SYSTEM_PROMPT,
            image_path=image_path,
            user_prompt=user_text,
            json_mode=True,
            model=MASKING_VALIDATE_MODEL,
            temperature=0.1,
            max_tokens=8192,
        )
    else:
        result = chat_completion(
            system_prompt=TEXT_MASKING_VALIDATE_SYSTEM_PROMPT,
            user_prompt=user_text,
            json_mode=True,
            model=MASKING_VALIDATE_MODEL,
            temperature=0.1,
            max_tokens=8192,
        )

    if isinstance(result, dict):
        is_valid = bool(result.get("is_valid", False))
        reason = result.get("reason", "No reason provided")
        return is_valid, reason
    raise ValueError(f"Text masking validation returned non-dict: {result}")
