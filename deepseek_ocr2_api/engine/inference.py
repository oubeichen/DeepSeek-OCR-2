"""
Inference module for DeepSeek-OCR-2 API Server.

Provides unified inference interface for both sync and async modes.
"""

import time
import logging
from typing import Optional, List, Dict, Any, Union, AsyncGenerator

from vllm import SamplingParams

from ..config import Settings, get_settings
from .manager import EngineManager  # This import sets up VLLM_DIR in sys.path

logger = logging.getLogger(__name__)

# Import from DeepSeek-OCR2-vllm (path already set by manager.py)
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor


def create_sampling_params(
    settings: Optional[Settings] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    ngram_size: Optional[int] = None,
    window_size: Optional[int] = None,
    skip_special_tokens: Optional[bool] = None,
    include_stop_str_in_output: bool = False,
    whitelist_token_ids: Optional[set] = None,
) -> SamplingParams:
    """
    Create SamplingParams with optional overrides.

    Args:
        settings: Base settings. If None, uses global settings.
        temperature: Override temperature.
        max_tokens: Override max tokens.
        ngram_size: Override N-gram size for repetition penalty.
        window_size: Override window size for N-gram check.
        skip_special_tokens: Override skip special tokens.
        include_stop_str_in_output: Include stop string in output (for PDF).
        whitelist_token_ids: Override whitelist token IDs.

    Returns:
        Configured SamplingParams instance.
    """
    settings = settings or get_settings()

    # Use provided values or fall back to settings
    _temperature = temperature if temperature is not None else settings.temperature
    _max_tokens = max_tokens if max_tokens is not None else settings.max_tokens
    _ngram_size = ngram_size if ngram_size is not None else settings.ngram_size
    _window_size = window_size if window_size is not None else settings.window_size
    _skip_special_tokens = skip_special_tokens if skip_special_tokens is not None else settings.skip_special_tokens
    _whitelist = whitelist_token_ids if whitelist_token_ids is not None else settings.whitelist_token_ids_set

    # Create logits processor for N-gram repetition penalty
    logits_processors = [
        NoRepeatNGramLogitsProcessor(
            ngram_size=_ngram_size,
            window_size=_window_size,
            whitelist_token_ids=_whitelist
        )
    ]

    return SamplingParams(
        temperature=_temperature,
        max_tokens=_max_tokens,
        logits_processors=logits_processors,
        skip_special_tokens=_skip_special_tokens,
        include_stop_str_in_output=include_stop_str_in_output,
    )


def sync_generate(
    prompts: Union[str, List[Dict[str, Any]]],
    sampling_params: Optional[SamplingParams] = None,
    settings: Optional[Settings] = None,
) -> List[str]:
    """
    Synchronous generation using LLM engine.

    Args:
        prompts: Single prompt string or list of prompt dicts with multi_modal_data.
        sampling_params: Sampling parameters. If None, creates default.
        settings: Settings for creating sampling params.

    Returns:
        List of generated text outputs.
    """
    manager = EngineManager.get_instance()

    if not manager.is_initialized():
        raise RuntimeError("Engine not initialized. Call initialize() first.")

    if manager.get_mode() != "sync":
        raise RuntimeError("Engine is in async mode. Use async_generate() instead.")

    engine = manager.get_engine()

    if sampling_params is None:
        sampling_params = create_sampling_params(settings)

    # Ensure prompts is a list
    if isinstance(prompts, str):
        prompts = [{"prompt": prompts}]
    elif isinstance(prompts, dict):
        prompts = [prompts]

    logger.info(f"Starting sync generation for {len(prompts)} prompt(s)...")
    start_time = time.time()

    outputs = engine.generate(prompts, sampling_params=sampling_params)

    elapsed = time.time() - start_time
    logger.info(f"Generation complete in {elapsed:.2f}s")

    # Extract text from outputs
    results = []
    for output in outputs:
        if output.outputs:
            text = output.outputs[0].text
            # Clean up end of sentence token if present
            if '<｜end▁of▁sentence｜>' in text:
                text = text.replace('<｜end▁of▁sentence｜>', '')
            results.append(text)
        else:
            results.append("")

    return results


async def async_generate(
    prompt: Union[str, Dict[str, Any]],
    sampling_params: Optional[SamplingParams] = None,
    settings: Optional[Settings] = None,
    stream: bool = False,
) -> Union[str, AsyncGenerator[str, None]]:
    """
    Asynchronous generation using AsyncLLMEngine.

    Args:
        prompt: Prompt string or dict with multi_modal_data.
        sampling_params: Sampling parameters. If None, creates default.
        settings: Settings for creating sampling params.
        stream: If True, yields tokens as they are generated.

    Returns:
        Generated text or async generator of tokens if stream=True.
    """
    manager = EngineManager.get_instance()

    if not manager.is_initialized():
        raise RuntimeError("Engine not initialized. Call initialize() first.")

    if manager.get_mode() != "async":
        raise RuntimeError("Engine is in sync mode. Use sync_generate() instead.")

    engine = manager.get_engine()

    if sampling_params is None:
        sampling_params = create_sampling_params(settings)

    # Prepare request
    if isinstance(prompt, str):
        request = {"prompt": prompt}
    else:
        request = prompt

    request_id = f"request-{int(time.time() * 1000)}"

    logger.info(f"Starting async generation (request_id={request_id})...")

    if stream:
        async def stream_generator():
            printed_length = 0
            async for request_output in engine.generate(request, sampling_params, request_id):
                if request_output.outputs:
                    full_text = request_output.outputs[0].text
                    new_text = full_text[printed_length:]
                    printed_length = len(full_text)
                    if new_text:
                        yield new_text

        return stream_generator()
    else:
        final_output = ""
        async for request_output in engine.generate(request, sampling_params, request_id):
            if request_output.outputs:
                final_output = request_output.outputs[0].text

        # Clean up end of sentence token
        if '<｜end▁of▁sentence｜>' in final_output:
            final_output = final_output.replace('<｜end▁of▁sentence｜>', '')

        return final_output


def prepare_image_input(
    image,
    prompt: str,
    crop_mode: bool = True,
    image_size: Optional[int] = None,
    base_size: Optional[int] = None,
    min_crops: Optional[int] = None,
    max_crops: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Prepare image input for inference.

    Args:
        image: PIL Image object.
        prompt: Text prompt.
        crop_mode: Whether to use dynamic cropping.
        image_size: Local view image size (overrides settings).
        base_size: Global view base size (overrides settings).
        min_crops: Minimum number of crops (overrides settings).
        max_crops: Maximum number of crops (overrides settings).

    Returns:
        Dict with prompt and multi_modal_data ready for inference.
    """
    manager = EngineManager.get_instance()
    processor = manager.get_processor()

    # Tokenize image with dynamic parameters
    image_features = processor.tokenize_with_images(
        images=[image],
        bos=True,
        eos=True,
        cropping=crop_mode,
        prompt=prompt,
        image_size=image_size,
        base_size=base_size,
        min_crops=min_crops,
        max_crops=max_crops,
    )

    return {
        "prompt": prompt,
        "multi_modal_data": {"image": image_features}
    }


def prepare_batch_inputs(
    images: List,
    prompt: str,
    crop_mode: bool = True,
    num_workers: int = 64,
    image_size: Optional[int] = None,
    base_size: Optional[int] = None,
    min_crops: Optional[int] = None,
    max_crops: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Prepare batch image inputs for inference using thread pool.

    Args:
        images: List of PIL Image objects.
        prompt: Text prompt (same for all images).
        crop_mode: Whether to use dynamic cropping.
        num_workers: Number of worker threads.
        image_size: Local view image size (overrides settings).
        base_size: Global view base size (overrides settings).
        min_crops: Minimum number of crops (overrides settings).
        max_crops: Maximum number of crops (overrides settings).

    Returns:
        List of dicts ready for batch inference.
    """
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    def process_single(image):
        return prepare_image_input(
            image, prompt, crop_mode,
            image_size=image_size,
            base_size=base_size,
            min_crops=min_crops,
            max_crops=max_crops,
        )

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        batch_inputs = list(tqdm(
            executor.map(process_single, images),
            total=len(images),
            desc="Preprocessing images"
        ))

    return batch_inputs
