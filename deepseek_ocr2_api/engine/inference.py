"""
Inference module for DeepSeek-OCR-2 API Server.

Provides unified inference interface for both sync and async modes.
"""

import time
import asyncio
import logging
from typing import Optional, List, Dict, Any, Union, AsyncGenerator

from vllm import SamplingParams

from ..config import Settings, get_settings
from .manager import EngineManager  # This import sets up VLLM_DIR in sys.path

logger = logging.getLogger(__name__)

# Import from DeepSeek-OCR2-vllm (path already set by manager.py)
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor

# Global semaphore for fair scheduling across all tasks
_inference_semaphore: Optional[asyncio.Semaphore] = None
_semaphore_lock = asyncio.Lock()


class SmartScheduler:
    """
    Smart scheduler for dynamic concurrency control.

    Dynamically allocates inference slots based on the number of active tasks:

    - 1 active task: can use all slots (e.g., 4 slots)
    - 2 active tasks: each can use half (e.g., 2 slots each)
    - 4+ active tasks: each gets at least 1 slot

    Key feature: When a new task arrives, existing tasks' quotas are reduced
    dynamically. Already-acquired slots are not forcibly released, but new
    slot acquisitions respect the new quota.
    """

    _instance: "SmartScheduler" = None
    _lock = asyncio.Lock()

    def __init__(self):
        self._global_semaphore: asyncio.Semaphore = None
        self._task_slots: Dict[str, int] = {}  # task_id -> current slots held
        self._task_lock = asyncio.Lock()
        self._max_concurrency: int = 4
        self._initialized = False

    @classmethod
    async def get_instance(cls) -> "SmartScheduler":
        """Get or create the singleton instance."""
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                await cls._instance._initialize()
            return cls._instance

    async def _initialize(self):
        """Initialize the scheduler with settings."""
        if self._initialized:
            return
        settings = get_settings()
        self._max_concurrency = settings.inference_concurrency
        # Use the global inference semaphore
        self._global_semaphore = await get_inference_semaphore()
        self._initialized = True
        logger.info(f"SmartScheduler initialized: max_concurrency={self._max_concurrency}")

    def _calculate_task_quota(self, active_task_count: int) -> int:
        """
        Calculate the maximum slots a single task can use.

        Formula: max(1, max_concurrency // active_task_count)

        Examples (with max_concurrency=4):
        - 1 task: max(1, 4//1) = 4
        - 2 tasks: max(1, 4//2) = 2
        - 3 tasks: max(1, 4//3) = 1
        - 4 tasks: max(1, 4//4) = 1
        """
        if active_task_count <= 0:
            active_task_count = 1
        return max(1, self._max_concurrency // active_task_count)

    async def register_task(self, task_id: str):
        """Register a new task with the scheduler."""
        async with self._task_lock:
            if task_id not in self._task_slots:
                self._task_slots[task_id] = 0
                logger.debug(
                    f"Task {task_id} registered. Active tasks: {len(self._task_slots)}"
                )

    async def unregister_task(self, task_id: str):
        """Unregister a task from the scheduler."""
        async with self._task_lock:
            if task_id in self._task_slots:
                del self._task_slots[task_id]
                logger.debug(
                    f"Task {task_id} unregistered. Active tasks: {len(self._task_slots)}"
                )

    async def acquire_slot(self, task_id: str) -> bool:
        """
        Try to acquire an inference slot for a task.

        Returns True if slot was acquired, False if task has reached its quota.
        """
        async with self._task_lock:
            if task_id not in self._task_slots:
                self._task_slots[task_id] = 0

            active_count = len(self._task_slots)
            quota = self._calculate_task_quota(active_count)
            current_slots = self._task_slots[task_id]

            if current_slots >= quota:
                # Task has reached its dynamic quota, must wait
                logger.debug(
                    f"Task {task_id} at quota ({current_slots}/{quota}), "
                    f"active_tasks={active_count}"
                )
                return False

            # Mark that we're about to acquire
            self._task_slots[task_id] = current_slots + 1

        # Acquire global semaphore (may block)
        await self._global_semaphore.acquire()
        logger.debug(f"Task {task_id} acquired slot")
        return True

    async def release_slot(self, task_id: str):
        """Release an inference slot."""
        async with self._task_lock:
            if task_id in self._task_slots and self._task_slots[task_id] > 0:
                self._task_slots[task_id] -= 1

        self._global_semaphore.release()
        logger.debug(f"Task {task_id} released slot")

    async def wait_for_slot(self, task_id: str):
        """
        Wait until a slot is available for this task.

        This handles the case where a task has reached its quota and needs
        to wait for either:
        1. One of its own slots to be released
        2. Other tasks to finish (increasing this task's quota)
        """
        while True:
            if await self.acquire_slot(task_id):
                return
            # Wait a bit before retrying
            await asyncio.sleep(0.05)

    def get_active_task_count(self) -> int:
        """Get the number of active tasks."""
        return len(self._task_slots)

    def get_task_slots(self, task_id: str) -> int:
        """Get the number of slots currently held by a task."""
        return self._task_slots.get(task_id, 0)


# Global smart scheduler instance
_smart_scheduler: SmartScheduler = None


async def get_smart_scheduler() -> SmartScheduler:
    """Get the global smart scheduler instance."""
    global _smart_scheduler
    if _smart_scheduler is None:
        _smart_scheduler = await SmartScheduler.get_instance()
    return _smart_scheduler


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

    # Check if engine has errored and attempt restart
    if manager.is_errored():
        logger.warning("Engine errored, attempting restart...")
        if manager.restart():
            logger.info("Engine restarted successfully")
        else:
            raise RuntimeError("Engine has errored and restart failed.")

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
            try:
                printed_length = 0
                async for request_output in engine.generate(request, sampling_params, request_id):
                    if request_output.outputs:
                        full_text = request_output.outputs[0].text
                        new_text = full_text[printed_length:]
                        printed_length = len(full_text)
                        if new_text:
                            yield new_text
            except Exception as e:
                error_msg = str(e).lower()
                if "loop is not running" in error_msg or "loop has errored" in error_msg:
                    logger.error(f"[{request_id}] vLLM background loop error detected: {e}")
                    manager.mark_errored()
                raise

        return stream_generator()
    else:
        try:
            final_output = ""
            async for request_output in engine.generate(request, sampling_params, request_id):
                if request_output.outputs:
                    final_output = request_output.outputs[0].text

            # Clean up end of sentence token
            if '<｜end▁of▁sentence｜>' in final_output:
                final_output = final_output.replace('<｜end▁of▁sentence｜>', '')

            return final_output
        except Exception as e:
            error_msg = str(e).lower()
            if "loop is not running" in error_msg or "loop has errored" in error_msg:
                logger.error(f"[{request_id}] vLLM background loop error detected: {e}")
                manager.mark_errored()
            raise


async def async_generate_batch(
    prompts: List[Dict[str, Any]],
    sampling_params: Optional[SamplingParams] = None,
    settings: Optional[Settings] = None,
) -> List[str]:
    """
    Batch asynchronous generation using AsyncLLMEngine.

    Args:
        prompts: List of prompt dicts with multi_modal_data.
        sampling_params: Sampling parameters. If None, creates default.
        settings: Settings for creating sampling params.

    Returns:
        List of generated text outputs.
    """
    import asyncio

    manager = EngineManager.get_instance()

    if not manager.is_initialized():
        raise RuntimeError("Engine not initialized. Call initialize() first.")

    # Check if engine has errored and attempt restart
    if manager.is_errored():
        logger.warning("Engine errored, attempting restart...")
        if manager.restart():
            logger.info("Engine restarted successfully")
        else:
            raise RuntimeError("Engine has errored and restart failed.")

    engine = manager.get_engine()

    if sampling_params is None:
        sampling_params = create_sampling_params(settings)

    logger.info(f"Starting async batch generation for {len(prompts)} prompt(s)...")
    start_time = time.time()

    async def generate_single(prompt: Dict[str, Any], idx: int) -> tuple:
        request_id = f"request-{int(time.time() * 1000)}-{idx}"
        try:
            final_output = ""
            async for request_output in engine.generate(prompt, sampling_params, request_id):
                if request_output.outputs:
                    final_output = request_output.outputs[0].text
            # Clean up end of sentence token
            if '<｜end▁of▁sentence｜>' in final_output:
                final_output = final_output.replace('<｜end▁of▁sentence｜>', '')
            return idx, final_output
        except Exception as e:
            error_msg = str(e).lower()
            if "loop is not running" in error_msg or "loop has errored" in error_msg:
                logger.error(f"[{request_id}] vLLM background loop error detected: {e}")
                manager.mark_errored()
            raise

    # Run all generations concurrently
    tasks = [generate_single(prompt, idx) for idx, prompt in enumerate(prompts)]
    results_with_idx = await asyncio.gather(*tasks)

    # Sort by index to maintain order
    results_with_idx.sort(key=lambda x: x[0])
    results = [r[1] for r in results_with_idx]

    elapsed = time.time() - start_time
    logger.info(f"Batch generation complete in {elapsed:.2f}s")

    return results


async def get_inference_semaphore() -> asyncio.Semaphore:
    """
    Get or create the global inference semaphore.

    The semaphore limits concurrent inference requests across all tasks,
    enabling fair scheduling where smaller files can complete faster
    even when larger files are being processed.
    """
    global _inference_semaphore

    async with _semaphore_lock:
        if _inference_semaphore is None:
            settings = get_settings()
            _inference_semaphore = asyncio.Semaphore(settings.inference_concurrency)
            logger.info(f"Created inference semaphore with concurrency={settings.inference_concurrency}")
        return _inference_semaphore


async def async_generate_single(
    prompt: Dict[str, Any],
    sampling_params: SamplingParams,
    request_id: str,
) -> str:
    """
    Generate output for a single prompt.

    Concurrency is managed externally by SmartScheduler.

    Args:
        prompt: Prompt dict with multi_modal_data.
        sampling_params: Sampling parameters.
        request_id: Unique request identifier.

    Returns:
        Generated text output.

    Raises:
        RuntimeError: If engine is not initialized or restart fails.
    """
    manager = EngineManager.get_instance()

    if not manager.is_initialized():
        raise RuntimeError("Engine not initialized. Call initialize() first.")

    # Check if engine has errored and attempt restart
    if manager.is_errored():
        logger.warning(f"[{request_id}] Engine errored, attempting restart...")
        if manager.restart():
            logger.info(f"[{request_id}] Engine restarted successfully")
        else:
            raise RuntimeError("Engine has errored and restart failed.")

    engine = manager.get_engine()

    logger.debug(f"[{request_id}] Starting inference")
    try:
        final_output = ""
        async for request_output in engine.generate(prompt, sampling_params, request_id):
            if request_output.outputs:
                final_output = request_output.outputs[0].text

        logger.debug(f"[{request_id}] Inference complete")
        return final_output
    except Exception as e:
        # Check if this is a "background loop errored" type error
        error_msg = str(e).lower()
        if "loop is not running" in error_msg or "loop has errored" in error_msg:
            logger.error(f"[{request_id}] vLLM background loop error detected: {e}")
            manager.mark_errored()
        raise


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
