"""
Task manager for asynchronous OCR processing.

Provides a queue-based task management system for the web interface.
"""

import asyncio
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import threading

from PIL import Image

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskType(str, Enum):
    """Task type enumeration."""
    IMAGE = "image"
    PDF = "pdf"


@dataclass
class TaskLog:
    """A single log entry for a task."""
    timestamp: str
    message: str


@dataclass
class OCRTask:
    """Represents an OCR task."""
    task_id: str
    filename: str
    task_type: TaskType
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    logs: List[TaskLog] = field(default_factory=list)
    ocr_params: Dict[str, Any] = field(default_factory=dict)

    # File paths
    input_file_path: Optional[str] = None
    output_dir: Optional[str] = None
    result_zip_path: Optional[str] = None

    def add_log(self, message: str):
        """Add a log entry."""
        self.logs.append(TaskLog(
            timestamp=datetime.now().isoformat(),
            message=message
        ))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "filename": self.filename,
            "task_type": self.task_type.value,
            "status": self.status.value,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error_message": self.error_message,
            "logs": [{"timestamp": log.timestamp, "message": log.message} for log in self.logs],
            "ocr_params": self.ocr_params,
        }


class TaskManager:
    """
    Manages OCR tasks with a processing queue.

    Singleton pattern to ensure only one instance exists.
    """

    _instance: Optional["TaskManager"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.tasks: Dict[str, OCRTask] = {}
        self.queue: asyncio.Queue = asyncio.Queue()
        self._processing = False
        self._worker_tasks: List[asyncio.Task] = []
        self._storage_dir = Path("/tmp/deepseek_ocr2_tasks")
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_instance(cls) -> "TaskManager":
        """Get the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def create_task(
        self,
        filename: str,
        task_type: TaskType,
        input_file_path: str,
        ocr_params: Dict[str, Any],
    ) -> OCRTask:
        """Create a new task and add it to the queue."""
        task_id = f"{task_type.value}_{uuid.uuid4().hex[:12]}"

        # Create task output directory
        output_dir = self._storage_dir / task_id
        output_dir.mkdir(parents=True, exist_ok=True)

        task = OCRTask(
            task_id=task_id,
            filename=filename,
            task_type=task_type,
            input_file_path=input_file_path,
            output_dir=str(output_dir),
            ocr_params=ocr_params,
        )
        task.add_log(f"Task created: {filename}")

        self.tasks[task_id] = task
        logger.info(f"Created task {task_id} for {filename}")

        return task

    async def enqueue_task(self, task: OCRTask):
        """Add a task to the processing queue."""
        await self.queue.put(task.task_id)
        task.add_log("Added to processing queue")
        logger.info(f"Task {task.task_id} added to queue")

    def get_task(self, task_id: str) -> Optional[OCRTask]:
        """Get a task by ID."""
        return self.tasks.get(task_id)

    def get_all_tasks(self, limit: int = 100, offset: int = 0) -> List[OCRTask]:
        """Get all tasks, sorted by creation time (newest first)."""
        sorted_tasks = sorted(
            self.tasks.values(),
            key=lambda t: t.created_at,
            reverse=True
        )
        return sorted_tasks[offset:offset + limit]

    def delete_task(self, task_id: str) -> bool:
        """Delete a task and its files."""
        task = self.tasks.get(task_id)
        if not task:
            return False

        # Clean up files
        if task.output_dir and os.path.exists(task.output_dir):
            shutil.rmtree(task.output_dir, ignore_errors=True)
        if task.input_file_path and os.path.exists(task.input_file_path):
            os.remove(task.input_file_path)

        del self.tasks[task_id]
        logger.info(f"Deleted task {task_id}")
        return True

    @property
    def queue_size(self) -> int:
        """Get the current queue size."""
        return self.queue.qsize()

    @property
    def total_tasks(self) -> int:
        """Get the total number of tasks."""
        return len(self.tasks)

    def start_worker(self, num_workers: Optional[int] = None):
        """Start the background workers.

        Args:
            num_workers: Number of concurrent workers. If None, uses config value.
        """
        from .config import get_settings

        if num_workers is None:
            num_workers = get_settings().task_workers

        # Clean up any finished workers
        self._worker_tasks = [t for t in self._worker_tasks if not t.done()]

        # Start workers if not enough running
        current_count = len(self._worker_tasks)
        if current_count < num_workers:
            self._processing = True
            for i in range(current_count, num_workers):
                worker_task = asyncio.create_task(self._process_queue(worker_id=i))
                self._worker_tasks.append(worker_task)
            logger.info(f"Started {num_workers - current_count} task worker(s), total: {num_workers}")

    def stop_worker(self):
        """Stop all background workers."""
        self._processing = False
        for task in self._worker_tasks:
            task.cancel()
        self._worker_tasks.clear()
        logger.info("All task workers stopped")

    @property
    def active_workers(self) -> int:
        """Get the number of active workers."""
        return len([t for t in self._worker_tasks if not t.done()])

    async def _process_queue(self, worker_id: int = 0):
        """Background worker to process tasks."""
        from .engine import EngineManager, async_generate_batch, prepare_image_input, prepare_batch_inputs, create_sampling_params
        from .config import get_settings
        from .processors.image import load_image
        from .processors.pdf import pdf_to_images, images_to_pdf
        from .processors.postprocess import process_output
        from .utils.packaging import create_result_package, create_pdf_result_package

        logger.info(f"Task worker {worker_id} running...")

        while self._processing:
            try:
                # Wait for a task with timeout
                try:
                    task_id = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                task = self.tasks.get(task_id)
                if not task:
                    continue

                # Update status
                task.status = TaskStatus.PROCESSING
                task.add_log(f"Processing started (worker {worker_id})")
                logger.info(f"[Worker {worker_id}] Processing task {task_id}")

                try:
                    # Check engine
                    manager = EngineManager.get_instance()
                    if not manager.is_initialized():
                        raise RuntimeError("Engine not initialized")

                    settings = get_settings()

                    # Get OCR parameters from task
                    params = task.ocr_params
                    prompt = params.get("prompt", settings.default_prompt)
                    crop_mode = params.get("crop_mode", settings.crop_mode)

                    # Image processing parameters
                    image_size = params.get("image_size")
                    base_size = params.get("base_size")
                    min_crops = params.get("min_crops")
                    max_crops = params.get("max_crops")

                    # Sampling parameters
                    temperature = params.get("temperature")
                    max_tokens = params.get("max_tokens")
                    ngram_size = params.get("ngram_size")
                    window_size = params.get("window_size")

                    # PDF parameters
                    dpi = params.get("dpi", settings.pdf_dpi)
                    page_separator = params.get("page_separator", settings.page_separator)
                    skip_repeat_pages = params.get("skip_repeat_pages", settings.skip_repeat_pages)

                    if task.task_type == TaskType.IMAGE:
                        # Process image
                        task.add_log("Loading image...")
                        image = load_image(task.input_file_path)
                        if image is None:
                            raise ValueError("Failed to load image")
                        image = image.convert('RGB')

                        task.add_log(f"Image size: {image.size}")

                        # Prepare input with all dynamic parameters
                        input_data = prepare_image_input(
                            image=image,
                            prompt=prompt,
                            crop_mode=crop_mode,
                            image_size=image_size,
                            base_size=base_size,
                            min_crops=min_crops,
                            max_crops=max_crops,
                        )

                        # Create sampling params with all dynamic parameters
                        sampling_params = create_sampling_params(
                            settings=settings,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            ngram_size=ngram_size,
                            window_size=window_size,
                        )

                        # Generate
                        task.add_log("Running OCR inference...")
                        outputs = await async_generate_batch([input_data], sampling_params, settings)
                        output_text = outputs[0] if outputs else ""

                        task.add_log("Processing output...")

                        # Post-process
                        result = process_output(
                            text=output_text,
                            image=image,
                            output_dir=task.output_dir,
                            page_index=0,
                            save_annotated=True,
                            extract_images=True,
                        )

                        # Package results
                        zip_path = create_result_package(
                            results=[result],
                            output_dir=task.output_dir,
                            package_name=task.task_id,
                        )
                        task.result_zip_path = zip_path

                    else:
                        # Process PDF
                        task.add_log("Converting PDF to images...")
                        images = pdf_to_images(task.input_file_path, dpi=dpi)
                        task.add_log(f"PDF has {len(images)} pages")

                        # Prepare batch inputs with all dynamic parameters
                        batch_inputs = prepare_batch_inputs(
                            images=images,
                            prompt=prompt,
                            crop_mode=crop_mode,
                            num_workers=settings.num_workers,
                            image_size=image_size,
                            base_size=base_size,
                            min_crops=min_crops,
                            max_crops=max_crops,
                        )

                        # Create sampling params with all dynamic parameters
                        # For PDF, use smaller window_size (50) if not specified
                        pdf_window_size = window_size if window_size is not None else 50
                        sampling_params = create_sampling_params(
                            settings=settings,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            ngram_size=ngram_size,
                            window_size=pdf_window_size,
                            include_stop_str_in_output=True,
                        )

                        # Generate
                        task.add_log("Running OCR inference...")
                        outputs = await async_generate_batch(batch_inputs, sampling_params, settings)

                        task.add_log("Processing outputs...")

                        # Post-process each page
                        results = []
                        annotated_images = []

                        for idx, (output_text, image) in enumerate(zip(outputs, images)):
                            task.add_log(f"Processing page {idx + 1}/{len(images)}")

                            result = process_output(
                                text=output_text,
                                image=image,
                                output_dir=task.output_dir,
                                page_index=idx,
                                save_annotated=True,
                                extract_images=True,
                            )
                            results.append(result)

                            if result.get("annotated_image_path"):
                                annotated_images.append(Image.open(result["annotated_image_path"]))

                        # Generate annotated PDF
                        annotated_pdf_path = None
                        if annotated_images:
                            annotated_pdf_path = os.path.join(task.output_dir, "annotated.pdf")
                            images_to_pdf(annotated_images, annotated_pdf_path)

                        # Package results with dynamic page_separator
                        original_name = os.path.splitext(task.filename)[0]
                        zip_path = create_pdf_result_package(
                            results=results,
                            output_dir=task.output_dir,
                            annotated_pdf_path=annotated_pdf_path,
                            original_filename=original_name,
                            page_separator=page_separator,
                        )
                        task.result_zip_path = zip_path

                    # Mark as completed
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now().isoformat()
                    task.add_log("Task completed successfully")
                    logger.info(f"[Worker {worker_id}] Task {task_id} completed")

                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error_message = str(e)
                    task.add_log(f"Error: {str(e)}")
                    logger.error(f"[Worker {worker_id}] Task {task_id} failed: {e}", exc_info=True)

                finally:
                    self.queue.task_done()

            except asyncio.CancelledError:
                logger.info(f"Task worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"[Worker {worker_id}] Worker error: {e}", exc_info=True)
                await asyncio.sleep(1)
