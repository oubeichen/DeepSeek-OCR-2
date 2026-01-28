"""
Task manager for asynchronous OCR processing.

Provides a queue-based task management system with concurrent processing
and fair scheduling across multiple files.
"""

import asyncio
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
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

    # Progress tracking
    total_pages: int = 0
    processed_pages: int = 0

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
            "total_pages": self.total_pages,
            "processed_pages": self.processed_pages,
            "input_file_path": self.input_file_path,
            "output_dir": self.output_dir,
            "result_zip_path": self.result_zip_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OCRTask":
        """Create an OCRTask from a dictionary."""
        task = cls(
            task_id=data["task_id"],
            filename=data["filename"],
            task_type=TaskType(data["task_type"]),
            status=TaskStatus(data["status"]),
            created_at=data.get("created_at", datetime.now().isoformat()),
            completed_at=data.get("completed_at"),
            error_message=data.get("error_message"),
            ocr_params=data.get("ocr_params", {}),
            input_file_path=data.get("input_file_path"),
            output_dir=data.get("output_dir"),
            result_zip_path=data.get("result_zip_path"),
            total_pages=data.get("total_pages", 0),
            processed_pages=data.get("processed_pages", 0),
        )
        # Restore logs
        for log_data in data.get("logs", []):
            task.logs.append(TaskLog(
                timestamp=log_data["timestamp"],
                message=log_data["message"]
            ))
        return task


class TaskManager:
    """
    Manages OCR tasks with concurrent processing and fair scheduling.

    Features:
    - Multiple concurrent task workers (configurable via task_concurrent_files)
    - Page-level fair scheduling via global inference semaphore
    - Smaller files can complete faster even when larger files are processing
    - Task persistence across server restarts
    - Automatic cleanup of old tasks

    Singleton pattern to ensure only one instance exists.
    """

    _instance: Optional["TaskManager"] = None
    _lock = threading.Lock()
    _tasks_index_file = "tasks_index.json"

    def __init__(self):
        from .config import get_settings
        settings = get_settings()

        self.tasks: Dict[str, OCRTask] = {}
        self.queue: asyncio.Queue = asyncio.Queue()
        self._processing = False
        self._worker_tasks: List[asyncio.Task] = []
        self._storage_dir = Path(settings.task_storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._persistence_enabled = settings.task_persistence
        self._retention_days = settings.task_retention_days
        self._retention_count = settings.task_retention_count
        self._last_cleanup_check: Optional[datetime] = None

        # Load persisted tasks on startup
        if self._persistence_enabled:
            self._load_tasks()

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
        # Trigger cleanup check on new task creation
        self.maybe_cleanup()

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

        # Save task state
        if self._persistence_enabled:
            self._save_tasks()

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
            try:
                os.remove(task.input_file_path)
            except OSError:
                pass

        del self.tasks[task_id]
        logger.info(f"Deleted task {task_id}")

        # Save updated task list
        if self._persistence_enabled:
            self._save_tasks()

        return True

    @property
    def queue_size(self) -> int:
        """Get the current queue size."""
        return self.queue.qsize()

    @property
    def total_tasks(self) -> int:
        """Get the total number of tasks."""
        return len(self.tasks)

    @property
    def active_workers(self) -> int:
        """Get the number of active workers."""
        return sum(1 for t in self._worker_tasks if not t.done())

    def start_worker(self):
        """Start the background workers."""
        from .config import get_settings
        settings = get_settings()

        # Calculate number of workers based on concurrency settings
        # At least 2 workers to allow fair scheduling between files
        # At most inference_concurrency workers (no point having more)
        num_workers = max(2, min(settings.inference_concurrency, 8))

        if not self._processing:
            self._processing = True
            # Clean up any done tasks
            self._worker_tasks = [t for t in self._worker_tasks if not t.done()]

            # Start workers up to the calculated limit
            current_workers = len(self._worker_tasks)
            for i in range(current_workers, num_workers):
                worker_task = asyncio.create_task(self._process_queue(worker_id=i))
                self._worker_tasks.append(worker_task)

            logger.info(f"Started {num_workers} task workers")

    def stop_worker(self):
        """Stop all background workers."""
        self._processing = False
        for worker_task in self._worker_tasks:
            worker_task.cancel()
        self._worker_tasks.clear()
        logger.info("All task workers stopped")

    async def _process_queue(self, worker_id: int = 0):
        """
        Background worker to process tasks.

        Each worker processes one task at a time, but uses the global
        inference semaphore for fair page-level scheduling.
        """
        from .engine import (
            EngineManager,
            async_generate_single,
            prepare_image_input,
            prepare_batch_inputs,
            create_sampling_params,
            get_smart_scheduler,
        )
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
                    self.queue.task_done()
                    continue

                # Update status
                task.status = TaskStatus.PROCESSING
                task.add_log(f"Processing started (worker {worker_id})")
                logger.info(f"[Worker {worker_id}] Processing task {task_id}")

                try:
                    # Check engine status and attempt restart if errored
                    manager = EngineManager.get_instance()
                    if manager.is_errored():
                        task.add_log("Engine error detected, attempting restart...")
                        logger.warning(f"[Worker {worker_id}] Engine errored, attempting restart")
                        if manager.restart():
                            task.add_log("Engine restarted successfully")
                        else:
                            raise RuntimeError("Engine restart failed")

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
                        # Process single image
                        task.total_pages = 1
                        task.add_log("Loading image...")
                        image = await asyncio.to_thread(load_image, task.input_file_path)
                        if image is None:
                            raise ValueError("Failed to load image")
                        image = image.convert('RGB')

                        task.add_log(f"Image size: {image.size}")

                        # Prepare input (run in thread pool to avoid blocking)
                        input_data = await asyncio.to_thread(
                            prepare_image_input,
                            image=image,
                            prompt=prompt,
                            crop_mode=crop_mode,
                            image_size=image_size,
                            base_size=base_size,
                            min_crops=min_crops,
                            max_crops=max_crops,
                        )

                        # Create sampling params
                        sampling_params = create_sampling_params(
                            settings=settings,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            ngram_size=ngram_size,
                            window_size=window_size,
                        )

                        # Generate with smart scheduling
                        task.add_log("Running OCR inference...")
                        scheduler = await get_smart_scheduler()
                        await scheduler.register_task(task_id)
                        try:
                            await scheduler.wait_for_slot(task_id)
                            try:
                                request_id = f"{task_id}-page-0"
                                output_text = await async_generate_single(
                                    input_data, sampling_params, request_id
                                )
                            finally:
                                await scheduler.release_slot(task_id)
                        finally:
                            await scheduler.unregister_task(task_id)
                        task.processed_pages = 1

                        # Clean EOS token before processing
                        clean_text = output_text.replace('<｜end▁of▁sentence｜>', '')

                        task.add_log("Processing output...")

                        # Post-process (run in thread pool to avoid blocking)
                        result = await asyncio.to_thread(
                            process_output,
                            text=clean_text,
                            image=image,
                            output_dir=task.output_dir,
                            page_index=0,
                            save_annotated=True,
                            extract_images=True,
                        )

                        # Package results (run in thread pool to avoid blocking)
                        result_format = params.get("result_format", "zip")
                        package_result = await asyncio.to_thread(
                            create_result_package,
                            results=[result],
                            output_dir=task.output_dir,
                            package_name=task.task_id,
                            result_format=result_format,
                        )
                        # For async tasks, save content to file if path is None
                        if package_result.path:
                            task.result_zip_path = package_result.path
                        else:
                            # Save content to file for later download
                            ext = ".md" if result_format == "markdown" else ".json"
                            result_file = os.path.join(task.output_dir, f"{task.task_id}{ext}")
                            content = package_result.content
                            if result_format == "json":
                                import json
                                content = json.dumps(content, indent=2, ensure_ascii=False)
                            with open(result_file, 'w', encoding='utf-8') as f:
                                f.write(content)
                            task.result_zip_path = result_file

                    else:
                        # Process PDF with page-level fair scheduling
                        task.add_log("Converting PDF to images...")
                        images = await asyncio.to_thread(
                            pdf_to_images, task.input_file_path, dpi=dpi
                        )
                        task.total_pages = len(images)
                        task.add_log(f"PDF has {len(images)} pages")

                        # Prepare batch inputs (run in thread pool to avoid blocking event loop)
                        task.add_log("Preprocessing pages...")
                        batch_inputs = await asyncio.to_thread(
                            prepare_batch_inputs,
                            images=images,
                            prompt=prompt,
                            crop_mode=crop_mode,
                            num_workers=settings.num_workers,
                            image_size=image_size,
                            base_size=base_size,
                            min_crops=min_crops,
                            max_crops=max_crops,
                        )

                        # Create sampling params
                        pdf_window_size = window_size if window_size is not None else 50
                        sampling_params = create_sampling_params(
                            settings=settings,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            ngram_size=ngram_size,
                            window_size=pdf_window_size,
                            include_stop_str_in_output=True,
                        )

                        # Generate each page with smart scheduling
                        # SmartScheduler dynamically allocates slots based on active task count
                        task.add_log("Running OCR inference (smart scheduling)...")

                        # Get the smart scheduler and register this task
                        scheduler = await get_smart_scheduler()
                        await scheduler.register_task(task_id)

                        async def process_page_with_limit(idx: int, input_data: Dict) -> tuple:
                            """Process a single page with smart scheduler."""
                            # Wait for a slot from the smart scheduler
                            await scheduler.wait_for_slot(task_id)
                            try:
                                request_id = f"{task_id}-page-{idx}"
                                result = await async_generate_single(
                                    input_data, sampling_params, request_id
                                )
                                task.processed_pages += 1
                                task.add_log(f"Inference progress: {task.processed_pages}/{len(images)}")
                                return idx, result
                            finally:
                                # Always release the slot
                                await scheduler.release_slot(task_id)

                        # Process pages concurrently with smart scheduler
                        page_tasks = [
                            process_page_with_limit(idx, input_data)
                            for idx, input_data in enumerate(batch_inputs)
                        ]
                        try:
                            results_with_idx = await asyncio.gather(*page_tasks)
                        finally:
                            # Unregister task from scheduler after all pages are done
                            await scheduler.unregister_task(task_id)

                        # Sort by index and extract outputs
                        results_with_idx.sort(key=lambda x: x[0])
                        outputs = [r[1] for r in results_with_idx]

                        task.add_log("Processing outputs...")

                        # Post-process each page (run in thread pool to avoid blocking)
                        async def process_page_output(idx: int, output_text: str, image) -> tuple:
                            """Process a single page output in thread pool."""
                            # Check for incomplete output (no EOS token)
                            has_eos = '<｜end▁of▁sentence｜>' in output_text
                            if not has_eos and skip_repeat_pages:
                                task.add_log(f"Page {idx + 1} appears incomplete, skipping")
                                return idx, None

                            # Clean EOS token before processing
                            clean_text = output_text.replace('<｜end▁of▁sentence｜>', '')

                            task.add_log(f"Processing page {idx + 1}/{len(images)}")

                            result = await asyncio.to_thread(
                                process_output,
                                text=clean_text,
                                image=image,
                                output_dir=task.output_dir,
                                page_index=idx,
                                save_annotated=True,
                                extract_images=True,
                            )
                            return idx, result

                        # Process all pages concurrently
                        process_tasks = [
                            process_page_output(idx, output_text, image)
                            for idx, (output_text, image) in enumerate(zip(outputs, images))
                        ]
                        processed_results = await asyncio.gather(*process_tasks)

                        # Collect results in order
                        results = []
                        annotated_images = []
                        for idx, result in sorted(processed_results, key=lambda x: x[0]):
                            if result is not None:
                                results.append(result)
                                if result.get("annotated_image_path"):
                                    annotated_images.append(Image.open(result["annotated_image_path"]))

                        # Generate annotated PDF (run in thread pool to avoid blocking)
                        annotated_pdf_path = None
                        if annotated_images:
                            annotated_pdf_path = os.path.join(task.output_dir, "annotated.pdf")
                            await asyncio.to_thread(images_to_pdf, annotated_images, annotated_pdf_path)

                        # Package results (run in thread pool to avoid blocking)
                        original_name = os.path.splitext(task.filename)[0]
                        result_format = params.get("result_format", "zip")
                        package_result = await asyncio.to_thread(
                            create_pdf_result_package,
                            results=results,
                            output_dir=task.output_dir,
                            annotated_pdf_path=annotated_pdf_path,
                            original_filename=original_name,
                            page_separator=page_separator,
                            result_format=result_format,
                        )
                        # For async tasks, save content to file if path is None
                        if package_result.path:
                            task.result_zip_path = package_result.path
                        else:
                            # Save content to file for later download
                            ext = ".md" if result_format == "markdown" else ".json"
                            result_file = os.path.join(task.output_dir, f"{original_name}{ext}")
                            content = package_result.content
                            if result_format == "json":
                                import json
                                content = json.dumps(content, indent=2, ensure_ascii=False)
                            with open(result_file, 'w', encoding='utf-8') as f:
                                f.write(content)
                            task.result_zip_path = result_file

                    # Mark as completed
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now().isoformat()
                    task.add_log("Task completed successfully")
                    logger.info(f"[Worker {worker_id}] Task {task_id} completed")

                    # Save task state
                    if self._persistence_enabled:
                        self._save_tasks()

                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error_message = str(e)
                    task.add_log(f"Error: {str(e)}")
                    logger.error(f"[Worker {worker_id}] Task {task_id} failed: {e}", exc_info=True)

                    # Save task state on failure too
                    if self._persistence_enabled:
                        self._save_tasks()

                finally:
                    self.queue.task_done()

            except asyncio.CancelledError:
                logger.info(f"Task worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"[Worker {worker_id}] Error: {e}", exc_info=True)
                await asyncio.sleep(1)

    def _save_tasks_sync(self):
        """Synchronously save all tasks to disk (internal use)."""
        try:
            index_path = self._storage_dir / self._tasks_index_file
            tasks_data = {
                task_id: task.to_dict()
                for task_id, task in self.tasks.items()
            }
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(tasks_data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved {len(tasks_data)} tasks to {index_path}")
        except Exception as e:
            logger.error(f"Failed to save tasks: {e}")

    def _save_tasks(self):
        """Save tasks in background thread to avoid blocking."""
        # Use a thread to avoid blocking the event loop
        thread = threading.Thread(target=self._save_tasks_sync, daemon=True)
        thread.start()

    def _load_tasks(self):
        """Load tasks from disk on startup."""
        try:
            index_path = self._storage_dir / self._tasks_index_file
            if not index_path.exists():
                logger.info("No persisted tasks found")
                return

            with open(index_path, "r", encoding="utf-8") as f:
                tasks_data = json.load(f)

            loaded_count = 0
            for task_id, task_dict in tasks_data.items():
                try:
                    task = OCRTask.from_dict(task_dict)
                    # Only load completed or failed tasks (not pending/processing)
                    # Processing tasks should be re-queued or marked as failed
                    if task.status in (TaskStatus.PENDING, TaskStatus.PROCESSING):
                        task.status = TaskStatus.FAILED
                        task.error_message = "Server restarted during processing"
                        task.add_log("Task interrupted by server restart")
                    self.tasks[task_id] = task
                    loaded_count += 1
                except Exception as e:
                    logger.warning(f"Failed to load task {task_id}: {e}")

            logger.info(f"Loaded {loaded_count} tasks from persistence")
        except Exception as e:
            logger.error(f"Failed to load tasks: {e}")

    def _cleanup_old_tasks_sync(self) -> int:
        """
        Synchronously remove old tasks based on retention settings (internal use).

        Cleanup rules:
        - Remove tasks older than retention_days (if > 0)
        - Keep at most retention_count tasks (if > 0), preserving the most recent ones

        Returns:
            Number of tasks deleted.
        """
        tasks_to_delete = []

        # Get all completed/failed tasks sorted by creation time (oldest first)
        cleanable_tasks = [
            (task_id, task) for task_id, task in self.tasks.items()
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
        ]
        cleanable_tasks.sort(key=lambda x: x[1].created_at)

        # Rule 1: Remove tasks older than retention_days
        if self._retention_days > 0:
            cutoff = datetime.now() - timedelta(days=self._retention_days)
            cutoff_str = cutoff.isoformat()
            for task_id, task in cleanable_tasks:
                if task.created_at < cutoff_str:
                    tasks_to_delete.append(task_id)

        # Rule 2: Keep at most retention_count tasks
        if self._retention_count > 0:
            # Count all tasks (including pending/processing)
            total_tasks = len(self.tasks)
            if total_tasks > self._retention_count:
                # Need to delete oldest cleanable tasks to get under limit
                excess = total_tasks - self._retention_count
                for task_id, task in cleanable_tasks:
                    if task_id not in tasks_to_delete:
                        tasks_to_delete.append(task_id)
                        excess -= 1
                        if excess <= 0:
                            break

        # Remove duplicates while preserving order
        tasks_to_delete = list(dict.fromkeys(tasks_to_delete))

        deleted_count = 0
        for task_id in tasks_to_delete:
            task = self.tasks.get(task_id)
            if not task:
                continue

            # Clean up files
            if task.output_dir and os.path.exists(task.output_dir):
                shutil.rmtree(task.output_dir, ignore_errors=True)
            if task.input_file_path and os.path.exists(task.input_file_path):
                try:
                    os.remove(task.input_file_path)
                except OSError:
                    pass

            del self.tasks[task_id]
            deleted_count += 1

        if deleted_count > 0:
            reasons = []
            if self._retention_days > 0:
                reasons.append(f"retention: {self._retention_days} days")
            if self._retention_count > 0:
                reasons.append(f"max: {self._retention_count} tasks")
            logger.info(f"Cleaned up {deleted_count} old tasks ({', '.join(reasons)})")
            # Save updated task list
            self._save_tasks_sync()

        return deleted_count

    def cleanup_old_tasks(self) -> None:
        """Run cleanup in background thread to avoid blocking."""
        thread = threading.Thread(target=self._cleanup_old_tasks_sync, daemon=True)
        thread.start()

    def maybe_cleanup(self):
        """
        Run cleanup if enough time has passed since last check.
        Called on new task creation to avoid needing a separate scheduler.
        """
        # Only check once per hour at most
        now = datetime.now()
        if self._last_cleanup_check is not None:
            if (now - self._last_cleanup_check).total_seconds() < 3600:
                return

        self._last_cleanup_check = now
        self.cleanup_old_tasks()
