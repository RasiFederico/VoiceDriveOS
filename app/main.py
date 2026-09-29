"""VoiceDriveOS application entry point and pipeline orchestration.

Startup sequence (project brief, section 23):

1. Load configuration.
2. Configure logging.
3. Initialize the state machine.
4. Load every heavy model once via :class:`ModelManager`.
5. Start the microphone + VAD worker thread.
6. Run the async command loop: each closed speech segment flows through
   transcription -> intent parsing -> (optional) vision + decision ->
   validation -> execution, then back to IDLE.

The microphone/VAD worker runs on a dedicated thread and never blocks (or is
blocked by) the asyncio event loop; it communicates with the async pipeline
exclusively through :class:`core.event_bus.EventBus`.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import threading
from collections.abc import Sequence
from pathlib import Path

from app.config import AppConfig
from app.state import PipelineState, StateMachine
from audio.microphone import MicrophoneWorker
from audio.vad import EnergyVAD, VADResult, VADState
from commands.command_router import command_to_direct_action, requires_screen
from commands.command_types import CommandType
from core.event_bus import EventBus
from core.exceptions import ValidationError, VoiceDriveError
from core.logger import configure_logging, get_logger
from core.performance import LatencyProfiler
from execution.executor import Executor
from intelligence.action_schema import Action, ClickAction, DragAction, MoveAction, NoAction
from intelligence.intent_parser import parse_intent
from intelligence.model_manager import ModelManager
from vision.element import UIElement
from vision.screen_capture import ScreenCapture
from vision.vision_cache import VisionCache

logger = get_logger(__name__)

_SPEECH_SEGMENT_TOPIC = "speech_segment"


class VoiceDriveApp:
    """Wires every subsystem together and drives the async command loop."""

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig.load()
        configure_logging(
            level=self.config.performance.logging_level,
            log_file=self.config.performance.log_file,
        )

        self.state = StateMachine()
        self.event_bus = EventBus()
        self.model_manager = ModelManager(self.config)

        self.microphone = MicrophoneWorker(self.config.audio, on_error=self._on_microphone_error)
        self.vad = EnergyVAD(self.config.audio)
        self.screen_capture = ScreenCapture(self.config.vision)
        self.vision_cache = VisionCache(self.config.vision)
        self.executor = Executor(self.config.execution)

        self._stop_event = threading.Event()
        self._mic_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._shutdown = False

    # ------------------------------------------------------------------ #
    # Startup / shutdown
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Load models and wire the executor's target resolver. Call once before :meth:`run`."""
        logger.info("VoiceDriveOS starting up...")
        self.model_manager.load_all()
        self.executor.set_target_resolver(self._resolve_target)
        logger.info("VoiceDriveOS READY")

    def _resolve_target(self, target_id: int) -> UIElement | None:
        """Look up a UI element by id in the last cached vision pass, for the executor."""
        elements = self.vision_cache.get()
        if not elements:
            return None
        return next((element for element in elements if element.id == target_id), None)

    def _on_microphone_error(self, exc: Exception) -> None:
        logger.error("microphone error: %s", exc)
        self.state.force_error(f"microphone error: {exc}")
        self.state.reset()

    async def run(self) -> None:
        """Bridge microphone/VAD (thread) into the async pipeline, forever, until stopped."""
        self._loop = asyncio.get_running_loop()
        self.event_bus.bind_loop(self._loop)
        try:
            if self._stop_event.is_set():
                return
            self.microphone.start()
            self.state.transition(PipelineState.LISTENING)
            self._mic_thread = threading.Thread(
                target=self._capture_loop, name="mic-vad-worker", daemon=True
            )
            self._mic_thread.start()

            logger.info("WAIT FOR SPEECH (press Ctrl+C to stop)")
            async for segment_result in self.event_bus.subscribe(_SPEECH_SEGMENT_TOPIC):
                if self._stop_event.is_set():
                    break
                await self._handle_segment(segment_result)
        finally:
            await self.shutdown()

    def _capture_loop(self) -> None:
        """Runs on a dedicated thread: reads mic chunks, feeds the VAD, publishes closed segments."""
        while not self._stop_event.is_set():
            chunk = self.microphone.read_chunk(timeout=0.1)
            if chunk is None:
                continue
            try:
                result = self.vad.process_chunk(chunk)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("VAD error: %s", exc)
                continue

            if result.segment is not None:
                self.event_bus.publish_threadsafe(_SPEECH_SEGMENT_TOPIC, result)

    # ------------------------------------------------------------------ #
    # Per-command pipeline
    # ------------------------------------------------------------------ #
    async def _handle_segment(self, segment_result: VADResult) -> None:
        assert self._loop is not None
        profiler = LatencyProfiler() if self.config.performance.profiling_enabled else None
        if profiler is not None and segment_result.speech_started_at is not None:
            profiler.mark("speech_start", at=segment_result.speech_started_at)
            profiler.mark("speech_end", at=segment_result.speech_ended_at)

        try:
            self.state.transition(PipelineState.TRANSCRIBING)
            transcript = await self._loop.run_in_executor(
                None, self.model_manager.stt.transcribe, segment_result.segment, profiler
            )

            command = parse_intent(transcript)
            logger.info("command_type=%s raw_text=%r", command.type.value, command.raw_text)

            if profiler is not None:
                profiler.mark("decision_start")

            if requires_screen(command) and not self.config.vision.enabled:
                self.state.transition(PipelineState.DECIDING)
                logger.info("screen command skipped: vision is disabled")
                action = NoAction(reason="Screen analysis is disabled in configuration")
            elif requires_screen(command):
                self.state.transition(PipelineState.ANALYZING_SCREEN)
                if profiler is not None:
                    profiler.mark("vision_start")
                elements = await self._loop.run_in_executor(None, self._capture_and_parse_screen)
                if profiler is not None:
                    profiler.mark("vision_end")

                self.state.transition(PipelineState.DECIDING)
                query = command.target_text or command.raw_text
                if command.type is CommandType.DRAG:
                    source = self.model_manager.decision_model.decide(
                        command.source_text or "", elements
                    )
                    destination = self.model_manager.decision_model.decide(
                        command.target_text or "", elements
                    )
                    if not isinstance(source, ClickAction) or not isinstance(destination, ClickAction):
                        action = NoAction(
                            reason="drag source or destination was not identified confidently",
                            confidence=min(source.confidence, destination.confidence),
                        )
                    else:
                        action = DragAction(
                            from_target_id=source.target_id,
                            to_target_id=destination.target_id,
                            confidence=min(source.confidence, destination.confidence),
                        )
                else:
                    selected = self.model_manager.decision_model.decide(query, elements)
                    if command.type is CommandType.CLICK and isinstance(selected, ClickAction):
                        action = ClickAction(
                            target_id=selected.target_id,
                            button=command.button,
                            clicks=command.clicks,
                            confidence=selected.confidence,
                        )
                    elif command.type is CommandType.MOVE and isinstance(selected, ClickAction):
                        action = MoveAction(
                            target_id=selected.target_id,
                            confidence=selected.confidence,
                        )
                    else:
                        action = selected
            else:
                self.state.transition(PipelineState.DECIDING)
                action = command_to_direct_action(command)

            if profiler is not None:
                profiler.mark("decision_end")

            self.state.transition(PipelineState.VALIDATING)
            if profiler is not None:
                profiler.mark("validation_start")
            validated = self._validate(action)
            if profiler is not None:
                profiler.mark("validation_end")

            self.state.transition(PipelineState.EXECUTING)
            if profiler is not None:
                profiler.mark("execution_start")
            await self._loop.run_in_executor(None, self.executor.execute, validated)
            if profiler is not None:
                profiler.mark("execution_end")
            self.vision_cache.invalidate()

            self.state.transition(PipelineState.IDLE)

        except VoiceDriveError as exc:
            logger.error("pipeline error: %s", exc)
            self.state.force_error(str(exc))
            self.state.reset()
        except Exception as exc:  # pragma: no cover - defensive catch-all, must never crash the loop
            logger.exception("unexpected pipeline failure: %s", exc)
            self.state.force_error(str(exc))
            self.state.reset()
        finally:
            if profiler is not None:
                report = profiler.report()
                logger.info(
                    "latency_report total_ms=%.1f stages=%s",
                    report.total_ms if report.total_ms is not None else -1.0,
                    {k: round(v, 1) for k, v in report.stage_latencies_ms.items()},
                )
            if self.state.current_state is PipelineState.IDLE and not self._stop_event.is_set():
                self.state.transition(PipelineState.LISTENING)

    def _capture_and_parse_screen(self) -> list[UIElement]:
        screenshot = self.screen_capture.capture()
        elements = self.model_manager.ui_parser.parse(screenshot)
        self.vision_cache.set(elements)
        return elements

    def _validate(self, action: Action) -> Action:
        """Final semantic gate before execution.

        The action's own ``__post_init__`` already enforced field-level
        validity; this stage additionally enforces the confidence threshold
        so a low-confidence guess never reaches the executor as a real click.
        """
        if isinstance(action, NoAction):
            logger.info("validation: NoAction passed through (reason=%s)", action.reason or "<none>")
            return action
        if action.confidence < self.config.intelligence.confidence_threshold:
            raise ValidationError(
                f"action confidence {action.confidence:.2f} below threshold "
                f"{self.config.intelligence.confidence_threshold:.2f}"
            )
        return action

    async def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        logger.info("shutting down...")
        self._stop_event.set()
        if self._loop is not None and self._loop.is_running():
            self.event_bus.close_all()
        if self._mic_thread is not None:
            self._mic_thread.join(timeout=2.0)
        self.microphone.stop()
        self.screen_capture.close()
        self.model_manager.shutdown()
        logger.info("shutdown complete")


def _list_audio_devices() -> int:
    """Print microphone device indices for VDOS_AUDIO_INPUT_DEVICE."""
    try:
        import sounddevice as sd  # type: ignore
    except ImportError:
        print("sounddevice is not installed. Install runtime dependencies first.")
        return 1

    try:
        devices = sd.query_devices()
        default_device = sd.default.device[0]
    except Exception as exc:
        print(f"Could not query audio devices: {exc}")
        return 1

    inputs = [
        (index, device)
        for index, device in enumerate(devices)
        if int(device.get("max_input_channels", 0)) > 0
    ]
    if not inputs:
        print("No microphone input devices were found.")
        return 1

    print("Microphone input devices (set VDOS_AUDIO_INPUT_DEVICE to an index):")
    for index, device in inputs:
        marker = " (default)" if index == default_device else ""
        print(f"  {index}: {device['name']} - {device['max_input_channels']} channel(s){marker}")
    return 0


def _download_model(model_name: str) -> int:
    """Explicitly fetch model weights for offline use on a later launch."""
    try:
        from faster_whisper.utils import download_model
    except ImportError:
        print("faster-whisper is not installed; use the source setup instructions.")
        return 1

    cache_value = os.environ.get("VDOS_AUDIO_WHISPER_CACHE_DIR")
    cache_dir = str(Path(cache_value).expanduser()) if cache_value else None
    print(f"Downloading Whisper '{model_name}'. This command needs internet access.")
    try:
        model_path = download_model(model_name, cache_dir=cache_dir, local_files_only=False)
    except Exception as exc:
        print(f"Could not download the model: {exc}")
        return 1
    print(f"Model '{model_name}' is ready at {model_path}")
    return 0


def _check_models() -> int:
    """Load configured models without starting the microphone or executor."""
    app: VoiceDriveApp | None = None
    try:
        app = VoiceDriveApp()
        app.model_manager.load_all()
    except Exception as exc:
        logger.error("model check failed: %s", exc)
        return 1
    finally:
        if app is not None:
            app.model_manager.shutdown()
    print("Configured models loaded successfully (no microphone opened).")
    return 0


def _check_vision() -> int:
    """Capture and parse one frame without starting the mic or executing actions."""
    app: VoiceDriveApp | None = None
    try:
        config = AppConfig.load()
        if not config.vision.enabled:
            print("Screen recognition is disabled (set VDOS_VISION_ENABLED=true to enable it).")
            return 1
        app = VoiceDriveApp(config)
        app.model_manager.load_all()
        elements = app._capture_and_parse_screen()
    except Exception as exc:
        logger.error("screen recognition check failed: %s", exc)
        return 1
    finally:
        if app is not None:
            app.screen_capture.close()
            app.model_manager.shutdown()
    print(f"Screen recognition is ready: {len(elements)} visible text targets found.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Process entry point: ``python -m app.main``."""
    parser = argparse.ArgumentParser(description="Offline VoiceDriveOS voice-to-action assistant")
    maintenance = parser.add_mutually_exclusive_group()
    maintenance.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="list microphone input devices and exit",
    )
    maintenance.add_argument(
        "--download-model",
        choices=("tiny", "base", "small", "medium", "large-v3"),
        metavar="MODEL",
        help="download Whisper weights for offline use (requires internet)",
    )
    maintenance.add_argument(
        "--check-models",
        action="store_true",
        help="load configured models and exit without opening the microphone",
    )
    maintenance.add_argument(
        "--check-vision",
        action="store_true",
        help="capture and OCR one screen without opening the microphone or performing actions",
    )
    args = parser.parse_args(argv)
    if args.list_audio_devices:
        return _list_audio_devices()
    if args.download_model is not None:
        return _download_model(args.download_model)
    if args.check_models:
        return _check_models()
    if args.check_vision:
        return _check_vision()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app: VoiceDriveApp | None = None
    exit_code = 0

    def _request_stop(*_args: object) -> None:
        if app is not None:
            app._stop_event.set()
            if app._loop is not None and app._loop.is_running():
                app._loop.call_soon_threadsafe(app.event_bus.close, _SPEECH_SEGMENT_TOPIC)

    try:
        signal.signal(signal.SIGINT, _request_stop)
        signal.signal(signal.SIGTERM, _request_stop)
    except (ValueError, AttributeError):  # pragma: no cover - depends on OS/thread
        pass

    try:
        app = VoiceDriveApp()
        app.start()
        loop.run_until_complete(app.run())
    except KeyboardInterrupt:
        logger.info("shutdown requested")
    except Exception as exc:
        logger.exception("VoiceDriveOS stopped with an error: %s", exc)
        exit_code = 1
    finally:
        if app is not None and not app._shutdown:
            loop.run_until_complete(app.shutdown())
        loop.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
