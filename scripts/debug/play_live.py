#!/usr/bin/env python3
"""Live LoL play inference script.

Captures screen, runs policy inference, and sends actions.

Usage:
    # Dry run (no inputs sent)
    python scripts/play_live.py \
        --tokenizer-checkpoint checkpoints/tokenizer_best.pt \
        --dynamics-checkpoint checkpoints/dynamics_best.pt \
        --policy-checkpoint checkpoints/policy_best.pt \
        --dry-run

    # Live play
    python scripts/play_live.py \
        --tokenizer-checkpoint checkpoints/tokenizer_best.pt \
        --dynamics-checkpoint checkpoints/dynamics_best.pt \
        --policy-checkpoint checkpoints/policy_best.pt

Pipeline:
    capture(20fps) -> resize(256x256) -> tokenize ->
    dynamics(hidden_state) -> policy_head ->
    decode_action -> send_keys
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ahriuwu.models import create_tokenizer, create_dynamics, PolicyHead
from ahriuwu.models.transformer_tokenizer import create_transformer_tokenizer
from ahriuwu.data.actions import decode_action, ActionSpace


# Key mapping for pynput
ABILITY_TO_KEY = {
    'Q': 'q',
    'W': 'w',
    'E': 'e',
    'R': 'r',
    'D': 'd',
    'F': 'f',
    'B': 'b',
    'item': '1',  # First item slot
}

# Movement direction to screen-relative offset
# 0=East, 90=North counter-clockwise
DIRECTION_OFFSETS = {
    i: (
        int(100 * np.cos(np.radians(i * 20))),  # x offset
        int(-100 * np.sin(np.radians(i * 20)))  # y offset (negative because screen Y is down)
    )
    for i in range(18)
}


def parse_args():
    parser = argparse.ArgumentParser(description="Live LoL inference")
    parser.add_argument(
        "--tokenizer-checkpoint",
        type=str,
        required=True,
        help="Path to tokenizer checkpoint",
    )
    parser.add_argument(
        "--dynamics-checkpoint",
        type=str,
        required=True,
        help="Path to dynamics checkpoint",
    )
    parser.add_argument(
        "--policy-checkpoint",
        type=str,
        default=None,
        help="Path to policy checkpoint (if separate from dynamics)",
    )
    parser.add_argument(
        "--tokenizer-type",
        type=str,
        default="transformer",
        choices=["cnn", "transformer"],
        help="Type of tokenizer",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="tiny",
        choices=["tiny", "small", "medium", "large"],
        help="Model size for dynamics",
    )
    parser.add_argument(
        "--context-frames",
        type=int,
        default=8,
        help="Number of frames for temporal context",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=20,
        help="Target inference FPS",
    )
    parser.add_argument(
        "--capture-region",
        type=str,
        default=None,
        help="Screen capture region as 'x,y,w,h' (default: full screen)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't send inputs, just show predictions",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (auto, cuda, mps, cpu)",
    )
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Show captured frames in window",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=256,
        help="Latent dimension (must match tokenizer)",
    )
    return parser.parse_args()


class ScreenCapture:
    """Fast screen capture using mss."""

    def __init__(self, region: tuple[int, int, int, int] | None = None):
        """Initialize screen capture.

        Args:
            region: (x, y, width, height) or None for full screen
        """
        try:
            import mss
        except ImportError:
            raise ImportError("mss required: pip install mss")

        self.sct = mss.mss()

        if region:
            self.monitor = {
                "left": region[0],
                "top": region[1],
                "width": region[2],
                "height": region[3],
            }
        else:
            # Use primary monitor
            self.monitor = self.sct.monitors[1]

        print(f"Capture region: {self.monitor}")

    def capture(self) -> np.ndarray:
        """Capture screen and return BGR image."""
        img = self.sct.grab(self.monitor)
        frame = np.array(img)
        # mss returns BGRA, convert to BGR
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


class InputController:
    """Send keyboard and mouse inputs using pynput."""

    def __init__(self, dry_run: bool = False, center: tuple[int, int] = (960, 540)):
        """Initialize input controller.

        Args:
            dry_run: If True, don't send actual inputs
            center: Screen center for relative mouse movement
        """
        self.dry_run = dry_run
        self.center = center

        if not dry_run:
            try:
                from pynput.keyboard import Controller as KeyboardController
                from pynput.mouse import Controller as MouseController, Button
                self.keyboard = KeyboardController()
                self.mouse = MouseController()
                self.Button = Button
            except ImportError:
                raise ImportError("pynput required: pip install pynput")

    def send_action(self, movement: int, ability: str | None):
        """Send action to game.

        Args:
            movement: Direction class 0-17
            ability: Ability key or None
        """
        if self.dry_run:
            return

        # Send movement (right-click at offset from center)
        if movement >= 0:
            dx, dy = DIRECTION_OFFSETS.get(movement, (0, 0))
            target_x = self.center[0] + dx
            target_y = self.center[1] + dy
            self.mouse.position = (target_x, target_y)
            self.mouse.click(self.Button.right)

        # Send ability
        if ability and ability in ABILITY_TO_KEY:
            key = ABILITY_TO_KEY[ability]
            self.keyboard.press(key)
            self.keyboard.release(key)


class LiveInference:
    """Live inference pipeline."""

    def __init__(
        self,
        tokenizer_path: str,
        dynamics_path: str,
        policy_path: str | None,
        tokenizer_type: str = "transformer",
        model_size: str = "tiny",
        latent_dim: int = 256,
        context_frames: int = 8,
        device: str = "auto",
    ):
        # Device selection
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        print(f"Device: {self.device}")

        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer_type = tokenizer_type
        self._load_tokenizer(tokenizer_path, tokenizer_type)

        # Load dynamics
        print("Loading dynamics...")
        self._load_dynamics(dynamics_path, model_size, latent_dim)

        # Load or create policy head
        print("Loading policy head...")
        self._load_policy(policy_path)

        # Frame buffer for temporal context
        self.context_frames = context_frames
        self.latent_buffer = deque(maxlen=context_frames)

        print(f"Models loaded. Context frames: {context_frames}")

    def _load_tokenizer(self, path: str, tokenizer_type: str):
        """Load tokenizer from checkpoint."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        if tokenizer_type == "cnn":
            self.tokenizer = create_tokenizer("small")
        else:
            self.tokenizer = create_transformer_tokenizer("small")

        self.tokenizer.load_state_dict(checkpoint["model_state_dict"])
        self.tokenizer = self.tokenizer.to(self.device).eval()

        # Freeze
        for param in self.tokenizer.parameters():
            param.requires_grad = False

        params = sum(p.numel() for p in self.tokenizer.parameters())
        print(f"  Tokenizer ({tokenizer_type}): {params:,} params")

    def _load_dynamics(self, path: str, model_size: str, latent_dim: int):
        """Load dynamics from checkpoint."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        # Get latent dim from checkpoint args if available
        args = checkpoint.get("args", {})
        ckpt_latent_dim = args.get("latent_dim", latent_dim)

        self.dynamics = create_dynamics(
            size=model_size,
            latent_dim=ckpt_latent_dim,
            use_agent_tokens=True,
            use_qk_norm=args.get("use_qk_norm", True),
            soft_cap=args.get("soft_cap", 50.0),
            num_register_tokens=args.get("num_register_tokens", 8),
        )

        # Load weights
        state_dict = checkpoint.get("dynamics_state_dict", checkpoint.get("model_state_dict", {}))
        self.dynamics.load_state_dict(state_dict, strict=False)
        self.dynamics = self.dynamics.to(self.device).eval()

        # Freeze
        for param in self.dynamics.parameters():
            param.requires_grad = False

        params = self.dynamics.get_num_params()
        print(f"  Dynamics ({model_size}): {params:,} params")

        self.latent_dim = ckpt_latent_dim

    def _load_policy(self, path: str | None):
        """Load policy head from checkpoint."""
        if path:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.policy = PolicyHead(
                input_dim=self.dynamics.model_dim,
                action_dim=128,
                hidden_dim=256,
                mtp_length=1,
            )
            self.policy.load_state_dict(checkpoint.get("policy_head_state_dict", {}))
        else:
            # Create fresh policy (for testing)
            self.policy = PolicyHead(
                input_dim=self.dynamics.model_dim,
                action_dim=128,
                hidden_dim=256,
                mtp_length=1,
            )

        self.policy = self.policy.to(self.device).eval()

        # Freeze
        for param in self.policy.parameters():
            param.requires_grad = False

        params = sum(p.numel() for p in self.policy.parameters())
        print(f"  Policy head: {params:,} params")

    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for tokenizer.

        Args:
            frame: BGR image from screen capture

        Returns:
            (1, 3, 256, 256) tensor in [0, 1]
        """
        # Resize to 256x256
        frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)
        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # To tensor
        tensor = torch.from_numpy(frame).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 256, 256)
        return tensor.to(self.device)

    def encode_frame(self, frame_tensor: torch.Tensor) -> torch.Tensor:
        """Encode frame to latent.

        Args:
            frame_tensor: (1, 3, 256, 256) tensor

        Returns:
            Latent tensor
        """
        with torch.no_grad():
            if self.tokenizer_type == "cnn":
                latent = self.tokenizer.encode(frame_tensor)  # (1, 256, 16, 16)
            else:
                latent = self.tokenizer.encode(frame_tensor)["latent"]  # (1, 256, 32)
                # Reshape for dynamics: (1, 256, 32) -> (1, 32, 16, 16)
                latent = latent.transpose(1, 2).view(1, 32, 16, 16)

        return latent

    @torch.no_grad()
    def infer(self, frame: np.ndarray) -> tuple[int, str | None, dict]:
        """Run inference on a single frame.

        Args:
            frame: BGR image from screen capture

        Returns:
            (action_idx, (movement, ability), timing_info)
        """
        timing = {}

        # Preprocess
        t0 = time.perf_counter()
        frame_tensor = self.preprocess_frame(frame)
        timing["preprocess"] = (time.perf_counter() - t0) * 1000

        # Encode
        t0 = time.perf_counter()
        latent = self.encode_frame(frame_tensor)
        timing["encode"] = (time.perf_counter() - t0) * 1000

        # Add to buffer
        self.latent_buffer.append(latent)

        # Build sequence
        t0 = time.perf_counter()
        if len(self.latent_buffer) < self.context_frames:
            # Pad with copies of first frame
            latents = list(self.latent_buffer)
            while len(latents) < self.context_frames:
                latents.insert(0, latents[0])
        else:
            latents = list(self.latent_buffer)

        # Stack: (T, 1, C, H, W) -> (1, T, C, H, W)
        latent_seq = torch.cat(latents, dim=0).unsqueeze(0)
        tau = torch.zeros(1, self.context_frames, device=self.device)
        timing["build_seq"] = (time.perf_counter() - t0) * 1000

        # Dynamics forward
        t0 = time.perf_counter()
        z_pred, agent_out = self.dynamics(latent_seq, tau)
        timing["dynamics"] = (time.perf_counter() - t0) * 1000

        # Policy forward (last frame only)
        t0 = time.perf_counter()
        action_logits = self.policy(agent_out[:, -1:, :])  # (1, 1, L, action_dim)
        action_idx = action_logits[:, :, 0, :].argmax(-1).item()  # First MTP step
        timing["policy"] = (time.perf_counter() - t0) * 1000

        # Decode action
        movement, ability = decode_action(action_idx)

        timing["total"] = sum(timing.values())

        return action_idx, (movement, ability), timing


def main():
    args = parse_args()

    print("=" * 60)
    print("LoL Live Play Inference")
    print("=" * 60)
    print(f"Tokenizer: {args.tokenizer_type}")
    print(f"Model size: {args.model_size}")
    print(f"Context frames: {args.context_frames}")
    print(f"Target FPS: {args.target_fps}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 60)

    # Parse capture region
    capture_region = None
    if args.capture_region:
        x, y, w, h = map(int, args.capture_region.split(","))
        capture_region = (x, y, w, h)

    # Initialize components
    print("\nInitializing...")
    capture = ScreenCapture(capture_region)

    # Get screen center for mouse movement
    screen_center = (
        capture.monitor["left"] + capture.monitor["width"] // 2,
        capture.monitor["top"] + capture.monitor["height"] // 2,
    )
    controller = InputController(dry_run=args.dry_run, center=screen_center)

    inference = LiveInference(
        tokenizer_path=args.tokenizer_checkpoint,
        dynamics_path=args.dynamics_checkpoint,
        policy_path=args.policy_checkpoint,
        tokenizer_type=args.tokenizer_type,
        model_size=args.model_size,
        latent_dim=args.latent_dim,
        context_frames=args.context_frames,
        device=args.device,
    )

    # Target frame time
    target_frame_time = 1.0 / args.target_fps

    print("\nStarting inference loop...")
    print("Press Ctrl+C to stop")
    print("-" * 60)

    frame_count = 0
    start_time = time.time()
    timing_history = deque(maxlen=100)

    try:
        while True:
            loop_start = time.perf_counter()

            # Capture frame
            t0 = time.perf_counter()
            frame = capture.capture()
            capture_time = (time.perf_counter() - t0) * 1000

            # Run inference
            action_idx, (movement, ability), timing = inference.infer(frame)

            # Send action
            t0 = time.perf_counter()
            controller.send_action(movement, ability)
            send_time = (time.perf_counter() - t0) * 1000

            # Track timing
            total_time = timing["total"] + capture_time + send_time
            timing_history.append(total_time)

            frame_count += 1

            # Print status
            angle = ActionSpace.direction_to_angle(movement)
            ability_str = ability if ability else "-"

            if frame_count % 20 == 0:  # Print every second at 20 FPS
                avg_time = sum(timing_history) / len(timing_history)
                actual_fps = 1000 / avg_time if avg_time > 0 else 0
                elapsed = time.time() - start_time

                print(
                    f"Frame {frame_count:5d} | "
                    f"Action: {action_idx:3d} (dir:{movement:2d}/{angle:3.0f}° {ability_str:4s}) | "
                    f"Time: {total_time:5.1f}ms | "
                    f"FPS: {actual_fps:4.1f} | "
                    f"Elapsed: {elapsed:.1f}s"
                )

            # Show preview
            if args.show_preview:
                # Draw info on frame
                preview = cv2.resize(frame, (512, 512))
                cv2.putText(
                    preview,
                    f"Action: {action_idx} | Dir: {angle:.0f} | Ability: {ability_str}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    preview,
                    f"Time: {total_time:.1f}ms | FPS: {1000/total_time:.1f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Live Preview", preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Rate limiting
            loop_time = time.perf_counter() - loop_start
            if loop_time < target_frame_time:
                time.sleep(target_frame_time - loop_time)

    except KeyboardInterrupt:
        print("\n\nStopping...")

    # Final stats
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Session Summary")
    print("=" * 60)
    print(f"Total frames: {frame_count}")
    print(f"Elapsed time: {elapsed:.1f}s")
    print(f"Average FPS: {frame_count / elapsed:.1f}")
    if timing_history:
        print(f"Average frame time: {sum(timing_history) / len(timing_history):.1f}ms")

    if args.show_preview:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
