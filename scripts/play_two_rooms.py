import argparse
from pathlib import Path
import sys
import tkinter as tk

import numpy as np
from PIL import Image, ImageTk
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pldm_envs.wall.wall import DotWall
from scripts.visualize_two_rooms import (
    action_toward,
    compose_frame,
    make_heuristic_waypoints,
)


KEY_TO_DIRECTION = {
    "Left": np.array([-1.0, 0.0], dtype=np.float32),
    "Right": np.array([1.0, 0.0], dtype=np.float32),
    "Up": np.array([0.0, -1.0], dtype=np.float32),
    "Down": np.array([0.0, 1.0], dtype=np.float32),
    "a": np.array([-1.0, 0.0], dtype=np.float32),
    "d": np.array([1.0, 0.0], dtype=np.float32),
    "w": np.array([0.0, -1.0], dtype=np.float32),
    "s": np.array([0.0, 1.0], dtype=np.float32),
}


class TwoRoomsViewer:
    def __init__(self, seed, scale, fps, autoplay):
        self.seed = seed
        self.scale = scale
        self.delay_ms = round(1000 / fps)
        self.autoplay = autoplay
        self.pressed = set()
        self.done = False
        self.truncated = False
        self.waypoints = []

        self.env = DotWall(device=torch.device("cpu"))
        self.obs, self.info = self.env.reset(seed=self.seed)
        self.target_obs = self.info["target_obs"]
        self.waypoints = make_heuristic_waypoints(self.env)

        self.root = tk.Tk()
        self.root.title("PLDM Two-Rooms")
        self.root.resizable(False, False)

        side = self.env.img_size * self.scale
        self.canvas = tk.Canvas(
            self.root, width=side, height=side, highlightthickness=0
        )
        self.canvas.pack()
        self.label = tk.Label(
            self.root, anchor="w", justify="left", font=("Consolas", 10)
        )
        self.label.pack(fill="x", padx=8, pady=6)

        self.image_id = None
        self.photo = None

        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.bind("<KeyRelease>", self.on_key_release)

    def on_key_press(self, event):
        key = event.keysym
        if key in KEY_TO_DIRECTION:
            self.pressed.add(key)
        elif key == "space":
            self.autoplay = not self.autoplay
        elif key.lower() == "r":
            self.reset()
        elif key == "Escape":
            self.root.destroy()

    def on_key_release(self, event):
        self.pressed.discard(event.keysym)

    def reset(self):
        self.seed += 1
        self.obs, self.info = self.env.reset(seed=self.seed)
        self.target_obs = self.info["target_obs"]
        self.waypoints = make_heuristic_waypoints(self.env)
        self.done = False
        self.truncated = False

    def manual_action(self):
        direction = np.zeros(2, dtype=np.float32)
        for key in self.pressed:
            direction += KEY_TO_DIRECTION[key]

        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return None

        return direction / norm * self.env.max_step_norm

    def autopilot_action(self):
        while len(self.waypoints) > 1:
            position = self.env.dot_position.detach().cpu().numpy()
            if np.linalg.norm(position - self.waypoints[0]) > 0.75:
                break
            self.waypoints.pop(0)
        return action_toward(self.env, self.waypoints[0])

    def step_env(self):
        if self.done or self.truncated:
            return

        action = self.autopilot_action() if self.autoplay else self.manual_action()
        if action is None:
            return

        self.obs, _, self.done, self.truncated, self.info = self.env.step(action)

    def draw(self):
        frame = compose_frame(self.obs, self.target_obs)
        image = Image.fromarray(frame).resize(
            (frame.shape[1] * self.scale, frame.shape[0] * self.scale),
            Image.Resampling.NEAREST,
        )
        self.photo = ImageTk.PhotoImage(image)

        if self.image_id is None:
            self.image_id = self.canvas.create_image(
                0, 0, anchor="nw", image=self.photo
            )
        else:
            self.canvas.itemconfig(self.image_id, image=self.photo)

        pos = self.info["dot_position"].detach().cpu().numpy()
        target = self.info["target_position"].detach().cpu().numpy()
        mode = "autopilot" if self.autoplay else "manual"
        status = "done" if self.done else "truncated" if self.truncated else "running"
        self.label.config(
            text=(
                f"mode={mode} status={status} seed={self.seed}\n"
                f"pos=({pos[0]:.2f}, {pos[1]:.2f}) "
                f"target=({target[0]:.2f}, {target[1]:.2f})\n"
                "WASD/arrows move | Space autopilot | R reset | Esc quit"
            )
        )

    def tick(self):
        self.step_env()
        self.draw()
        self.root.after(self.delay_ms, self.tick)

    def run(self):
        self.draw()
        self.root.after(self.delay_ms, self.tick)
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Real-time PLDM Two-Rooms viewer.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=int, default=10)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--autoplay", action="store_true")
    args = parser.parse_args()

    TwoRoomsViewer(
        seed=args.seed,
        scale=args.scale,
        fps=args.fps,
        autoplay=args.autoplay,
    ).run()


if __name__ == "__main__":
    main()
