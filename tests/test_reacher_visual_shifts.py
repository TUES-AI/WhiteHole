import unittest

import numpy as np

from scripts.visualize_reacher_shifts import (
    DYNAMIC_CAMERA_PERIOD,
    SOURCE_CAMERA_FOVY,
    SOURCE_CAMERA_POS,
    SOURCE_CAMERA_QUAT,
    VARIANTS,
    camera_phase_from_seed,
    canonicalize_dynamic_frame,
    dynamic_to_source_homography,
    dynamic_camera_parameters,
    hide_reacher_foreground,
    look_at_quat,
    make_env,
    project_world_to_image,
    render_source_background,
)


class DynamicCameraTest(unittest.TestCase):
    def test_variant_is_registered(self):
        self.assertIn("dynamic_camera", VARIANTS)

    def test_seed_phase_is_deterministic_and_episode_specific(self):
        self.assertEqual(camera_phase_from_seed(42), camera_phase_from_seed(42))
        self.assertNotEqual(camera_phase_from_seed(42), camera_phase_from_seed(43))
        self.assertGreaterEqual(camera_phase_from_seed(42), 0.0)
        self.assertLess(camera_phase_from_seed(42), 2.0 * np.pi)

    def test_schedule_is_smooth_bounded_and_periodic(self):
        samples = [dynamic_camera_parameters(step) for step in np.linspace(0, 48, 193)]
        positions = np.stack([sample["position"] for sample in samples])
        targets = np.stack([sample["target"] for sample in samples])
        fovys = np.asarray([sample["fovy"] for sample in samples])
        radii = np.linalg.norm(positions[:, :2], axis=1)

        self.assertTrue(np.all((radii >= 0.199) & (radii <= 0.281)))
        self.assertTrue(np.all((positions[:, 2] >= 0.639) & (positions[:, 2] <= 0.841)))
        self.assertTrue(np.all((fovys >= 39.99) & (fovys <= 52.01)))
        self.assertTrue(np.all(np.linalg.norm(targets[:, :2], axis=1) <= 0.026))

        start = dynamic_camera_parameters(0.0)
        one_orbit = dynamic_camera_parameters(DYNAMIC_CAMERA_PERIOD)
        self.assertTrue(np.allclose(start["position"][:2], one_orbit["position"][:2]))

        frame_deltas = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        self.assertLess(float(frame_deltas.max()), 0.04)

    def test_look_at_quaternion_is_normalized(self):
        params = dynamic_camera_parameters(7.0, camera_phase_from_seed(123))
        quat = look_at_quat(params["position"], params["target"])
        self.assertAlmostEqual(float(np.linalg.norm(quat)), 1.0, places=6)

    def test_dynamic_homography_matches_plane_projections(self):
        import cv2

        step = 9.0
        phase_offset = camera_phase_from_seed(123)
        points = np.array(
            [
                [-0.2, -0.1, 0.015],
                [0.2, -0.1, 0.015],
                [0.1, 0.2, 0.015],
                [-0.2, 0.15, 0.015],
            ]
        )
        params = dynamic_camera_parameters(step, phase_offset)
        dynamic_pixels = project_world_to_image(
            points,
            np.asarray(params["position"]),
            look_at_quat(
                np.asarray(params["position"]), np.asarray(params["target"])
            ),
            float(params["fovy"]),
            224,
            224,
        )
        source_pixels = project_world_to_image(
            points,
            SOURCE_CAMERA_POS,
            SOURCE_CAMERA_QUAT,
            SOURCE_CAMERA_FOVY,
            224,
            224,
        )
        homography = dynamic_to_source_homography(
            step, phase_offset, image_width=224, image_height=224
        )
        mapped = cv2.perspectiveTransform(
            dynamic_pixels[None].astype(np.float32), homography
        )[0]
        self.assertTrue(np.allclose(mapped, source_pixels, atol=1e-3))

    def test_canonicalization_preserves_shape_and_dtype(self):
        frame = np.full((32, 32, 3), 127, dtype=np.uint8)
        background = np.full_like(frame, 64)
        canonical = canonicalize_dynamic_frame(
            frame,
            step=4.0,
            phase_offset=0.3,
            source_background=background,
        )
        self.assertEqual(canonical.shape, frame.shape)
        self.assertEqual(canonical.dtype, frame.dtype)

    def test_completion_background_contains_no_movable_state(self):
        expected = render_source_background(64)
        env = make_env("source")
        physics = env.env.physics
        hide_reacher_foreground(physics)
        try:
            for qpos, target in (
                ([-1.2, 1.7], [-0.1, 0.15]),
                ([0.2, -0.5], [0.12, -0.08]),
            ):
                env.set_state(np.asarray(qpos), np.zeros(2))
                env.set_target_qpos(np.asarray(target))
                actual = env.render(width=64, height=64)
                self.assertTrue(np.array_equal(actual, expected))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
