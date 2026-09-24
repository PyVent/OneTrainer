import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from modules.util.concept_stats import folder_scan, init_concept_stats
from modules.util.config.ConceptConfig import ConceptConfig

import cv2
from PIL import Image


class ConceptStatisticsTest(unittest.TestCase):
    def test_advanced_image_statistics_count_each_caption_line(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (4, 2)).save(root / "sample.png")
            (root / "sample-masklabel.png").touch()
            (root / "sample.txt").write_text("red fox\nblue sky\n", encoding="utf-8")
            (root / "orphan.txt").write_text("unused", encoding="utf-8")
            concept = ConceptConfig.default_values()
            concept.path = str(root)

            stats = folder_scan(
                str(root),
                init_concept_stats(True),
                True,
                concept,
                time.perf_counter(),
                30,
                threading.Event(),
            )

            self.assertFalse(stats["force_cancelled"])
            self.assertEqual(stats["directory_count"], 1)
            self.assertEqual(stats["image_count"], 1)
            self.assertEqual(stats["image_with_mask_count"], 1)
            self.assertEqual(stats["mask_count"], 1)
            self.assertEqual(stats["paired_masks"], 1)
            self.assertEqual(stats["image_with_caption_count"], 1)
            self.assertEqual(stats["caption_count"], 2)
            self.assertEqual(stats["paired_captions"], 1)
            self.assertEqual(stats["unpaired_captions"], 1)
            self.assertEqual(stats["subcaption_count"], 2)
            self.assertEqual(stats["avg_caption_length"], [7.5, 2])
            self.assertEqual(stats["avg_pixels"], 8)

    def test_missing_directory_is_reported_as_cancelled(self):
        concept = ConceptConfig.default_values()
        stats = folder_scan(
            "missing-concept-directory",
            init_concept_stats(False),
            False,
            concept,
            time.perf_counter(),
            30,
            threading.Event(),
        )
        self.assertTrue(stats["force_cancelled"])
        self.assertEqual(stats["image_count"], 0)

    def test_unreadable_video_does_not_crash_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "broken.mp4").touch()
            concept = ConceptConfig.default_values()
            concept.path = str(root)
            for opened in (False, True):
                with self.subTest(opened=opened):
                    capture = Mock()
                    capture.isOpened.return_value = opened
                    capture.get.return_value = 0
                    with patch("modules.util.concept_stats.cv2.VideoCapture", return_value=capture):
                        stats = folder_scan(
                            str(root),
                            init_concept_stats(True),
                            True,
                            concept,
                            time.perf_counter(),
                            30,
                            threading.Event(),
                        )

                    self.assertEqual(stats["video_count"], 1)
                    self.assertEqual(stats["avg_pixels"], 0)
                    capture.release.assert_called_once()
                    if not opened:
                        capture.get.assert_not_called()

    def test_video_statistics_include_metadata_and_all_caption_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "clip.mp4").touch()
            (root / "clip.txt").write_text("one\ntwo words\n", encoding="utf-8")
            concept = ConceptConfig.default_values()
            concept.path = str(root)
            capture = Mock()
            capture.isOpened.return_value = True
            metadata = {
                cv2.CAP_PROP_FRAME_WIDTH: 16,
                cv2.CAP_PROP_FRAME_HEIGHT: 8,
                cv2.CAP_PROP_FRAME_COUNT: 3,
                cv2.CAP_PROP_FPS: 12,
            }
            capture.get.side_effect = metadata.__getitem__

            with patch("modules.util.concept_stats.cv2.VideoCapture", return_value=capture):
                stats = folder_scan(
                    str(root),
                    init_concept_stats(True),
                    True,
                    concept,
                    time.perf_counter(),
                    30,
                    threading.Event(),
                )

            self.assertEqual(stats["video_count"], 1)
            self.assertEqual(stats["video_with_caption_count"], 1)
            self.assertEqual(stats["subcaption_count"], 2)
            self.assertEqual(stats["avg_caption_length"], [6, 1.5])
            self.assertEqual(stats["avg_pixels"], 128)
            self.assertEqual(stats["avg_length"], 3)
            self.assertEqual(stats["avg_fps"], 12)
            capture.release.assert_called_once()

    def test_unreadable_video_does_not_lower_average_image_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            broken = root / "broken"
            broken.mkdir()
            (broken / "clip.mp4").touch()
            valid = root / "valid"
            valid.mkdir()
            Image.new("RGB", (4, 2)).save(valid / "sample.png")
            concept = ConceptConfig.default_values()
            concept.path = str(root)

            with patch("modules.util.concept_stats.cv2.VideoCapture") as capture:
                capture.return_value.isOpened.return_value = False
                stats = folder_scan(
                    str(broken), init_concept_stats(True), True, concept,
                    time.perf_counter(), 30, threading.Event(),
                )
            stats = folder_scan(
                str(valid), stats, True, concept,
                time.perf_counter(), 30, threading.Event(),
            )

            self.assertEqual((stats["video_count"], stats["image_count"]), (1, 1))
            self.assertEqual(stats["avg_pixels"], 8)

    def test_unreadable_video_does_not_lower_valid_video_averages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            broken = root / "broken"
            broken.mkdir()
            (broken / "bad.mp4").touch()
            valid = root / "valid"
            valid.mkdir()
            (valid / "good.mp4").touch()
            concept = ConceptConfig.default_values()
            concept.path = str(root)
            bad_capture = Mock()
            bad_capture.isOpened.return_value = False
            good_capture = Mock()
            good_capture.isOpened.return_value = True
            metadata = {
                cv2.CAP_PROP_FRAME_WIDTH: 16,
                cv2.CAP_PROP_FRAME_HEIGHT: 8,
                cv2.CAP_PROP_FRAME_COUNT: 3,
                cv2.CAP_PROP_FPS: 12,
            }
            good_capture.get.side_effect = metadata.__getitem__

            with patch("modules.util.concept_stats.cv2.VideoCapture", side_effect=[bad_capture, good_capture]):
                stats = folder_scan(
                    str(broken), init_concept_stats(True), True, concept,
                    time.perf_counter(), 30, threading.Event(),
                )
                stats = folder_scan(
                    str(valid), stats, True, concept,
                    time.perf_counter(), 30, threading.Event(),
                )

            self.assertEqual(stats["video_count"], 2)
            self.assertEqual(stats["avg_pixels"], 128)
            self.assertEqual(stats["avg_length"], 3)
            self.assertEqual(stats["avg_fps"], 12)


if __name__ == "__main__":
    unittest.main()
