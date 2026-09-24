import unittest
from unittest.mock import patch

from modules.util.enum.TimeUnit import TimeUnit
from modules.util.TimedActionMixin import TimedActionMixin
from modules.util.TrainProgress import TrainProgress


class TrainingTimingTest(unittest.TestCase):
    def test_progress_counts_steps_samples_and_epochs(self):
        progress = TrainProgress()
        progress.next_step(3)
        progress.next_step(2)
        self.assertEqual((progress.epoch_step, progress.epoch_sample, progress.global_step), (2, 5, 2))
        self.assertEqual(progress.filename_string(), "2-0-2")

        progress.next_epoch()
        self.assertEqual(
            (progress.epoch, progress.epoch_step, progress.epoch_sample, progress.global_step), (1, 0, 0, 2)
        )

    def test_step_and_epoch_repeating_actions(self):
        timer = TimedActionMixin()
        progress = TrainProgress()
        self.assertTrue(timer.repeating_action_needed("step", 3, TimeUnit.STEP, progress))
        self.assertFalse(timer.repeating_action_needed("accumulate", 3, TimeUnit.STEP, progress, False))
        progress.next_step(1)
        self.assertFalse(timer.repeating_action_needed("step", 3, TimeUnit.STEP, progress))
        self.assertFalse(timer.repeating_action_needed("accumulate", 3, TimeUnit.STEP, progress, False))
        progress.next_step(1)
        self.assertTrue(timer.repeating_action_needed("accumulate", 3, TimeUnit.STEP, progress, False))

        self.assertFalse(timer.repeating_action_needed("epoch", 1, TimeUnit.EPOCH, progress))
        progress.next_epoch()
        self.assertTrue(timer.repeating_action_needed("epoch", 1, TimeUnit.EPOCH, progress, False))

    def test_first_time_action_can_run_at_monotonic_zero(self):
        progress = TrainProgress()
        with patch("modules.util.TimedActionMixin.time.monotonic", return_value=0):
            timer = TimedActionMixin()
            self.assertTrue(timer.repeating_action_needed("sample", 10, TimeUnit.SECOND, progress))
            self.assertFalse(timer.repeating_action_needed("backup", 10, TimeUnit.SECOND, progress, False))

    def test_delayed_action_runs_at_exact_interval(self):
        progress = TrainProgress()
        with patch("modules.util.TimedActionMixin.time.monotonic", return_value=0) as clock:
            timer = TimedActionMixin()
            self.assertFalse(timer.repeating_action_needed("backup", 5, TimeUnit.SECOND, progress, False))
            clock.return_value = 4.9
            self.assertFalse(timer.repeating_action_needed("backup", 5, TimeUnit.SECOND, progress, False))
            clock.return_value = 5
            self.assertTrue(timer.repeating_action_needed("backup", 5, TimeUnit.SECOND, progress, False))
            self.assertFalse(timer.repeating_action_needed("backup", 5, TimeUnit.SECOND, progress, False))

    def test_minutes_and_hours_use_the_same_monotonic_clock(self):
        progress = TrainProgress()
        with patch("modules.util.TimedActionMixin.time.monotonic", return_value=0) as clock:
            timer = TimedActionMixin()
            self.assertFalse(timer.repeating_action_needed("minute", 1, TimeUnit.MINUTE, progress, False))
            self.assertFalse(timer.repeating_action_needed("hour", 1, TimeUnit.HOUR, progress, False))
            clock.return_value = 60
            self.assertTrue(timer.repeating_action_needed("minute", 1, TimeUnit.MINUTE, progress, False))
            self.assertFalse(timer.repeating_action_needed("hour", 1, TimeUnit.HOUR, progress, False))
            clock.return_value = 3600
            self.assertTrue(timer.repeating_action_needed("hour", 1, TimeUnit.HOUR, progress, False))

    def test_single_action_uses_elapsed_time_and_never_always(self):
        progress = TrainProgress()
        with patch("modules.util.TimedActionMixin.time.monotonic", return_value=0) as clock:
            timer = TimedActionMixin()
            self.assertFalse(timer.single_action_elapsed("delay", 5, TimeUnit.SECOND, progress))
            clock.return_value = 5
            self.assertTrue(timer.single_action_elapsed("delay", 5, TimeUnit.SECOND, progress))
            self.assertFalse(timer.repeating_action_needed("never", 1, TimeUnit.NEVER, progress))
            self.assertTrue(timer.repeating_action_needed("always", 1, TimeUnit.ALWAYS, progress))


if __name__ == "__main__":
    unittest.main()
