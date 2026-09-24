import unittest

from modules.util.lr_scheduler_util import (
    lr_lambda_constant,
    lr_lambda_cosine,
    lr_lambda_cosine_with_hard_restarts,
    lr_lambda_cosine_with_restarts,
    lr_lambda_linear,
    lr_lambda_rex,
    lr_lambda_warmup,
)


class LearningRateScheduleTest(unittest.TestCase):
    def test_warmup_transitions_to_the_requested_schedule(self):
        schedule = lr_lambda_warmup(4, lr_lambda_linear(4, 0.2))
        for step, expected in ((0, 0), (2, 0.5), (3, 0.75), (4, 1), (6, 0.6), (8, 0.2)):
            with self.subTest(step=step):
                self.assertAlmostEqual(schedule(step), expected)

    def test_constant_and_linear_boundaries(self):
        self.assertEqual([lr_lambda_constant()(step) for step in (0, 10, 100)], [1, 1, 1])
        linear = lr_lambda_linear(10, 0.2)
        self.assertAlmostEqual(linear(5), 0.6)
        self.assertEqual([linear(step) for step in (10, 11, 30)], [0.2, 0.2, 0.2])

    def test_cosine_stays_at_minimum_after_schedule_ends(self):
        cosine = lr_lambda_cosine(10, 0.2)
        self.assertAlmostEqual(cosine(0), 1)
        self.assertAlmostEqual(cosine(5), 0.6)
        for step in (10, 11, 20, 30):
            with self.subTest(step=step):
                self.assertAlmostEqual(cosine(step), 0.2)

    def test_restart_schedules_end_at_minimum(self):
        for factory in (lr_lambda_cosine_with_restarts, lr_lambda_cosine_with_hard_restarts):
            with self.subTest(schedule=factory.__name__):
                schedule = factory(10, 2, 0.2)
                self.assertAlmostEqual(schedule(0), 1)
                self.assertGreater(schedule(5), 0.2)
                for step in (10, 11, 20):
                    self.assertAlmostEqual(schedule(step), 0.2)

    def test_no_decay_steps_use_minimum_instead_of_dividing_by_zero(self):
        factories = (
            lambda: lr_lambda_linear(0, 0.2),
            lambda: lr_lambda_cosine(0, 0.2),
            lambda: lr_lambda_cosine_with_restarts(0, 1, 0.2),
            lambda: lr_lambda_cosine_with_hard_restarts(0, 1, 0.2),
            lambda: lr_lambda_rex(0, 0.2),
        )
        for factory in factories:
            with self.subTest(schedule=factory):
                schedule = factory()
                self.assertEqual([schedule(step) for step in (0, 1)], [0.2, 0.2])


if __name__ == "__main__":
    unittest.main()
