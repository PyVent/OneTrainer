import time

from modules.util.enum.TimeUnit import TimeUnit
from modules.util.TrainProgress import TrainProgress


class TimedActionMixin:
    def __init__(self):
        super().__init__()
        self.__previous_action = {}
        self.__start_time = time.monotonic() # resist system clock changes

    def repeating_action_needed(
            self,
            name: str,
            interval: float,
            unit: TimeUnit,
            train_progress: TrainProgress,
            start_at_zero: bool = True,
    ):
        if name not in self.__previous_action:
            self.__previous_action[name] = -1

        match unit:
            case TimeUnit.EPOCH:
                if int(interval) == 0:
                    return False
                if start_at_zero:
                    return train_progress.epoch % int(interval) == 0 and train_progress.epoch_step == 0
                else:
                    # should actually be the last step of each epoch, but we don't know how many steps an epoch has
                    return train_progress.epoch % int(interval) == 0 and train_progress.epoch_step == 0 \
                        and train_progress.epoch > 0
            case TimeUnit.STEP:
                if int(interval) == 0:
                    return False
                if start_at_zero:
                    return train_progress.global_step % int(interval) == 0
                else:
                    return (train_progress.global_step + 1) % int(interval) == 0
            case TimeUnit.SECOND | TimeUnit.MINUTE | TimeUnit.HOUR:
                seconds = interval * {
                    TimeUnit.SECOND: 1,
                    TimeUnit.MINUTE: 60,
                    TimeUnit.HOUR: 3600,
                }[unit]
                now = time.monotonic()
                previous = self.__previous_action[name]
                if previous < 0:
                    self.__previous_action[name] = now
                    return start_at_zero
                if now - previous >= seconds:
                    self.__previous_action[name] = now
                    return True
                return False
            case TimeUnit.NEVER:
                return False
            case TimeUnit.ALWAYS:
                return True
            case _:
                return False

    def single_action_elapsed(
            self,
            name: str,
            delay: float,
            unit: TimeUnit,
            train_progress: TrainProgress,
    ):
        if name not in self.__previous_action:
            self.__previous_action[name] = time.monotonic()

        match unit:
            case TimeUnit.EPOCH:
                return (train_progress.epoch + 1) > int(delay)
            case TimeUnit.STEP:
                return (train_progress.global_step + 1) > int(delay)
            case TimeUnit.SECOND:
                seconds_since_start = time.monotonic() - self.__start_time
                return seconds_since_start >= delay
            case TimeUnit.MINUTE:
                seconds_since_start = time.monotonic() - self.__start_time
                return seconds_since_start >= (delay * 60)
            case TimeUnit.HOUR:
                seconds_since_start = time.monotonic() - self.__start_time
                return seconds_since_start >= (delay * 60 * 60)
            case TimeUnit.NEVER:
                return False
            case TimeUnit.ALWAYS:
                return True
            case _:
                return False
