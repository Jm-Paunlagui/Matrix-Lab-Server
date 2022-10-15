from datetime import datetime

import pytz

"""This module contains functions to get the current time in the local timezone."""

timezone = pytz.timezone("Asia/Manila")
timezone_current_time = timezone.localize(datetime.now())
