from datetime import datetime

import pytz

# @desc: The timezone of the application
timezone = pytz.timezone("Asia/Manila")
timezone_current_time = timezone.localize(datetime.now())
