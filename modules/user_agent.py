from datetime import datetime

from flask import request
from ua_parser import user_agent_parser
from werkzeug.user_agent import UserAgent
from werkzeug.utils import cached_property


class ParsedUserAgent(UserAgent):
    """
    This class is a wrapper around the UserAgent class from Werkzeug.
    It parses the user agent string and provides access to the browser
    and operating system properties. It also provides a method to
    return a dictionary of the parsed user agent string.
    """

    @cached_property
    def _details(self):
        """
        Parse the user agent string and return a dictionary of the
        parsed user agent string.
        """
        return user_agent_parser.Parse(self.string)

    @property
    def platform(self):
        """Return the operating system name."""
        return self._details['os']['family']

    @property
    def browser(self):
        """Return the browser name."""
        return self._details['user_agent']['family']

    @property
    def version(self):
        """Return the browser version."""
        return '.'.join(
            part
            for key in ('major', 'minor', 'patch')
            if (part := self._details['user_agent'][key]) is not None
        )

    @property
    def os_version(self):
        """Return the operating system version."""
        return '.'.join(
            part
            for key in ('major', 'minor', 'patch')
            if (part := self._details['os'][key]) is not None
        )


def get_os_browser_versions():
    """Get the User's browser and OS from the request header."""
    user_agent = ParsedUserAgent(request.headers.get('User-Agent'))
    return user_agent.platform, user_agent.os_version, user_agent.browser, user_agent.version, datetime.now()
