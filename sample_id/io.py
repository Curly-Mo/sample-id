import logging

logger = logging.getLogger(__name__)


class Track(object):
    def __init__(self, id: str, path: str):
        self.id = id
        self.path = path
