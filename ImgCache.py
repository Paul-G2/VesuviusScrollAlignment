import sys
import time
import threading
import psutil


class ImgCache(object):
    """
    Maintains a cache of recently-used images, to minimize unnecessary
    re-loading from disk.
    """

    def __init__(self, max_gb=None, purge_interval_secs=1800):
        """
        Constructor.
        :param max_gb: The maximum size of the cache, in gigabytes
        :param purge_interval_secs: The time interval between cache purges, in seconds
        """
        self.max_bytes = max_gb * (2**30) if max_gb is not None else None
        self.images_stored = 0
        self.bytes_stored = 0
        self.purge_interval_secs = purge_interval_secs
        self.cache = {}

        self.purge_timer = None
        if purge_interval_secs is not None:
            self.purge_timer = threading.Timer(purge_interval_secs, self._timed_purge)
            self.purge_timer.start()


    def __del__(self):
        """
        Destructor
        """
        if self.purge_timer is not None:
            self.purge_timer.cancel()


    def clear(self):
        """
        Empties the cache
        """
        self.cache.clear()
        self.bytes_stored = 0
        self.images_stored = 0


    def get_image(self, key):
        """
        Gets an image from the cache.
        """
        cached_item = self.cache.get(key)
        if cached_item is not None:
            cached_item['lastUsed'] = time.time()
            return cached_item['img']
        else:
           return None


    def add_image(self, img, key):
        """
        Adds an image to the cache.
        """
        # Remove the oldest image if we are at capacity
        img_bytes = sys.getsizeof(img)
        if (self.max_bytes is not None and self.bytes_stored + img_bytes > self.max_bytes) or \
                (psutil.virtual_memory().percent > 92):
            if len(self.cache) > 0:
                oldest_key = min(self.cache, key=lambda key: self.cache[key]['lastUsed'])
                to_remove = self.cache[oldest_key]
                self.bytes_stored -= sys.getsizeof(to_remove['img'])
                self.images_stored -= 1
                del self.cache[oldest_key]

        # Add to cache
        self.cache[key] = {'img':img, 'lastUsed':time.time()}
        self.bytes_stored += img_bytes
        self.images_stored += 1


    def _timed_purge(self):
        """
        Removes images that haven't been used in a while.
        """
        now = time.time()
        to_remove = [key for key, value in self.cache.items() if now - value['lastUsed'] > self.purge_interval_secs]
        for key in to_remove:
            self.bytes_stored -= sys.getsizeof(self.cache[key]['img'])
            self.images_stored -= 1
            del self.cache[key]

        self.purge_timer = threading.Timer(self.purge_interval_secs, self._timed_purge)
        self.purge_timer.start()