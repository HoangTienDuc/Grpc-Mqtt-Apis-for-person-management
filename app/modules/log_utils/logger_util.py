from logging.config import ConvertingList, ConvertingDict, valid_ident
from logging.handlers import QueueHandler, QueueListener
import logging
from queue import Queue
import atexit
import yaml


# import config

def _resolve_handlers(l):
    if not isinstance(l, ConvertingList):
        return l

    # Indexing the list performs the evaluation.
    return [l[i] for i in range(len(l))]


def _resolve_queue(q):
    if not isinstance(q, ConvertingDict):
        return q
    if '__resolved_value__' in q:
        return q['__resolved_value__']

    cname = q.pop('class')
    klass = q.configurator.resolve(cname)
    props = q.pop('.', None)
    kwargs = {k: q[k] for k in q if valid_ident(k)}
    result = klass(**kwargs)
    if props:
        for name, value in props.items():
            setattr(result, name, value)

    q['__resolved_value__'] = result
    return result


class QueueListenerHandler(QueueHandler):

    def __init__(self, handlers, respect_handler_level=True, auto_run=True, queue=Queue(-1)):
        queue = _resolve_queue(queue)
        super().__init__(queue)
        handlers = _resolve_handlers(handlers)
        self._listener = QueueListener(
            self.queue,
            *handlers,
            respect_handler_level=respect_handler_level)
        if auto_run:
            self.start()
            atexit.register(self.stop)


    def start(self):
        self._listener.start()


    def stop(self):
        self._listener.stop()

    def emit(self, record):
        return super().emit(record)

class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.INFO

class WarningFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.WARNING

class DebugFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.DEBUG

class CriticalFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.CRITICAL

class ErrorFilter(logging.Filter):
    def filter(self, record):
        return record.levelno >= logging.ERROR

