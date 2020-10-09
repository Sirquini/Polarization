import sys

def progress_bar(value, end, length=20, status=''):
    """Prints a progress indicator to `stderr`, with the current `value`
    and a bar of width `length`.

    This function flushes the `stderr` stream.

    Args:
      value: the number of the current iteration.
      end: the max number.
      length: optional width of the progress bar in characters, defaults to 20.
      status: optional message, max 10 characters
    """
    percent = value * 100 // end
    bar = '=' * (percent * length // 100)
    spaces = ' ' * (length - len(bar))
    print("\rProgress: [{}] {}/{} {:10.10}".format(bar + spaces, value, end, status), end='', file=sys.stderr, flush=True)

class ProgressRange():
    """A class wrapper for `range(stop)` that prints the progress status.
    
    Behaves like an iterator, as such may be usen in a for loop like one
    would use `range(stop)`.

    The status bar output uses the `progress_bar` format.
    """
    def __init__(self, stop, status=''):
        """Creates a new iterable like `range`.

        Generates values from `0` up to, but not including, `stop`.       
        Optionaly, displays a status message during iteration.
        
        Args:
            stop (int): The non-inclusive upper limit.
            status (str, optional): A message next to the progress bar.
        """
        self.sequence = iter(range(stop))
        self.max = stop
        self.status = status

    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            n = next(self.sequence)
        except StopIteration as stop:
            progress_bar(self.max, self.max, status="Done")
            print()
            raise stop
        progress_bar(n, self.max, status=self.status)
        return n
