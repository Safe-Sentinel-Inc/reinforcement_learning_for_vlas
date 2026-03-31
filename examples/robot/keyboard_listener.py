import logging
import select
import sys
import termios
import threading
import time
import tty

logger = logging.getLogger(__name__)


class KeyboardListener:
    """Non-blocking keyboard reader that runs on a background thread."""

    def __init__(self):
        self.reset_flag = False
        self.quit_flag = False
        self.start_flag = False
        self.listener_thread = None
        self.running = False
        self.old_settings = None

    def start(self):
        self.running = True
        self.old_settings = termios.tcgetattr(sys.stdin)
        self.listener_thread = threading.Thread(target=self._listen, daemon=True)
        self.listener_thread.start()
        logger.info("Keyboard listener started. Press 'Enter' to start, 'R' to reset, 'Q' to quit.")

    def _listen(self):
        try:
            tty.setcbreak(sys.stdin.fileno())
            while self.running:
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1)
                    if key.lower() == "r":
                        self.reset_flag = True
                        logger.info("Reset requested!")
                    elif key.lower() == "q":
                        self.quit_flag = True
                        logger.info("Quit requested!")
                    elif key in {"\n", "\r"}:
                        self.start_flag = True
                        logger.info("Start requested!")
                time.sleep(0.01)
        except Exception as exc:
            logger.error("Keyboard listener error: %s", exc)
        finally:
            if self.old_settings:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)

    def check_reset(self):
        if self.reset_flag:
            self.reset_flag = False
            return True
        return False

    def check_quit(self):
        return self.quit_flag

    def check_start(self):
        if self.start_flag:
            self.start_flag = False
            return True
        return False

    def stop(self):
        self.running = False
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
