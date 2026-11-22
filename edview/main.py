import sys
import signal
import argparse
import platform
import multiprocessing
from PySide6.QtCore import QTimer
from edview.lib.interactions import EdviewInteraction
from edview.lib.rendering import GL_AVAILABLE, GL_ERROR
import edview.lib.utilities as util


exit_printed = False


def print_exit_message(*_args):
    global exit_printed
    if exit_printed:
        return
    exit_printed = True
    print("\n======================")
    print(" Exit the EDView app! ")
    print("======================\n", flush=True)


def main():
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="EDView: viewer of EDDT "
                                                 "data processed by Algodif")
    parser.add_argument("-b", dest="base_folder", default=None,
                        help="Base folder of the data browser")
    args = parser.parse_args()
    if platform.system() == "Windows":
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "algodif.edview")
    util.suppress_warnings()
    if not GL_AVAILABLE:
        print(f"Warning: 3D views are disabled ({GL_ERROR}). The 4D viewer "
              f"and the multiple-lines viewer use 2D plots instead. Install "
              f"PyOpenGL to enable them: python -m pip install PyOpenGL",
              file=sys.stderr, flush=True)
    controller = EdviewInteraction(args.base_folder)
    controller.app.aboutToQuit.connect(print_exit_message)

    def _handle_signal(_signum=None, _frame=None):
        print_exit_message()
        controller.app.quit()

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            signal.signal(sig, _handle_signal)
    timer = QTimer()
    timer.start(250)
    timer.timeout.connect(lambda: None)
    try:
        sys.exit(controller.run())
    except KeyboardInterrupt:
        controller.shutdown()
        print_exit_message()
        sys.exit(0)


if __name__ == "__main__":
    main()
