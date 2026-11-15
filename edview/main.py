import sys
import signal
import argparse
import platform
import warnings
from PySide6.QtCore import QTimer
from edview.lib.interactions import EdviewInteraction

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
    parser = argparse.ArgumentParser(description="EDView: viewer of EDXD "
                                                 "data processed by Algodif")
    parser.add_argument("-b", dest="base_folder", default=None,
                        help="Base folder of the data browser")
    args = parser.parse_args()
    if platform.system() == "Windows":
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "algodif.edview")
    warnings.filterwarnings("ignore", message="Covariance of the parameters "
                                              "could not be estimated")
    controller = EdviewInteraction(args.base_folder)
    controller.app.aboutToQuit.connect(print_exit_message)

    def _handle_signal(_signum=None, _frame=None):
        print_exit_message()
        controller.app.quit()

    for name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, name, None)
        if sig is not None:
            signal.signal(sig, _handle_signal)
    # Let Python handle Ctrl+C while the Qt event loop runs
    timer = QTimer()
    timer.start(250)
    timer.timeout.connect(lambda: None)
    try:
        sys.exit(controller.run())
    except KeyboardInterrupt:
        print_exit_message()
        sys.exit(0)


if __name__ == "__main__":
    main()
