# fusion/gui/runner.py

from PyQt5 import QtWidgets
import multiprocessing

def _as_dict(config_like):
    """Accept a ConfigManager or a plain dict; return a dict."""
    try:
        return config_like.as_dict()
    except AttributeError:
        return config_like or {}

def launch_gui(config_like):
    """
    Launch the GUI with the provided config (mirrors run_simulation pattern):
      - normalize config
      - create QApplication + MainWindow
      - seed window with config dict (expects {'s1': {...}})
      - create and attach shared progress dict (matches legacy run_gui behavior)
      - show and exec the event loop
    """
    from fusion.gui.main_window import MainWindow  # local import keeps CLI lightweight

    cfg = _as_dict(config_like)

    app = QtWidgets.QApplication([])
    window = MainWindow()

    # Seed the GUI if we have a config (no logic change if empty)
    if isinstance(cfg, dict) and cfg:
        try:
            # 1) Inject full simulation config
            window.set_simulation_config(cfg)

            # 2) Create shared_progress_dict like the original run_gui did
            try:
                first_key = next(iter(cfg))
                sim_conf = cfg[first_key]

                if 'erlangs' in sim_conf:
                    erlangs = sim_conf['erlangs']
                    erlang_start = erlangs['start']
                    erlang_stop = erlangs['stop']
                    erlang_step = erlangs['step']
                else:
                    erlang_start = sim_conf['erlang_start']
                    erlang_stop = sim_conf['erlang_stop']
                    erlang_step = sim_conf['erlang_step']

                total_erlangs = len(range(erlang_start, erlang_stop, erlang_step))

                manager = multiprocessing.Manager()
                shared_progress_dict = manager.dict({i: 0 for i in range(total_erlangs)})
                window.set_shared_progress_dict(shared_progress_dict)
            except Exception:
                # Keep GUI resilient even if config is partial/malformed
                pass

        except Exception:
            # Don’t explode on startup if something’s missing
            pass

    window.show()
    app.exec_()
