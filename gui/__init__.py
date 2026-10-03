"""Strata Manager: an optional local GUI management layer for a Strata install (Windows and Linux).

    python gui/manager.py            (or START-MANAGER.bat / ./start-manager.sh)
    python -m unittest gui.test_manager

The Manager reuses setup.py and the strata-*.json configs; it does not duplicate Strata's setup or engine.
The only platform-specific code is the small launcher adapter in gui/platforms/ (gui/launcher.py).
See docs/GUI_MANAGER.md.
"""