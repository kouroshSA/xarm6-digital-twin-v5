# SpaceMouse setup (spacenavd)

What to install before `scripts/run_spacemouse.py` will work, and how to tell
which part is broken when it does not. Verified on Linux Mint 22.1 with a
**3Dconnexion SpaceMouse Pro** (`046d:c62b`) on 2026-09-21.

## The chain

There are three pieces, and they fail differently:

```
device --> spacenavd (daemon) --> libspnav.so (client lib) --> spnav (Python)
```

`spnav` is **not** a HID library. It is a client that talks to the spacenavd
daemon over a Unix socket, so the daemon must be running or `spnav_open()`
fails. That surprises people because the failure looks like a Python problem.

## Install

```bash
# 1. the daemon
sudo apt install spacenavd
sudo systemctl enable --now spacenavd
systemctl is-active spacenavd            # must print: active

# 2. the CLIENT LIBRARY — this step is easy to miss
sudo apt install libspnav0 libspnav-dev

# 3. the Python binding
pip install "spnav @ git+https://github.com/kazoo-osaro/spnav"
```

> **Installing `spacenavd` alone is not enough.** The `spacenavd` package ships
> the daemon, not `libspnav.so`, and the Python binding `dlopen()`s that
> library. Without step 2 the import dies with
> `OSError: libspnav.so: cannot open shared object file`. This bites everyone
> exactly once.

There is **no 3Dconnexion udev rule to hunt for.** spacenavd handles device
access itself. (The UFACTORY LeRobot repo's `rules/` directory contains rules
for Vive, RealSense and XVisio — none apply here.)

## Verify

```bash
python -c "import spnav; spnav.spnav_open(); print('ok'); spnav.spnav_close()"
python scripts/spacemouse_probe.py --seconds 30
```

The probe should print six-axis values that respond to push / pull / twist /
tilt, and a distinct `bnum` for each button pressed.

## Discovering the button map

Button indices differ per model — the Compact has 2, the Pro reports 15 — and
the protocol carries no model string, so the map cannot be autodetected. Press
every key and record what comes back:

```bash
python scripts/spacemouse_probe.py --quiet-motion --seconds 60
```

Then edit `BUTTON_MAPS` / `HOLD_MAPS` in `teleop_sm/buttons.py`. Note the
daemon logs `Device 046d:c62b reports 15 buttons before disjointed button
remapping` — "before remapping" is load-bearing, which is why the shipped Pro
map is provisional until this has been run.

`--dump events.jsonl` saves the session; `teleop_sm.device.events_from_dump()`
replays it as a test fixture, so a hardware session becomes a regression test.

## Confirming the axis mapping

`teleop_sm/config.py::AXIS_PERMUTATION` ships the vendor's matrix, which
encodes *their* bench orientation, not ours. Worse, spacenavd applies its own
transform first — it logs `device flags: swap y-z invert y-z` for the Pro — so
any matrix derived from raw HID reports is one transform too many.

**Push the cap forward and confirm the twin's TCP moves +X.** If an axis runs
backwards, flip its entry in `AXIS_SIGNS` rather than re-deriving the matrix.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `OSError: libspnav.so: cannot open shared object file` | client library missing | `sudo apt install libspnav0 libspnav-dev` |
| `spnav_open() failed` | daemon not running | `sudo systemctl enable --now spacenavd` |
| Probe runs, reports 0 events | daemon has not bound the device | `sudo systemctl restart spacenavd`, then check `journalctl -u spacenavd` for `found usb device` |
| Arm moves along the wrong axis | permutation not yet confirmed | fix `AXIS_SIGNS` / `AXIS_PERMUTATION` |
| Buttons do nothing | wrong `--device`, or provisional Pro map | run the probe, fix `BUTTON_MAPS` |
| Two devices plugged in | spacenavd binds one | the runner warns; unplug the other |

### Checking the daemon actually grabbed the device

```bash
sudo journalctl -u spacenavd --no-pager | grep -iE "found usb|using device"
sudo ls -l /proc/$(systemctl show -p MainPID --value spacenavd)/fd | grep event
```

The second command should show an `/dev/input/eventN` handle. If it does not,
the daemon is running but holding nothing, and no events will ever arrive.

A harmless line you will see in the log: `failed to open X11 display`. The
daemon tries to serve X11 clients too; that path is irrelevant to us because
this package uses the socket protocol.
