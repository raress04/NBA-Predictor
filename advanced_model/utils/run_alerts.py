"""
utils/run_alerts.py
===================
W2 (Plans/W2_UNATTENDED_FAILURES_AUDIT.md): unattended failures are loud and non-fatal.

Every problem in a nightly run is written as one line in a fixed form, so a night can be
checked with one grep:

    [ALERT ERROR] <source>: <what happened> -> <what the run did instead>
    [ALERT WARN] <source>: ...

alert() prints the line (it reaches logs/scheduled/pipeline_<date>.log through the
scheduled task) and appends it to logs/scheduled/alerts_<slate>.log.

At the end of the night, `python3 -m utils.run_alerts --summary` prints the night status:
    NIGHT STATUS: OK | DEGRADED (n errors, m warnings) | FAILED
and shows a Windows pop-up when the night is not OK.

CLI (used by scripts/run_pipeline.sh):
    python3 -m utils.run_alerts --start                    # marks the start of a run
    python3 -m utils.run_alerts ERROR <source> <message>   # one alert
    python3 -m utils.run_alerts --summary [--failed]       # status line + pop-up

Env vars:
    PIPELINE_ALERTS_DIR  folder of the alerts file (default: advanced_model/logs/scheduled)
    PIPELINE_NOTIFY=0    no Windows pop-up (tests)
"""
import os
import shutil
import subprocess
import sys
from datetime import datetime

LEVELS = ('ERROR', 'WARN')
RUN_MARK = '=== run started'
_MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _slate() -> str:
    try:
        from utils.slate_date import slate_date
    except ImportError:
        sys.path.insert(0, _MODEL_DIR)
        from utils.slate_date import slate_date
    return slate_date()


def alerts_path(slate: str = None) -> str:
    folder = os.environ.get('PIPELINE_ALERTS_DIR') or os.path.join(_MODEL_DIR, 'logs', 'scheduled')
    return os.path.join(folder, f"alerts_{slate or _slate()}.log")


def _append(line: str, slate: str = None) -> None:
    path = alerts_path(slate)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except OSError as e:  # an alert must never break the run
        print(f"[ALERT WARN] run_alerts: could not write {path}: {e}")


def format_alert(level: str, source: str, message: str) -> str:
    level = level.upper()
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    return f"[ALERT {level}] {source}: {message}"


def alert(level: str, source: str, message: str, slate: str = None) -> str:
    """Print one alert line and append it (with a timestamp) to the night's alerts file."""
    line = format_alert(level, source, message)
    print(line, flush=True)
    _append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {line}", slate)
    return line


def start_run(slate: str = None) -> None:
    """Mark the start of a run. The summary counts only alerts after the last mark, so a
    re-run of the same night starts clean (nothing is deleted)."""
    _append(f"{RUN_MARK} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===", slate)


def current_alerts(slate: str = None) -> list:
    """Alert lines of the latest run of the slate."""
    path = alerts_path(slate)
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        lines = [l.rstrip('\n') for l in f]
    last = max((i for i, l in enumerate(lines) if l.startswith(RUN_MARK)), default=-1)
    return [l for l in lines[last + 1:] if '[ALERT ' in l]


def night_status(slate: str = None, failed: bool = False):
    """('OK' | 'DEGRADED' | 'FAILED', errors, warnings) for the latest run."""
    rows = current_alerts(slate)
    errors = sum('[ALERT ERROR]' in l for l in rows)
    warns = sum('[ALERT WARN]' in l for l in rows)
    if failed:
        return 'FAILED', errors, warns
    return ('DEGRADED' if errors or warns else 'OK'), errors, warns


def _counts(errors: int, warns: int) -> str:
    return (f"{errors} error{'s' if errors != 1 else ''}, "
            f"{warns} warning{'s' if warns != 1 else ''}")


def status_line(status: str, errors: int, warns: int) -> str:
    if status == 'OK':
        return "NIGHT STATUS: OK"
    return f"NIGHT STATUS: {status} ({_counts(errors, warns)})"


# ── Windows pop-up ────────────────────────────────────────────────
_TOAST_PS = r"""
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
$t = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$x = $t.GetElementsByTagName('text')
$x.Item(0).AppendChild($t.CreateTextNode('__TITLE__')) | Out-Null
$x.Item(1).AppendChild($t.CreateTextNode('__BODY__')) | Out-Null
$app = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app).Show([Windows.UI.Notifications.ToastNotification]::new($t))
"""
_PS_FALLBACK = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'


def _ps_quote(text: str) -> str:
    return text.replace("'", "''").replace('\n', ' ')


def toast_command(title: str, body: str) -> list:
    exe = shutil.which('powershell.exe') or _PS_FALLBACK
    script = _TOAST_PS.replace('__TITLE__', _ps_quote(title)).replace('__BODY__', _ps_quote(body))
    return [exe, '-NoProfile', '-NonInteractive', '-Command', script]


def notify(title: str, body: str) -> bool:
    """Show a Windows pop-up from WSL. Never raises; False when it could not be shown."""
    if os.environ.get('PIPELINE_NOTIFY', '1') == '0':
        return False
    cmd = toast_command(title, body)
    if not os.path.exists(cmd[0]):
        print(f"[ALERT WARN] notify: powershell.exe not found, no pop-up")
        return False
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            print(f"[ALERT WARN] notify: pop-up failed: {(r.stderr or '').strip()[:200]}")
            return False
        return True
    except Exception as e:
        print(f"[ALERT WARN] notify: pop-up failed: {e}")
        return False


def summarize(slate: str = None, failed: bool = False) -> str:
    """Print the night status line and the alerts behind it; pop-up when not OK."""
    slate = slate or _slate()
    status, errors, warns = night_status(slate, failed)
    line = status_line(status, errors, warns)
    print("=========================================")
    print(line)
    for l in current_alerts(slate):
        print("   " + l.split(' ', 2)[-1] if l[:4].isdigit() else "   " + l)
    if status != 'OK':
        print(f"   Alerts file: {alerts_path(slate)}")
        body = (f"{_counts(errors, warns)}. " if (errors or warns) else '') + "See logs/scheduled."
        if failed:
            body = "No predictions tonight. " + body
        notify(f"NBA Prediction Engine: {slate} {status}", body)
    print("=========================================")
    return status


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['--start']:
        start_run()
        return 0
    if argv[:1] == ['--summary']:
        summarize(failed='--failed' in argv)
        return 0
    if len(argv) >= 3 and argv[0].upper() in LEVELS:
        alert(argv[0], argv[1], ' '.join(argv[2:]))
        return 0
    print(__doc__)
    return 2


if __name__ == '__main__':
    sys.exit(main())
