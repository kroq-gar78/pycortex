import cortex.export.headless as headless


def test_gpu_rendering_available_non_linux(monkeypatch):
    monkeypatch.setattr(headless.sys, "platform", "darwin")
    assert headless._gpu_rendering_available() is True


def test_gpu_rendering_available_linux_with_accessible_node(monkeypatch):
    monkeypatch.setattr(headless.sys, "platform", "linux")
    monkeypatch.setattr(headless.os, "listdir", lambda _: ["renderD129"])

    accessible = {"/dev/dri/renderD129"}

    monkeypatch.setattr(headless.os.path, "exists", lambda path: path in accessible)
    monkeypatch.setattr(headless.os, "access", lambda path, mode: path in accessible)

    assert headless._gpu_rendering_available() is True


def test_gpu_rendering_available_linux_without_nodes(monkeypatch):
    monkeypatch.setattr(headless.sys, "platform", "linux")

    def _raise(_):
        raise OSError("missing /dev/dri")

    monkeypatch.setattr(headless.os, "listdir", _raise)
    monkeypatch.setattr(headless.os.path, "exists", lambda _: False)
    monkeypatch.setattr(headless.os, "access", lambda *_: False)

    assert headless._gpu_rendering_available() is False


def test_chromium_launch_args_gpu_mode():
    args = headless._chromium_launch_args(True)
    assert "--enable-webgl" in args
    assert "--enable-gpu" in args
    assert "--ignore-gpu-blocklist" in args
    assert "--use-gl=swiftshader" not in args


def test_chromium_launch_args_swiftshader_mode():
    args = headless._chromium_launch_args(False)
    assert "--enable-webgl" in args
    assert "--use-gl=swiftshader" in args
    assert "--enable-gpu" not in args
