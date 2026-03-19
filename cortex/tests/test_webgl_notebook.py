from cortex.webgl import serve, view


def test_make_iframe_html_formats_dimensions_and_escapes_url():
    html = view._make_iframe_html(
        'http://localhost:1234/mixer.html?x="1"',
        width=800,
        height="60vh",
        title='Viewer "A"',
    )

    assert 'style="width:800px;height:60vh;border:0;display:block;"' in html
    assert 'src="http://localhost:1234/mixer.html?x=&quot;1&quot;"' in html
    assert 'title="Viewer &quot;A&quot;"' in html


def test_show_in_notebook_returns_server_and_url(monkeypatch):
    server = serve.WebApp([], 43123)

    def fake_show(*args, **kwargs):
        assert kwargs["open_browser"] is False
        assert kwargs["display_url"] is False
        assert kwargs["autoclose"] is False
        return server

    monkeypatch.setattr(view, "show", fake_show)

    returned_server, url = view.show_in_notebook(
        data={},
        display=False,
    )

    assert returned_server is server
    assert url.endswith(":43123/mixer.html")
