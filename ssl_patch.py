"""Windows SSL workaround for malformed system certificate stores."""

import ssl


def patch_ssl_context():
    """Make default server TLS contexts use certifi instead of Windows store."""
    if getattr(ssl, "_anollm_certifi_patched", False):
        return

    try:
        import certifi
    except Exception:
        return

    create_default_context = ssl.create_default_context

    def create_default_context_with_certifi(
        purpose=ssl.Purpose.SERVER_AUTH,
        *,
        cafile=None,
        capath=None,
        cadata=None,
    ):
        if (
            purpose == ssl.Purpose.SERVER_AUTH
            and cafile is None
            and capath is None
            and cadata is None
        ):
            cafile = certifi.where()
        return create_default_context(
            purpose=purpose,
            cafile=cafile,
            capath=capath,
            cadata=cadata,
        )

    ssl.create_default_context = create_default_context_with_certifi
    ssl._anollm_certifi_patched = True
