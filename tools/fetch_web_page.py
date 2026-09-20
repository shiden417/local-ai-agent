from __future__ import annotations

from html.parser import HTMLParser
import ipaddress
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from agent.observation import truncate_text


MAX_URL_CHARS = 2_000
MAX_CONTENT_CHARS = 6_000
DEFAULT_TIMEOUT_SECONDS = 10





class _SafeRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_public_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


urlopen = build_opener(_SafeRedirectHandler).open


def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http:// and https:// URLs are supported")
    if parsed.username or parsed.password:
        raise ValueError("URLs with embedded credentials are not supported")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must include a hostname")

    if hostname.lower() in {"localhost", "localhost.localdomain"}:
        raise ValueError("Localhost URLs are not allowed for web fetching")

    try:
        literal = ipaddress.ip_address(hostname)
        addresses = [literal]
    except ValueError:
        try:
            infos = socket.getaddrinfo(hostname, parsed.port or 443)
        except OSError as exc:
            raise ValueError(f"Could not resolve web host: {exc}") from exc
        addresses = [ipaddress.ip_address(info[4][0]) for info in infos]

    if not addresses:
        raise ValueError("Web host resolved to no addresses")

    if any(
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
        for address in addresses
    ):
        raise ValueError(
            "Web fetching to private, local, reserved, or multicast addresses is blocked"
        )

class _PageTextParser(HTMLParser):
    _SKIP_TAGS = {"script", "style", "noscript", "svg", "template"}

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.title = ""
        self._in_title = False
        self._title_parts: list[str] = []
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = True
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = " ".join(data.split())
        if not text:
            return
        if self._in_title:
            self._title_parts.append(text)
        self._parts.append(text)

    def finish(self) -> None:
        self.title = " ".join(self._title_parts).strip()

    @property
    def text(self) -> str:
        return "\n".join(self._parts)


def fetch_web_page(
    url: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    url = str(url).strip()
    if not url:
        return {"ok": False, "error": "url must not be empty"}
    if len(url) > MAX_URL_CHARS:
        return {"ok": False, "error": f"url must be {MAX_URL_CHARS} characters or fewer"}
    if not url.lower().startswith(("http://", "https://")):
        return {"ok": False, "error": "Only http:// and https:// URLs are supported"}

    try:
        timeout_seconds = int(timeout_seconds)
    except (TypeError, ValueError):
        return {"ok": False, "error": "timeout_seconds must be an integer"}
    if not 1 <= timeout_seconds <= 20:
        return {"ok": False, "error": "timeout_seconds must be between 1 and 20"}

    try:
        _validate_public_url(url)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "url": url, "retryable": False}

    request = Request(
        url,
        headers={
            "User-Agent": "local-ai-agent/1.0",
            "Accept": "text/html, text/plain;q=0.9, */*;q=0.5",
        },
    )

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status = getattr(response, "status", None)
            final_url = response.geturl()
            content_type = response.headers.get_content_type()
            raw = response.read(512_000)
    except HTTPError as exc:
        return {
            "ok": False,
            "error": f"HTTP {exc.code}: {exc.reason}",
            "status_code": exc.code,
            "url": url,
            "retryable": exc.code in {408, 429, 500, 502, 503, 504},
        }
    except (URLError, TimeoutError) as exc:
        return {
            "ok": False,
            "error": f"Webページの取得に失敗しました: {exc}",
            "url": url,
            "retryable": True,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Webページの取得に失敗しました: {type(exc).__name__}: {exc}",
            "url": url,
            "retryable": False,
        }

    encoding = "utf-8"
    try:
        charset = response.headers.get_content_charset()
        if charset:
            encoding = charset
    except Exception:
        pass

    text = raw.decode(encoding, errors="replace")

    if "html" in content_type.lower():
        parser = _PageTextParser()
        try:
            parser.feed(text)
            parser.close()
            parser.finish()
            title = parser.title
            body = parser.text
        except Exception:
            title = ""
            body = text
    else:
        title = ""
        body = text

    body, truncated = truncate_text(body, MAX_CONTENT_CHARS)
    return {
        "ok": True,
        "url": url,
        "final_url": final_url,
        "status_code": status,
        "content_type": content_type,
        "title": title[:500],
        "content": body,
        "truncated": truncated,
    }
