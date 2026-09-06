"""
OneDrive / SharePoint storage for AFM cell records, over Microsoft Graph.

Same job as ``box_store``, same public interface, different service: one
folder per cell holding ``record.json``, ``curve.csv``, the morphology frame
and optionally the video, plus an ``index.csv`` at the root so the gallery
can be listed without walking every folder.

Why this exists alongside Box
-----------------------------
Box on an enterprise tenant needs an administrator to authorise the app
before it will do anything, and that queue can be long. A personal or
university OneDrive is storage the user already has, and the device-code
flow below lets them authorise it themselves in a browser without anyone's
permission.

Authentication
--------------
Two ways in, in order of how much help you need from anyone else:

1. **Device code** (``client_id`` + ``tenant``, then a one-time sign-in).
   You run :func:`begin_device_login`, open the URL it prints, type the code,
   and get a refresh token back to paste into secrets. No admin involved for
   a personal OneDrive; for a university tenant an admin may still have to
   allow the app, but many allow user consent by default.
2. **Client credentials** (``client_id`` + ``client_secret`` + ``tenant``).
   App-only, unattended, and needs admin consent plus a ``drive_id``, since
   an app with no user has no "my" drive to write to.

Put whichever you have in ``.streamlit/secrets.toml``::

    [onedrive]
    account = "personal"              # or "work", or "either"
    client_id = "..."
    refresh_token = "..."             # from the device-code flow
    root_folder = "AFM cells"         # created if missing
    # tenant = "..."                  # work accounts only, never personal

    # or, app-only:
    # client_secret = "..."
    # drive_id = "b!..."

Refresh tokens
--------------
Microsoft returns a new refresh token on every exchange and the old one
keeps working for a while, so unlike Box this does not have to be written
back after each call. It does expire if unused for about 90 days, which for
a tool used weekly is not a problem, and the error says plainly what to do.
"""

from __future__ import annotations

import io
import json
import mimetypes
import time
import urllib.error
import urllib.parse
import urllib.request

GRAPH = "https://graph.microsoft.com/v1.0"
LOGIN = "https://login.microsoftonline.com"

# Personal and work accounts do not take the same scopes or the same tenant,
# and getting either wrong fails in a way whose error message names neither.
#
# tenant: a personal account does not exist in any directory, so signing it in
# against a directory (tenant) id is refused. "consumers" is the personal
# endpoint, "organizations" the work one, "common" accepts both.
#
# scopes: Files.ReadWrite.All is a work-account scope. A personal account has
# no shared drives or SharePoint libraries for the .All to reach, and asking
# for it is rejected or silently unconsented. Files.ReadWrite is the personal
# equivalent and covers the whole of that OneDrive.
ACCOUNTS = {
    "personal": {
        "tenant": "consumers",
        "scopes": "offline_access Files.ReadWrite User.Read",
        "label": "Personal Microsoft account (outlook.com, hotmail, live)",
    },
    "work": {
        "tenant": "organizations",
        "scopes": "offline_access Files.ReadWrite.All User.Read",
        "label": "Work or school account (university, company)",
    },
    "either": {
        "tenant": "common",
        "scopes": "offline_access Files.ReadWrite User.Read",
        "label": "Either, let Microsoft decide",
    },
}
DEFAULT_ACCOUNT = "personal"

# Kept for callers that still import it.
SCOPES = ACCOUNTS["work"]["scopes"]


def settings_for(account="personal", tenant=None):
    """The tenant and scopes to use, with an explicit tenant winning."""
    profile = ACCOUNTS.get(account or DEFAULT_ACCOUNT, ACCOUNTS[DEFAULT_ACCOUNT])
    return {
        "tenant": tenant or profile["tenant"],
        "scopes": profile["scopes"],
    }


def _looks_like_a_directory_id(tenant):
    """True for a GUID, which is a work tenant and never a personal one."""
    text = str(tenant or "")
    return len(text) == 36 and text.count("-") == 4


def explain(error_text, account="personal", tenant=None):
    """
    Turn a Microsoft sign-in error into the thing to actually change.

    The AADSTS codes are precise but they describe Microsoft's model, not
    yours. These are the four that a personal account runs into, and what
    each one really means.
    """
    text = str(error_text)
    if "AADSTS50020" in text or "does not exist in tenant" in text:
        return (
            "That is a personal Microsoft account, but the app is set to sign "
            "in against a work directory. Set the account type to Personal, "
            "which uses the `consumers` endpoint, and remove any tenant id "
            "from your secrets: a personal account does not live in a "
            "directory and cannot be found in one."
        )
    if "AADSTS700016" in text or "AADSTS50194" in text or "was not found in the directory" in text:
        return (
            "The app registration does not accept this kind of account. In "
            "Azure, open the app, go to Authentication, and under Supported "
            "account types choose the option that includes **personal "
            "Microsoft accounts**. Registering as single-tenant shuts "
            "personal accounts out entirely."
        )
    if "AADSTS7000218" in text or "client_assertion" in text:
        return (
            "The app is not marked as a public client. In Azure, open the "
            "app, go to Authentication, scroll to Advanced settings, and set "
            "**Allow public client flows** to Yes."
        )
    if "AADSTS65001" in text or "consent" in text.lower():
        return (
            "The permissions have not been consented to. Sign in again and "
            "accept the prompt, or ask an administrator to grant consent for "
            "a work account."
        )
    if "invalid_scope" in text or "AADSTS70011" in text:
        return (
            "Those permissions are not valid for this account type. A "
            "personal account uses Files.ReadWrite, not Files.ReadWrite.All. "
            "Set the account type to Personal and try again."
        )
    if _looks_like_a_directory_id(tenant) and account == "personal":
        return (
            "A tenant id is set, but personal accounts do not belong to a "
            "tenant. Clear `tenant` from the [onedrive] secrets."
        )
    return ""

# Files above this go up in chunks; Graph rejects a simple PUT beyond ~4 MB
# and a compression video is comfortably larger.
SIMPLE_UPLOAD_LIMIT = 4 * 1024 * 1024
CHUNK = 5 * 1024 * 1024          # must be a multiple of 320 KiB


class OneDriveError(RuntimeError):
    """Raised with a message meant to be shown to the user as-is."""


def _request(method, url, token=None, data=None, headers=None, timeout=60,
             raw=False):
    """
    One Graph call, with errors turned into something readable.

    ``raw`` returns the bytes untouched. Downloads must use it: a stored
    record.json is valid JSON, so parsing by content would hand back a dict
    where the caller asked for the file, and the file would look empty.
    """
    headers = dict(headers or {})
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            if raw:
                return body
            if not body:
                return {}
            try:
                return json.loads(body.decode("utf-8"))
            except ValueError:
                return {"_raw": body}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        try:
            parsed = json.loads(detail)
            message = parsed.get("error", {})
            if isinstance(message, dict):
                detail = message.get("message", detail)
            else:
                detail = parsed.get("error_description", detail)
        except ValueError:
            pass
        raise OneDriveError(f"OneDrive said {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise OneDriveError(f"Could not reach OneDrive: {exc.reason}") from exc


# --------------------------------------------------------- device sign-in ---


def begin_device_login(client_id, tenant=None, account=DEFAULT_ACCOUNT):
    """
    Start the device-code flow.

    Returns the dict Microsoft sends back, which carries ``user_code``,
    ``verification_uri`` and ``device_code``, plus the scopes used, so the
    refresh later asks for exactly the same ones. Asking for different scopes
    on refresh than on sign-in is its own quiet failure.
    """
    config = settings_for(account, tenant)
    data = urllib.parse.urlencode(
        {"client_id": client_id, "scope": config["scopes"]}
    ).encode()
    try:
        payload = _request(
            "POST", f"{LOGIN}/{config['tenant']}/oauth2/v2.0/devicecode",
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    except OneDriveError as exc:
        hint = explain(exc, account, tenant)
        raise OneDriveError(f"{exc}\n\n{hint}" if hint else str(exc)) from exc
    payload["_scopes"] = config["scopes"]
    payload["_tenant"] = config["tenant"]
    return payload


def poll_device_login(client_id, device_code, tenant=None,
                      account=DEFAULT_ACCOUNT):
    """
    Ask once whether the person has finished signing in.

    Returns the token payload when they have, None while they have not, and
    raises when the attempt has failed for good.
    """
    data = urllib.parse.urlencode(
        {
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            "client_id": client_id,
            "device_code": device_code,
        }
    ).encode()
    config = settings_for(account, tenant)
    try:
        return _request(
            "POST", f"{LOGIN}/{config['tenant']}/oauth2/v2.0/token", data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
    except OneDriveError as exc:
        text = str(exc)
        # Still waiting is the normal state, not a failure.
        if "authorization_pending" in text or "slow_down" in text:
            return None
        if "expired_token" in text:
            raise OneDriveError(
                "The code expired before sign-in finished. Start again."
            ) from exc
        hint = explain(text, account, tenant)
        raise OneDriveError(f"{text}\n\n{hint}" if hint else text) from exc


class OneDriveStore:
    """One folder per cell in a OneDrive or SharePoint document library."""

    def __init__(self, client_id=None, client_secret=None, tenant=None,
                 refresh_token=None, drive_id=None, root_folder="AFM cells",
                 account=DEFAULT_ACCOUNT):
        self.client_id = client_id
        self.client_secret = client_secret
        self.account = account or DEFAULT_ACCOUNT
        config = settings_for(self.account, tenant)
        self.tenant = config["tenant"]
        self.scopes = config["scopes"]
        self.refresh_token = refresh_token
        self.drive_id = drive_id
        self.root_folder = root_folder or "AFM cells"
        self._token = None
        self._expires_at = 0.0
        self._root_id = None

    # ------------------------------------------------------------ auth

    def auth_method(self):
        who = ACCOUNTS.get(self.account, {}).get("label", self.account)
        if self.refresh_token:
            return f"signed in as you · {who}"
        if self.client_secret:
            return f"app only · {who}"
        return "none configured"

    def token(self):
        """A valid access token, refreshed when it is close to expiring."""
        if self._token and time.time() < self._expires_at - 60:
            return self._token

        if self.refresh_token:
            fields = {
                "client_id": self.client_id,
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token,
                # The same scopes the sign-in asked for. A refresh that asks
                # for more than was granted is refused.
                "scope": self.scopes,
            }
            if self.client_secret:
                fields["client_secret"] = self.client_secret
        elif self.client_secret:
            fields = {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "client_credentials",
                "scope": "https://graph.microsoft.com/.default",
            }
        else:
            raise OneDriveError(
                "No OneDrive credentials. Add a [onedrive] section to "
                "`.streamlit/secrets.toml` with client_id and either a "
                "refresh_token from the sign-in below or a client_secret."
            )

        try:
            payload = _request(
                "POST", f"{LOGIN}/{self.tenant}/oauth2/v2.0/token",
                data=urllib.parse.urlencode(fields).encode(),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        except OneDriveError as exc:
            hint = explain(exc, self.account, self.tenant)
            raise OneDriveError(f"{exc}\n\n{hint}" if hint else str(exc)) from exc
        self._token = payload.get("access_token")
        self._expires_at = time.time() + float(payload.get("expires_in", 3600))
        # Microsoft hands back a fresh refresh token each time. Keeping the
        # newest one in memory means a long session does not run into the
        # 90-day idle expiry on the one that came from secrets.
        if payload.get("refresh_token"):
            self.refresh_token = payload["refresh_token"]
        if not self._token:
            raise OneDriveError("OneDrive returned no access token.")
        return self._token

    def _drive(self):
        """The drive to write into, as a Graph path prefix."""
        if self.drive_id:
            return f"/drives/{self.drive_id}"
        if self.client_secret and not self.refresh_token:
            raise OneDriveError(
                "App-only access has no personal drive. Add drive_id to the "
                "[onedrive] secrets, or sign in with the device code instead."
            )
        return "/me/drive"

    def check(self):
        """Confirm the credentials work and say whose drive this is."""
        try:
            info = _request("GET", f"{GRAPH}{self._drive()}", token=self.token())
        except OneDriveError as exc:
            return {"ok": False, "detail": str(exc)}
        owner = (info.get("owner") or {}).get("user", {}).get("displayName", "")
        quota = info.get("quota") or {}
        free = quota.get("remaining")
        detail = f"{info.get('driveType', 'drive')}"
        if owner:
            detail += f" belonging to {owner}"
        if free:
            detail += f", {free / 1e9:.1f} GB free"
        return {"ok": True, "detail": detail, "drive_id": info.get("id", "")}

    # ---------------------------------------------------------- folders

    def _by_path(self, path):
        """Graph addressing for a path relative to the drive root."""
        if not path:
            return f"{GRAPH}{self._drive()}/root"
        quoted = urllib.parse.quote(path.strip("/"))
        return f"{GRAPH}{self._drive()}/root:/{quoted}"

    def ensure_folder(self, name, parent_path=""):
        """Create a folder if it is not already there; return its path."""
        path = f"{parent_path.strip('/')}/{name}".strip("/")
        try:
            _request("GET", self._by_path(path), token=self.token())
            return path
        except OneDriveError as exc:
            if "404" not in str(exc) and "not found" not in str(exc).lower():
                raise
        parent = self._by_path(parent_path)
        _request(
            "POST", f"{parent}/children" if parent_path else f"{parent}/children",
            token=self.token(),
            data=json.dumps(
                {
                    "name": name,
                    "folder": {},
                    "@microsoft.graph.conflictBehavior": "replace",
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        return path

    def cells_folder(self):
        return self.ensure_folder("cells", self.ensure_folder(self.root_folder))

    # ---------------------------------------------------------- uploads

    def upload(self, folder_path, name, content):
        """Write one file, choosing a simple or chunked upload by size."""
        if isinstance(content, str):
            content = content.encode("utf-8")
        content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        path = f"{folder_path.strip('/')}/{name}".strip("/")

        if len(content) <= SIMPLE_UPLOAD_LIMIT:
            return _request(
                "PUT", f"{self._by_path(path)}:/content", token=self.token(),
                data=content, headers={"Content-Type": content_type},
            )
        return self._upload_large(path, content)

    def _upload_large(self, path, content):
        """
        Chunked upload, which anything over about 4 MB has to use.

        Graph gives back a session URL that takes the chunks without an
        Authorization header, and each chunk names the byte range it covers.
        """
        session = _request(
            "POST", f"{self._by_path(path)}:/createUploadSession",
            token=self.token(),
            data=json.dumps(
                {"item": {"@microsoft.graph.conflictBehavior": "replace"}}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        url = session.get("uploadUrl")
        if not url:
            raise OneDriveError("OneDrive would not start an upload session.")

        total = len(content)
        sent = 0
        result = {}
        while sent < total:
            chunk = content[sent:sent + CHUNK]
            end = sent + len(chunk) - 1
            result = _request(
                "PUT", url, data=chunk,
                headers={
                    "Content-Length": str(len(chunk)),
                    "Content-Range": f"bytes {sent}-{end}/{total}",
                },
                timeout=300,
            )
            sent += len(chunk)
        return result

    def _download(self, path):
        return _request(
            "GET", f"{self._by_path(path)}:/content", token=self.token(), raw=True
        )

    def shared_link(self, path, scope="organization"):
        """A link to a stored file. Falls back to a private webUrl."""
        try:
            payload = _request(
                "POST", f"{self._by_path(path)}:/createLink", token=self.token(),
                data=json.dumps({"type": "view", "scope": scope}).encode(),
                headers={"Content-Type": "application/json"},
            )
            return (payload.get("link") or {}).get("webUrl", ""), scope
        except OneDriveError:
            item = _request("GET", self._by_path(path), token=self.token())
            return item.get("webUrl", ""), "private"

    # ------------------------------------------------------------ cells

    def save_cell(self, record, curve_csv=None, thumbnail_png=None,
                  video_bytes=None, video_name="video.mp4"):
        """
        Write one cell's folder, then add it to the index.

        The record goes up first: if anything later fails, the cell still
        exists and is complete enough to be recovered by rebuilding the index.
        """
        cell_id = str(record["cell_id"]).strip()
        if not cell_id:
            raise OneDriveError("The cell needs a name before it can be saved.")
        # Characters OneDrive refuses in a name, replaced rather than rejected.
        safe = "".join("_" if ch in '\\/:*?"<>|' else ch for ch in cell_id)

        folder = self.ensure_folder(safe, self.cells_folder())
        record = dict(record)
        record["onedrive_path"] = folder

        if curve_csv is not None:
            self.upload(folder, "curve.csv", curve_csv)
        if thumbnail_png is not None:
            self.upload(folder, "thumbnail.png", thumbnail_png)
            record["thumbnail_path"] = f"{folder}/thumbnail.png"
        if video_bytes is not None:
            self.upload(folder, video_name, video_bytes)
            record["video_path"] = f"{folder}/{video_name}"
            try:
                url, _ = self.shared_link(f"{folder}/{video_name}")
                record["video_url"] = url
            except OneDriveError:
                record["video_url"] = ""

        self.upload(
            folder, "record.json", json.dumps(record, indent=2, default=str)
        )
        self.append_to_index(record)
        return record

    def load_cell(self, cell_id):
        """Return a cell's record and its curve, or None if it is not there."""
        safe = "".join("_" if ch in '\\/:*?"<>|' else ch for ch in str(cell_id))
        folder = f"{self.cells_folder()}/{safe}"
        try:
            record = json.loads(
                self._download(f"{folder}/record.json").decode("utf-8")
            )
        except OneDriveError:
            return None
        try:
            record["curve_csv"] = self._download(f"{folder}/curve.csv").decode("utf-8")
        except OneDriveError:
            record["curve_csv"] = None
        return record

    def thumbnail_bytes(self, path):
        try:
            return self._download(path)
        except Exception:
            return None

    # ------------------------------------------------------------- index

    INDEX_NAME = "index.csv"

    def index_columns(self):
        return [
            "cell_id", "date", "cell_type", "Em_MPa", "Ec_kPa", "En_kPa",
            "r_squared", "chi_squared_reduced", "onedrive_path", "video_url",
            "saved_at",
        ]

    def _index_path(self):
        return f"{self.ensure_folder(self.root_folder)}/{self.INDEX_NAME}"

    def load_index(self):
        """The index as a DataFrame, empty if there is not one yet."""
        import pandas as pd

        try:
            text = self._download(self._index_path()).decode("utf-8")
            return pd.read_csv(io.StringIO(text))
        except Exception:
            return pd.DataFrame(columns=self.index_columns())

    def append_to_index(self, record):
        """Add or replace one row, keeping the newest save of each cell."""
        import pandas as pd

        frame = self.load_index()
        row = {key: record.get(key, "") for key in self.index_columns()}
        if not frame.empty and "cell_id" in frame.columns:
            frame = frame[frame["cell_id"].astype(str) != str(row["cell_id"])]
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
        self.upload(
            self.ensure_folder(self.root_folder), self.INDEX_NAME,
            frame.to_csv(index=False),
        )
        return frame

    def rebuild_index(self):
        """Walk the cells folder and rebuild the index from the records."""
        import pandas as pd

        rows = []
        listing = _request(
            "GET", f"{self._by_path(self.cells_folder())}:/children",
            token=self.token(),
        )
        for item in listing.get("value", []):
            if "folder" not in item:
                continue
            try:
                record = json.loads(
                    self._download(
                        f"{self.cells_folder()}/{item['name']}/record.json"
                    ).decode("utf-8")
                )
            except Exception:
                continue
            rows.append({key: record.get(key, "") for key in self.index_columns()})
        frame = pd.DataFrame(rows, columns=self.index_columns())
        self.upload(
            self.ensure_folder(self.root_folder), self.INDEX_NAME,
            frame.to_csv(index=False),
        )
        return frame


def store_from_secrets(st, root_folder=None):
    """Build a store from ``st.secrets['onedrive']``, or None when unset."""
    try:
        config = dict(st.secrets.get("onedrive", {}))
    except Exception:
        config = {}
    if not config:
        return None
    return OneDriveStore(
        client_id=config.get("client_id"),
        client_secret=config.get("client_secret"),
        account=config.get("account", DEFAULT_ACCOUNT),
        tenant=config.get("tenant"),
        refresh_token=config.get("refresh_token"),
        drive_id=config.get("drive_id"),
        root_folder=root_folder or config.get("root_folder", "AFM cells"),
    )
