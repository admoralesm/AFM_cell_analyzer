"""OneDrive store against a fake Microsoft Graph."""
import json, sys
sys.path.insert(0, "/root/AFM_cell_analyzer")
import onedrive_store as od

FAILURES = []
def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond else f" {detail}"))
    if not cond: FAILURES.append(name)

class FakeGraph:
    """Records calls; stores files by path."""
    def __init__(self):
        self.files, self.folders, self.calls = {}, set(), []
    def __call__(self, method, url, token=None, data=None, headers=None,
                 timeout=60, raw=False):
        self.calls.append((method, url))
        if "oauth2/v2.0/token" in url:
            return {"access_token": "tok", "expires_in": 3600,
                    "refresh_token": "newer-refresh"}
        if url.endswith("/me/drive"):
            return {"id": "drive1", "driveType": "business",
                    "owner": {"user": {"displayName": "David"}},
                    "quota": {"remaining": 900e9}}
        # by-path addressing
        if ":/content" in url:
            path = url.split("root:/")[1].split(":/content")[0]
            from urllib.parse import unquote
            path = unquote(path)
            if method == "PUT":
                self.files[path] = data
                return {"id": path, "name": path.split("/")[-1]}
            if method == "GET":
                if path not in self.files:
                    raise od.OneDriveError("OneDrive said 404: not found")
                body = self.files[path]
                return body if raw else json.loads(body)
        if ":/createLink" in url:
            return {"link": {"webUrl": "https://onedrive/x"}}
        if url.endswith("/children") and method == "POST":
            name = json.loads(data)["name"]
            base = ""
            if "root:/" in url:
                from urllib.parse import unquote
                base = unquote(url.split("root:/")[1].split("/children")[0].rstrip(":"))
            self.folders.add(f"{base}/{name}".strip("/"))
            return {"id": name}
        if method == "GET" and "root:/" in url:
            from urllib.parse import unquote
            path = unquote(url.split("root:/")[1])
            if path in self.folders:
                return {"id": path}
            raise od.OneDriveError("OneDrive said 404: not found")
        if method == "GET" and url.endswith("/root"):
            return {"id": "root"}
        raise AssertionError(f"unexpected call {method} {url}")

fake = FakeGraph()
od._request = fake

store = od.OneDriveStore(client_id="cid", refresh_token="r0", tenant="common")

print("credentials and drive")
status = store.check()
check("check reports the drive", status["ok"], str(status))
check("it names the owner", "David" in status["detail"], status["detail"])
check("it reports free space", "GB free" in status["detail"], status["detail"])
check("auth method is named", "refresh token" in store.auth_method())

print("saving a cell")
record = {"cell_id": "Cell 04/trial 2", "Em_MPa": 0.6, "Ec_kPa": 1.2,
          "En_kPa": 0.0, "r_squared": 0.9997, "chi_squared_reduced": 1.1}
saved = store.save_cell(record, curve_csv="a,b\n1,2\n",
                        thumbnail_png=b"\x89PNG", video_bytes=None)
check("the slash in the name was replaced",
      "Cell 04_trial 2" in saved["onedrive_path"], saved["onedrive_path"])
paths = sorted(fake.files)
check("curve.csv written", any(p.endswith("curve.csv") for p in paths), str(paths))
check("record.json written", any(p.endswith("record.json") for p in paths))
check("thumbnail written", any(p.endswith("thumbnail.png") for p in paths))
check("index written", any(p.endswith("index.csv") for p in paths))

print("reading it back")
back = store.load_cell("Cell 04/trial 2")
check("the record came back", back is not None)
if back:
    check("moduli survived the round trip",
          abs(back["Em_MPa"] - 0.6) < 1e-9, str(back.get("Em_MPa")))
    check("the curve came back as text", back["curve_csv"].startswith("a,b"),
          repr(back.get("curve_csv")))
check("a missing cell returns None", store.load_cell("nope") is None)

print("the index")
frame = store.load_index()
check("one row in the index", len(frame) == 1, str(len(frame)))
check("chi squared is in the index",
      "chi_squared_reduced" in frame.columns, str(list(frame.columns)))
store.save_cell(dict(record, Em_MPa=0.9), curve_csv="a,b\n3,4\n")
frame = store.load_index()
check("re-saving replaces the row, not appends", len(frame) == 1, str(len(frame)))
check("and keeps the newer value",
      abs(float(frame.iloc[0]["Em_MPa"]) - 0.9) < 1e-9, str(frame.iloc[0]["Em_MPa"]))

print("large files")
big = b"x" * (od.SIMPLE_UPLOAD_LIMIT + 10)
sessions = []
def fake2(method, url, token=None, data=None, headers=None, timeout=60, raw=False):
    if ":/createUploadSession" in url:
        sessions.append(url); return {"uploadUrl": "https://upload/session"}
    if url == "https://upload/session":
        sessions.append(headers.get("Content-Range")); return {"id": "big"}
    return fake(method, url, token, data, headers, timeout, raw)
od._request = fake2
store._upload_large("cells/x/video.mp4", big)
check("a large file used an upload session", len(sessions) >= 2, str(sessions[:1]))
check("chunks carry a byte range",
      any(str(r).startswith("bytes ") for r in sessions[1:]), str(sessions[1:]))
od._request = fake

print("refresh token rotation")
check("the newest refresh token is kept",
      store.refresh_token == "newer-refresh", store.refresh_token)

print("no credentials")
bare = od.OneDriveStore(client_id="cid")
try:
    bare.token(); check("bare store refuses to authenticate", False)
except od.OneDriveError as exc:
    check("bare store explains what is missing", "secrets" in str(exc), str(exc))

print()
if FAILURES:
    print(f"{len(FAILURES)} failing: {FAILURES}"); sys.exit(1)
print("all passing")
