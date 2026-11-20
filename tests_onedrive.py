"""OneDrive store against a fake Microsoft Graph."""
import io
import json, sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
        if method == "GET" and url.endswith(":/children"):
            from urllib.parse import unquote
            base = unquote(url.split("root:/")[1][:-len(":/children")])
            names = {f.split("/")[-1] for f in self.folders
                     if f.rsplit("/", 1)[0] == base and "/" in f}
            files = {f.split("/")[-1]: data for f, data in self.files.items()
                     if f.rsplit("/", 1)[0] == base}
            return {"value": [{"name": n, "folder": {}} for n in sorted(names)]
                    + [{"name": n, "file": {}, "size": len(d),
                        "webUrl": f"https://onedrive/{base}/{n}"}
                       for n, d in sorted(files.items())]}
        if method == "GET" and "root:/" in url:
            from urllib.parse import unquote
            path = unquote(url.split("root:/")[1])
            if path in self.folders:
                return {"id": path, "webUrl": f"https://onedrive/{path}"}
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
check("auth method is named", "signed in" in store.auth_method(),
      store.auth_method())

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

print("saving a stacked analysis")
fake.calls.clear()
record = {"stack_id": "2026-11-09 1622 Cell 3: compressions",
          "name": "Cell 3: compressions", "saved_at": "2026-11-09T16:22:00",
          "notes": "two files", "state": {"sc_n": 2},
          "files": [{"slot": 0, "name": "a.csv", "path": "files/1_a.csv"},
                    {"slot": 1, "name": "b.csv", "path": "files/2_b.csv"}]}
saved = store.save_stack(record, {
    "files/1_a.csv": "t,F\n0,1\n", "files/2_b.csv": b"t,F\n0,2\n",
    "plot.html": "<html></html>", "aligned.csv": "x\n1\n"})
folder = saved["onedrive_path"]
check("its folder is under stacks, the colon replaced",
      folder.endswith("stacks/2026-11-09 1622 Cell 3_ compressions"), folder)
check("the files went into files/",
      f"{folder}/files/1_a.csv" in fake.files
      and f"{folder}/files/2_b.csv" in fake.files, str(sorted(fake.files)))
check("the plot and table beside them",
      f"{folder}/plot.html" in fake.files and f"{folder}/aligned.csv" in fake.files)
puts = [u for m, u in fake.calls if m == "PUT"]
check("analysis.json after the files, the index last",
      "analysis.json" in puts[-2] and "stacks_index.csv" in puts[-1],
      str(puts[-3:]))
check("the folder's address is kept (no sharing link made)",
      saved["folder_url"].startswith("https://onedrive/")
      and not any("createLink" in u for _, u in fake.calls),
      saved.get("folder_url"))

print("reading it back")
back = store.load_stack("2026-11-09 1622 Cell 3: compressions")
check("the record comes back", back is not None and back["state"] == {"sc_n": 2},
      str(back))
check("a file comes back by its path",
      store.stack_file(back["stack_id"], "files/1_a.csv") == b"t,F\n0,1\n",
      repr(store.stack_file(back["stack_id"], "files/1_a.csv")))
check("a missing analysis is None", store.load_stack("nope") is None)
index = store.load_stack_index()
check("one row in the list, with its files named",
      len(index) == 1 and index.iloc[0]["files"] == "a.csv; b.csv"
      and str(index.iloc[0]["n_files"]) == "2", index.to_dict("records"))
store.save_stack(dict(record, notes="again"), {"plot.html": "<html/>"})
index = store.load_stack_index()
check("saving it again replaces its row",
      len(index) == 1 and index.iloc[0]["notes"] == "again",
      index.to_dict("records"))
store.save_stack(dict(record, stack_id="second", saved_at="2026-11-10T09:00"),
                 {"plot.html": "<html/>"})
index = store.load_stack_index()
check("the newest first", list(index["stack_id"]) ==
      ["second", "2026-11-09 1622 Cell 3_ compressions"], list(index["stack_id"]))
fake.files.pop(store.stack_index_path())
check("the list is rebuilt from the folders",
      len(store.rebuild_stack_index()) == 2)
try:
    store.save_stack({"stack_id": " "}, {}); check("a nameless save is refused", False)
except od.OneDriveError:
    check("a nameless save is refused", True)

print("saving a beat analysis")
fake.calls.clear()
beat = {"beat_id": "2026-11-17 2230 Cell 3 beats", "name": "Cell 3 beats",
        "saved_at": "2026-11-17T22:30:00", "notes": "control",
        "source": "AFM output analysis", "file": "X202509220920.ibw",
        "results": {"Full beats": 15, "Positive peaks": 16,
                    "Energy, all peaks (total) (aJ)": 8650.5,
                    "Positive peak height, mean (µN)": 0.052},
        "settings": {"hi": 0.005}}
saved = store.save_beats(beat, {"data.csv": "t,F\n0,1\n",
                                "results.csv": "Parameter,Value,Unit\n"})
folder = saved["onedrive_path"]
check("its folder is under beats",
      folder.endswith("beats/2026-11-17 2230 Cell 3 beats"), folder)
check("its files beside analysis.json",
      f"{folder}/data.csv" in fake.files
      and f"{folder}/analysis.json" in fake.files, str(sorted(fake.files)))
puts = [u for m, u in fake.calls if m == "PUT"]
check("analysis.json after the files, the index last",
      "analysis.json" in puts[-2] and "beats_index.csv" in puts[-1],
      str(puts[-3:]))
back = store.load_beats(beat["beat_id"])
check("the record and a file come back",
      back is not None and back["settings"] == {"hi": 0.005}
      and store.beat_file(back["beat_id"], "data.csv") == b"t,F\n0,1\n")
index = store.load_beat_index()
check("one line in the index, with every result as a column",
      len(index) == 1 and index.iloc[0]["Full beats"] == 15
      and abs(index.iloc[0]["Energy, all peaks (total) (aJ)"] - 8650.5) < 1e-9
      and list(index.columns)[:3] == ["beat_id", "name", "saved_at"]
      and list(index.columns)[-1] == "folder_url", list(index.columns))
store.save_beats(dict(beat, beat_id="second", saved_at="2026-11-18T09:00",
                      results=dict(beat["results"], **{"Beats per minute (/min)": 15.0})),
                 {"data.csv": "t,F\n"})
index = store.load_beat_index()
check("a second line, newest first, a new result a new column",
      list(index["beat_id"]) == ["second", "2026-11-17 2230 Cell 3 beats"]
      and "Beats per minute (/min)" in index.columns, index.to_dict("records"))
store.save_beats(dict(beat, notes="again"), {"data.csv": "t,F\n"})
check("saving it again replaces its line",
      len(store.load_beat_index()) == 2)
fake.files.pop(store.beat_index_path())
check("the index is rebuilt from the folders",
      len(store.rebuild_beat_index()) == 2)

print("a cell's own analyses, in its folder")
fake.calls.clear()
item = {"item_id": "2026-11-18 164000 Plot 1", "cell": "Cell 7", "kind": "beats",
        "name": "Plot 1", "saved_at": "2026-11-18T16:40:00", "notes": "day 3",
        "source": "AFM output analysis", "file": "X1.ibw",
        "results": {"Full beats": 12, "Energy, all peaks (total) (J)": 6.5e-15},
        "settings": {"hi": 0.005}}
saved = store.save_item(item, {"data.csv": "t,F\n0,1\n", "report.txt": "ok"})
folder = saved["onedrive_path"]
check("it goes in the cell's folder, under its tab",
      folder.endswith("cells/Cell 7/Beat analysis/2026-11-18 164000 Plot 1"),
      folder)
check("its files beside item.json",
      f"{folder}/data.csv" in fake.files and f"{folder}/item.json" in fake.files)
puts = [u for m, u in fake.calls if m == "PUT"]
check("item.json after the files, the index last",
      "item.json" in puts[-2] and "cell_items_index.csv" in puts[-1], puts[-3:])
back = store.load_item("Cell 7", "beats", saved["item_id"])
check("it comes back, with a file",
      back is not None and back["settings"] == {"hi": 0.005}
      and store.item_file("Cell 7", "beats", saved["item_id"], "data.csv")
      == b"t,F\n0,1\n")
store.save_item(dict(item, item_id="2026-11-18 170000 Stack", kind="stack",
                     name="Stack", results={}, saved_at="2026-11-18T17:00"),
                {"plot.html": "<html/>"})
store.save_item(dict(item, cell="Cell 8", item_id="x", kind="fourier",
                     saved_at="2026-11-18T18:00"), {"spectrum.csv": "f,A\n"})
index = store.load_items_index()
check("one line each, newest first, results as columns",
      list(index["cell"]) == ["Cell 8", "Cell 7", "Cell 7"]
      and list(index["kind"]) == ["fourier", "stack", "beats"]
      and "Full beats" in index.columns, index[["cell", "kind"]].to_dict("records"))
store.save_item(dict(item, notes="again"), {"data.csv": "t,F\n"})
check("saving it again replaces its line", len(store.load_items_index()) == 3)
check("the cells, from the analyses and the fitted cells",
      "Cell 7" in store.cell_names() and "Cell 8" in store.cell_names()
      and "Cell 04/trial 2" in store.cell_names(), store.cell_names())
fake.files.pop(store.items_index_path())
check("the index is rebuilt from the folders",
      len(store.rebuild_items_index()) == 3)
store.add_cell_files("Cell 7", {"X1.ibw": b"raw", "video.mp4": b"mp4"})
check("raw files in the cell's Files folder",
      sorted(f["name"] for f in store.cell_files("Cell 7")) == ["X1.ibw", "video.mp4"]
      and store.cell_file("Cell 7", "X1.ibw") == b"raw",
      store.cell_files("Cell 7"))
print("backups")
data, left = store.cell_archive("Cell 7")
import zipfile as _zf
names = sorted(_zf.ZipFile(io.BytesIO(data)).namelist())
check("a cell's ZIP holds its analyses and raw files, laid out as in OneDrive",
      "Files/X1.ibw" in names and "Beat analysis/2026-11-18 164000 Plot 1/data.csv"
      in names and "Beat analysis/2026-11-18 164000 Plot 1/item.json" in names
      and not left, names)
data, left = store.cell_archive("Cell 7", max_file=2)
names = _zf.ZipFile(io.BytesIO(data)).namelist()
check("files too big stay out and are named",
      "Files/video.mp4" not in names and "LEFT_OUT.txt" in names
      and any("video.mp4" in x for x in left), left)
data, _ = store.everything_archive()
names = _zf.ZipFile(io.BytesIO(data)).namelist()
check("everything: every cell, the older saves and the indexes",
      "cell_items_index.csv" in names and "beats_index.csv" in names
      and any(n.startswith("cells/Cell 8/") for n in names)
      and any(n.startswith("stacks/") for n in names), sorted(names)[:8])
saved = store.save_item(dict(item, item_id="who", analysed_by="A. Morales"),
                        {"a.csv": "x"})
check("who analysed it is kept, and is a column of the index",
      store.load_items_index().set_index("item_id").loc["who", "analysed_by"]
      == "A. Morales")
try:
    store.save_item(dict(item, cell=" "), {}); check("a cell is needed", False)
except od.OneDriveError:
    check("a cell is needed", True)

print("client id")
check("a real Application (client) ID, spaces round it or not",
      od.looks_like_client_id(" 1a2b3c4d-1111-2222-3333-444455556666 "))
check("the example text, or an id cut short, is not one",
      not od.looks_like_client_id("the ID from step 2")
      and not od.looks_like_client_id("1a2b3c4d-1111-2222-3333-44445555666")
      and not od.looks_like_client_id(None))
said = ("AADSTS700016: Application with identifier 'the ID from step 2' was "
        "not found in the directory '9188040d-6c67-4c5b-b112-36a304b66dad'.")
check("Microsoft sent the example text back: say to replace it",
      "not an ID" in od.explain(said)
      and "the ID from step 2" in od.explain(said), od.explain(said))
check("a real id it does not know: check it, then the account types",
      "personal Microsoft accounts" in od.explain(said.replace(
          "the ID from step 2", "1a2b3c4d-1111-2222-3333-444455556666")))

print()
if FAILURES:
    print(f"{len(FAILURES)} failing: {FAILURES}"); sys.exit(1)
print("all passing")
