import fasthtml.common as ft
import monsterui.all as mui
import os
import json
import glob

DATASET_DIR = os.path.join(os.path.dirname(__file__), "traces")

app, rt = mui.fast_app(hdrs=mui.Theme.blue.headers())

def list_traces():
    files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.json')]
    files.sort()  # Changed to sort in ascending order
    items = []
    for fname in files:
        path = os.path.join(DATASET_DIR, fname)
        with open(path) as f:
            data = json.load(f)
        msg = data["request"]["messages"][0]["content"]
        dt = fname.split('_')[1] + ' ' + fname.split('_')[2]
        has_open_coding = bool(data.get("open_coding", ""))
        has_axial_coding = bool(data.get("axial_coding_code", ""))
        check_mark = "✅ " if has_open_coding else ""
        check_mark += "✅ " if has_axial_coding else ""
        items.append(
            ft.Li(ft.A(f"{check_mark}{dt}: {msg[:60]}...", href=annotate.to(fname=fname), cls=mui.AT.classic))
        )
    return ft.Ul(*items, cls=mui.ListT.bullet)

@rt
def index():
    return mui.Container(
        mui.H2("Golden Dataset Traces"),
        list_traces()
    )

def chat_bubble(m):
    is_user = m["role"] == "user"
    if m["role"] == "system":
        return ft.Details(
            ft.Summary("System Prompt"),
            ft.Div(
                mui.render_md(m["content"]),
                cls="chat-bubble chat-bubble-secondary"
            ),
            cls="chat chat-start"
        )
    return ft.Div(
        ft.Div(
            mui.render_md(m["content"]),
            cls=f"chat-bubble {'chat-bubble-primary' if is_user else 'chat-bubble-secondary'}"
        ),
        cls=f"chat {'chat-end' if is_user else 'chat-start'}"
    )

def get_unique_open_coding_codes():
    codes = set()
    for fname in glob.glob(os.path.join(DATASET_DIR, '*.json')):
        with open(fname) as f:
            data = json.load(f)
        note = data.get('open_coding', '')
        if note and note.strip().lower() != 'n/a':
            # Split on double space or newlines for multiple codes, else treat as one code
            for code in note.split('\n'):
                code = code.strip()
                if code:
                    codes.add(code)
    return sorted(codes)

def get_unique_axial_coding_codes():
    codes = set()
    for fname in glob.glob(os.path.join(DATASET_DIR, '*.json')):
        with open(fname) as f:
            data = json.load(f)
        code = data.get('axial_coding_code', '')
        if code and code.strip():
            codes.add(code.strip())
    return sorted(codes)

@rt
def annotate(fname:str):
    path = os.path.join(DATASET_DIR, fname)
    with open(path) as f:
        data = json.load(f)
    chat =  data["response"]["messages"]
    bubbles = [chat_bubble(m) for m in chat]
    notes = data.get("open_coding", "")
    axial_code = data.get("axial_coding_code", "")
    axial_code_options = get_unique_axial_coding_codes()
    # Get next and previous files
    files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.json')]
    files.sort()  # Changed to sort in ascending order
    current_idx = files.index(fname)
    next_file = files[current_idx + 1] if current_idx < len(files) - 1 else files[0]
    prev_file = files[current_idx - 1] if current_idx > 0 else files[-1]
    return mui.Container(
        mui.DivFullySpaced(
            ft.A("Previous", href=annotate.to(fname=prev_file), cls=mui.AT.classic),
            ft.A("Home", href=index.to(), cls=mui.AT.classic),
            ft.A("Next", href=annotate.to(fname=next_file), cls=mui.AT.classic),
            cls="my-4"
        ),
        mui.Grid(
            ft.Div(*bubbles),
            mui.Form(
                mui.Select(
                    *[ft.Option(code, value=code) for code in axial_code_options],
                    id="axial_coding_code", icon=True, insertable=True, multiple=False,
                    value=axial_code, placeholder="Select or add axial coding (failure mode)..."
                ),
                mui.TextArea(notes, name="notes", value=notes, rows=20),
                mui.Button("Save", type="submit"),
                action=save_annotation.to(fname=fname), method="post",
                cls='w-full flex flex-col gap-2'
            ),
        ),
    )

@rt
def save_annotation(fname:str, notes:str, axial_coding_code:str=None):
    path = os.path.join(DATASET_DIR, fname)
    with open(path) as f:
        data = json.load(f)
    data["open_coding"] = notes
    if axial_coding_code is not None:
        data["axial_coding_code"] = axial_coding_code
    with open(path, "w") as f:
        json.dump(data, f)
    return ft.Redirect(annotate.to(fname=fname))

@rt
def theme():
    return mui.ThemePicker()

ft.serve()
