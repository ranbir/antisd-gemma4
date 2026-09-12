"""Render HUGGINGFACE_BLOG_POST.md to a self-contained HTML preview (local only).

Math is rendered client-side with KaTeX, figures are embedded as base64, and the
[FIGURE: path] / [TODO-RUN2: ...] placeholders are shown as visible callouts.
    python make_html.py  -> HUGGINGFACE_BLOG_POST.html
"""
import base64, html, pathlib, re
import markdown

src = pathlib.Path("HUGGINGFACE_BLOG_POST.md").read_text()

# 1. protect math from the markdown parser
maths = []
def stash(kind, body):
    maths.append((kind, body)); return f"\x00MATH{len(maths)-1}\x00"
src = re.sub(r"\$\$\n(.*?)\n\$\$", lambda m: stash("display", m.group(1)), src, flags=re.S)
src = re.sub(r"\\\\\((.+?)\\\\\)", lambda m: stash("inline", m.group(1).strip()), src)

# 2. figures and TODO callouts
def fig(m):
    p = pathlib.Path(m.group(1))
    if p.exists():
        b64 = base64.b64encode(p.read_bytes()).decode()
        return f'<img src="data:image/png;base64,{b64}" alt="{p.name}">'
    return f'<div class="todo">Figure to add: {p} (not rendered yet)</div>'
src = re.sub(r"^\[FIGURE: (\S+)\]$", fig, src, flags=re.M)
src = re.sub(r"^\[TODO-RUN2: (.*?)\]$", lambda m: f'<div class="todo">TODO after run 2: {html.escape(m.group(1))}</div>', src, flags=re.M | re.S)

body = markdown.markdown(src, extensions=["tables", "fenced_code"])
body = body.replace("TODO-RUN2", '<span class="todo-inline">TODO-RUN2</span>')

def unstash(m):
    kind, tex = maths[int(m.group(1))]
    tag = "div" if kind == "display" else "span"
    return f'<{tag} class="math-{kind}">{html.escape(tex)}</{tag}>'
body = re.sub("\x00MATH(\\d+)\x00", unstash, body)

page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Teaching Gemma 4 to Hesitate</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<style>
 body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;line-height:1.6;color:#1f2328;background:#fff;margin:0;padding:32px 16px}}
 main{{max-width:820px;margin:0 auto}} h1{{font-size:2rem;line-height:1.25}} h2{{margin-top:2.2em;border-bottom:1px solid #d0d7de;padding-bottom:.3em}}
 table{{border-collapse:collapse;margin:1em 0;font-size:.95em;display:block;overflow-x:auto}} th,td{{border:1px solid #d0d7de;padding:6px 12px}} th{{background:#f6f8fa}}
 pre{{background:#f6f8fa;padding:14px;border-radius:6px;overflow-x:auto;font-size:.9em}} code{{background:#f6f8fa;padding:1px 4px;border-radius:4px;font-size:.92em}} pre code{{background:none;padding:0}}
 img{{max-width:100%;border:1px solid #e3e3e3;border-radius:6px;margin:.5em 0}} hr{{border:0;border-top:1px solid #d0d7de;margin:2em 0}}
 .math-display{{margin:1em 0;text-align:center;overflow-x:auto}}
 .todo,.todo-inline{{background:#fff8c5;border:1px solid #d4a72c;color:#5a4400;border-radius:6px}} .todo{{padding:8px 12px;margin:.8em 0;font-size:.9em}} .todo-inline{{padding:0 4px;font-size:.85em}}
 blockquote{{border-left:4px solid #d0d7de;margin:0;padding:0 1em;color:#57606a}}
</style></head><body><main>
{body}
</main>
<script>
document.addEventListener("DOMContentLoaded",()=>{{
 for(const el of document.querySelectorAll(".math-inline,.math-display")){{
  const tex=el.textContent; try{{katex.render(tex,el,{{displayMode:el.classList.contains("math-display"),throwOnError:false}})}}catch(e){{el.textContent=tex}}
 }}
}});
</script></body></html>"""
pathlib.Path("HUGGINGFACE_BLOG_POST.html").write_text(page)
print("wrote HUGGINGFACE_BLOG_POST.html", len(page)//1024, "KB; math spans:", len(maths))
