"""The web app's stylesheet. No web fonts or scripts: the page makes no
request beyond the app itself (Atkinson Hyperlegible is used when it's
installed locally)."""

from __future__ import annotations

CSS = """
:root {
  --bg: #fcfcfe; --surface: #f3f4f9; --raised: #ffffff; --hover: #f3f4f9; --selected: #eef0f7;
  --ink: #1b1d33; --body: #2a2d47; --soft: #464a66; --faint: #62668a;
  --line: #e2e4ef; --line-strong: #c9ccdf; --hairline: #eef0f7;
  --indigo: #2f3173; --link: #383cab; --madder: #b0412e; --framework: #8f93b3; --pale: #c9ccdf;
  --code-bg: #181b3b; --code-ink: #dfe2f6; --on-accent: #ffffff;
  --sans: "Atkinson Hyperlegible Next", "Atkinson Hyperlegible", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "Atkinson Hyperlegible Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0e1024; --surface: #151834; --raised: #14173a; --hover: #1c2042; --selected: #232850;
    --ink: #e8e9f4; --body: #d5d7e8; --soft: #b3b6cf; --faint: #9a9ec0;
    --line: #252a4c; --line-strong: #363b66; --hairline: #1f2346;
    --indigo: #a7abff; --link: #a7abff; --madder: #f08b74; --framework: #8f93b3; --pale: #3d4270;
    --code-bg: #0a0c1d; --code-ink: #dfe2f6; --on-accent: #0e1024;
    color-scheme: dark;
  }
}
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--sans); }
a { color: var(--link); }
a:hover { color: var(--indigo); }
:focus-visible { outline: 2px solid var(--link); outline-offset: 2px; }
.mono { font-family: var(--mono); }
.num { font-variant-numeric: tabular-nums; text-align: right; }
.faint { color: var(--faint); }
.soft { color: var(--soft); }
.shell { display: flex; flex-wrap: wrap; min-height: 100vh; }
.nav { flex: 1 1 220px; max-width: 260px; background: var(--surface); border-right: 1px solid var(--line);
  padding: 20px 14px; display: flex; flex-direction: column; gap: 22px; }
.brand { display: flex; align-items: center; gap: 10px; padding: 0 8px; font-weight: 700; font-size: 16px;
  color: var(--ink); text-decoration: none; }
.nav-group { display: flex; flex-direction: column; gap: 2px; }
.nav-label { font-size: 12px; color: var(--faint); padding: 0 10px 4px; }
.nav a.item { display: flex; align-items: center; min-height: 40px; padding: 0 10px; border-radius: 8px;
  color: var(--body); text-decoration: none; font-size: 15px; }
.nav a.item:hover { background: var(--hover); }
.nav a.item[aria-current="page"] { background: var(--raised); color: var(--ink); font-weight: 600;
  box-shadow: inset 3px 0 0 var(--indigo); }
.nav-foot { margin-top: auto; padding: 0 10px; display: flex; flex-direction: column; gap: 4px;
  font-size: 13px; color: var(--faint); overflow-wrap: anywhere; }
main { flex: 999 1 560px; min-width: 0; padding: 26px 36px 56px; display: flex; flex-direction: column; gap: 20px; }
h1 { margin: 0; font-size: 26px; font-weight: 700; letter-spacing: -0.01em; overflow-wrap: anywhere; }
h2 { margin: 0; font-size: 17px; font-weight: 700; }
h3 { margin: 0; font-size: 13px; font-weight: 600; color: var(--soft); }
p { margin: 0; }
.lede { font-size: 15px; color: var(--soft); max-width: 760px; }
.head { display: flex; flex-wrap: wrap; align-items: flex-end; justify-content: space-between; gap: 16px; }
.crumbs { display: flex; flex-wrap: wrap; gap: 8px; font-size: 14px; color: var(--soft); }
.row-gap { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.btn { display: inline-flex; align-items: center; min-height: 40px; padding: 0 14px; border: 1px solid var(--line-strong);
  border-radius: 8px; background: var(--raised); color: var(--ink); font: inherit; font-size: 14px;
  text-decoration: none; cursor: pointer; }
.btn:hover { background: var(--hover); color: var(--ink); }
.btn.primary { background: var(--indigo); border-color: var(--indigo); color: var(--on-accent); font-weight: 600; }
.chip { display: inline-flex; align-items: center; min-height: 36px; padding: 0 14px; border: 1px solid var(--line-strong);
  border-radius: 999px; background: var(--raised); color: var(--body); font-size: 14px; text-decoration: none; }
.chip[aria-current="true"] { background: var(--indigo); border-color: var(--indigo); color: var(--on-accent); font-weight: 600; }
.filters { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
.filters input[type="search"], .filters select { min-height: 42px; padding: 0 12px; border: 1px solid var(--line-strong);
  border-radius: 8px; background: var(--raised); color: var(--ink); font: inherit; font-size: 14px; }
.filters input[type="search"] { flex: 1 1 300px; min-width: 0; }
.check { display: inline-flex; align-items: center; gap: 6px; font-size: 14px; color: var(--body); }
.panel { border: 1px solid var(--line); border-radius: 10px; background: var(--raised); }
.panel.pad { padding: 18px; display: flex; flex-direction: column; gap: 14px; }
.scroll { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 14px; }
th { text-align: left; font-size: 13px; font-weight: 600; color: var(--soft); padding: 11px 14px;
  border-bottom: 1px solid var(--line); white-space: nowrap; }
td { padding: 10px 14px; border-bottom: 1px solid var(--hairline); vertical-align: middle; }
tr.link:hover td { background: var(--hover); }
td a.cell { color: inherit; text-decoration: none; display: block; }
td.clip { max-width: 420px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 999px; background: var(--indigo); }
.status { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; white-space: nowrap; }
.status.error, .status.denied { padding: 2px 9px; border-radius: 999px; background: var(--madder); color: var(--on-accent); font-weight: 600; }
.status.interrupted { padding: 1px 8px; border-radius: 999px; border: 1px solid var(--indigo); color: var(--indigo); font-weight: 600; }
.status.running { padding: 1px 8px; border-radius: 999px; border: 1px dashed var(--faint); color: var(--soft); }
.ver { font-family: var(--mono); font-size: 12px; color: var(--faint); margin-left: 6px; }
dl.meta { margin: 0; display: flex; flex-wrap: wrap; gap: 10px 28px; font-size: 14px; }
dl.meta div, dl.grid div { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
dl.meta dt, dl.grid dt { font-size: 12px; color: var(--faint); }
dl.meta dd, dl.grid dd { margin: 0; overflow-wrap: anywhere; }
dl.grid { margin: 0; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px 18px; font-size: 14px; }
.split { display: flex; flex-wrap: wrap; gap: 20px; align-items: flex-start; }
.split > .wide { flex: 3 1 600px; min-width: 0; }
.split > .side { flex: 2 1 380px; min-width: 0; display: flex; flex-direction: column; gap: 20px; }
.split > .list { flex: 1 1 250px; max-width: 340px; min-width: 0; }
.panel-head { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px;
  padding: 14px 16px; border-bottom: 1px solid var(--line); }
.legend { display: flex; flex-wrap: wrap; gap: 14px; font-size: 13px; color: var(--soft); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.swatch { display: inline-block; width: 18px; height: 4px; border-radius: 2px; }
.tl { min-width: 640px; }
.tl-row { display: grid; grid-template-columns: minmax(200px, 270px) minmax(0, 1fr) 72px; gap: 0 12px; align-items: center;
  min-height: 38px; padding: 0 16px; border-bottom: 1px solid var(--hairline); color: var(--body); text-decoration: none; font-size: 14px; }
.tl-row:hover { background: var(--hover); color: var(--body); }
.tl-row[aria-current="true"] { background: var(--selected); box-shadow: inset 3px 0 0 var(--indigo); }
.tl-axis { font-size: 12px; color: var(--faint); min-height: 30px; }
.tl-name { display: flex; align-items: center; gap: 8px; min-width: 0; }
.tl-name span.label { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.track { position: relative; height: 20px; }
.bar { position: absolute; top: 8px; height: 4px; border-radius: 2px; min-width: 2px; }
.bar.root { top: 7px; height: 6px; }
.mark { position: absolute; top: 4px; width: 10px; height: 10px; transform: translateX(-5px) rotate(45deg); border-radius: 2px; }
.k-model { background: var(--indigo); } .k-tool { background: var(--madder); }
.k-agent, .k-workflow { background: var(--pale); } .k-turn, .k-step, .k-span, .k-event { background: var(--framework); }
.k-error { background: var(--madder); }
pre.code { margin: 0; padding: 12px; border-radius: 8px; background: var(--code-bg); color: var(--code-ink);
  font-family: var(--mono); font-size: 13px; line-height: 1.55; white-space: pre-wrap; overflow-wrap: anywhere; max-height: 520px; overflow: auto; }
.msg { display: grid; grid-template-columns: 78px minmax(0, 1fr); gap: 4px 12px; padding: 10px 0; border-top: 1px solid var(--hairline); font-size: 14px; }
.msg .role { font-family: var(--mono); font-size: 12px; color: var(--faint); padding-top: 2px; }
.msg .body { white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.5; }
details summary { cursor: pointer; font-size: 13px; color: var(--soft); }
.notice { padding: 12px 14px; border-radius: 8px; border: 1px solid var(--madder); color: var(--ink); font-size: 14px; overflow-wrap: anywhere; }
.empty { padding: 28px; text-align: left; display: flex; flex-direction: column; gap: 10px; }
.score-row { display: grid; grid-template-columns: minmax(110px, 160px) 1fr; gap: 2px 14px; padding: 10px 0; border-top: 1px solid var(--hairline); font-size: 14px; }
.better { color: var(--indigo); font-weight: 600; white-space: nowrap; }
.worse { color: var(--madder); font-weight: 600; white-space: nowrap; }
.pager { display: flex; justify-content: space-between; gap: 12px; padding: 12px 14px; font-size: 14px; color: var(--soft); }
.agent-pick { display: flex; flex-direction: column; gap: 3px; padding: 12px 16px; border-bottom: 1px solid var(--hairline);
  color: var(--body); text-decoration: none; }
.agent-pick:hover { background: var(--hover); color: var(--body); }
.agent-pick[aria-current="true"] { background: var(--selected); box-shadow: inset 3px 0 0 var(--indigo); }
.sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
@media (max-width: 720px) { main { padding: 20px 16px 40px; } .nav { max-width: none; } dl.grid { grid-template-columns: minmax(0, 1fr); } }
"""
