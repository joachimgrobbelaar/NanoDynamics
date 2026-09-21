import re, subprocess, tempfile
with open("visualization_3d/static/index.html", "r") as f: html = f.read()
scripts = re.findall(r'<script>(.*?)</script>', html, re.DOTALL)
with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as temp_js:
    temp_js.write(scripts[-1])
    temp_path = temp_js.name
res = subprocess.run(['node', '--check', temp_path], capture_output=True, text=True)
if res.returncode != 0:
    print(res.stderr)
else:
    print("OK")
