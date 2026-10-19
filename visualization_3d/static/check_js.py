import re, subprocess, tempfile
with open("index.html", "r") as f: html = f.read()
scripts = re.findall(r'<script>(.*?)</script>', html, re.DOTALL)
with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as temp_js:
    temp_js.write(scripts[-1])
    temp_path = temp_js.name
print(subprocess.run(['node', '--check', temp_path], capture_output=True, text=True).stderr)
