#!/usr/bin/env python
"""Assemble the report parts into one HTML file and render it to PDF."""
import os, datetime
from weasyprint import HTML, CSS

HERE = os.path.dirname(os.path.abspath(__file__))
parts = [f"report_part{i}.html" for i in range(1, 7)]
body = "\n".join(open(os.path.join(HERE, p)).read() for p in parts)

html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>The agent that could not walk to lane</title>
<link rel="stylesheet" href="report.css">
</head><body>
{body}
</body></html>"""

out_html = os.path.join(HERE, "report.html")
open(out_html, "w").write(html)

date = datetime.date.today().isoformat()
pdf = os.path.join(HERE, f"ahriuwu_report_{date}.pdf")
HTML(filename=out_html, base_url=HERE).write_pdf(pdf)
print("wrote", pdf)
