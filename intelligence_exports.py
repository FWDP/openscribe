"""Export reviewed Intelligence findings with source references."""
import json
from pathlib import Path
from xml.sax.saxutils import escape

from transcribe import timestamp

ATTRIBUTION = "Generated with OpenScribe, an open-source project by FWDP."


def export_analysis(analysis, sources, suffix, attribution, stream):
    lookup = {source["id"]: source for source in sources}
    title = analysis["title"]
    note = "AI-assisted draft. Verify findings against the source recording before use."
    groups = []
    for section in analysis["sections"]:
        lines = []
        for finding in section["findings"]:
            refs = ", ".join(f"#{ref} {timestamp(lookup[ref]['start'])}" for ref in finding["sources"])
            kind = "user edited" if "generated_text" in finding else finding["kind"]
            lines.append(f"[{kind}] {finding['text']} ({refs})")
        groups.append((section["title"], lines))
    if suffix == "json":
        value = dict(analysis)
        cited = {ref for section in analysis["sections"] for finding in section["findings"] for ref in finding["sources"]}
        value["source_references"] = [
            {key: item for key, item in source.items() if key != "text"}
            for source in sources if source["id"] in cited
        ]
        if attribution:
            value["attribution"] = ATTRIBUTION
        return json.dumps(value, ensure_ascii=False, indent=2).encode(), "application/json"
    if suffix in {"txt", "md"}:
        md = suffix == "md"
        text = [("# " if md else "") + title, note]
        for section, lines in groups:
            text.extend((("## " if md else "") + section, "\n".join(("- " if md else "• ") + line for line in lines)))
        if attribution:
            text.append(ATTRIBUTION)
        return ("\n\n".join(text) + "\n").encode(), "text/markdown" if md else "text/plain"
    if suffix == "docx":
        from docx import Document

        document = Document()
        document.add_heading(title, 0)
        document.add_paragraph(note)
        for heading, lines in groups:
            document.add_heading(heading, level=1)
            for line in lines:
                document.add_paragraph(line, style="List Bullet")
        if attribution:
            document.add_paragraph(ATTRIBUTION)
        document.save(stream)
        return stream.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if suffix != "pdf":
        raise KeyError(suffix)

    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate

    styles = getSampleStyleSheet()
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if font.is_file():
        pdfmetrics.registerFont(TTFont("OpenScribe PDF", str(font)))
        for style in ("Title", "Heading2", "BodyText"):
            styles[style].fontName = "OpenScribe PDF"
    elements = [Paragraph(escape(title), styles["Title"]), Paragraph(note, styles["BodyText"])]
    for heading, lines in groups:
        elements.append(Paragraph(escape(heading), styles["Heading2"]))
        elements.extend(Paragraph(escape("• " + line), styles["BodyText"]) for line in lines)
    if attribution:
        elements.append(Paragraph(ATTRIBUTION, styles["BodyText"]))
    SimpleDocTemplate(stream, title=title).build(elements)
    return stream.getvalue(), "application/pdf"
